# -*- coding: utf-8 -*-
"""
機上演算法(= 未來 ESP32 韌體的 1:1 藍本)。
鐵律:本檔案只准使用 SensorPacket 與影像,嚴禁引用任何真值。
"""
import numpy as np
import config as C

TWO_PI = 2 * np.pi
DEG_PER_PX = 360.0 / C.PANO_W


def norm(img):
    """光度正規化:零均值/單位變異,抵抗全域光照增益與偏移漂移。"""
    m = img.mean()
    s = img.std() + 1e-6
    return (img - m) / s


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


# ---------- 視覺羅盤 RIDF(FFT 圓周相關,約快 10x) ----------
def _ssd_curve(cur, ref, ref_fft=None, ref_sq=None):
    """curve[s] = mean over pixels of (roll_left(cur,s) - ref)^2。"""
    Fc = np.fft.rfft(cur, axis=1)
    Fr = np.fft.rfft(ref, axis=1) if ref_fft is None else ref_fft
    corr = np.fft.irfft(Fc * np.conj(Fr), n=cur.shape[1], axis=1).sum(0)
    sq = np.sum(cur * cur) + (np.sum(ref * ref) if ref_sq is None else ref_sq)
    return (sq - 2 * corr) / cur.size


def ridf(cur, ref, ref_fft=None, ref_sq=None):
    """回傳 (航向增量 rad, 信心, min-SSD)。符號:左轉為正。"""
    curve = _ssd_curve(cur, ref, ref_fft, ref_sq)
    W = cur.shape[1]
    s = int(np.argmin(curve))
    y0, y1, y2 = curve[(s - 1) % W], curve[s], curve[(s + 1) % W]
    denom = (y0 - 2 * y1 + y2)
    frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    shift = s + np.clip(frac, -0.5, 0.5)
    if shift > W / 2:
        shift -= W
    conf = float((np.mean(curve) - np.min(curve)) / (np.mean(curve) + 1e-9))
    return np.deg2rad(-shift * (360.0 / W)), conf, float(np.min(curve))


def img_diff(cur, ref, ref_fft=None, ref_sq=None):
    """旋轉不變影像差(Zeil 的 IDF):所有旋轉中的最小 SSD。"""
    return float(np.min(_ssd_curve(cur, ref, ref_fft, ref_sq)))


# ---------- 航向估計器(三來源 + 融合) ----------
class HeadingBank:
    GYRO_GATE = np.deg2rad(80)            # 轉太快時視覺幀因拖影不可信

    def __init__(self, mode='fused'):
        self.mode = mode                  # 'enc' | 'gyro' | 'fused'
        self.nest_locked = False
        self.th_enc = 0.0
        self.th_gyro = 0.0
        self.th_fused = 0.0
        self.ref_img = None
        self.last_conf = 0.0
        self._last_rate = 0.0

    def on_packet(self, p):
        d_enc = (p.enc_dR - p.enc_dL) / C.TRACK
        self.th_enc = wrap(self.th_enc + d_enc)
        self.th_gyro = wrap(self.th_gyro + p.gyro_z * C.DT)
        self._last_rate = p.gyro_z
        if self.mode == 'enc':
            self.th_fused = wrap(self.th_fused + d_enc)
        else:
            self.th_fused = wrap(self.th_fused + p.gyro_z * C.DT)

    def on_frame(self, img):
        self.nest_locked = False
        img = norm(img)
        if self.ref_img is None:
            self.ref_img = img.copy()     # 開機於巢,面向定義為 0°(已正規化)
            band = norm(self.ref_img[:C.COMPASS_ROWS])
            self._ref_fft = np.fft.rfft(band, axis=1)
            self._ref_sq = float(np.sum(band * band))
            self._full_fft = np.fft.rfft(self.ref_img, axis=1)
            self._full_sq = float(np.sum(self.ref_img ** 2))
            self.nest_locked = True
            return
        if self.mode != 'fused':
            return
        if abs(self._last_rate) > self.GYRO_GATE:
            return                        # 陀螺閘控:IMU 的第四個用途
        band = norm(img[:C.COMPASS_ROWS])  # 帶域獨立正規化(遠景),抑制視差
        dth, conf, dmin = ridf(band, self.ref_img[:C.COMPASS_ROWS],
                               self._ref_fft, self._ref_sq)
        self.last_conf = conf
        if dmin <= C.COMPASS_NEAR:
            # 鎖巢判定:旋轉不變的全圖 IDF,不受角度混疊影響,獨立於融合閘
            if img_diff(img, self.ref_img, self._full_fft, self._full_sq) < C.NEST_LOCK_FULL:
                self.nest_locked = True
            # 航向錨定的四重門檻:信心(黑暗/模糊)、近距(視差)、
            # 陀螺閘控(拖影)、陀螺先驗一致(牆面天際線準週期混疊會給出
            # 自信但大錯的角度;與現行估計差 >30° 一律拒收)
            if (conf >= C.RIDF_CONF_MIN
                    and abs(wrap(dth - self.th_fused)) < C.COMPASS_CONSIST):
                self.th_fused = wrap(self.th_fused + C.FUSE_K * wrap(dth - self.th_fused))

    @property
    def theta(self):
        return self.th_fused


# ---------- 路徑積分(回家向量) ----------
class PathIntegrator:
    def __init__(self, heading: HeadingBank, use_flow_rej=False):
        self.h = heading
        self.use_flow_rej = use_flow_rej
        self.p = np.zeros(2)              # 相對巢的估計位移
        self.slip_flags = 0

    def on_packet(self, p):
        ds_enc = 0.5 * (p.enc_dL + p.enc_dR)
        ds = ds_enc
        if self.use_flow_rej and p.flow_ok and abs(ds_enc - p.flow_dx) > C.SLIPREJ_THRESH:
            ds = p.flow_dx                # 編碼器與光流吵架 → 信看得到地面的那個
            self.slip_flags += 1
        th = self.h.theta
        self.p += ds * np.array([np.cos(th), np.sin(th)])

    def reset(self):
        self.p[:] = 0.0                   # 螞蟻回巢:積分器歸零

    def home_vector(self):
        return -self.p                    # 指向巢

    def dist(self):
        return float(np.hypot(*self.p))


# ---------- 行為控制器 ----------
class Controller:
    """觸鬚反射永遠最高優先。三相狀態機:
    BACK(倒退+轉開)→ SIDE(直走繞行,打破 P 控制器的死鎖對稱)→ 還控制權。"""
    def __init__(self, rng):
        self.rng = rng
        self.phase = None          # None | 'back' | 'side'
        self.pt = 0.0
        self.om_away = 0.0

    def mix(self, p, v_want, om_want):
        if self.phase is None and (p.whisk_L or p.whisk_R):
            self.om_away = -2.2 if p.whisk_L else 2.2
            if p.whisk_L and p.whisk_R:
                self.om_away = self.rng.choice([-2.2, 2.2])
            self.phase, self.pt = 'back', self.rng.uniform(0.35, 0.55)
        if self.phase == 'back':
            v_want, om_want = -0.10, self.om_away
            self.pt -= C.DT
            if self.pt <= 0:
                self.phase, self.pt = 'side', self.rng.uniform(0.9, 1.5)
        elif self.phase == 'side':
            v_want, om_want = 0.14, 0.0    # 沿新方向繞行,別立刻轉回去
            self.pt -= C.DT
            if self.pt <= 0:
                self.phase = None
            if p.whisk_L or p.whisk_R:     # 繞行中再碰 → 重新反射
                self.phase = None
        vL = v_want - om_want * C.TRACK / 2
        vR = v_want + om_want * C.TRACK / 2
        return vL, vR

    @property
    def busy(self):
        return self.phase is not None


class Forager:
    """相關隨機遊走(OU 過程轉向)= 覓食。"""
    def __init__(self, rng):
        self.rng = rng
        self.om = 0.0

    def act(self):
        self.om += (-self.om * 0.5 + 1.4 * self.rng.standard_normal()) * C.DT * 4
        self.om = np.clip(self.om, -1.5, 1.5)
        return C.FORAGE_V, self.om


class Homer:
    """沿回家向量的 P 控制。"""
    def __init__(self, pi: PathIntegrator):
        self.pi = pi

    def act(self):
        hv = self.pi.home_vector()
        if self.pi.dist() < C.HOME_STOP_EST:
            return 0.0, 0.0, True
        tgt = np.arctan2(hv[1], hv[0])
        err = wrap(tgt - self.pi.h.theta)
        om = np.clip(2.5 * err, -2.0, 2.0)
        v = C.HOMING_V * max(0.15, np.cos(err))
        return v, om, False


class SearchSpiral:
    """Wehner 式系統性搜尋:在 PI 座標系繞「相信的巢」畫擴張同心圈。
    PI 閉環 → 打滑不會讓螺旋漂走;這正是沙漠螞蟻的搜尋幾何。"""
    def __init__(self, pi: PathIntegrator, v=0.12, r0=0.05, growth=0.008):
        self.pi = pi
        self.v = v
        self.r = r0
        self.growth = growth
        self.phi = np.arctan2(self.pi.p[1], self.pi.p[0])  # 從目前方位角切入

    def act(self):
        self.r += self.growth * C.DT
        self.phi += (self.v / max(self.r, 0.04)) * C.DT
        target = self.r * np.array([np.cos(self.phi), np.sin(self.phi)])
        err_v = target - self.pi.p
        tgt_th = np.arctan2(err_v[1], err_v[0])
        e = wrap(tgt_th - self.pi.h.theta)
        om = np.clip(3.0 * e, -1.6, 1.6)
        v = self.v * max(0.2, np.cos(e))
        return v, om


class SnapshotHomer:
    """run-and-tumble:沿旋轉不變影像差(IDF)的梯度下降。
    判準:與「本段航程最佳值」比,惡化 >6% 才翻滾(≈直行 6cm 的梯度量,
    遠高於幀雜訊);翻滾=快速短促重定向;停滯 >6s 觸發大翻滾逃離平原。"""
    MARGIN = 1.06
    def __init__(self, rng, snapshot):
        self.rng = rng
        self.snap = norm(snapshot)
        self.sfft = np.fft.rfft(self.snap, axis=1)
        self.ssq = float(np.sum(self.snap ** 2))
        self.leg_best = np.inf            # 本段航程最佳 D
        self.glob_best = np.inf
        self.stall = 0.0                  # 距上次全域改善的時間
        self.tumble_t = 0.0
        self.tumble_om = 0.0
        self.arrived = False

    def on_frame(self, img):
        D = img_diff(norm(img), self.snap, self.sfft, self.ssq)
        if D < 0.040:                     # 到家門檻(正規化):3cm→0.027, 6cm→0.055
            self.arrived = True
        if D < self.glob_best * 0.98:
            self.glob_best = D
            self.stall = 0.0
        if D > self.leg_best * self.MARGIN and self.tumble_t <= 0:
            self.tumble_t = self.rng.uniform(0.30, 0.60)
            self.tumble_om = self.rng.choice([-1, 1]) * self.rng.uniform(1.8, 3.0)
            self.leg_best = D             # 新段航程以當前值起算
        else:
            self.leg_best = min(self.leg_best, D)

    def act(self):
        if self.arrived:
            return 0.0, 0.0
        self.stall += C.DT
        if self.stall > 6.0:              # 平原逃脫:大翻滾 + 重設基準
            self.tumble_t = self.rng.uniform(0.5, 0.9)
            self.tumble_om = self.rng.choice([-1, 1]) * self.rng.uniform(2.0, 3.2)
            self.leg_best = np.inf
            self.stall = 0.0
        if self.tumble_t > 0:
            self.tumble_t -= C.DT
            return 0.03, self.tumble_om
        return 0.17, 0.0


class RouteFollower:
    """完美記憶版熟悉度循路(Baddeley 2012 的基準線):
    對 ±67.5° 的候選航向捲動當前影像,問「哪個方向最眼熟」。"""
    def __init__(self, views):
        self.views = np.stack([norm(v) for v in views])    # (N,48,64) 正規化
        self.vfft = np.fft.rfft(self.views, axis=2)        # 預算 FFT
        self.vsq = np.sum(self.views ** 2, axis=(1, 2))    # (N,)
        self.shifts = np.arange(-12, 13)                   # px, 5.625°/px

    def steer(self, img):
        img = norm(img)
        Fc = np.fft.rfft(img, axis=1)                      # (48,33)
        corr = np.fft.irfft(Fc[None] * np.conj(self.vfft), n=C.PANO_W,
                            axis=2).sum(1)                 # (N,64) 對每個左旋 s
        ssd = (np.sum(img * img) + self.vsq[:, None] - 2 * corr) / img.size
        # 左旋 s 像素 = 假想左轉 s*5.625°;取 shifts 視窗
        cols = self.shifts % C.PANO_W
        window = ssd[:, cols]                              # (N,25)
        fam = window.min(axis=0)                           # (25,)
        k = int(np.argmin(fam))
        dth = np.deg2rad(self.shifts[k] * DEG_PER_PX)
        return dth, float(fam[k])
