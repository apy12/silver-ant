"""場地幾何(World)與全景相機(PanoCamera)。

架構:
  World      = 幾何 + 隨時間演化的狀態(光照增益)。
               狀態只由 step(dt) 推進,與拍照無關。
               純查詢:wall_hit / skyline_at / clearance。
  PanoCamera = 感測器參數 + 純函數 render(world, pose):
               讀世界、算影像,不改動任何狀態。

用法:
    world = World()
    cam   = PanoCamera()                      # 參數預設取自 config
    ...
    world.step(dt, rng)                       # 每個模擬 tick 推進世界
    img, valid = cam.render(world, x, y, theta, omega, rng)

影像約定:48x64 灰階,行 0 = 仰角最高;col 0 = 車頭,欄位隨逆時針方位遞增。
黑色剪影在亮背景上,所有物體同一種黑 → 剪影取聯集即可,不需深度排序。
"""
import numpy as np
import config as C


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ======================================================================
# 世界:幾何 + 隨時間演化的狀態
# ======================================================================
class World:
    def __init__(self, lights_on=True):
        self.lights_on = lights_on
        rng = np.random.default_rng(C.SKYLINE_SEED)
        # 牆面稜線:沿四面牆周長參數 s 的固定隨機輪廓(印好貼上的紙帶)
        self.perim = 2 * (C.ARENA_W + C.ARENA_H)
        n = 256
        prof = np.zeros(n)
        for k in range(1, 7):
            prof += rng.uniform(0.2, 1.0) / k * np.sin(
                2 * np.pi * k * np.arange(n) / n + rng.uniform(0, 2 * np.pi))
        prof = prof / np.max(np.abs(prof))
        self.sky_s = np.linspace(0, self.perim, n, endpoint=False)
        self.sky_h = C.WALL_H_BASE + C.WALL_H_AMP * prof
        self.cyl = np.array(C.CYLINDERS)  # (N,4):cx, cy, r, h
        self._gain = 1.0                  # 光照慢漂移狀態

    # ---- 狀態演化(唯一會改動狀態的入口)----
    def reset(self):
        """episode 之間重置光照狀態。"""
        self._gain = 1.0

    def step(self, dt, rng):
        """推進世界 dt 秒。光照增益依 OU 過程漂移:
        g += −(g−1)·dt/τ + σ·√(2dt/τ)·ξ
        → 穩態 std = LIGHT_GAIN_STD,回復時間常數 = LIGHT_TAU 秒。"""
        self._gain += (-(self._gain - 1.0) * dt / C.LIGHT_TAU
                       + C.LIGHT_GAIN_STD * np.sqrt(2 * dt / C.LIGHT_TAU)
                       * rng.standard_normal())

    @property
    def gain(self):
        return self._gain

    # ---- 純查詢:幾何 ----
    def wall_hit(self, x, y, alpha):
        """向量化射線投射:方位角 alpha(K,) 對四面牆的最近命中。
        回傳 (距離, 周長參數 s)。s 沿下→右→上→左牆逆時針累計,牆角連續。
        量的是視線長度,不是車身淨空(淨空用 clearance)。"""
        cx, cy = np.cos(alpha), np.sin(alpha)
        eps = 1e-9
        ds, ss = [], []
        # 濾網 1:方向——光朝該牆飛才解該牆
        with np.errstate(divide='ignore', invalid='ignore'):
            t_r = np.where(cx > eps, (C.ARENA_W - x) / cx, np.inf)
            t_l = np.where(cx < -eps, (0.0 - x) / cx, np.inf)
            t_t = np.where(cy > eps, (C.ARENA_H - y) / cy, np.inf)
            t_b = np.where(cy < -eps, (0.0 - y) / cy, np.inf)
        for t, which in ((t_b, 0), (t_r, 1), (t_t, 2), (t_l, 3)):
            # 濾網 2:負 t——不撞身後(僅位置在場外時可能觸發)
            t = np.where(t >= 0.0, t, np.inf)
            tf = np.where(np.isfinite(t), t, 0.0)
            hx, hy = x + tf * cx, y + tf * cy
            ok = np.isfinite(t)
            # 濾網 3:牆段——交點須落在實際牆段內(1e-6 容忍牆角浮點誤差)
            if which == 0:
                ok &= (hx >= -1e-6) & (hx <= C.ARENA_W + 1e-6)
                s = np.clip(hx, 0, C.ARENA_W)
            elif which == 1:
                ok &= (hy >= -1e-6) & (hy <= C.ARENA_H + 1e-6)
                s = C.ARENA_W + np.clip(hy, 0, C.ARENA_H)
            elif which == 2:
                ok &= (hx >= -1e-6) & (hx <= C.ARENA_W + 1e-6)
                s = C.ARENA_W + C.ARENA_H + (C.ARENA_W - np.clip(hx, 0, C.ARENA_W))
            else:
                ok &= (hy >= -1e-6) & (hy <= C.ARENA_H + 1e-6)
                s = 2 * C.ARENA_W + C.ARENA_H + (C.ARENA_H - np.clip(hy, 0, C.ARENA_H))
            ds.append(np.where(ok, t, np.inf))
            ss.append(s)
        ds = np.stack(ds)          # (4,K)
        ss = np.stack(ss)
        j = np.argmin(ds, axis=0)
        k = np.arange(alpha.shape[0])
        return ds[j, k], ss[j, k]

    def skyline_at(self, s):
        """周長位置 s 的牆頂高度(等距取樣 → floor 到所在區間)。"""
        step = self.perim / len(self.sky_s)
        idx = np.floor((s % self.perim) / step).astype(int) % len(self.sky_s)
        return self.sky_h[idx]

    def clearance(self, x, y):
        """點 (x,y) 到最近障礙物表面的距離(供物理 / 碰撞偵測用)。
        檢查頻率應跟物理步,而非視覺幀率。"""
        d_wall = min(x, C.ARENA_W - x, y, C.ARENA_H - y)
        d_cyl = min((float(np.hypot(cx - x, cy - y) - r)
                     for cx, cy, r, _h in self.cyl), default=np.inf)
        return min(d_wall, d_cyl)


# ======================================================================
# 相機:感測器參數 + 純 render
# ======================================================================
class PanoCamera:
    """全景剪影相機。render() 是純函數:不改動 world,也不改動自己。

    參數預設取自 config,可逐台覆寫(除錯相機、域隨機化)。
    參數在建構時複製——執行期改 config 不影響既有相機,要不同參數就建新相機。

        width, height   影像欄 / 行數
        cam_h           鏡頭離地高度 [m]
        el_top, el_bot  行 0 與最末行的仰角 [rad]
        blur_exposure   旋轉模糊等效曝光 [s]
        pix_noise       高斯像素雜訊 std
        quantize        是否做 8-bit 量化
        levels          (背景, 地板, 剪影, 關燈) 亮度
    """

    def __init__(self, width=None, height=None, cam_h=None,
                 el_top=None, el_bot=None, blur_exposure=None,
                 pix_noise=None, quantize=True,
                 levels=(0.80, 0.48, 0.10, 0.02)):
        self.W = C.PANO_W if width is None else width
        self.H = C.PANO_H if height is None else height
        self.cam_h = C.CAM_H if cam_h is None else cam_h
        self.el_top = C.EL_TOP if el_top is None else el_top
        self.el_bot = C.EL_BOT if el_bot is None else el_bot
        self.blur_exposure = (C.BLUR_EXPOSURE if blur_exposure is None
                              else blur_exposure)
        self.pix_noise = C.PIX_NOISE if pix_noise is None else pix_noise
        self.quantize = quantize
        self.lv_bg, self.lv_floor, self.lv_dark, self.lv_off = levels

    def render(self, world, x, y, theta, omega=0.0, rng=None):
        """回傳 (img[H x W] float 0..1, valid_light)。
        valid_light:影像是否含有效光照訊號(關燈時 False)。"""
        if rng is None:
            rng = np.random.default_rng()
        K = self.W
        beta = np.arange(K) * 2 * np.pi / K   # 車體方位,col 0 = 車頭,逆時針遞增
        alpha = theta + beta                  # 世界方位
        # 牆剪影:射線得距離與周長位置 → 查稜線 → 換算頂/底仰角
        dw, sw = world.wall_hit(x, y, alpha)
        hw = world.skyline_at(sw)
        top_w = np.arctan2(hw - self.cam_h, dw)
        bot_w = np.arctan2(0.0 - self.cam_h, dw)
        tops = [top_w]; bots = [bot_w]; cover = [np.ones(K, bool)]
        # 圓筒剪影:閉式解——角半寬 arcsin(r/dc)、距離 dc−r(平頂近似)
        for (cx0, cy0, r, h) in world.cyl:
            dx, dy = cx0 - x, cy0 - y
            dc = np.hypot(dx, dy)
            if dc <= r + 1e-6:                # 車在筒內:交由物理保證不發生
                continue
            phi = np.arctan2(dy, dx)
            gam = np.arcsin(np.clip(r / dc, 0, 1))
            cov = np.abs(wrap(alpha - phi)) <= gam
            d = max(dc - r, 1e-3)
            tops.append(np.full(K, np.arctan2(h - self.cam_h, d)))
            bots.append(np.full(K, np.arctan2(0.0 - self.cam_h, d)))
            cover.append(cov)
        el = np.linspace(self.el_top, self.el_bot, self.H)[:, None]   # (H,1)
        dark = np.zeros((self.H, K), bool)
        for t, b, cov in zip(tops, bots, cover):
            dark |= cov[None, :] & (el <= t[None, :]) & (el >= b[None, :])
        img = np.full((self.H, K), self.lv_bg)
        img[el[:, 0] < 0, :] = self.lv_floor
        img[dark] = self.lv_dark
        # 光照:讀世界狀態,不推進它(gain 屬場景亮度,故在感測雜訊之前)
        if world.lights_on:
            img = img * world.gain
        else:
            img = np.full_like(img, self.lv_off)
        # 旋轉動態模糊:曝光期間特徵掃過欄位 [β, β+ωT] → 拖影方向與 ω 同號
        blur_px = abs(omega) * self.blur_exposure / (2 * np.pi / K)
        n = int(blur_px)
        if n >= 1:
            sgn = 1 if omega >= 0 else -1
            acc = img.copy()
            for i in range(1, n + 1):
                acc += np.roll(img, sgn * i, axis=1)
            img = acc / (n + 1)
        # 感測雜訊 + 8-bit 量化
        if self.pix_noise > 0:
            img = img + rng.standard_normal(img.shape) * self.pix_noise
        img = np.clip(img, 0, 1)
        if self.quantize:
            img = np.round(img * 255) / 255.0
        return img, world.lights_on
