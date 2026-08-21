# -*- coding: utf-8 -*-
"""
物理 + 感測。演算法端(brain)只拿得到 SensorPacket 與 Frame,拿不到真值。
真值只交給實驗腳本當裁判(= 天花板相機的角色)。

[相容性修補 2026-08] 對齊新版 world.py(World / PanoCamera 拆分):
  1. render 搬到 PanoCamera,sim 自建相機並改呼叫簽名。
  2. render 回傳 (img, valid_light);grab_frame 解包,對 brain 仍只回傳 img,
     光照有效旗標存於 self.last_frame_valid。
  3. 每個物理 tick 呼叫 world.step(DT, rng) 推進光照 OU 漂移
     (新架構下不呼叫 = 增益永遠凍在 1.0,教訓 3 的雜訊被偷偷閹掉)。
  4. 動態模糊改用「打滑後」的車體真實角速度(舊版用輪面差速,打滑中拖影量錯)。
  5. 補圓筒滑動碰撞:新 render 假設車永不在筒內(在筒內該筒直接消失),
     這個契約必須由物理保證;觸鬚只看前方 ±60°,盲倒原本可倒穿圓筒。
  6. 光流改量「約束後」的真實車體位移(投影到航向)。舊實作用約束前的
     v·DT,頂著牆/圓筒時車沒動、光流卻回報在動——恰好是光流最該抓的
     情境。無接觸時數值與原本逐位元相同,只有接觸事件變誠實。
"""
import numpy as np
from dataclasses import dataclass
import config as C
from world import World, PanoCamera, wrap


@dataclass
class SensorPacket:            # 200 Hz,對應實機序列協定上行封包
    t: float
    enc_dL: float              # 左輪編碼器換算的輪面位移(含量化)
    enc_dR: float
    gyro_z: float              # rad/s
    flow_dx: float             # 車體前向位移 (m),squal 低時為 0
    flow_ok: bool
    whisk_L: bool
    whisk_R: bool
    vbat: float


class Sim:
    def __init__(self, seed=0, start=None, heading=0.0, lights_on=True):
        self.rng = np.random.default_rng(seed)
        self.world = World(lights_on=lights_on)
        self.cam = PanoCamera()            # [修補 1] 相機與世界解耦
        p = C.HOME.copy() if start is None else np.array(start, float)
        self.x, self.y, self.th = float(p[0]), float(p[1]), heading
        self.t = 0.0
        self.wL = self.wR = 0.0            # 實際輪面速度
        self.cmdL = self.cmdR = 0.0
        # 每台固定的個體差(出廠校正殘差)
        self.gyro_scale = 1.0 + self.rng.normal(0, C.GYRO_SCALE_ERR)
        self.gyro_bias = self.rng.normal(0, C.GYRO_BIAS0_STD)
        self.flow_scale = 1.0 + self.rng.normal(0, C.FLOW_SCALE_STD)
        # 打滑/掉線事件狀態
        self.slip = {'L': [0.0, 1.0], 'R': [0.0, 1.0]}   # [剩餘時間, 因子]
        self.flow_drop = 0.0
        self.encL_cnt = 0; self.encR_cnt = 0
        self.encL_pos = 0.0; self.encR_pos = 0.0
        self.vbat = 4.1
        self._frame_due = 0.0
        self._om_true = 0.0                # [修補 4] 打滑後車體真實角速度(供模糊)
        self.last_frame_valid = True       # [修補 2] 最近一幀的光照有效旗標
        self.gt_path = []                  # 裁判用軌跡

    # ---- 事件 ----
    def _slip_update(self, key, omega):
        st = self.slip[key]
        if st[0] > 0:
            st[0] -= C.DT
            if st[0] <= 0:
                st[1] = 1.0
        else:
            rate = C.SLIP_BASE_RATE + C.SLIP_TURN_GAIN * abs(omega)
            if self.rng.random() < rate * C.DT:
                st[0] = self.rng.uniform(*C.SLIP_DUR)
                st[1] = self.rng.uniform(*C.SLIP_FACTOR)
        return st[1] * (1.0 + self.rng.normal(0, C.SLIP_MICRO_STD))

    # ---- 主步進 ----
    def step(self, cmdL, cmdR):
        self.cmdL = np.clip(cmdL, -C.V_MAX, C.V_MAX)
        self.cmdR = np.clip(cmdR, -C.V_MAX, C.V_MAX)
        a = C.DT / C.A_TAU
        self.wL += (self.cmdL - self.wL) * a
        self.wR += (self.cmdR - self.wR) * a
        omega_pre = (self.wR - self.wL) / C.TRACK
        gL = self.wL * self._slip_update('L', omega_pre)
        gR = self.wR * self._slip_update('R', omega_pre)
        v = 0.5 * (gL + gR)
        om = (gR - gL) / C.TRACK
        self._om_true = om                 # [修補 4]
        # [修補 3] 推進世界狀態(光照 OU 漂移)。新架構下這是唯一入口。
        self.world.step(C.DT, self.rng)
        # 位姿積分 + 牆面滑動碰撞
        ox, oy, oth = self.x, self.y, self.th        # [修補 6] 約束前位姿
        nx = self.x + v * np.cos(self.th) * C.DT
        ny = self.y + v * np.sin(self.th) * C.DT
        nx = float(np.clip(nx, C.BODY_R, C.ARENA_W - C.BODY_R))
        ny = float(np.clip(ny, C.BODY_R, C.ARENA_H - C.BODY_R))
        # [修補 5] 圓筒滑動碰撞:徑向推出到表面(render 契約:車永不在筒內)。
        for (cx0, cy0, r, _h) in C.CYLINDERS:
            dx, dy = nx - cx0, ny - cy0
            d = float(np.hypot(dx, dy))
            keep = r + C.BODY_R
            if d < keep:
                if d < 1e-9:               # 幾何退化:任意方向推出
                    dx, dy, d = 1.0, 0.0, 1.0
                nx = cx0 + dx / d * keep
                ny = cy0 + dy / d * keep
        # 推出後再夾一次牆(目前場地圓筒離牆都 > BODY_R,此步為保險)
        nx = float(np.clip(nx, C.BODY_R, C.ARENA_W - C.BODY_R))
        ny = float(np.clip(ny, C.BODY_R, C.ARENA_H - C.BODY_R))
        self.x, self.y = nx, ny
        self.th = wrap(self.th + om * C.DT)
        self.t += C.DT
        self.gt_path.append((self.t, self.x, self.y, self.th))
        # [修補 6] 約束後的真實前向位移(供光流;無接觸時 == v*DT)
        ds_true = (self.x - ox) * np.cos(oth) + (self.y - oy) * np.sin(oth)
        # ---- 感測 ----
        # 編碼器量的是「輪子轉了多少」(馬達有轉就有數),量化到 CPR
        self.encL_pos += self.wL * C.DT
        self.encR_pos += self.wR * C.DT
        circ = 2 * np.pi * C.WHEEL_R
        newL = int(self.encL_pos / circ * C.CPR)
        newR = int(self.encR_pos / circ * C.CPR)
        dL = (newL - self.encL_cnt) * circ / C.CPR
        dR = (newR - self.encR_cnt) * circ / C.CPR
        self.encL_cnt, self.encR_cnt = newL, newR
        # 陀螺
        self.gyro_bias += self.rng.normal(0, C.GYRO_RW_STD * np.sqrt(C.DT))
        gz = om * self.gyro_scale + self.gyro_bias \
            + self.rng.normal(0, C.GYRO_NOISE_STD)
        # 光流:看的是「車體對地」真實位移(含牆/圓筒接觸時的靜止)
        if self.flow_drop > 0:
            self.flow_drop -= C.DT
            fdx, fok = 0.0, False
        else:
            if self.rng.random() < C.FLOW_DROP_RATE * C.DT:
                self.flow_drop = self.rng.uniform(*C.FLOW_DROP_DUR)
            # [修補 7] 橋一:光流裝在轉心外 → dx 讀到切向污染 −ω·mount_y
            ds_at_sensor = ds_true - self._om_true * C.FLOW_MOUNT_Y * C.DT
            fdx = ds_at_sensor * self.flow_scale + self.rng.normal(0, C.FLOW_NOISE_STD * C.DT)
            fok = True
        # 觸鬚:牆或圓筒在左右前 60° 扇區內進入觸及範圍
        wl = wr = False
        for sgn, name in ((+1, 'L'), (-1, 'R')):
            tip = np.array([self.x, self.y]) + C.WHISKER_REACH * np.array(
                [np.cos(self.th + sgn * 0.6), np.sin(self.th + sgn * 0.6)])
            hit = (tip[0] < 0.01 or tip[0] > C.ARENA_W - 0.01
                   or tip[1] < 0.01 or tip[1] > C.ARENA_H - 0.01)
            if not hit:
                for (cx0, cy0, r, h) in C.CYLINDERS:
                    if np.hypot(tip[0] - cx0, tip[1] - cy0) < r + 0.01:
                        hit = True; break
            if name == 'L':
                wl = hit
            else:
                wr = hit
        self.vbat -= 1.2e-6 * (abs(self.wL) + abs(self.wR)) / C.V_MAX
        return SensorPacket(self.t, dL, dR, gz, fdx, fok, wl, wr, self.vbat)

    def frame_ready(self):
        if self.t >= self._frame_due:
            self._frame_due = self.t + 1.0 / C.VISION_FPS
            return True
        return False

    def grab_frame(self):
        # [修補 1,2,4] 新簽名 cam.render(world, ...);解包 tuple;用真實 ω 算拖影。
        img, ok = self.cam.render(self.world, self.x, self.y, self.th,
                                  self._om_true, self.rng)
        self.last_frame_valid = ok
        return img

    # ---- 裁判 ----
    def gt_pose(self):
        return self.x, self.y, self.th

    def gt_dist_home(self):
        return float(np.hypot(self.x - C.HOME[0], self.y - C.HOME[1]))
