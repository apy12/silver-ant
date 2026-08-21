"""模擬器常數。"""
import math

# ---- 場地 ----
ARENA_W = 4.0          # m
ARENA_H = 3.0          # m
SKYLINE_SEED = 7       # 稜線種子 = 場地版本號:改了等於換一個世界的地標
WALL_H_BASE = 0.50     # m
WALL_H_AMP = 0.15      # m

# (cx, cy, r, h)
CYLINDERS = [
    (1.0, 1.0, 0.20, 0.45),
    (3.0, 2.2, 0.25, 0.60),
    (2.2, 0.8, 0.15, 0.35),
]

# ---- 相機 / 全景(PanoCamera 的預設值)----
PANO_W = 64
PANO_H = 48
CAM_H = 0.06                    # m
EL_TOP = math.radians(35.0)     # 行 0
EL_BOT = math.radians(-25.0)    # 最末行
VISION_FPS = 30.0

# ---- 光照 / 雜訊 ----
LIGHT_TAU = 5.0        # s:光照 OU 回復時間常數(漂多慢)
LIGHT_GAIN_STD = 0.05  # 光照增益穩態標準差(漂多大)
PIX_NOISE = 0.01
BLUR_EXPOSURE = 0.020  # s
