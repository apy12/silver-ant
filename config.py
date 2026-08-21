# -*- coding: utf-8 -*-
"""
蟻級單元模擬參數。每一項雜訊都對應實機的真實缺陷來源。
單位:公尺、秒、弧度(介面註明者除外)。

[2026-08 合併說明] world.py review 版附帶的 config 曾把下列世界參數改掉:
場地 4×3、圓筒 3 支大筒、CAM_H 0.06、EL 35/−25°、PIX_NOISE 0.01、
LIGHT_TAU 5、BLUR 0.020、VISION_FPS 30。全部退回本檔的凍結規格值——
README 的門檻表與四張實驗表都是在這組值下校定/實測的。
本檔已含新 world.py(World/PanoCamera 拆分)需要的全部欄位,無需增補。
"""
import numpy as np

DT = 0.005                 # 物理/即時層 200 Hz
VISION_FPS = 12            # 全景視覺幀率(ESP32-S3 實測可行區間)
DECISION_HZ = 5            # 導航決策層

# ---- 機構(對照凍結規格) ----
WHEEL_R   = 0.017          # N20 輪半徑 34mm/2
TRACK     = 0.100          # 輪距
CPR       = 700            # 編碼器 counts/圈(7ppr x 1:100)
V_MAX     = 0.30           # 輪面極速(N20 1:100 有載保守值)
A_TAU     = 0.08           # 馬達一階響應時間常數
BODY_R    = 0.065          # 車體半徑 13cm/2
WHISKER_REACH = 0.10       # 觸鬚尖端離車心距離
# 已知盲區:觸鬚在 ±0.6 rad,尖端橫偏 ~5.6cm;正面直衝最小圓筒
# (觸發半徑 0.05)可從兩鬚之間溜過。實機考慮第三根中央觸鬚。

# ---- 打滑模型(里程計的頭號敵人) ----
SLIP_BASE_RATE   = 0.06    # 每秒發生打滑事件的基礎機率
SLIP_TURN_GAIN   = 0.5     # 角速度每 rad/s 增加的事件率
SLIP_DUR         = (0.10, 0.30)   # 事件持續 (s)
SLIP_FACTOR      = (0.35, 0.80)   # 事件中輪子對地有效位移比例
SLIP_MICRO_STD   = 0.006   # 常態微打滑(每步乘性雜訊 σ)

# ---- MPU6050 Z 軸陀螺(開機靜置 2s 校零後的殘差) ----
GYRO_BIAS0_STD   = np.deg2rad(0.03)    # 校零殘餘偏置 σ (rad/s)
GYRO_RW_STD      = np.deg2rad(0.012)   # 偏置隨機游走 σ,含溫漂 (rad/s/√s)
GYRO_NOISE_STD   = np.deg2rad(0.06)    # 取樣白雜訊 σ (rad/s @200Hz)
GYRO_SCALE_ERR   = 0.01                # 刻度誤差 ±1%(每台固定)

# ---- PAA5100 光流(車腹朝下,離地 22mm) ----
FLOW_SCALE_STD   = 0.03    # 每台安裝高度造成的刻度誤差 σ
FLOW_NOISE_STD   = 0.0025  # 速度量測白雜訊 σ (m/s)
FLOW_DROP_RATE   = 0.02    # 表面品質(squal)掉線事件率 (每秒)
FLOW_DROP_DUR    = (0.2, 0.8)
# 安裝位置(相對旋轉中心,車體座標;校定常數,地位同 TRACK)
# 旋轉時感測點速度 = (v − ω·y, ω·x) → dx 通道污染 = −ω·FLOW_MOUNT_Y
# 前向偏移 x 只污染側向通道(封包未用);側向偏移 y 直接毒 dx。
# 校定法:smoke_test 第 5 段(原地旋轉,flow_dx/DT 對 gyro 回歸,斜率=−y)
FLOW_MOUNT_X     = 0.030   # m,佔位:由機構圖回填
FLOW_MOUNT_Y     = 0.020   # m,佔位:由機構圖/第5段校定回填

# ---- 全景視覺(OV2640 朝天 + 球面鏡,展開 64x48 灰階) ----
PANO_W, PANO_H   = 64, 48
CAM_H            = 0.12                 # 鏡球等效視點高度
EL_TOP, EL_BOT   = np.deg2rad(55), np.deg2rad(-5)
PIX_NOISE        = 4.0 / 255.0          # 像素雜訊 σ
LIGHT_GAIN_STD   = 0.05                 # 慢變全域增益漂移
LIGHT_TAU        = 8.0                  # 光照漂移時間常數 (s)
BLUR_EXPOSURE    = 1.0 / 30.0           # 等效曝光,快轉時造成水平拖影
RIDF_CONF_MIN    = 0.22    # (mean-min)/mean 低於此值視為不可信
COMPASS_ROWS     = 30      # 羅盤只用上半帶(仰角 >~17.5°),抑制近物視差
# ⚠ COMPASS_ROWS 的語義綁定 EL_TOP/EL_BOT 的仰角柵格:
#   第 30 行仰角 = 55 − 30×(60/47) ≈ 16.7°。改 EL 範圍必須重推此值,
#   否則帶域會含地板/近物,教訓 1 的視差抑制失效(且不會報錯)。
COMPASS_NEAR     = 0.60    # 上半帶 min-SSD(正規化單位):錨定半徑約 6cm
NEST_LOCK_FULL   = 0.060   # 全圖 IDF(正規化):視覺確認在巢→重置 PI,半徑約 6cm

# ---- 融合與導航 ----
FUSE_K           = 0.15    # 視覺羅盤互補濾波增益(每有效幀)
COMPASS_CONSIST  = np.deg2rad(30)  # 視覺角度須與陀螺先驗一致才融合(防混疊)
SLIPREJ_THRESH   = 0.0004  # |ds_enc - ds_flow| 超過即改信光流 (m/步 @200Hz)
HOME_STOP_EST    = 0.03    # 回家向量模長低於此值即宣告到家
FORAGE_V         = 0.20
HOMING_V         = 0.18

# ---- 場地 ----
ARENA_W, ARENA_H = 3.0, 2.0
HOME             = np.array([0.60, 0.60])
# 黑色圓筒地標 (x, y, 半徑, 高)
CYLINDERS = [
    (1.05, 1.45, 0.050, 0.30),
    (2.20, 0.50, 0.050, 0.30),
    (2.45, 1.55, 0.060, 0.35),
    (1.60, 1.00, 0.040, 0.25),
    (0.55, 1.55, 0.045, 0.28),
]
WALL_H_BASE      = 0.32    # 牆面「山稜線」紙帶平均高
WALL_H_AMP       = 0.14
SKYLINE_SEED     = 7       # 稜線輪廓固定亂數種子(印出來就不會變)
