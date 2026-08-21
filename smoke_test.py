# -*- coding: utf-8 -*-
"""校定工具:符號自檢、FFT 正確性、正規化門檻量測、打滑統計。
實機版流程相同:把機器人放在巢、原地已知角度旋轉、再已知小位移平移,
用本腳本的量測邏輯重新填 config 裡的三個門檻。"""
import numpy as np
import config as C
from sim import Sim
from brain import norm, ridf, img_diff, _ssd_curve

rng = np.random.default_rng(1)
s = Sim(seed=1, start=(1.5, 1.0), heading=0.0)
ref = s.world.render(1.5, 1.0, 0.0, 0.0, rng)

print('--- 1) RIDF 符號與精度(左轉為正) ---')
idx = (np.arange(64)[None, :] + np.arange(64)[:, None]) % 64
cur30 = s.world.render(1.5, 1.0, np.deg2rad(30), 0.0, rng)
brute = np.mean((norm(cur30)[:, idx] - norm(ref)[:, None, :]) ** 2, axis=(0, 2))
assert np.allclose(brute, _ssd_curve(norm(cur30), norm(ref)), atol=1e-10), 'FFT != 暴力法'
print('FFT 圓周相關 == 暴力法  OK')
for d in [10, 30, -25, 80]:
    cur = s.world.render(1.5, 1.0, np.deg2rad(d), 0.0, rng)
    dth, conf, _ = ridf(norm(cur), norm(ref))
    print(f'  true {d:+4d}°  ridf {np.rad2deg(dth):+7.2f}°  conf {conf:.3f}')

print('--- 2) 正規化門檻校定(增益 0.92~1.08 掃描) ---')
refb = norm(ref[:C.COMPASS_ROWS]); reff = norm(ref)
sH = Sim(seed=4)
refH = sH.world.render(*C.HOME, 0.0, 0.0, rng)
refHb = norm(refH[:C.COMPASS_ROWS]); refHf = norm(refH)
print(f'{"位移":>6} {"帶域min-SSD":>12} {"全圖IDF":>9} {"conf":>6}')
for off in [0.0, 0.03, 0.06, 0.10, 0.20, 0.40]:
    dm, fi, cf = [], [], []
    for g in [0.92, 1.0, 1.08]:
        sH.world._gain = g
        cur = sH.world.render(C.HOME[0] + off * 0.7, C.HOME[1] + off * 0.71, 0.5, 0.0, rng)
        _, c, d = ridf(norm(cur[:C.COMPASS_ROWS]), refHb)
        dm.append(d); fi.append(img_diff(norm(cur), refHf)); cf.append(c)
    print(f'{off*100:5.0f}cm {min(dm):.3f}-{max(dm):.3f} {np.mean(fi):9.3f} {min(cf):6.2f}')
print(f'-> COMPASS_NEAR={C.COMPASS_NEAR}  NEST_LOCK_FULL={C.NEST_LOCK_FULL}  到家門檻=0.040')

dark = Sim(seed=5, start=(1.5, 1.0), lights_on=False)
c = dark.world.render(1.5, 1.0, 0.2, 0.0, rng)
_, cf, _ = ridf(norm(c[:C.COMPASS_ROWS]), refb)
print(f'--- 3) 關燈 conf={cf:.3f}(門檻 {C.RIDF_CONF_MIN} 正確拒收) ---')

print('--- 4) 打滑對編碼器航向的殺傷(20s 混合行駛) ---')
s2 = Sim(seed=3, start=(1.5, 1.0)); th_enc = 0.0
for i in range(4000):
    p = s2.step(0.20, 0.20 if (i // 400) % 2 == 0 else -0.20)
    th_enc += (p.enc_dR - p.enc_dL) / C.TRACK
print(f'編碼器航向誤差: {np.rad2deg(abs((th_enc - s2.th + np.pi) % (2*np.pi) - np.pi)):.1f}°')
