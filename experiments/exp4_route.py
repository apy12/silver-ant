# -*- coding: utf-8 -*-
"""實驗四:熟悉度循路(完美記憶版,Baddeley 2012 的基準線)。
訓練:GT 遙控沿 S 曲線行駛,每 8cm 存一張全景(共 ~35 張)。
回放:只用視覺問「哪個方向最眼熟」,從 3 個起點(含 ±14–17cm 偏移、
+20° 航向偏移)重走路線。裁判量測對訓練路線的橫向偏差。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C
from sim import Sim
from brain import RouteFollower, Controller, wrap

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
WPS = [(0.45, 0.50), (1.00, 0.42), (1.50, 0.50), (1.95, 0.75),
       (2.00, 1.10), (2.20, 1.35), (2.60, 1.30)]
END = np.array(WPS[-1])

class _B: whisk_L = whisk_R = False

def train():
    sim = Sim(seed=11, start=WPS[0], heading=0.0)
    ctrl = Controller(np.random.default_rng(999))
    lp = _B(); views = []; route = []
    k = 1; acc = 0.0; last = np.array(WPS[0]); nxt_store = 0.0
    last_frame = None
    while k < len(WPS) and sim.t < 60:
        tgt = np.array(WPS[k])
        d = np.hypot(*(tgt - [sim.x, sim.y]))
        if d < 0.12:
            k += 1; continue
        des = np.arctan2(tgt[1] - sim.y, tgt[0] - sim.x)
        e = wrap(des - sim.th)                     # 訓練者=人手遙控,可用真值
        lp = sim.step(*ctrl.mix(lp, 0.13 * max(0.3, np.cos(e)),
                                np.clip(3.0 * e, -2.0, 2.0)))
        p = np.array([sim.x, sim.y])
        acc += np.hypot(*(p - last)); last = p
        route.append(p)
        if sim.frame_ready():
            last_frame = sim.grab_frame()
        if acc >= nxt_store and last_frame is not None:
            views.append(last_frame); nxt_store += 0.08
    return views, np.array(route)

def recall(views, start, heading, seed):
    sim = Sim(seed=seed, start=start, heading=heading)
    rf = RouteFollower(views)
    ctrl = Controller(np.random.default_rng(seed + 5))
    lp = _B(); traj = []; last_frame = None
    om_hold = 0.0; next_dec = 0.0
    reached = False
    while sim.t < 55.0:
        if sim.frame_ready():
            last_frame = sim.grab_frame()
        if sim.t >= next_dec and last_frame is not None:
            dth, fam = rf.steer(last_frame)
            om_hold = np.clip(2.5 * dth, -1.8, 1.8)
            next_dec = sim.t + 0.2
        lp = sim.step(*ctrl.mix(lp, 0.14, om_hold))
        traj.append((sim.x, sim.y))
        if np.hypot(sim.x - END[0], sim.y - END[1]) < 0.18:
            reached = True; break
    return np.array(traj), reached, sim.t

def main():
    views, route = train()
    print(f'訓練完成:路線 {len(route)} 點、視圖庫 {len(views)} 張')
    starts = [((0.45, 0.50), 0.0, 'on-route start'),
              ((0.45, 0.64), 0.0, '+14 cm lateral'),
              ((0.50, 0.33), np.deg2rad(20), '-17 cm, +20° heading')]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for (cx, cy, r, h) in C.CYLINDERS:
        ax.add_patch(plt.Circle((cx, cy), r, color='k'))
    ax.plot(route[:, 0], route[:, 1], 'k--', lw=1.4, label='trained route')
    ax.plot(*END, marker='*', ms=16, color='goldenrod', mec='k', ls='none')
    cols = ['#1f77b4', '#2ca02c', '#c0392b']
    stats = []
    # [協定升級 2026-08] 舊版每起點 n=1;驗屍發現 −17cm+20° 的「成功」是
    # 幸運抽籤(新舊堆疊 5 seeds 皆僅 1/5)。改為每起點 5 seeds,
    # 圖上畫「中位表現」那一次,統計報 到達率 + 偏差中位。
    for (st, hd, lab), col in zip(starts, cols):
        runs = []
        for k in range(5):
            traj, ok, tt = recall(views, st, hd, seed=int(st[1] * 1000) + k * 100)
            seg = np.cumsum(np.r_[0, np.hypot(*np.diff(traj, axis=0).T)])
            m = seg > 0.30
            dev = np.array([np.min(np.hypot(*(route - p).T)) for p in traj[m]])
            runs.append((traj, ok, tt, dev))
        n_ok = sum(r[1] for r in runs)
        means = [r[3].mean() for r in runs]
        traj, ok, tt, dev = runs[int(np.argsort(means)[len(runs) // 2])]
        stats.append((lab, n_ok, np.median(means), max(r[3].max() for r in runs), tt))
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.6,
                label=f'{lab}: reached {n_ok}/5, median dev {np.median(means)*100:.1f} cm')
        ax.plot(*st, marker='o', color=col, ms=7, mec='k')
    ax.set_xlim(0, C.ARENA_W); ax.set_ylim(0, C.ARENA_H); ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Familiarity route following (perfect-memory, n=5 seeds/start; '
                 'median-performing run shown)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_route.png'), dpi=140)
    for lab, n_ok, dm, dx, tt in stats:
        print(f'{lab:22s} 到達 {n_ok}/5  偏差中位 {dm*100:5.1f}cm  最大 {dx*100:5.1f}cm')

if __name__ == '__main__':
    main()
