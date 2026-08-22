# -*- coding: utf-8 -*-
"""實驗一:中央覓食(central-place foraging)。
每 trial 五趟:{覓食 25s → PI 歸巢 → (C/D) 螺旋搜尋至視覺鎖巢} × 5。
指標:每趟「宣告到家」瞬間與真巢的距離。A 純編碼器 / B +陀螺 /
C +巢錨定視覺 / D +光流打滑抑制。裁判 = 天花板相機真值。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C
from sim import Sim
from brain import (HeadingBank, PathIntegrator, Controller, Forager, Homer,
                   SearchSpiral)

CONFIGS = [('A enc-only', 'enc', False), ('B +gyro', 'gyro', False),
           ('C +nest-anchored vision', 'fused', False), ('D C+flow slip-rej', 'fused', True)]
TRIPS, T_FORAGE, SEEDS = 5, 25.0, 12
T_HOME_MAX, T_SEARCH_MAX = 40.0, 40.0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')


class _Blank: whisk_L = whisk_R = False


def run_trial(seed, mode, flow_rej, record=False):
    sim = Sim(seed=seed, heading=0.0)
    hb = HeadingBank(mode); pi = PathIntegrator(hb, flow_rej)
    ctrl = Controller(np.random.default_rng(seed + 999))
    homer = Homer(pi)
    vision = (mode == 'fused')
    lp = _Blank(); traj = []
    errs, t_marks = [], []
    for trip in range(TRIPS):
        forager = Forager(np.random.default_rng(seed * 31 + trip))
        # --- 覓食 ---
        t0 = sim.t
        while sim.t - t0 < T_FORAGE:
            v, om = forager.act()
            lp = sim.step(*ctrl.mix(lp, v, om)); hb.on_packet(lp); pi.on_packet(lp)
            if vision and sim.frame_ready(): hb.on_frame(sim.grab_frame())
            if record: traj.append((sim.x, sim.y, trip, 0))
        # --- PI 歸巢 ---
        t0 = sim.t
        while sim.t - t0 < T_HOME_MAX:
            v, om, done = homer.act()
            lp = sim.step(*ctrl.mix(lp, v, om)); hb.on_packet(lp); pi.on_packet(lp)
            if vision and sim.frame_ready(): hb.on_frame(sim.grab_frame())
            if record: traj.append((sim.x, sim.y, trip, 1))
            if done or (vision and hb.nest_locked): break   # 視覺鎖定即靠泊
        # --- 到家判定 ---
        if not vision:
            errs.append(sim.gt_dist_home()); pi.reset()   # 只能相信 PI 說到家
        else:
            locked = hb.nest_locked                        # 進場途中可能已鎖
            search = SearchSpiral(pi)                       # PI 閉環的 Wehner 螺旋
            t0 = sim.t
            while not locked and sim.t - t0 < T_SEARCH_MAX:
                v, om = search.act()
                lp = sim.step(*ctrl.mix(lp, v, om))
                hb.on_packet(lp); pi.on_packet(lp)
                if sim.frame_ready():
                    hb.on_frame(sim.grab_frame())
                    if hb.nest_locked: locked = True
                if record: traj.append((sim.x, sim.y, trip, 2))
            errs.append(sim.gt_dist_home())
            if locked: pi.reset()                          # 視覺確認才歸零
        # 巢口停頓 0.8s:讓羅盤在錨定半徑內把航向拉正(螞蟻也會停)
        t0 = sim.t
        while sim.t - t0 < 0.8:
            lp = sim.step(*ctrl.mix(lp, 0.0, 0.0))
            hb.on_packet(lp); pi.on_packet(lp)
            if vision and sim.frame_ready(): hb.on_frame(sim.grab_frame())
        t_marks.append(sim.t)
    return errs, traj


def main():
    t0 = time.time()
    res = {name: np.zeros((SEEDS, TRIPS)) for name, _, _ in CONFIGS}
    for name, mode, fr in CONFIGS:
        for s in range(SEEDS):
            errs, _ = run_trial(2000 + s, mode, fr)
            res[name][s] = errs
        print(f'{name:26s} done  {time.time()-t0:.0f}s', flush=True)

    errA, trajA = run_trial(2004, 'enc', False, record=True)
    errD, trajD = run_trial(2004, 'fused', True, record=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, TRIPS))
    tD = np.array(trajD)
    for trip in range(TRIPS):
        m = (tD[:, 2] == trip)
        ax[0].plot(tD[m, 0], tD[m, 1], color=cols[trip], lw=0.8, alpha=0.8)
    for (cx, cy, r, h) in C.CYLINDERS:
        ax[0].add_patch(plt.Circle((cx, cy), r, color='k'))
    ax[0].plot(*C.HOME, marker='*', ms=18, color='goldenrod', mec='k', ls='none')
    ax[0].set_xlim(0, C.ARENA_W); ax[0].set_ylim(0, C.ARENA_H); ax[0].set_aspect('equal')
    ax[0].set_title('Config D, 5 trips (color = trip index), star = nest')

    colmap = {'A enc-only': '#c0392b', 'B +gyro': '#e67e22',
              'C +nest-anchored vision': '#2ca02c', 'D C+flow slip-rej': '#1f77b4'}
    x = np.arange(1, TRIPS + 1)
    for name, _, _ in CONFIGS:
        med = np.median(res[name], axis=0) * 100
        q1 = np.percentile(res[name], 25, axis=0) * 100
        q3 = np.percentile(res[name], 75, axis=0) * 100
        ax[1].plot(x, med, 'o-', color=colmap[name], label=name)
        ax[1].fill_between(x, q1, q3, color=colmap[name], alpha=0.15)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('trip number'); ax[1].set_xticks(x)
    ax[1].set_ylabel('docking error: true distance to nest (cm)')
    ax[1].set_title(f'Central-place foraging, {T_FORAGE:.0f}s trips (median+IQR, n={SEEDS})')
    ax[1].legend(); ax[1].grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_homing.png'), dpi=140)

    print('\n=== 每趟到家誤差中位數 (cm) ===')
    print('trip:   ', ' '.join(f'{i:6d}' for i in x))
    for name, _, _ in CONFIGS:
        print(f'{name:26s}', ' '.join(f'{v:6.1f}' for v in np.median(res[name], 0) * 100))
    np.save(os.path.join(OUT, 'exp1.npy'), res, allow_pickle=True)


if __name__ == '__main__':
    main()
