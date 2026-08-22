# -*- coding: utf-8 -*-
"""實驗三:快照歸巢(run-and-tumble 沿 IDF 梯度)集水區地圖。
11×7 起點網格、隨機初始航向、50s 逾時。裁判記錄最終真實距巢距離。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C
from sim import Sim
from brain import SnapshotHomer, Controller

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
TIMEOUT = 50.0

class _B: whisk_L = whisk_R = False

def main():
    rng = np.random.default_rng(0)
    s0 = Sim(seed=0)
    snap = s0.cam.render(s0.world, *C.HOME, 0.0, 0.0, rng)[0]  # [相容性修補] render 已拆到 PanoCamera,回傳 (img, valid)
    xs = np.linspace(0.25, 2.75, 11)
    ys = np.linspace(0.25, 1.75, 7)
    M = np.full((len(ys), len(xs)), np.nan)
    t0 = time.time()
    n_ok = n_run = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if any(np.hypot(x - cx, y - cy) < r + 0.13 for cx, cy, r, h in C.CYLINDERS):
                continue
            seed = 300 + j * 20 + i
            sim = Sim(seed=seed, start=(x, y),
                      heading=np.random.default_rng(seed).uniform(-np.pi, np.pi))
            homer = SnapshotHomer(np.random.default_rng(seed + 1), snap)
            ctrl = Controller(np.random.default_rng(seed + 2))
            lp = _B()
            while sim.t < TIMEOUT and not homer.arrived:
                v, om = homer.act()
                lp = sim.step(*ctrl.mix(lp, v, om))
                if sim.frame_ready():
                    homer.on_frame(sim.grab_frame())
            d = sim.gt_dist_home()
            M[j, i] = d
            n_run += 1
            n_ok += (d < 0.10)
        print(f'row {j+1}/{len(ys)}  {time.time()-t0:.0f}s', flush=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(np.minimum(M, 0.6) * 100, origin='lower', cmap='RdYlGn_r',
                   extent=[xs[0]-0.125, xs[-1]+0.125, ys[0]-0.125, ys[-1]+0.125],
                   vmin=0, vmax=60, aspect='equal')
    ok = M < 0.10
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if np.isnan(M[j, i]):
                ax.plot(x, y, 'x', color='#888', ms=6)
            elif ok[j, i]:
                ax.plot(x, y, '.', color='k', ms=4)
    for (cx, cy, r, h) in C.CYLINDERS:
        ax.add_patch(plt.Circle((cx, cy), r, color='k'))
    ax.plot(*C.HOME, marker='*', ms=20, color='white', mec='k')
    ax.set_xlim(0, C.ARENA_W); ax.set_ylim(0, C.ARENA_H)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('final distance to nest (cm), capped 60')
    succ = 100 * n_ok / max(n_run, 1)
    med_ok = np.nanmedian(M[ok]) * 100 if ok.any() else float('nan')
    ax.set_title(f'Snapshot-homing catchment map — success {succ:.0f}% of {n_run} starts '
                 f'(dot = docked <10 cm, median {med_ok:.1f} cm)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_catchment.png'), dpi=140)
    print(f'成功率 {succ:.0f}%  ({n_ok}/{n_run}),成功案例最終距離中位數 {med_ok:.1f}cm')
    np.save(os.path.join(OUT, 'exp3.npy'), M)

if __name__ == '__main__':
    main()
