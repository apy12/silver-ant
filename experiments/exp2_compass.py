# -*- coding: utf-8 -*-
"""實驗二:三個航向估計器的時間線 + 關燈時段(30–55s)。
同一台機器人、同一串封包,三個 HeadingBank 並行(enc/gyro/fused),
fused 負責實際控制(中央覓食短趟)。展示:優雅退化與復錨。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C
from sim import Sim
from brain import (HeadingBank, PathIntegrator, Controller, Forager, Homer,
                   SearchSpiral, wrap)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
DARK = (30.0, 55.0)
T_END = 135.0

class _B: whisk_L = whisk_R = False

def main():
    seed = 7
    sim = Sim(seed=seed, heading=0.0)
    hbs = {m: HeadingBank(m) for m in ('enc', 'gyro', 'fused')}
    hf = hbs['fused']
    pi = PathIntegrator(hf, use_flow_rej=True)
    ctrl = Controller(np.random.default_rng(seed + 999))
    homer = Homer(pi)
    lp = _B()
    rec = []          # t, err_e, err_g, err_f, conf, lights
    locks = []
    state, st_t = 'forage', 0.0
    forager = Forager(np.random.default_rng(seed * 31))
    search = None

    def all_packet(p):
        for hb in hbs.values():
            hb.on_packet(p)
        pi.on_packet(p)

    while sim.t < T_END:
        sim.world.lights_on = not (DARK[0] <= sim.t <= DARK[1])
        if state == 'forage':
            v, om = forager.act()
            if sim.t - st_t > 12.0:
                state, st_t = 'home', sim.t
        elif state == 'home':
            v, om, done = homer.act()
            if done or hf.nest_locked or sim.t - st_t > 30.0:
                if hf.nest_locked:
                    state, st_t = 'dwell', sim.t
                    locks.append(sim.t); pi.reset()
                else:
                    state, st_t = 'search', sim.t
                    search = SearchSpiral(pi)
        elif state == 'search':
            v, om = search.act()
            if hf.nest_locked:
                state, st_t = 'dwell', sim.t
                locks.append(sim.t); pi.reset()
            elif sim.t - st_t > 20.0:
                state, st_t = 'forage', sim.t
                forager = Forager(np.random.default_rng(int(sim.t * 13)))
        else:  # dwell
            v, om = 0.0, 0.0
            if sim.t - st_t > 0.8:
                state, st_t = 'forage', sim.t
                forager = Forager(np.random.default_rng(int(sim.t * 13)))
        lp = sim.step(*ctrl.mix(lp, v, om))
        all_packet(lp)
        if sim.frame_ready():
            was = hf.nest_locked
            hf.on_frame(sim.grab_frame())
            if hf.nest_locked and not was and state in ('home', 'search'):
                pass  # 狀態機下一步會處理
        rec.append((sim.t,
                    np.rad2deg(wrap(hbs['enc'].th_enc - sim.th)),
                    np.rad2deg(wrap(hbs['gyro'].th_gyro - sim.th)),
                    np.rad2deg(wrap(hf.theta - sim.th)),
                    hf.last_conf, sim.world.lights_on))
    r = np.array(rec)

    fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    ax[0].axvspan(*DARK, color='k', alpha=0.13, label='lights OFF')
    ax[0].plot(r[:, 0], r[:, 1], color='#c0392b', lw=1.1, label='encoder-only')
    ax[0].plot(r[:, 0], r[:, 2], color='#e67e22', lw=1.1, label='gyro-only')
    ax[0].plot(r[:, 0], r[:, 3], color='#1f77b4', lw=1.6, label='fused (gyro + nest anchor)')
    for i, t in enumerate(locks):
        ax[0].axvline(t, color='#2ca02c', lw=0.9, alpha=0.7,
                      label='nest lock / re-anchor' if i == 0 else None)
    ax[0].set_ylabel('heading error (deg)')
    ax[0].legend(loc='upper left', fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[0].set_title('Heading error vs time — graceful degradation in darkness, snap-back on re-anchor')
    ax[1].axvspan(*DARK, color='k', alpha=0.13)
    ax[1].plot(r[:, 0], r[:, 4], color='#555', lw=0.9)
    ax[1].axhline(C.RIDF_CONF_MIN, color='#2ca02c', ls='--', lw=1)
    ax[1].set_ylabel('RIDF conf'); ax[1].set_xlabel('time (s)')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_compass.png'), dpi=140)
    seg = lambda a, b: [np.abs(r[(r[:,0]>=a)&(r[:,0]<=b), k]).max() for k in (1,2,3)]
    print('最大|航向誤差|° [enc, gyro, fused]  關燈前:', [f'{v:.1f}' for v in seg(0, 30)],
          ' 關燈中:', [f'{v:.1f}' for v in seg(*DARK)],
          ' 復燈後:', [f'{v:.1f}' for v in seg(60, T_END)])
    print('鎖巢/復錨事件時刻:', [f'{t:.0f}s' for t in locks])

if __name__ == '__main__':
    main()
