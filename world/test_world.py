#!/usr/bin/env python3
"""
test_world.py — 全景模擬器(World + PanoCamera)的回歸測試與診斷視覺化。

用法:
    python3 test_world.py [輸出資料夾]

輸出:
    終端機列印 PASS / FAIL 報告(任何 FAIL → exit code 1)
    fig1_map_and_skyline.png   俯視圖 + 牆面稜線輪廓
    fig2_panoramas.png         多個位姿的全景圖(含關燈)
    fig3_blur_and_gain.png     ±ω 動態模糊比較 + 光照增益漂移統計
    fig4_two_cameras.png       感測器相機 vs 高解析除錯相機(同一世界瞬間)

檢查採「獨立實作對照」,避免循環驗證:
    牆距離     → 逐步射線行進重算
    周長參數 s → 反推回牆面座標比對命中點
    圓筒角寬   → 切線公式 2·asin(r/dc)
    淨空       → 對所有障礙物邊界密集取樣求最近距離
    OU 統計    → 長時間 step 量穩態 std 與自相關時間
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import config as C
from world import World, PanoCamera, wrap

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    line = f"[{'PASS' if ok else 'FAIL':>4}] {name}"
    if detail:
        line += f"  —  {detail}"
    print(line)


# 幾何檢查用的決定性相機:無雜訊;世界不 step → gain 固定 1.0
CAM0 = dict(pix_noise=0.0)


def cylinder_pose():
    """正對 CYLINDERS[1] 的位姿(圓筒剪影應貼齊 col 0)。"""
    cx0, cy0, r, h = C.CYLINDERS[1]
    x, y = 2.0, 1.5
    dc = np.hypot(cx0 - x, cy0 - y)
    th = np.arctan2(cy0 - y, cx0 - x)
    return x, y, th, dc, r, h


def probe_row(world, cam, x, y, th, dc, r, h):
    """挑一個仰角行:高於所有牆頂、低於目標圓筒頂 → 該行僅圓筒類為暗。"""
    alpha = th + np.arange(cam.W) * 2 * np.pi / cam.W
    dw, sw = world.wall_hit(x, y, alpha)
    wall_top = np.arctan2(world.skyline_at(sw) - cam.cam_h, dw).max()
    cyl_top = np.arctan2(h - cam.cam_h, dc - r)
    assert cyl_top > wall_top + 0.03, "測試場景無效:圓筒不夠高,請調整 config"
    els = np.linspace(cam.el_top, cam.el_bot, cam.H)
    return int(np.argmin(np.abs(els - 0.5 * (wall_top + cyl_top))))


def run_containing_col0(dark):
    """包含 col 0 的連續暗段(含環繞)。同行可能有其他圓筒,不能整行計數。"""
    K = len(dark)
    run = []
    if dark[0]:
        c = 0
        while dark[c % K] and len(run) < K:
            run.append(c % K); c += 1
        c = -1
        while dark[c % K] and len(run) < K:
            run.append(c % K); c -= 1
    return run


def dark_circ_centroid(row):
    """暗度加權的欄位圓形平均(rad)。"""
    K = len(row)
    ang = np.arange(K) * 2 * np.pi / K
    wgt = np.clip(0.5 - row, 0, None)
    if wgt.sum() < 1e-9:
        return np.nan
    return np.arctan2((wgt * np.sin(ang)).sum(), (wgt * np.cos(ang)).sum())


# ======================================================================
# 1. 基本輸出格式
# ======================================================================
def test_basic():
    img, valid = PanoCamera().render(World(), 2.0, 1.5, 0.3,
                                     rng=np.random.default_rng(0))
    check("輸出形狀 (PANO_H, PANO_W)", img.shape == (C.PANO_H, C.PANO_W),
          f"got {img.shape}")
    check("回傳 (img, valid_light) 且 valid 為 bool",
          isinstance(valid, (bool, np.bool_)), f"valid = {valid!r}")
    check("像素值域 [0,1]", (img >= 0).all() and (img <= 1).all(),
          f"min={img.min():.3f} max={img.max():.3f}")
    q = img * 255
    check("8-bit 量化(所有值為 k/255)", np.allclose(q, np.round(q), atol=1e-9))
    check("行序:頂行均值 > 底行均值(天空亮於地板)",
          float(img[0].mean()) > float(img[-1].mean()),
          f"top={img[0].mean():.2f} bottom={img[-1].mean():.2f}")


def test_determinism_and_purity():
    w, cam = World(), PanoCamera()
    a, _ = cam.render(w, 1.2, 2.0, 1.0, rng=np.random.default_rng(42))
    b, _ = cam.render(w, 1.2, 2.0, 1.0, rng=np.random.default_rng(42))
    check("決定性(相同 seed → 相同影像)", np.array_equal(a, b))
    g0 = w.gain
    cam.render(w, 1.2, 2.0, 1.0, rng=np.random.default_rng(0))
    check("render 純函數:不改動世界狀態(gain 不變)", w.gain == g0)


def test_light_decoupled_from_rendering():
    """光照只認模擬時間:甲每步拍照、乙完全不拍 → 增益軌跡相同。"""
    wa, wb, cam = World(), World(), PanoCamera()
    ra, rb = np.random.default_rng(3), np.random.default_rng(3)
    dt = 1.0 / C.VISION_FPS
    for _ in range(300):
        wa.step(dt, ra); cam.render(wa, 2.0, 1.5, 0.0, rng=np.random.default_rng(0))
        wb.step(dt, rb)
    check("光照與拍照解耦(拍 300 張 vs 0 張,增益相同)", wa.gain == wb.gain,
          f"gain = {wa.gain:.4f}")


# ======================================================================
# 2. wall_hit 對照獨立射線行進
# ======================================================================
def _march(x, y, a, coarse=0.02, fine=1e-4):
    cx, cy = np.cos(a), np.sin(a)
    t = 0.0
    while (0 <= x + t * cx <= C.ARENA_W) and (0 <= y + t * cy <= C.ARENA_H):
        t += coarse
    t -= coarse
    while (0 <= x + t * cx <= C.ARENA_W) and (0 <= y + t * cy <= C.ARENA_H):
        t += fine
    return t


def test_wall_hit():
    rng = np.random.default_rng(1)
    w = World()
    n_ray = 300
    xs = rng.uniform(0.1, C.ARENA_W - 0.1, n_ray)
    ys = rng.uniform(0.1, C.ARENA_H - 0.1, n_ray)
    aa = rng.uniform(-np.pi, np.pi, n_ray)
    errs = [abs(w.wall_hit(x, y, np.array([a]))[0][0] - _march(x, y, a))
            for x, y, a in zip(xs, ys, aa)]
    check(f"wall_hit 距離 vs 射線行進({n_ray} 條隨機射線)",
          max(errs) < 5e-4, f"max err = {max(errs):.2e} m")

    # 周長參數 s 反推回牆面座標,應等於命中點(牆角連續性也一併覆蓋)
    x, y = 2.0, 1.5
    alpha = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    d, s = w.wall_hit(x, y, alpha)
    hx, hy = x + d * np.cos(alpha), y + d * np.sin(alpha)
    W_, H_ = C.ARENA_W, C.ARENA_H
    px = np.empty_like(s); py = np.empty_like(s)
    m0 = s <= W_
    m1 = (s > W_) & (s <= W_ + H_)
    m2 = (s > W_ + H_) & (s <= 2 * W_ + H_)
    m3 = s > 2 * W_ + H_
    px[m0], py[m0] = s[m0], 0.0
    px[m1], py[m1] = W_, s[m1] - W_
    px[m2], py[m2] = 2 * W_ + H_ - s[m2], H_
    px[m3], py[m3] = 0.0, 2 * W_ + 2 * H_ - s[m3]
    err = np.hypot(px - hx, py - hy).max()
    check("周長參數 s ↔ 命中點座標互相一致", err < 1e-6, f"max err = {err:.2e} m")


def test_outside_arena():
    d, _ = World().wall_hit(-0.05, 1.5,
                            np.linspace(-np.pi, np.pi, 64, endpoint=False))
    check("場外位置不回傳負牆距", d.min() >= 0, f"min distance = {d.min():.3f} m")


def test_skyline():
    w = World()
    k = 10
    step = w.perim / len(w.sky_s)
    v = w.skyline_at(np.array([w.sky_s[k] + 0.01 * step]))[0]
    check("skyline_at 取樣對齊(樣本 k 區間 → sky_h[k])",
          np.isclose(v, w.sky_h[k]))
    hh = w.skyline_at(np.array([1.234, 1.234 + w.perim]))
    check("skyline_at 週期性(s 與 s+perim 相同)", hh[0] == hh[1])


# ======================================================================
# 3. 圓筒剪影與欄位約定
# ======================================================================
def test_cylinder():
    x, y, th, dc, r, h = cylinder_pose()
    w, cam = World(), PanoCamera(**CAM0)
    img, _ = cam.render(w, x, y, th)
    row = probe_row(w, cam, x, y, th, dc, r, h)
    run = run_containing_col0(img[row] < 0.3)
    expect = 2 * np.arcsin(r / dc) / (2 * np.pi / cam.W)
    check("圓筒剪影角寬 ≈ 2·asin(r/dc)(取含 col 0 的暗段)",
          abs(len(run) - expect) <= 1.5, f"量到 {len(run)} 欄,理論 {expect:.2f} 欄")
    centered = len(run) > 0 and all(c <= 3 or c >= cam.W - 3 for c in run)
    check("欄位約定:車頭方向 = col 0(暗段貼齊 col 0)", centered,
          f"run = {sorted(run)}")


def test_lights_off():
    img, valid = PanoCamera(**CAM0).render(World(lights_on=False), 2.0, 1.5, 0.0)
    check("關燈 → 全畫面 ≈ 關燈亮度,valid_light = False",
          np.allclose(img, np.round(0.02 * 255) / 255, atol=1e-6) and not valid,
          f"unique = {np.unique(img)}, valid = {valid}")


# ======================================================================
# 4. 動態模糊
# ======================================================================
def test_blur():
    x, y, th, dc, r, h = cylinder_pose()
    w, cam = World(), PanoCamera(**CAM0)
    om = 12.0
    a, _ = cam.render(w, x, y, th, +om)
    b, _ = cam.render(w, x, y, th, -om)
    sharp, _ = cam.render(w, x, y, th, 0.0)
    check("模糊有作用(|ω|=12 與 ω=0 影像不同)", not np.array_equal(a, sharp))
    check("±ω 影像不同(拖影方向跟隨 ω 正負)", not np.array_equal(a, b))
    # 方向物理正確性:曝光期間特徵掃過 [β, β+ωT] → +ω 拖向高欄位、−ω 拖向低欄位
    row = probe_row(w, cam, x, y, th, dc, r, h)
    c0 = dark_circ_centroid(sharp[row])
    dp = wrap(dark_circ_centroid(a[row]) - c0)
    dm = wrap(dark_circ_centroid(b[row]) - c0)
    col_w = 2 * np.pi / cam.W
    check("拖影方向:+ω 重心位移為正、−ω 為負",
          dp > 0.2 * col_w and dm < -0.2 * col_w,
          f"+ω → {dp:+.3f} rad,−ω → {dm:+.3f} rad")
    return a, b


# ======================================================================
# 5. 光照 OU 統計(直接 step,量穩態 std 與自相關時間)
# ======================================================================
def test_ou_stats(n=400_000):
    w = World()
    rng = np.random.default_rng(5)
    dt = 1.0 / C.VISION_FPS
    g = np.empty(n)
    for i in range(n):
        w.step(dt, rng)
        g[i] = w.gain
    burn = int(5 * C.LIGHT_TAU / dt)
    g = g[burn:]
    sig, tau = C.LIGHT_GAIN_STD, C.LIGHT_TAU
    std = g.std()
    check("OU 穩態 std ≈ LIGHT_GAIN_STD(±10%)",
          abs(std - sig) < 0.10 * sig, f"量到 {std:.4f},目標 {sig:.4f}")
    # 自相關時間:lag = τ 時相關性應 ≈ e⁻¹(驗證時間常數是「秒」不是「幀」)
    lag = int(tau / dt)
    ac = np.corrcoef(g[:-lag], g[lag:])[0, 1]
    check("OU 自相關:lag = τ 秒時 ≈ e⁻¹(時間常數以秒計)",
          0.25 < ac < 0.50, f"量到 {ac:.3f},理論 {np.exp(-1):.3f}")
    return g


# ======================================================================
# 6. clearance 對照障礙物邊界密集取樣
# ======================================================================
def test_clearance():
    w = World()
    # 獨立求法:把所有障礙物「表面」取樣成點雲,淨空 = 到點雲的最近距離
    pts = []
    for u in np.linspace(0, 1, 400):
        pts += [(u * C.ARENA_W, 0.0), (u * C.ARENA_W, C.ARENA_H),
                (0.0, u * C.ARENA_H), (C.ARENA_W, u * C.ARENA_H)]
    for cx, cy, r, _h in C.CYLINDERS:
        for a in np.linspace(0, 2 * np.pi, 400, endpoint=False):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    pts = np.array(pts)
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(50):
        x = rng.uniform(0.3, C.ARENA_W - 0.3)
        y = rng.uniform(0.3, C.ARENA_H - 0.3)
        if any(np.hypot(cx - x, cy - y) < r + 0.05 for cx, cy, r, _h in C.CYLINDERS):
            continue
        brute = np.hypot(pts[:, 0] - x, pts[:, 1] - y).min()
        errs.append(abs(w.clearance(x, y) - brute))
    check(f"clearance vs 邊界點雲最近距離({len(errs)} 個隨機點)",
          max(errs) < 5e-3, f"max err = {max(errs):.2e} m")


# ======================================================================
# 7. 多相機
# ======================================================================
def test_multi_camera():
    w = World()
    dbg = PanoCamera(width=256, height=96, pix_noise=0.0, quantize=False)
    img, _ = dbg.render(w, 2.0, 1.5, 0.3)
    check("除錯相機:自訂解析度生效", img.shape == (96, 256), f"got {img.shape}")
    q = img * 255
    check("除錯相機:quantize=False 生效(存在非 k/255 的值)",
          not np.allclose(q, np.round(q), atol=1e-9))
    return dbg


# ======================================================================
# 診斷圖
# ======================================================================
def setup_cjk_font():
    import glob
    from matplotlib import font_manager as fm
    for pat in ("/usr/share/fonts/**/*CJK*.*", "/usr/share/fonts/**/*uming*.*",
                "/System/Library/Fonts/PingFang.ttc", "C:/Windows/Fonts/msjh.ttc"):
        for p in glob.glob(pat, recursive=True):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass
    cjk = sorted({f.name for f in fm.fontManager.ttflist
                  if any(k in f.name for k in ("CJK", "PingFang", "JhengHei", "Ming"))})
    matplotlib.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def show_pano(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def fig_map(poses):
    w = World()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1, 1.4])
    ax0.add_patch(Rectangle((0, 0), C.ARENA_W, C.ARENA_H, fill=False, lw=2))
    for (cx0, cy0, r, h) in C.CYLINDERS:
        ax0.add_patch(Circle((cx0, cy0), r, color="0.25"))
        ax0.annotate(f"h={h}", (cx0, cy0), color="w", ha="center", va="center",
                     fontsize=7)
    for i, (x, y, th) in enumerate(poses):
        ax0.plot(x, y, "o", color=f"C{i}")
        ax0.annotate(f"P{i}", (x, y), textcoords="offset points", xytext=(6, 6),
                     color=f"C{i}")
        ax0.arrow(x, y, 0.3 * np.cos(th), 0.3 * np.sin(th),
                  head_width=0.07, color=f"C{i}", length_includes_head=True)
    ax0.set_xlim(-0.3, C.ARENA_W + 0.3); ax0.set_ylim(-0.3, C.ARENA_H + 0.3)
    ax0.set_aspect("equal"); ax0.set_title("俯視圖:場地 / 圓筒 / 測試位姿")
    ax0.set_xlabel("x [m]"); ax0.set_ylabel("y [m]")
    ax1.plot(w.sky_s, w.sky_h, lw=1)
    for s0, lab in ((0, "下牆"), (C.ARENA_W, "右牆"),
                    (C.ARENA_W + C.ARENA_H, "上牆"),
                    (2 * C.ARENA_W + C.ARENA_H, "左牆")):
        ax1.axvline(s0, color="0.8", lw=0.8)
        ax1.text(s0, ax1.get_ylim()[1], lab, fontsize=8, va="bottom")
    ax1.set_title(f"牆面稜線 sky_h(s)(seed={C.SKYLINE_SEED})")
    ax1.set_xlabel("周長參數 s [m]"); ax1.set_ylabel("高度 [m]")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_map_and_skyline.png", dpi=150)
    plt.close(fig)


def fig_panoramas(poses):
    cam = PanoCamera()
    fig, axes = plt.subplots(len(poses) + 1, 1,
                             figsize=(8, 2.0 * (len(poses) + 1)))
    w = World()
    for ax, (x, y, th), i in zip(axes, poses, range(len(poses))):
        img, _ = cam.render(w, x, y, th, rng=np.random.default_rng(i))
        show_pano(ax, img, f"P{i}:(x={x}, y={y}, θ={np.degrees(th):.0f}°)  燈開")
    img, _ = cam.render(World(lights_on=False), *poses[0],
                        rng=np.random.default_rng(9))
    show_pano(axes[-1], img, "P0 燈關(全黑 + 雜訊)")
    axes[-1].set_xlabel("車體方位角 β(col 0 = 車頭,往左遞增)", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_panoramas.png", dpi=150)
    plt.close(fig)


def fig_blur_gain(img_p, img_m, gains):
    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.3])
    x, y, th = cylinder_pose()[:3]
    w, cam = World(), PanoCamera(**CAM0)
    for j, om in enumerate((0.0, +12.0, -12.0)):
        ax = fig.add_subplot(gs[0, j])
        show_pano(ax, cam.render(w, x, y, th, om)[0], f"ω = {om:+.0f} rad/s")
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(np.abs(img_p - img_m), cmap="magma", vmin=0, vmax=0.2, aspect="auto")
    ax.set_title("|+ω 影像 − −ω 影像|:拖影方向跟隨 ω 正負", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2, :])
    n_show = min(len(gains), 3000)
    t = np.arange(n_show) / C.VISION_FPS
    ax.plot(t, gains[:n_show], lw=0.7, label=f"World.step 實測增益  std={gains.std():.4f}")
    for sgn in (+1, -1):
        ax.axhline(1 + sgn * C.LIGHT_GAIN_STD, color="0.6", ls="--", lw=0.8)
    ax.text(t[-1], 1 + C.LIGHT_GAIN_STD, f" 目標 ±{C.LIGHT_GAIN_STD}",
            fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("時間 [s]"); ax.set_ylabel("增益 gain")
    ax.set_title("光照增益 OU 漂移(穩態 std 應 = LIGHT_GAIN_STD,時間常數 = LIGHT_TAU 秒)",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_blur_and_gain.png", dpi=150)
    plt.close(fig)


def fig_two_cameras(dbg):
    w, cam = World(), PanoCamera()
    pose = cylinder_pose()[:3]
    sensor, _ = cam.render(w, *pose, rng=np.random.default_rng(7))
    debug, _ = dbg.render(w, *pose, rng=np.random.default_rng(7))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4.6))
    show_pano(ax0, sensor, "感測器相機:64×48,含雜訊與 8-bit 量化(訓練用觀測)")
    show_pano(ax1, debug, "除錯相機:256×96,無雜訊不量化 — 同一世界、同一瞬間")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_two_cameras.png", dpi=150)
    plt.close(fig)


# ======================================================================
def main():
    setup_cjk_font()
    poses = [
        (2.0, 1.5, 0.0),
        (0.5, 0.5, np.pi / 4),
        (3.5, 2.5, np.pi),
        cylinder_pose()[:3],
    ]

    print("=" * 72)
    print("World + PanoCamera 回歸測試")
    print("=" * 72)
    test_basic()
    test_determinism_and_purity()
    test_light_decoupled_from_rendering()
    test_wall_hit()
    test_outside_arena()
    test_skyline()
    test_cylinder()
    test_lights_off()
    img_p, img_m = test_blur()
    gains = test_ou_stats()
    test_clearance()
    dbg = test_multi_camera()
    print("-" * 72)

    fig_map(poses)
    fig_panoramas(poses)
    fig_blur_gain(img_p, img_m, gains)
    fig_two_cameras(dbg)
    print(f"圖片已輸出到 {OUTDIR}/fig1..fig4")

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"總結:{len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
