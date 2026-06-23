#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset generator using Sobol sampling, but with a "complexity reducer":
- primitive weights are forced to one-hot (pick one primitive)
- hollow/split/invert/layer are forced to binary toggles with controllable probabilities
- optional lower num_layers / higher resolution support
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


# ----------------------------------------------------------------------
# Generator (your logic, with symmetric_split option)
# ----------------------------------------------------------------------
def generate_super_shape(
    w_circle, w_square, w_cross, w_dual,
    hollow_weight, split_weight, invert_weight,
    r, s, cross_w, cross_l, rect_x, rect_w, rect_h, sq_x, sq_s,
    thickness, gap_width,
    layer_weight=0.0, num_layers=3, layer_thickness=0.08, line_thickness=0.015,
    resolution=20,
    symmetric_split=False,
):
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)

    # --- base SDF primitives (signed-ish fields) ---
    d_circle = np.sqrt(X**2 + Y**2) - r
    d_square = np.maximum(np.abs(X) - s, np.abs(Y) - s)

    d_cross_v = np.maximum(np.abs(X) - cross_w, np.abs(Y) - cross_l)
    d_cross_h = np.maximum(np.abs(X) - cross_l, np.abs(Y) - cross_w)
    d_cross = np.minimum(d_cross_v, d_cross_h)

    d_rect = np.maximum(np.abs(X - rect_x) - rect_w, np.abs(Y) - rect_h)
    d_sq = np.maximum(np.abs(X - sq_x) - sq_s, np.abs(Y) - sq_s)
    d_dual = np.minimum(d_rect, d_sq)

    # --- sharpened primitive blend (dominant wins) ---
    weights = np.array([w_circle, w_square, w_cross, w_dual], dtype=np.float64)
    if weights.sum() == 0:
        weights[0] = 1.0
    weights = weights ** 3
    weights = weights / weights.sum()

    d_base = (
        weights[0] * d_circle +
        weights[1] * d_square +
        weights[2] * d_cross +
        weights[3] * d_dual
    )

    # --- hollowing (frame / square-in-square) ---
    d_hollowed = np.abs(d_base) - thickness
    d_current = (1.0 - hollow_weight) * d_base + hollow_weight * d_hollowed

    # --- layers: HARD toggle -> concentric outlines via dist = abs(d_base) ---
    if layer_weight > 0.5:
        dist = np.abs(d_base)
        outlines = np.zeros_like(d_base, dtype=bool)
        for k in range(num_layers):
            offset = k * layer_thickness
            outlines |= (np.abs(dist - offset) <= line_thickness)
        # inside becomes 1 after your thresholding convention (d_final<=0)
        d_current = np.where(outlines, -1.0, 1.0)

    # --- splitting ---
    if symmetric_split:
        d_gap = np.maximum(gap_width - np.abs(X), gap_width - np.abs(Y))
    else:
        d_gap = gap_width - np.abs(X)

    d_split = np.maximum(d_current, d_gap)
    d_current = (1.0 - split_weight) * d_current + split_weight * d_split

    # --- inversion ---
    d_final = d_current * (1.0 - 2.0 * invert_weight)

    return (d_final <= 0).astype(np.uint8)


def params_to_shape(
    params,
    resolution=20,
    num_layers=4,
    symmetric_split=False,
):
    # params length = 21
    # [:8] = w_circ,w_sq,w_cross,w_dual, hollow, split, inv, layer
    # [8:19] = r,s,cross_w,cross_l,rect_x,rect_w,rect_h,sq_x,sq_s,thickness,gap_width
    # [19] = layer_thickness
    # [20] = line_thickness
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv, m_layer = params[:8]
    mapped_constants = list(params[8:19])
    layer_thickness = params[19]
    line_thickness = params[20]

    return generate_super_shape(
        w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
        *mapped_constants,
        layer_weight=m_layer,
        num_layers=num_layers,
        layer_thickness=layer_thickness,
        line_thickness=line_thickness,
        resolution=resolution,
        symmetric_split=symmetric_split,
    )


# ----------------------------------------------------------------------
# Bounds and structured Sobol sampler
# ----------------------------------------------------------------------
resolution = 20

bounds_geom = [
    (0.1, 0.45), (0.1, 0.45), (0.025, 0.2), (0.2, 0.475),
    (-0.4, -0.05), (0.05, 0.2), (0.1, 0.4), (0.05, 0.4),
    (0.05, 0.2), (0.025, 0.15), (0.025, 0.2)
]

def make_param_bounds():
    # order matches params_to_shape
    bounds = (
        ([(0.0, 1.0)] * 7) +           # 7 dims: w_circle,w_square,w_cross,w_dual, hollow, split, invert
        ([(0.0, 1.0)] * 1) +           # m_layer
        bounds_geom +                  # 11 geometry params (r,s,...,gap_width)
        [(0.03, 0.15)] +              # layer_thickness
        [(0.005, 0.03)]              # line_thickness
    )
    assert len(bounds) == 21
    return bounds


def sample_structured_sobol(
    n_samples,
    seed=0,
    scramble=True,
    primitive_onehot=True,
    # probabilities for binary toggles (evaluated using Sobol u values)
    p_hollow=0.35,
    p_split=0.10,
    p_invert=0.08,
    p_layer=0.25,
):
    """
    Uses Sobol for all dims, then post-processes:
    - primitive weights -> one-hot (choose argmax among first 4 u's)
    - hollow/split/invert/layer -> binary with probabilities
    """
    bounds = make_param_bounds()
    d = len(bounds)
    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)  # (n, d) in [0,1]

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    params = lo + u * (hi - lo)

    # --- complexity reduction: primitive blend -> one-hot ---
    if primitive_onehot:
        w_u = u[:, 0:4]  # which primitive wins
        idx = np.argmax(w_u, axis=1)
        params[:, 0:4] = 0.0
        params[np.arange(n_samples), idx] = 1.0

    # --- complexity reduction: effects -> binary toggles ---
    params[:, 4] = (u[:, 4] < p_hollow).astype(np.float64)  # hollow_weight
    params[:, 5] = (u[:, 5] < p_split).astype(np.float64)   # split_weight
    params[:, 6] = (u[:, 6] < p_invert).astype(np.float64)  # invert_weight
    params[:, 7] = (u[:, 7] < p_layer).astype(np.float64)   # layer_weight (threshold at >0.5)

    return params


def generate_dataset(
    n_samples,
    seed=0,
    scramble=True,
    primitive_onehot=True,
    p_hollow=0.35,
    p_split=0.10,
    p_invert=0.08,
    p_layer=0.25,
    symmetric_split=False,
    num_layers=3,
    render_resolution=20,
):
    params = sample_structured_sobol(
        n_samples,
        seed=seed,
        scramble=scramble,
        primitive_onehot=primitive_onehot,
        p_hollow=p_hollow,
        p_split=p_split,
        p_invert=p_invert,
        p_layer=p_layer,
    )

    bitmaps = np.empty((n_samples, render_resolution, render_resolution), dtype=np.uint8)
    for i in range(n_samples):
        bitmaps[i] = params_to_shape(
            params[i],
            resolution=render_resolution,
            num_layers=num_layers,
            symmetric_split=symmetric_split,
        )
    return params, bitmaps


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_random_samples(params, bitmaps, k=12, seed=123, title="Generated"):
    rng = np.random.default_rng(seed)
    n = bitmaps.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)

    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(idx):]:
        ax.axis("off")

    for ax, i in zip(axes[:len(idx)], idx):
        ax.imshow(bitmaps[i], cmap="gray_r", origin="lower", interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"{title}\nidx={i}\n"
            f"hollow={params[i][4]:.0f} split={params[i][5]:.0f} inv={params[i][6]:.0f} layer={params[i][7]:.0f}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    N = 300

    # Try these first (tighter = less weird)
    params, bitmaps = generate_dataset(
        N,
        seed=np.random.randint(1e6),
        scramble=True,
        primitive_onehot=True,  # biggest simplifier
        p_hollow=0.35,
        p_split=0.10,
        p_invert=0.08,
        p_layer=0.25,
        symmetric_split=False,  # match your original splitting
        num_layers=1,           # reduce a bit
        render_resolution=20,   # raise to 40/64 if you want cleaner edges
    )

    print("params:", params.shape, "(21D)")
    print("bitmaps:", bitmaps.shape)
    print("unique bitmap values:", np.unique(bitmaps))

    plot_random_samples(params, bitmaps, k=12, seed=42, title="Structured Sobol samples")