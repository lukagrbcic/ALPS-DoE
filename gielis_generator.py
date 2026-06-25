#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


# -----------------------------
# Gielis superformula (polar)
# -----------------------------
def gielis_r(theta, m, a, b, n1, n2, n3, eps=1e-12):
    ct = np.abs(np.cos(m * theta / 4.0) / a)
    st = np.abs(np.sin(m * theta / 4.0) / b)
    denom = (ct ** n2) + (st ** n3)
    denom = np.maximum(denom, eps)
    return denom ** (-1.0 / n1)


def gielis_bitmap(
    m, a, b, n1, n2, n3,
    invert, vsplit, hsplit,
    resolution=20,
    split_gap=0.06,   # <-- fixed, smaller but not too small
    frame_limit=0.4, # <-- central 80%: with unit cell walls at +/-0.5, keep +/-0.4
    rmax_angles=2048,
):
    # Grid in [-0.5, 0.5]
    x = np.linspace(-0.5, 0.5, resolution, dtype=np.float64)
    y = np.linspace(-0.5, 0.5, resolution, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    # Frame mask enforces "10% margin from all side walls"
    frame_mask = (np.abs(X) <= frame_limit) & (np.abs(Y) <= frame_limit)

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Scale so the curve fits within radius frame_limit (circle fit)
    angles_dense = np.linspace(-np.pi, np.pi, rmax_angles, dtype=np.float64)
    r_dense = gielis_r(angles_dense, m, a, b, n1, n2, n3)
    r_max = float(np.max(r_dense))
    r_max = max(r_max, 1e-12)

    r_theta = gielis_r(theta, m, a, b, n1, n2, n3)
    r_scaled = r_theta * (frame_limit / r_max)

    # Base region (inside polar curve) + frame constraint
    mask = frame_mask & (rho <= r_scaled)

    # Vertical split (carve vertical strip around x=0)
    if int(vsplit) == 1:
        mask &= ~(np.abs(X) <= split_gap)

    # Horizontal split (carve horizontal strip around y=0)
    if int(hsplit) == 1:
        mask &= ~(np.abs(Y) <= split_gap)

    # Inversion, but only within the central frame (so no foreground in margins)
    if int(invert) == 1:
        mask = frame_mask & (~mask)

    return mask.astype(np.uint8)


# -----------------------------
# Params -> bitmap
# -----------------------------
def params_to_shape(params, resolution=20, split_gap=0.06, frame_limit=0.4):
    """
    params layout (9 dims):
    [0]=m, [1]=a, [2]=b, [3]=n1, [4]=n2, [5]=n3,
    [6]=invert (0/1),
    [7]=vsplit (0/1),
    [8]=hsplit (0/1)
    """
    m = int(round(params[0]))
    a, b, n1, n2, n3 = params[1:6]
    invert, vsplit, hsplit = params[6:9]

    return gielis_bitmap(
        m=m, a=a, b=b, n1=n1, n2=n2, n3=n3,
        invert=invert, vsplit=vsplit, hsplit=hsplit,
        resolution=resolution,
        split_gap=split_gap,
        frame_limit=frame_limit,
    )


# -----------------------------
# Sobol sampling
# -----------------------------
def make_param_bounds(m_min=1, m_max=12):
    return [
        (m_min, m_max),     # m (rounded after)
        (0.25, 1.20),       # a
        (0.25, 1.20),       # b
        (0.20, 6.00),       # n1
        (0.20, 6.00),       # n2
        (0.20, 6.00),       # n3
        (0.0, 1.0),         # invert toggle (thresholded)
        (0.0, 1.0),         # vsplit toggle (thresholded)
        (0.0, 1.0),         # hsplit toggle (thresholded)
    ]


def sample_structured_sobol(
    n_samples,
    seed=0,
    scramble=True,
    p_invert=0.08,
    p_vsplit=0.10,
    p_hsplit=0.10,
    m_min=1,
    m_max=12,
):
    bounds = make_param_bounds(m_min=m_min, m_max=m_max)
    d = len(bounds)

    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    u = sampler.random(n_samples)  # [0,1]^d

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    params = lo + u * (hi - lo)

    # m -> integer-like
    params[:, 0] = np.clip(np.round(params[:, 0]), m_min, m_max)

    # threshold toggles using their u-values
    params[:, 6] = (u[:, 6] < p_invert).astype(np.float64)
    params[:, 7] = (u[:, 7] < p_vsplit).astype(np.float64)
    params[:, 8] = (u[:, 8] < p_hsplit).astype(np.float64)

    return params


# -----------------------------
# Dataset generation
# -----------------------------
def generate_dataset(
    n_samples,
    seed=0,
    scramble=True,
    p_invert=0.08,
    p_vsplit=0.10,
    p_hsplit=0.10,
    resolution=20,
    split_gap=0.06,
    frame_limit=0.4,
    m_min=1,
    m_max=12,
):
    params = sample_structured_sobol(
        n_samples,
        seed=seed,
        scramble=scramble,
        p_invert=p_invert,
        p_vsplit=p_vsplit,
        p_hsplit=p_hsplit,
        m_min=m_min,
        m_max=m_max,
    )

    bitmaps = np.empty((n_samples, resolution, resolution), dtype=np.uint8)
    for i in range(n_samples):
        bitmaps[i] = params_to_shape(
            params[i],
            resolution=resolution,
            split_gap=split_gap,
            frame_limit=frame_limit,
        )

    return params, bitmaps


# -----------------------------
# Plotting helper
# -----------------------------
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

        m = int(round(params[i, 0]))
        invert = int(params[i, 6])
        vs = int(params[i, 7])
        hs = int(params[i, 8])

        ax.set_title(
            f"{title}\nidx={i}\n"
            f"m={m} inv={invert} vs={vs} hs={hs}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    N = 300
    RES = 60

    # Tune split_gap here if you want a slightly different fixed gap size:
    # e.g. 0.05 (smaller), 0.07 (larger)
    split_gap = 0.06
    frame_limit = 0.4  # central 80%

    params, bitmaps = generate_dataset(
        N,
        seed=np.random.randint(1_000_000),
        scramble=True,
        p_invert=0.08,
        p_vsplit=0.10,
        p_hsplit=0.10,
        resolution=RES,
        split_gap=split_gap,
        frame_limit=frame_limit,
        m_min=1,
        m_max=12,
    )

    print("params:", params.shape, "(9D)")
    print("bitmaps:", bitmaps.shape)
    print("unique bitmap values:", np.unique(bitmaps))

    plot_random_samples(params, bitmaps, k=12, seed=42, title="Gielis + fixed splits + central frame")