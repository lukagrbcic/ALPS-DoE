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
    shape_mode=0,        # 0 = none, 1 = circle, 2 = square
    resolution=20,
    split_gap=0.06,
    frame_limit=0.45,
    rmax_angles=2048,
):
    # Grid in [-0.5, 0.5]
    x = np.linspace(-0.5, 0.5, resolution, dtype=np.float64)
    y = np.linspace(-0.5, 0.5, resolution, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    frame_mask = (np.abs(X) <= frame_limit) & (np.abs(Y) <= frame_limit)

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    angles_dense = np.linspace(-np.pi, np.pi, rmax_angles, dtype=np.float64)
    r_dense = gielis_r(angles_dense, m, a, b, n1, n2, n3)
    r_max = float(np.max(r_dense))
    r_max = max(r_max, 1e-12)

    r_theta = gielis_r(theta, m, a, b, n1, n2, n3)
    r_scaled = r_theta * (frame_limit / r_max)

    gielis_mask = frame_mask & (rho <= r_scaled)

    # Build the split "cut" strips once, to be applied only to the inner shape
    vstrip = np.abs(X) <= split_gap
    hstrip = np.abs(Y) <= split_gap

    shape_mode = int(shape_mode)

    if shape_mode == 0:
        # No primitive: the Gielis shape itself IS the inner shape.
        # Carve the splits directly out of it.
        inner = gielis_mask
        if int(vsplit) == 1:
            inner = inner & (~vstrip)
        if int(hsplit) == 1:
            inner = inner & (~hstrip)

        mask = inner
        if int(invert) == 1:
            mask = ~mask

    else:
        if shape_mode == 1:
            primitive_mask = frame_mask & (rho <= frame_limit)   # circle
        elif shape_mode == 2:
            primitive_mask = frame_mask.copy()                   # square
        else:
            raise ValueError(f"Unknown shape_mode: {shape_mode}")

        # The inner shape is the Gielis "hole" living inside the primitive.
        # Apply the splits to THIS region only, before computing the ring,
        # so the primitive's outer boundary is never touched by the split.
        inner = gielis_mask & primitive_mask
        if int(vsplit) == 1:
            inner = inner & (~vstrip)
        if int(hsplit) == 1:
            inner = inner & (~hstrip)

        # ring = primitive material minus the (now split-modified) inner hole
        ring = primitive_mask & (~inner)

        if int(invert) == 0:
            mask = ring          # background=0, ring=1, interior=0 -> W-B-W
        else:
            mask = ~ring         # background=1, ring=0, interior=1 -> B-W-B

    return mask.astype(np.uint8)

# -----------------------------
# Params -> bitmap
# -----------------------------
def params_to_shape(params, resolution=20, split_gap=0.06, frame_limit=0.4):
    """
    params layout (10 dims):
    [0]=m, [1]=a, [2]=b, [3]=n1, [4]=n2, [5]=n3,
    [6]=invert (0/1),
    [7]=vsplit (0/1),
    [8]=hsplit (0/1),
    [9]=shape_mode (0=none, 1=circle, 2=square)
    """
    m = int(round(params[0]))
    a, b, n1, n2, n3 = params[1:6]
    invert, vsplit, hsplit, shape_mode = params[6:10]

    return gielis_bitmap(
        m=m, a=a, b=b, n1=n1, n2=n2, n3=n3,
        invert=invert, vsplit=vsplit, hsplit=hsplit,
        shape_mode=shape_mode,
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
        (0.0, 3.0),         # shape_mode raw (thresholded into 0/1/2)
    ]


def sample_structured_sobol(
    n_samples,
    seed=0,
    scramble=True,
    p_invert=0.08,
    p_vsplit=0.10,
    p_hsplit=0.10,
    p_shape_circle=0.30,
    p_shape_square=0.30,
    m_min=1,
    m_max=12,
    period_min=200,
    period_max=300,
    period_delta=10,
):
    # --- Sobol ONLY for continuous geometry dims: m, a, b, n1, n2, n3 ---
    bounds_geom = make_param_bounds(m_min=m_min, m_max=m_max)[:6]  # just the 6 continuous ones
    d = len(bounds_geom)  # 6

    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    u_geom = sampler.random(n_samples)  # [0,1]^6

    lo = np.array([b[0] for b in bounds_geom], dtype=np.float64)
    hi = np.array([b[1] for b in bounds_geom], dtype=np.float64)
    geom_vals = lo + u_geom * (hi - lo)   # shape (n_samples, 6)
    geom_vals[:, 0] = np.clip(np.round(geom_vals[:, 0]), m_min, m_max)  # m

    # --- Independent RNG for all discrete/categorical toggles ---
    # (decoupled from Sobol so probabilities behave like true independent draws,
    #  avoiding correlation artifacts between dims)
    rng = np.random.default_rng(seed if seed is not None else None)

    invert = (rng.random(n_samples) < p_invert).astype(np.float64)
    vsplit = (rng.random(n_samples) < p_vsplit).astype(np.float64)
    hsplit = (rng.random(n_samples) < p_hsplit).astype(np.float64)

    p_shape_none = 1.0 - p_shape_circle - p_shape_square
    if p_shape_none < 0:
        raise ValueError("p_shape_circle + p_shape_square must be <= 1.0")

    shape_rand = rng.random(n_samples)
    shape_mode = np.zeros(n_samples, dtype=np.float64)
    shape_mode[(shape_rand >= p_shape_none) &
               (shape_rand < p_shape_none + p_shape_circle)] = 1.0
    shape_mode[shape_rand >= p_shape_none + p_shape_circle] = 2.0

    # --- periodicity: discrete values period_min..period_max step period_delta ---
    n_steps = int(round((period_max - period_min) / period_delta))
    k = rng.integers(0, n_steps + 1, size=n_samples)
    periodicity = (period_min + period_delta * k).astype(int)
    periodicity = np.clip(periodicity, period_min, period_max)

    # --- assemble full 10D geometry param array ---
    params = np.empty((n_samples, 10), dtype=np.float64)
    params[:, 0:6] = geom_vals
    params[:, 6] = invert
    params[:, 7] = vsplit
    params[:, 8] = hsplit
    params[:, 9] = shape_mode

    return params, periodicity

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
    p_shape_circle=0.333,
    p_shape_square=0.333,
    resolution=20,
    split_gap=0.06,
    frame_limit=0.4,
    m_min=1,
    m_max=12,
    period_min=200,
    period_max=300,
    period_delta=10,
):
    params_geom, periodicity = sample_structured_sobol(
        n_samples,
        seed=seed,
        scramble=scramble,
        p_invert=p_invert,
        p_vsplit=p_vsplit,
        p_hsplit=p_hsplit,
        p_shape_circle=p_shape_circle,
        p_shape_square=p_shape_square,
        m_min=m_min,
        m_max=m_max,
        period_min=period_min,
        period_max=period_max,
        period_delta=period_delta,
    )

    bitmaps = np.empty((n_samples, resolution, resolution), dtype=np.uint8)
    for i in range(n_samples):
        bitmaps[i] = params_to_shape(
            params_geom[i],
            resolution=resolution,
            split_gap=split_gap,
            frame_limit=frame_limit,
        )

    return params_geom, bitmaps, periodicity


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
        sh = int(params[i, 9])

        ax.set_title(
            f"{title}\nidx={i}\n"
            f"m={m} inv={invert} vs={vs} hs={hs} sh={sh}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    N = 1000 #set the size of the dataset
    RES = 100 #set the resolution per dimension, total pixels will be RESxRES

    split_gap = 0.06 #gap in % when we apply the horizontal and/or vertical slices
    frame_limit = 0.45  # central 90%, 5% buffer on all sides of the unit cell

    params, bitmaps, periodicity = generate_dataset(
        N,
        seed=np.random.randint(1_000_000),
        scramble=True,
        p_invert=0.5, #chance of applying inversion
        p_vsplit=0.25, #chance of vertical splitting
        p_hsplit=0.25, #chance of horizontal splitting
        p_shape_circle=0.333, #chance of placing shape into a circle primitive
        p_shape_square=0.333, #chance of placing shape into a square primitive
        resolution=RES,
        split_gap=split_gap,
        frame_limit=frame_limit,
        m_min=1,
        m_max=12,
    )

    print("params:", params.shape, "(10D)")
    print("bitmaps:", bitmaps.shape)
    print("unique bitmap values:", np.unique(bitmaps))

    from PIL import Image
    import os
    import csv

    os.makedirs("bitmaps_test", exist_ok=True)

    saved = 0
    skipped = 0

    csv_rows = []

    for idx, i in enumerate(bitmaps):
        m = int(round(params[idx, 0]))
        a, b, n1, n2, n3 = params[idx, 1:6]
        invert = int(params[idx, 6])
        vsplit = int(params[idx, 7])
        hsplit = int(params[idx, 8])
        shape_mode = int(params[idx, 9])
        per = int(periodicity[idx])

        is_empty = np.all(i == 0) or np.all(i == 1)
        filename = f"bitmap_{idx}_periodicity_{per}.png" if not is_empty else ""

        # record every sample's parameters regardless of whether it was saved
        csv_rows.append({
            "index": idx,
            "filename": filename,
            "saved": int(not is_empty),
            "m": m,
            "a": a,
            "b": b,
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "invert": invert,
            "vsplit": vsplit,
            "hsplit": hsplit,
            "shape_mode": shape_mode,
            "periodicity": per,
        })

        if is_empty:
            skipped += 1
            continue

        bitmap_matrix = (i * 255).astype(np.uint8)
        img = Image.fromarray(bitmap_matrix, mode='L')
        img.save(os.path.join("bitmaps_test", filename))

        saved += 1
        print(f"Saved {filename}")

    print("saved:", saved, "skipped:", skipped)

    # Write CSV with all parameters, keyed by index (matches bitmap filenames)
    csv_path = os.path.join("bitmaps_test", "dataset_params.csv")
    fieldnames = [
        "index", "filename", "saved",
        "m", "a", "b", "n1", "n2", "n3",
        "invert", "vsplit", "hsplit", "shape_mode",
        "periodicity",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved parameter CSV to {csv_path}")

    plot_random_samples(params, bitmaps, k=12, seed=42, title="Gielis + fixed splits + central frame")