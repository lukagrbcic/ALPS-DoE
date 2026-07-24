#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import differential_evolution


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
    resolution=100,
    split_gap=0.06,
    frame_limit=0.45,
    rmax_angles=2048,
):
    x = np.linspace(-0.5, 0.5, resolution, dtype=np.float64)
    y = np.linspace(-0.5, 0.5, resolution, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    frame_mask = (np.abs(X) <= frame_limit) & (np.abs(Y) <= frame_limit)

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    angles_dense = np.linspace(-np.pi, np.pi, rmax_angles, dtype=np.float64)
    r_dense = gielis_r(angles_dense, m, a, b, n1, n2, n3)
    r_max = max(float(np.max(r_dense)), 1e-12)

    r_theta = gielis_r(theta, m, a, b, n1, n2, n3)
    r_scaled = r_theta * (frame_limit / r_max)

    mask = frame_mask & (rho <= r_scaled)

    if int(vsplit) == 1:
        mask &= ~(np.abs(X) <= split_gap)

    if int(hsplit) == 1:
        mask &= ~(np.abs(Y) <= split_gap)

    if int(invert) == 1:
        mask = ~mask

    return mask.astype(np.uint8)


# -----------------------------
# Exact detection of invert/vsplit/hsplit
# -----------------------------
def detect_invert(bitmap):
    corners = [bitmap[0, 0], bitmap[0, -1], bitmap[-1, 0], bitmap[-1, -1]]
    return int(round(float(np.mean(corners))))


def detect_split(bitmap, axis, resolution, split_gap, invert):
    x = np.linspace(-0.5, 0.5, resolution)
    idx = np.where(np.abs(x) <= split_gap)[0]
    if len(idx) == 0:
        return 0
    strip = bitmap[:, idx] if axis == 0 else bitmap[idx, :]
    return int(np.all(strip == invert))


# -----------------------------
# Fit m, a, b, n1, n2, n3 for a single bitmap
# -----------------------------
def fit_bitmap(bitmap, resolution=100, split_gap=0.06, frame_limit=0.45,
               m_min=1, m_max=12, maxiter=60, popsize=20, seed=0):

    invert = detect_invert(bitmap)
    vsplit = detect_split(bitmap, axis=0, resolution=resolution,
                           split_gap=split_gap, invert=invert)
    hsplit = detect_split(bitmap, axis=1, resolution=resolution,
                           split_gap=split_gap, invert=invert)

    bounds = [(0.25, 1.20), (0.25, 1.20), (0.20, 6.00), (0.20, 6.00), (0.20, 6.00)]

    best = None
    for m in range(m_min, m_max + 1):
        def loss(p, m=m):
            pred = gielis_bitmap(
                m, p[0], p[1], p[2], p[3], p[4],
                invert, vsplit, hsplit,
                resolution=resolution, split_gap=split_gap, frame_limit=frame_limit,
            )
            return float(np.mean(pred != bitmap))

        res = differential_evolution(
            loss, bounds, maxiter=maxiter, popsize=popsize,
            tol=1e-9, seed=seed, polish=True, workers=1,
        )

        if best is None or res.fun < best[0]:
            best = (res.fun, m, res.x.copy())

        if best[0] == 0.0:
            break

    err, m, (a, b, n1, n2, n3) = best

    return dict(
        m=m, a=a, b=b, n1=n1, n2=n2, n3=n3,
        invert=invert, vsplit=vsplit, hsplit=hsplit,
        pixel_error=err,
    )


# -----------------------------
# Load one real PNG and fit it
# -----------------------------
if __name__ == "__main__":
    FOLDER = "bitmaps"
    RESOLUTION = 100
    SPLIT_GAP = 0.06
    FRAME_LIMIT = 0.45

    files = [f for f in os.listdir(FOLDER) if f.endswith(".png")]
    if not files:
        raise RuntimeError(f"No PNG files found in '{FOLDER}'")

    fname = sorted(files)[0]  # pick the first one alphabetically
    fname = 'bitmap_338_periodicity_290.png'
    print(f"Loading: {fname}")

    match = re.match(r"bitmap_(\d+)_periodicity_(\d+)\.png", fname)
    
    idx, periodicity = (int(match.group(1)), int(match.group(2))) if match else (None, None)

    img = Image.open(os.path.join(FOLDER, fname)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    bitmap = (arr > 127).astype(np.uint8)

    print(f"idx={idx}, periodicity={periodicity}, shape={bitmap.shape}")

    recovered = fit_bitmap(
        bitmap, resolution=RESOLUTION, split_gap=SPLIT_GAP, frame_limit=FRAME_LIMIT,
    )

    print("\nRECOVERED PARAMS:")
    for k, v in recovered.items():
        print(f"  {k}: {v}")

    reconstructed = gielis_bitmap(
        m=recovered["m"], a=recovered["a"], b=recovered["b"],
        n1=recovered["n1"], n2=recovered["n2"], n3=recovered["n3"],
        invert=recovered["invert"], vsplit=recovered["vsplit"], hsplit=recovered["hsplit"],
        resolution=RESOLUTION, split_gap=SPLIT_GAP, frame_limit=FRAME_LIMIT,
    )

    diff = np.abs(bitmap.astype(int) - reconstructed.astype(int))
    print(f"\nPixel error: {recovered['pixel_error']:.6f}")
    print(f"Mismatched pixels: {diff.sum()} / {diff.size}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(bitmap, cmap="gray_r", origin="lower")
    axes[0].set_title(f"Original\n{fname}")
    axes[0].axis("off")

    axes[1].imshow(reconstructed, cmap="gray_r", origin="lower")
    axes[1].set_title("Reconstructed (fitted)")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="Reds", origin="lower")
    axes[2].set_title(f"Difference ({diff.sum()} px)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()