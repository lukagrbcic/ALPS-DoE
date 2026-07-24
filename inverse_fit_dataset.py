#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import time

# -----------------------------
# Gielis superformula (polar) - identical to the generation script
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
# Exact detection of invert/vsplit/hsplit (no fitting needed)
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
               m_min=1, m_max=12, maxiter=120, popsize=20, seed=0):

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
            break  # exact pixel match found, stop searching other m values

    err, m, (a, b, n1, n2, n3) = best

    return dict(
        m=m, a=a, b=b, n1=n1, n2=n2, n3=n3,
        invert=invert, vsplit=vsplit, hsplit=hsplit,
        pixel_error=err,
    )


# -----------------------------
# Filename parsing / loading
# -----------------------------
FNAME_RE = re.compile(r"bitmap_(\d+)_periodicity_(\d+)\.png")


def load_bitmap(path, resolution=100):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != (resolution, resolution):
        raise ValueError(f"Unexpected shape {arr.shape} for {path}")
    return (arr > 127).astype(np.uint8)  # back to 0/1


def parse_filename(fname):
    match = FNAME_RE.match(fname)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def process_file(fname, folder, resolution, split_gap, frame_limit, m_min, m_max):
    parsed = parse_filename(fname)
    if parsed is None:
        return None
    idx, periodicity = parsed
    t0 = time.time()

    bitmap = load_bitmap(os.path.join(folder, fname), resolution=resolution)
    fit = fit_bitmap(bitmap, resolution=resolution, split_gap=split_gap,
                      frame_limit=frame_limit, m_min=m_min, m_max=m_max)
    
    elapsed = time.time() - t0

    fit["idx"] = idx
    fit["periodicity"] = periodicity
    
    # fit["idx"] = idx
    # fit["periodicity"] = periodicity

    status = "OK" if fit["pixel_error"] == 0.0 else "WARN"
    print(f"[{status}] idx={idx:4d}  {fname:35s}  "
          f"m={fit['m']:2d}  pixel_error={fit['pixel_error']:.6f}  "
          f"time={elapsed:.2f}s")

    return fit


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    FOLDER = "bitmaps"        # change if your folder has a different name
    RESOLUTION = 100
    SPLIT_GAP = 0.06
    FRAME_LIMIT = 0.45
    M_MIN, M_MAX = 1, 12
    N_JOBS = 6                # use all CPU cores

    files = [f for f in os.listdir(FOLDER) if f.endswith(".png")]
    print(f"Found {len(files)} PNG files in '{FOLDER}'")

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_file)(fname, FOLDER, RESOLUTION, SPLIT_GAP, FRAME_LIMIT, M_MIN, M_MAX)
        for fname in files
    )
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["idx"])

    columns = ["idx", "m", "a", "b", "n1", "n2", "n3",
               "invert", "vsplit", "hsplit", "periodicity", "pixel_error"]
    df = pd.DataFrame(results)[columns]

    n_total = int(df["idx"].max()) + 1
    df = df.set_index("idx").reindex(range(n_total)).reset_index()

    df.to_csv("recovered_params.csv", index=False)

    param_cols = ["m", "a", "b", "n1", "n2", "n3", "invert", "vsplit", "hsplit", "periodicity"]
    arr = df[param_cols].to_numpy(dtype=np.float64)
    np.save("recovered_params.npy", arr)

    n_bad = int((df["pixel_error"] > 0).sum())
    print(f"Done. {len(df)} bitmaps processed, {n_bad} did not reach an exact (0-error) match.")
    print("Saved recovered_params.csv and recovered_params.npy")
