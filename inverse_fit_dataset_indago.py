#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import time
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from indago import PSO, DE, MSGD


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
# Single PSO run for a fixed m
# -----------------------------
def run_pso(loss_fn, lb, ub, iterations=20000, swarm_size=40, seed=None):
    if seed is not None:
        np.random.seed(seed)  

    optimizer = DE()
    optimizer.dimensions = len(lb)
    optimizer.lb = np.array(lb, dtype=float)
    optimizer.ub = np.array(ub, dtype=float)
    optimizer.evaluation_function = loss_fn
    optimizer.max_iterations = iterations
    optimizer.params['pop_init'] = swarm_size

    result = optimizer.optimize()
    return np.array(result.X, dtype=float), float(result.f)


# -----------------------------
# Fit m, a, b, n1, n2, n3 for a single bitmap, with retries
# -----------------------------
def fit_bitmap_once(bitmap, resolution, split_gap, frame_limit,
                     m_min, m_max, iterations, swarm_size, seed):

    invert = detect_invert(bitmap)
    vsplit = detect_split(bitmap, axis=0, resolution=resolution,
                           split_gap=split_gap, invert=invert)
    hsplit = detect_split(bitmap, axis=1, resolution=resolution,
                           split_gap=split_gap, invert=invert)

    lb = [0.25, 0.25, 0.20, 0.20, 0.20]
    ub = [1.20, 1.20, 6.00, 6.00, 6.00]

    best = None
    for m in range(m_min, m_max + 1):
        def loss(x, m=m):
            a, b, n1, n2, n3 = x
            pred = gielis_bitmap(
                m, a, b, n1, n2, n3, invert, vsplit, hsplit,
                resolution=resolution, split_gap=split_gap, frame_limit=frame_limit,
            )
            return float(np.mean(pred != bitmap))

        x_best, err = run_pso(loss, lb, ub, iterations=iterations,
                               swarm_size=swarm_size, seed=seed)

        if best is None or err < best[0]:
            best = (err, m, x_best.copy())

        if best[0] == 0.0:
            break

    err, m, (a, b, n1, n2, n3) = best

    return dict(
        m=m, a=a, b=b, n1=n1, n2=n2, n3=n3,
        invert=invert, vsplit=vsplit, hsplit=hsplit,
        pixel_error=err,
    )


def fit_bitmap(bitmap, resolution=100, split_gap=0.06, frame_limit=0.45,
               m_min=1, m_max=12, iterations=60, swarm_size=20,
               max_retries=5):

    fit = None
    for attempt in range(max_retries):
        seed = attempt
        fit = fit_bitmap_once(bitmap, resolution, split_gap, frame_limit,
                               m_min, m_max, iterations, swarm_size, seed)
        if fit["pixel_error"] == 0.0:
            fit["attempts"] = attempt + 1
            return fit

    fit["attempts"] = max_retries
    return fit


# -----------------------------
# Filename parsing / loading
# -----------------------------
FNAME_RE = re.compile(r"bitmap_(\d+)_periodicity_(\d+)\.png")


def load_bitmap(path, resolution=100):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != (resolution, resolution):
        raise ValueError(f"Unexpected shape {arr.shape} for {path}")
    return (arr > 127).astype(np.uint8)


def parse_filename(fname):
    match = FNAME_RE.match(fname)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def process_file(fname, folder, resolution, split_gap, frame_limit,
                  m_min, m_max, iterations, swarm_size, max_retries):
    parsed = parse_filename(fname)
    if parsed is None:
        return None
    idx, periodicity = parsed

    t0 = time.time()
    bitmap = load_bitmap(os.path.join(folder, fname), resolution=resolution)
    fit = fit_bitmap(bitmap, resolution=resolution, split_gap=split_gap,
                      frame_limit=frame_limit, m_min=m_min, m_max=m_max,
                      iterations=iterations, swarm_size=swarm_size,
                      max_retries=max_retries)
    elapsed = time.time() - t0

    fit["idx"] = idx
    fit["periodicity"] = periodicity
    fit["fname"] = fname

    status = "OK" if fit["pixel_error"] == 0.0 else "WARN"
    print(f"[{status}] idx={idx:4d}  {fname:35s}  "
          f"m={fit['m']:2d}  pixel_error={fit['pixel_error']:.6f}  "
          f"attempts={fit['attempts']}  time={elapsed:.2f}s")

    return fit


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    FOLDER = "bitmaps"
    RESOLUTION = 100
    SPLIT_GAP = 0.06
    FRAME_LIMIT = 0.45
    M_MIN, M_MAX = 1, 12
    ITERATIONS = 120
    SWARM_SIZE = 30
    MAX_RETRIES = 6
    N_JOBS = 8
    OUTPUT_CSV = "recovered_params.csv"

    columns = ["idx", "m", "a", "b", "n1", "n2", "n3",
               "invert", "vsplit", "hsplit", "periodicity", "pixel_error", "attempts"]

    files = [f for f in os.listdir(FOLDER) if f.endswith(".png")]
    print(f"Found {len(files)} PNG files in '{FOLDER}'")

    # --- report missing indices (skipped during generation) ---
    parsed = [parse_filename(f) for f in files]
    found_idx = sorted(p[0] for p in parsed if p is not None)
    if found_idx:
        expected = set(range(found_idx[0], found_idx[-1] + 1))
        missing = sorted(expected - set(found_idx))
        if missing:
            print(f"Missing {len(missing)} indices (likely skipped as empty during generation): {missing}")
        else:
            print("No missing indices in the found range.")

    # --- stream results to CSV as each fit completes ---
    results_iter = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(process_file)(fname, FOLDER, RESOLUTION, SPLIT_GAP, FRAME_LIMIT,
                               M_MIN, M_MAX, ITERATIONS, SWARM_SIZE, MAX_RETRIES)
        for fname in files
    )

    all_results = []
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for fit in results_iter:
            if fit is None:
                continue
            writer.writerow([fit[c] for c in columns])
            f.flush()
            all_results.append(fit)
            print(f"  -> appended idx={fit['idx']} to {OUTPUT_CSV}")

    # --- build final reindexed npy (fills gaps for missing/skipped bitmaps) ---
    df = pd.DataFrame(all_results)[columns]
    n_total = int(df["idx"].max()) + 1
    df = df.set_index("idx").reindex(range(n_total)).reset_index()

    param_cols = ["m", "a", "b", "n1", "n2", "n3", "invert", "vsplit", "hsplit", "periodicity"]
    arr = df[param_cols].to_numpy(dtype=np.float64)
    np.save("recovered_params.npy", arr)

    n_bad = int((df["pixel_error"] > 0).sum())
    print(f"\nDone. {len(all_results)} bitmaps processed, "
          f"{n_bad} did not reach an exact (0-error) match after {MAX_RETRIES} attempts.")
    print(f"Saved {OUTPUT_CSV} (streamed) and recovered_params.npy (final)")