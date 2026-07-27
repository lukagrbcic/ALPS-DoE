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
from indago import PSO, DE, FWA, BA


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
    ITERATIONS = 2000
    SWARM_SIZE = 100
    MAX_RETRIES = 3
    N_JOBS = 8

    OUTPUT_CSV = "recovered_params.csv"
    OUTPUT_NPY = "recovered_params.npy"

    columns = ["idx", "m", "a", "b", "n1", "n2", "n3",
               "invert", "vsplit", "hsplit", "periodicity", "pixel_error", "attempts"]

    files = [f for f in os.listdir(FOLDER) if f.endswith(".png")]
    print(f"Found {len(files)} PNG files in '{FOLDER}'")

    def parse_idx_only(fname):
        p = parse_filename(fname)
        return p[0] if p else None

    # Map idx -> filename (so we re-optimize by idx)
    files_by_idx = {}
    duplicates = 0
    for f in files:
        idx = parse_idx_only(f)
        if idx is None:
            continue
        if idx in files_by_idx:
            duplicates += 1
            continue
        files_by_idx[idx] = f
    if duplicates:
        print(f"WARNING: {duplicates} duplicate idx found in PNG names; keeping first occurrence.")

    csv_path = OUTPUT_CSV
    existing_df = None
    bad_idxs = None

    if os.path.exists(csv_path):
        print(f"Loading existing '{csv_path}' ...")
        existing_df = pd.read_csv(csv_path)

        required = {"idx", "pixel_error"}
        missing_cols = required - set(existing_df.columns)
        if missing_cols:
            raise RuntimeError(f"Existing CSV missing columns: {missing_cols}")

        # Treat pixel_error exactly as "need re-opt" when > 0
        # Also re-opt NaN rows (if any)
        existing_df["idx"] = existing_df["idx"].astype(int)
        existing_df["pixel_error"] = pd.to_numeric(existing_df["pixel_error"], errors="coerce")

        bad_mask = existing_df["pixel_error"].isna() | (existing_df["pixel_error"] > 0.009)
        bad_idxs = existing_df.loc[bad_mask, "idx"].astype(int).tolist()

        print(f"Existing CSV rows: {len(existing_df)}")
        print(f"Re-optimizing bad bitmaps: {len(bad_idxs)} (pixel_error > 0 or NaN)")
    else:
        print(f"'{csv_path}' not found -> optimizing ALL bitmaps (this will be slow).")
        bad_idxs = None

    if bad_idxs is None:
        files_to_process = list(files_by_idx.values())
        print(f"Optimizing {len(files_to_process)} bitmap(s) total.")
    else:
        files_to_process = [files_by_idx[i] for i in bad_idxs if i in files_by_idx]
        missing = [i for i in bad_idxs if i not in files_by_idx]

        print(f"Files matched for re-optimization: {len(files_to_process)}")
        if missing:
            print(f"WARNING: {len(missing)} bad idx have no corresponding PNG file; they will NOT be replaced:")
            print(f"  missing idx (first 20): {missing[:20]}")

    if not files_to_process:
        print("Nothing to optimize. Exiting.")
        raise SystemExit(0)

    print(f"Starting optimization with N_JOBS={N_JOBS} ...")

    # Run parallel; show progress from the MAIN process as results return
    results_iter = Parallel(n_jobs=N_JOBS, return_as="generator", verbose=10)(
        delayed(process_file)(
            fname, FOLDER, RESOLUTION, SPLIT_GAP, FRAME_LIMIT,
            M_MIN, M_MAX, ITERATIONS, SWARM_SIZE, MAX_RETRIES
        )
        for fname in files_to_process
    )

    new_results = []
    for k, fit in enumerate(results_iter, 1):
        if fit is None:
            continue
        new_results.append(fit)
        print(f"[{k}/{len(files_to_process)}] Re-optimized idx={fit['idx']} "
              f"pixel_error={fit['pixel_error']:.6f} attempts={fit.get('attempts','?')}")

    # Replace in CSV: drop old rows for the idx we re-optimized, then append new
    new_df = pd.DataFrame(new_results)
    for c in columns:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[columns]
    new_df["idx"] = new_df["idx"].astype(int)

    if existing_df is None:
        updated_df = new_df.sort_values("idx").reset_index(drop=True)
    else:
        processed_idxs = set(new_df["idx"].tolist())
        keep_df = existing_df[~existing_df["idx"].isin(processed_idxs)].copy()
        # Ensure columns
        for c in columns:
            if c not in keep_df.columns:
                keep_df[c] = np.nan

        updated_df = pd.concat([keep_df, new_df], ignore_index=True)
        updated_df = updated_df[columns].sort_values("idx").reset_index(drop=True)

    updated_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Updated '{OUTPUT_CSV}'. Replaced {len(new_df)} rows.")

    # Rebuild npy from updated CSV
    if len(updated_df) > 0:
        n_total = int(updated_df["idx"].max()) + 1
        df_reindexed = updated_df.set_index("idx").reindex(range(n_total)).reset_index()

        param_cols = ["m", "a", "b", "n1", "n2", "n3",
                       "invert", "vsplit", "hsplit", "periodicity"]
        arr = df_reindexed[param_cols].to_numpy(dtype=np.float64)
        np.save(OUTPUT_NPY, arr)
        print(f"Saved '{OUTPUT_NPY}'")