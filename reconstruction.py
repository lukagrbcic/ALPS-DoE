#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


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
# IO helpers
# -----------------------------
FNAME_RE = re.compile(r"bitmap_(\d+)_periodicity_(\d+)\.png")


def parse_filename(fname):
    match = FNAME_RE.match(fname)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def load_bitmap(path, resolution=100):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != (resolution, resolution):
        raise ValueError(f"Unexpected shape {arr.shape} for {path}")
    return (arr > 127).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="recovered_params.csv", help="Path to recovered_params.csv")
    ap.add_argument("--bitmaps", default="bitmaps", help="Folder containing bitmap_*.png")
    ap.add_argument("--out", default="bitmap_comparison", help="Output folder for comparisons")
    ap.add_argument("--resolution", type=int, default=100)
    ap.add_argument("--split-gap", type=float, default=0.06)
    ap.add_argument("--frame-limit", type=float, default=0.45)
    ap.add_argument("--rmax-angles", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = ["idx", "m", "a", "b", "n1", "n2", "n3", "invert", "vsplit", "hsplit"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    # Build idx -> filename map from PNGs
    png_files = [f for f in os.listdir(args.bitmaps) if f.endswith(".png")]
    idx_to_png = {}
    for f in png_files:
        p = parse_filename(f)
        if p is None:
            continue
        idx, _per = p
        # keep first if duplicates
        idx_to_png.setdefault(idx, f)

    df = df.sort_values("idx")
    param_cols = ["m", "a", "b", "n1", "n2", "n3", "invert", "vsplit", "hsplit"]

    print(f"Loaded CSV with {len(df)} rows.")
    print(f"Found {len(png_files)} PNGs, mapped {len(idx_to_png)} unique idx->file.")

    errors = []

    for i, row in df.iterrows():
        idx = int(row["idx"])

        # Pred params
        if row[param_cols].isna().any():
            print(f"[SKIP] idx={idx} has NaNs in params.")
            continue

        m = int(round(float(row["m"])))
        a = float(row["a"])
        b = float(row["b"])
        n1 = float(row["n1"])
        n2 = float(row["n2"])
        n3 = float(row["n3"])
        invert = int(round(float(row["invert"])))
        vsplit = int(round(float(row["vsplit"])))
        hsplit = int(round(float(row["hsplit"])))

        pred = gielis_bitmap(
            m, a, b, n1, n2, n3,
            invert, vsplit, hsplit,
            resolution=args.resolution,
            split_gap=args.split_gap,
            frame_limit=args.frame_limit,
            rmax_angles=args.rmax_angles,
        )

        fname = idx_to_png.get(idx, None)
        if fname is None:
            # still save predicted, but leave actual blank
            actual = np.zeros_like(pred)
            actual_loaded = False
        else:
            actual_path = os.path.join(args.bitmaps, fname)
            actual = load_bitmap(actual_path, resolution=args.resolution)
            actual_loaded = True

        pixel_error = float(np.mean(pred != actual))
        errors.append(pixel_error)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"idx={idx}  pixel_error={pixel_error:.6f}")

        axes[0].imshow(actual, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Actual")
        axes[0].axis("off")
        if actual_loaded:
            axes[0].text(0.01, -0.08, fname, transform=axes[0].transAxes, fontsize=7)

        axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted")
        axes[1].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        out_path = os.path.join(args.out, f"comparison_idx_{idx:05d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        if (len(errors) % 50) == 0:
            print(f"Processed {len(errors)} / {len(df)} ... last idx={idx} err={pixel_error:.6f}")

    if errors:
        print(f"\nDone. Saved comparisons to: {args.out}")
        print(f"Pixel error stats across plotted rows: mean={np.mean(errors):.6f}, max={np.max(errors):.6f}")
    else:
        print("Done, but no comparisons were produced (possible NaNs/skips).")


if __name__ == "__main__":
    main()