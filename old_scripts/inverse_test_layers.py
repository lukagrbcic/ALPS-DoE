#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concentric-squares shape fitting via differential evolution.

Built for targets that are independent nested square outlines plus an
optional filled center (e.g. outer ring + inner ring + filled middle).
Uses independent radii so all features are reachable, with no toggle cliffs.
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from PIL import Image


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMAGE_PATH = "uc1.png"
resolution = 20


# ----------------------------------------------------------------------
# Generator: independent nested square bands + optional filled center
# ----------------------------------------------------------------------
def concentric_squares(params, resolution=20):
    """
    params = [h_outer, h_inner, edge, fill_half, fill_on, cx, cy]
        h_outer   : half-size (center-to-edge) of the outer square band
        h_inner   : half-size of the inner square band
        edge      : half-thickness of each band's line
        fill_half : half-size of the solid filled center block
        fill_on   : > 0.5 enables the filled center
        cx, cy    : center offset of the whole figure
    Returns a (resolution, resolution) int bitmap (1 = ink, 0 = background).
    """
    h_outer, h_inner, edge, fill_half, fill_on, cx, cy = params
    x = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, x)
    cheb = np.maximum(np.abs(X - cx), np.abs(Y - cy))

    inside  = np.abs(cheb - h_outer) <= edge          # outer ring
    inside |= np.abs(cheb - h_inner) <= edge          # inner ring
    if fill_on > 0.5:
        inside |= (cheb <= fill_half)                 # filled center

    return inside.astype(int)


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
class CSObjective:
    def __init__(self, target):
        self.target = target.astype(np.float64)
    def __call__(self, p):
        try:
            shape = concentric_squares(p, resolution).astype(np.float64)
        except Exception:
            return 1.0
        return np.mean(shape != self.target)


def fit_concentric(target, maxiter=2000, popsize=25, seed=0,
                   workers=2, verbose=True):
    assert target.shape == (resolution, resolution)
    bounds = [
        (0.15, 0.48),   # h_outer
        (0.05, 0.35),   # h_inner
        (0.01, 0.10),   # edge half-thickness
        (0.02, 0.20),   # fill_half
        (0.0, 1.0),     # fill_on (toggle at 0.5)
        (-0.15, 0.15),  # cx
        (-0.15, 0.15),  # cy
    ]
    objective = CSObjective(target)
    result = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7, seed=seed,
        polish=False, workers=workers,
        updating='deferred' if workers != 1 else 'immediate', disp=verbose,
    )
    fitted = concentric_squares(result.x, resolution)
    return result.x, result.fun, fitted


# ----------------------------------------------------------------------
# Load target
# ----------------------------------------------------------------------
img = Image.open(IMAGE_PATH).convert("L")   # load as grayscale
arr = np.array(img)
print("min/max:", arr.min(), arr.max())
print("unique values (first 20):", np.unique(arr)[:20])
print("fraction white:", (arr == 255).mean())

# Threshold to a clean bitmap (ink = 1). Resize to the working resolution
# with nearest-neighbour so we don't smear the thin lines.
bitmap_full = (arr < 128).astype(np.uint8)
bitmap_img = Image.fromarray(bitmap_full * 255).resize(
    (resolution, resolution), Image.NEAREST
)
target_bitmap = (np.array(bitmap_img) > 127).astype(int)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
best_params, best_score, fitted = fit_concentric(
    target_bitmap, maxiter=2000, popsize=25, seed=0, workers=2
)

print("Pixel mismatch fraction:", best_score)
print("Best params [h_outer, h_inner, edge, fill_half, fill_on, cx, cy]:")
print(best_params)
print("fill_on > 0.5 ?", best_params[4] > 0.5)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target_bitmap, cmap='gray_r', origin='lower')
axes[0].set_title("Target")
axes[1].imshow(fitted, cmap='gray_r', origin='lower')
axes[1].set_title(f"Fitted ({best_score:.3f})")
axes[2].imshow(np.abs(fitted.astype(int) - target_bitmap.astype(int)),
               cmap='Reds', origin='lower')
axes[2].set_title("Difference")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()