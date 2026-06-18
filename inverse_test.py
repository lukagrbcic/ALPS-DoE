#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:49:34 2026

@author: luka
"""

from generator import generate_super_shape

import numpy as np
from scipy.optimize import differential_evolution


resolution = 209

bounds_geom = [
    (0.1, 0.45), (0.1, 0.45), (0.025, 0.2), (0.2, 0.475),
    (-0.4, -0.05), (0.05, 0.2), (0.1, 0.4), (0.05, 0.4),
    (0.05, 0.2), (0.025, 0.15), (0.025, 0.2)
]


def params_to_shape(params):
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv, m_layer = params[:8]
    mapped_constants = list(params[8:19])
    return generate_super_shape(
        w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
        *mapped_constants,
        layer_weight=m_layer,
        resolution=resolution
    )


class ShapeObjective:
    """Top-level picklable objective for differential_evolution."""
    def __init__(self, target):
        self.target = target.astype(np.float64)

    def __call__(self, params):
        try:
            shape = params_to_shape(params).astype(np.float64)
        except Exception:
            return 1.0
        return np.mean(shape != self.target)


def fit_shape(target, maxiter=200, popsize=20, seed=None, workers=-1, verbose=True):
    assert target.shape == (resolution, resolution), \
        f"target must be {resolution}x{resolution}"

    bounds = [(0.0, 1.0)] * 8 + bounds_geom
    objective = ShapeObjective(target)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=False,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate',
        disp=verbose,
    )

    best_params = result.x
    best_score = result.fun
    fitted = params_to_shape(best_params)
    return best_params, best_score, fitted

import matplotlib.pyplot as plt


from PIL import Image

img = Image.open("unitCell.jpeg").convert("L")   # load as grayscale
arr = np.array(img)
print(arr.min(), arr.max())
print("fraction white:", (arr == 255).mean())
bitmap = (arr < 128).astype(int)   # cross = 1


target_bitmap = bitmap  # or load your own bitmap as a (209, 209) int array

best_params, best_score, fitted = fit_shape(
    target_bitmap, maxiter=150, popsize=20, seed=0, workers=-1
)

print("Pixel mismatch fraction:", best_score)
print("Best params:", best_params)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target_bitmap, cmap='gray_r', origin='lower'); axes[0].set_title("Target")
axes[1].imshow(fitted,        cmap='gray_r', origin='lower'); axes[1].set_title(f"Fitted ({best_score:.3f})")
axes[2].imshow(np.abs(fitted.astype(int) - target_bitmap.astype(int)),
               cmap='Reds', origin='lower'); axes[2].set_title("Difference")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()