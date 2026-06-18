#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:49:34 2026

@author: luka
"""


import numpy as np
from scipy.optimize import differential_evolution

def generate_super_shape(
    w_circle, w_square, w_cross, w_dual, hollow_weight, split_weight, invert_weight,
    r, s, cross_w, cross_l, rect_x, rect_w, rect_h, sq_x, sq_s, thickness, gap_width,
    layer_weight=0.0, num_layers=3, layer_thickness=0.08, line_thickness=0.015,
    resolution=64
):
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)

    # --- base SDF primitives (unchanged) ---
    d_circle = np.sqrt(X**2 + Y**2) - r
    d_square = np.maximum(np.abs(X) - s, np.abs(Y) - s)
    d_cross_v = np.maximum(np.abs(X) - cross_w, np.abs(Y) - cross_l)
    d_cross_h = np.maximum(np.abs(X) - cross_l, np.abs(Y) - cross_w)
    d_cross = np.minimum(d_cross_v, d_cross_h)
    d_rect = np.maximum(np.abs(X - rect_x) - rect_w, np.abs(Y) - rect_h)
    d_sq = np.maximum(np.abs(X - sq_x) - sq_s, np.abs(Y) - sq_s)
    d_dual = np.minimum(d_rect, d_sq)

    total_base_weight = w_circle + w_square + w_cross + w_dual
    if total_base_weight == 0:
        w_circle = 1.0; total_base_weight = 1.0
    w_c = w_circle / total_base_weight
    w_s = w_square / total_base_weight
    w_cr = w_cross / total_base_weight
    w_d = w_dual / total_base_weight
    d_base = (w_c*d_circle) + (w_s*d_square) + (w_cr*d_cross) + (w_d*d_dual)

    # --- hollowing ---
    d_hollowed = np.abs(d_base) - thickness
    d_current = ((1.0 - hollow_weight)*d_base) + (hollow_weight*d_hollowed)

    # --- concentric outline layers (REWRITTEN) ---
    # Draw `num_layers` thin outlines stepping inward from the boundary,
    # spaced `layer_thickness` apart, each `line_thickness` wide.
    # An outline at offset k is the set where |d_base + k*spacing| <= line_thickness.
    if layer_weight > 0:
        outlines = np.zeros_like(d_base, dtype=bool)
        for k in range(num_layers):
            offset = k * layer_thickness
            outlines |= (np.abs(d_base + offset) <= line_thickness)
        d_layered = np.where(outlines, -1.0, 1.0)
        # hard blend so high layer_weight fully selects the outlines
        d_current = ((1.0 - layer_weight)*d_current) + (layer_weight*d_layered)

    # --- splitting ---
    d_gap = gap_width - np.abs(X)
    d_split = np.maximum(d_current, d_gap)
    d_final = ((1.0 - split_weight)*d_current) + (split_weight*d_gap if False else split_weight*d_split)

    # --- inversion ---
    d_final = d_final * (1.0 - 2.0*invert_weight)

    bitmap = (d_final <= 0).astype(int)
    return bitmap

resolution = 209

bounds_geom = [
    (0.1, 0.45), (0.1, 0.45), (0.025, 0.2), (0.2, 0.475),
    (-0.4, -0.05), (0.05, 0.2), (0.1, 0.4), (0.05, 0.4),
    (0.05, 0.2), (0.025, 0.15), (0.025, 0.2)
]

def params_to_shape(params):
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv, m_layer = params[:8]
    mapped_constants = list(params[8:19])
    layer_thickness = params[19]
    line_thickness  = params[20]   # new
    m_hollow = 0.0
    return generate_super_shape(
        w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
        *mapped_constants,
        layer_weight=m_layer, num_layers=3,
        layer_thickness=layer_thickness, line_thickness=line_thickness,
        resolution=resolution
    )




class ShapeObjective:
    def __init__(self, target):
        self.target = target.astype(np.float64)
    def __call__(self, params):
        try:
            shape = params_to_shape(params).astype(np.float64)
        except Exception:
            return 1.0
        return np.mean(shape != self.target)


def fit_shape(target, maxiter=300, popsize=25, seed=None, workers=1, verbose=True):
    assert target.shape == (resolution, resolution)
    # bounds: add line_thickness range
    bounds = ([(0.0,1.0)]*7 + [(0.85,1.0)] + bounds_geom
          + [(0.03, 0.15)]    # layer_thickness (spacing)
          + [(0.005, 0.03)])  # line_thickness
    objective = ShapeObjective(target)
    result = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7, seed=seed,
        polish=False, workers=workers,
        updating='deferred' if workers != 1 else 'immediate', disp=verbose,
    )
    fitted = params_to_shape(result.x)
    return result.x, result.fun, fitted

import matplotlib.pyplot as plt


# from PIL import Image

# img = Image.open("unitCell.jpeg").convert("L")   # load as grayscale
# arr = np.array(img)
# print(arr.min(), arr.max())
# print("fraction white:", (arr == 255).mean())
# bitmap = (arr < 128).astype(int)   # cross = 1


# target_bitmap = bitmap  # or load your own bitmap as a (209, 209) int array


# resolution = 209

# def make_circle_bitmap(resolution=209, radius_frac=0.3, cx_frac=0.5, cy_frac=0.5):
#     """
#     Returns a (resolution, resolution) int bitmap: 1 inside the circle, 0 outside.
#     Coordinates are normalized to [0, 1] then scaled, matching origin='lower'.
#     radius_frac, cx_frac, cy_frac are fractions of the image size.
#     """
#     yy, xx = np.mgrid[0:resolution, 0:resolution]
#     # normalize pixel coords to [0, 1]
#     x = xx / (resolution - 1)
#     y = yy / (resolution - 1)
#     cx, cy = cx_frac, cy_frac
#     dist = np.sqrt((x - cx)**2 + (y - cy)**2)
#     bitmap = (dist <= radius_frac).astype(int)
#     return bitmap

# target_bitmap = make_circle_bitmap(resolution, radius_frac=0.3)



resolution = 209

# def make_square_in_square(resolution=209, outer_half=0.30, inner_half=0.15,
#                           cx_frac=0.5, cy_frac=0.5, filled_inner=False):
#     """
#     Returns a (resolution, resolution) int bitmap.
#     By default: 1 in the region between the outer and inner square (a frame), 0 elsewhere.
#     Set filled_inner=True to also fill the inner square (concentric solid squares look,
#     which on a binary bitmap is just a solid square unless you want a distinct value).

#     outer_half, inner_half: half-side lengths as fractions of image size.
#     """
#     yy, xx = np.mgrid[0:resolution, 0:resolution]
#     x = xx / (resolution - 1)
#     y = yy / (resolution - 1)
#     cx, cy = cx_frac, cy_frac

#     # Chebyshev (L-inf) distance from center -> square level sets
#     cheb = np.maximum(np.abs(x - cx), np.abs(y - cy))

#     outer = cheb <= outer_half
#     inner = cheb <= inner_half

#     if filled_inner:
#         bitmap = outer.astype(int)          # solid outer square
#     else:
#         bitmap = (outer & ~inner).astype(int)  # frame: outer minus inner hole
#     return bitmap

# target_bitmap = make_square_in_square(resolution, outer_half=0.30, inner_half=0.15)


def make_two_square_outlines_with_mid_line(resolution=209,
                                           outer_half=0.30, inner_half=0.12,
                                           edge_thick=0.012,
                                           mid_half=0.21, line_thick=0.008,
                                           cx_frac=0.5, cy_frac=0.5):
    """
    Two thin square OUTLINES (at outer_half and inner_half) with a third thin
    square LINE centered between them at mid_half. All on a black background.
    1 = white line/outline, 0 = background.
    """
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    x = xx / (resolution - 1)
    y = yy / (resolution - 1)
    cx, cy = cx_frac, cy_frac
    cheb = np.maximum(np.abs(x - cx), np.abs(y - cy))

    outer_line = np.abs(cheb - outer_half) <= edge_thick
    inner_line = np.abs(cheb - inner_half) <= edge_thick
    mid_line   = np.abs(cheb - mid_half)   <= line_thick

    bitmap = (outer_line | inner_line | mid_line).astype(int)
    return bitmap

target_bitmap = make_two_square_outlines_with_mid_line(resolution)


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