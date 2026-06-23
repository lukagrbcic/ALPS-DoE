#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape fitting via differential evolution.
Handles plain-circle and square-in-square (frame) targets.
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------
def generate_super_shape(
    w_circle, w_square, w_cross, w_dual, hollow_weight, split_weight, invert_weight,
    r, s, cross_w, cross_l, rect_x, rect_w, rect_h, sq_x, sq_s, thickness, gap_width,
    layer_weight=0.0, num_layers=3, layer_thickness=0.08, line_thickness=0.015,
    resolution=64
):
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)
    

    # --- base SDF primitives (signed-ish fields) ---
    d_circle = np.sqrt(X**2 + Y**2) - r
    d_square = np.maximum(np.abs(X) - s, np.abs(Y) - s)
    d_cross_v = np.maximum(np.abs(X) - cross_w, np.abs(Y) - cross_l)
    d_cross_h = np.maximum(np.abs(X) - cross_l, np.abs(Y) - cross_w)
    d_cross = np.minimum(d_cross_v, d_cross_h)
    d_rect = np.maximum(np.abs(X - rect_x) - rect_w, np.abs(Y) - rect_h)
    d_sq = np.maximum(np.abs(X - sq_x) - sq_s, np.abs(Y) - sq_s)
    d_dual = np.minimum(d_rect, d_sq)

    # --- sharpened primitive blend (dominant shape wins -> no squircle/octagon) ---
    weights = np.array([w_circle, w_square, w_cross, w_dual], dtype=np.float64)
    if weights.sum() == 0:
        weights[0] = 1.0
    weights = weights ** 3
    weights = weights / weights.sum()
    d_base = (weights[0] * d_circle + weights[1] * d_square
              + weights[2] * d_cross + weights[3] * d_dual)

    # --- hollowing (frame / square-in-square) ---
    d_hollowed = np.abs(d_base) - thickness
    d_current = ((1.0 - hollow_weight) * d_base) + (hollow_weight * d_hollowed)

    # --- layers: HARD toggle -> concentric outlines via dist = abs(d_base) ---
    # Concentric outline bands: abs(dist - offset) <= line_thickness
    # where dist = abs(d_base) is ~ distance-to-boundary magnitude for this generator.
    if layer_weight > 0.5:
        dist = np.abs(d_base)
        outlines = np.zeros_like(d_base, dtype=bool)
        for k in range(num_layers):
            offset = k * layer_thickness
            outlines |= (np.abs(dist - offset) <= line_thickness)
        d_layered = np.where(outlines, -1.0, 1.0)  # inside=1 after final threshold
        d_current = d_layered

    # --- splitting ---
    d_gap = gap_width - np.abs(X)
    d_split = np.maximum(d_current, d_gap)
    d_final = ((1.0 - split_weight) * d_current) + (split_weight * d_split)

    # --- inversion ---
    d_final = d_final * (1.0 - 2.0 * invert_weight)

    return (d_final <= 0).astype(int)


# ----------------------------------------------------------------------
# Parameter mapping
# ----------------------------------------------------------------------
resolution = 20

bounds_geom = [
    (0.1, 0.45), (0.1, 0.45), (0.025, 0.2), (0.2, 0.475),
    (-0.4, -0.05), (0.05, 0.2), (0.1, 0.4), (0.05, 0.4),
    (0.05, 0.2), (0.025, 0.15), (0.025, 0.2)
]

def params_to_shape(params):
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv, m_layer = params[:8]
    mapped_constants = list(params[8:19])
    layer_thickness = params[19]
    line_thickness = params[20]
    return generate_super_shape(
        w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
        *mapped_constants,
        layer_weight=m_layer, num_layers=4,
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


def fit_shape(target, maxiter=150, popsize=20, seed=0, workers=-1, verbose=True):
    assert target.shape == (resolution, resolution)
    bounds = (
        ([(0.0, 1.0)] * 7      # 4 weights + hollow + split + invert
         + [(0.0, 1.0)]        # m_layer (full range; hard toggle at 0.5)
         + bounds_geom         # 11 geometry params
         + [(0.03, 0.15)]     # layer_thickness
         + [(0.005, 0.03)])   # line_thickness
    )
    objective = ShapeObjective(target)
    result = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.7, seed=seed,
        polish=False, workers=workers,
        updating='deferred' if workers != 1 else 'immediate', disp=verbose,
    )
    fitted = params_to_shape(result.x)
    return result.x, result.fun, fitted


# ----------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------
def make_circle_bitmap(resolution=209, radius_frac=0.3, cx_frac=0.5, cy_frac=0.5):
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    x = xx / (resolution - 1)
    y = yy / (resolution - 1)
    dist = np.sqrt((x - cx_frac)**2 + (y - cy_frac)**2)
    return (dist <= radius_frac).astype(int)


def make_square_in_square(resolution=209, outer_half=0.30, inner_half=0.15,
                          cx_frac=0.5, cy_frac=0.5, filled_inner=False):
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    x = xx / (resolution - 1)
    y = yy / (resolution - 1)
    cheb = np.maximum(np.abs(x - cx_frac), np.abs(y - cy_frac))
    outer = cheb <= outer_half
    inner = cheb <= inner_half
    if filled_inner:
        return outer.astype(int)
    return (outer & ~inner).astype(int)


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
    mid_line = np.abs(cheb - mid_half) <= line_thick

    bitmap = (outer_line | inner_line | mid_line).astype(int)
    return bitmap


def make_octagon_bitmap(resolution=209, radius_frac=0.30,
                        cx_frac=0.5, cy_frac=0.5):
    """
    Returns a (resolution, resolution) int bitmap: 1 inside a regular
    octagon, 0 outside. `radius_frac` is the circumradius (center-to-vertex)
    as a fraction of image size. Octagon is flat-top oriented.
    """
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    x = xx / (resolution - 1) - cx_frac
    y = yy / (resolution - 1) - cy_frac

    # Regular octagon as intersection of 8 half-planes.
    # Apothem (center-to-edge) for a regular octagon: a = R * cos(pi/8).
    apothem = radius_frac * np.cos(np.pi / 8)

    inside = np.ones_like(x, dtype=bool)
    for k in range(8):
        ang = np.pi / 8 + k * (np.pi / 4)   # edge normal directions
        nx, ny = np.cos(ang), np.sin(ang)
        inside &= (x * nx + y * ny) <= apothem

    return inside.astype(int)


def make_circle_in_octagon(resolution=209,
                           oct_radius_frac=0.30, circ_radius_frac=0.15,
                           cx_frac=0.5, cy_frac=0.5):
    """
    Returns a (resolution, resolution) int bitmap:
    1 in the region inside the octagon but OUTSIDE the inner circle
    (an octagonal ring around a circular hole), 0 elsewhere.
    """
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    x = xx / (resolution - 1) - cx_frac
    y = yy / (resolution - 1) - cy_frac

    # --- octagon as intersection of 8 half-planes ---
    apothem = oct_radius_frac * np.cos(np.pi / 8)
    inside_oct = np.ones_like(x, dtype=bool)
    for k in range(8):
        ang = np.pi / 8 + k * (np.pi / 4)
        nx, ny = np.cos(ang), np.sin(ang)
        inside_oct &= (x * nx + y * ny) <= apothem

    # --- inner circle ---
    inside_circ = (np.sqrt(x**2 + y**2) <= circ_radius_frac)

    # octagon minus the circular hole
    bitmap = (inside_oct & ~inside_circ).astype(int)
    return bitmap


from PIL import Image

img = Image.open("uc1.png").convert("L")   # load as grayscale
arr = np.array(img)
print(arr.min(), arr.max())
print("fraction white:", (arr == 255).mean())
bitmap = (arr < 128).astype(int)  # cross = 1


# ----------------------------------------------------------------------
# Run  --  pick ONE target
# ----------------------------------------------------------------------
target_bitmap = make_circle_bitmap(resolution, radius_frac=0.3)
target_bitmap = make_square_in_square(resolution, outer_half=0.30, inner_half=0.15)
target_bitmap = make_two_square_outlines_with_mid_line(resolution)
# target_bitmap = make_octagon_bitmap(resolution, radius_frac=0.30)
target_bitmap = make_circle_in_octagon(resolution, oct_radius_frac=0.30, circ_radius_frac=0.15)

target_bitmap = bitmap  # or load your own bitmap as a (209, 209) int array

best_params, best_score, fitted = fit_shape(
    target_bitmap, maxiter=5000, popsize=20, seed=0, workers=2
)

print("Pixel mismatch fraction:", best_score)
print("Best params:", best_params)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target_bitmap, cmap='gray_r', origin='lower'); axes[0].set_title("Target")
axes[1].imshow(fitted, cmap='gray_r', origin='lower'); axes[1].set_title(f"Fitted ({best_score:.3f})")
axes[2].imshow(np.abs(fitted.astype(int) - target_bitmap.astype(int)),
               cmap='Reds', origin='lower'); axes[2].set_title("Difference")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()