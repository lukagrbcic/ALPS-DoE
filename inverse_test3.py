#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape fitting via CMA-ES.
Linear primitive blend (octagons / squircles reachable), IoU loss.
Requires: pip install cma
"""

import numpy as np
import cma
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

    # --- base SDF primitives ---
    d_circle = np.sqrt(X**2 + Y**2) - r
    d_square = np.maximum(np.abs(X) - s, np.abs(Y) - s)
    d_cross_v = np.maximum(np.abs(X) - cross_w, np.abs(Y) - cross_l)
    d_cross_h = np.maximum(np.abs(X) - cross_l, np.abs(Y) - cross_w)
    d_cross = np.minimum(d_cross_v, d_cross_h)
    d_rect = np.maximum(np.abs(X - rect_x) - rect_w, np.abs(Y) - rect_h)
    d_sq = np.maximum(np.abs(X - sq_x) - sq_s, np.abs(Y) - sq_s)
    d_dual = np.minimum(d_rect, d_sq)

    # --- linear blend (octagons / squircles reachable) ---
    weights = np.array([w_circle, w_square, w_cross, w_dual], dtype=np.float64)
    if weights.sum() == 0:
        weights[0] = 1.0
    weights = weights / weights.sum()
    d_base = (weights[0]*d_circle + weights[1]*d_square
              + weights[2]*d_cross + weights[3]*d_dual)

    # --- hollowing ---
    d_hollowed = np.abs(d_base) - thickness
    d_current = ((1.0 - hollow_weight)*d_base) + (hollow_weight*d_hollowed)

    # --- layers: HARD toggle, true off-state below 0.5 ---
    if layer_weight > 0.5:
        outlines = np.zeros_like(d_base, dtype=bool)
        for k in range(num_layers):
            offset = k * layer_thickness
            outlines |= (np.abs(d_base + offset) <= line_thickness)
        d_layered = np.where(outlines, -1.0, 1.0)
        d_current = d_layered

    # --- splitting ---
    d_gap = gap_width - np.abs(X)
    d_split = np.maximum(d_current, d_gap)
    d_final = ((1.0 - split_weight)*d_current) + (split_weight*d_split)

    # --- inversion ---
    d_final = d_final * (1.0 - 2.0*invert_weight)

    return (d_final <= 0).astype(int)


# ----------------------------------------------------------------------
# Parameter mapping + bounds
# ----------------------------------------------------------------------
resolution = 209

# (lower, upper) for all 21 params
bounds_lo = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # weights + hollow/split/invert
    0.0,                                  # m_layer
    0.1, 0.1, 0.025, 0.2, -0.4, 0.05, 0.1, 0.05, 0.05, 0.025, 0.025,  # geom
    0.03,                                 # layer_thickness
    0.005,                                # line_thickness
])
bounds_hi = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
    0.45, 0.45, 0.2, 0.475, -0.05, 0.2, 0.4, 0.4, 0.2, 0.15, 0.2,
    0.15,
    0.03,
])
N_PARAMS = len(bounds_lo)


def params_to_shape(params):
    p = np.clip(params, bounds_lo, bounds_hi)  # safety clip
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv, m_layer = p[:8]
    mapped_constants = list(p[8:19])
    layer_thickness = p[19]
    line_thickness  = p[20]
    return generate_super_shape(
        w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
        *mapped_constants,
        layer_weight=m_layer, num_layers=3,
        layer_thickness=layer_thickness, line_thickness=line_thickness,
        resolution=resolution
    )


def iou_loss(params, target):
    try:
        shape = params_to_shape(params)
    except Exception:
        return 1.0
    inter = np.logical_and(shape, target).sum()
    union = np.logical_or(shape, target).sum()
    return 1.0 - (inter / union if union > 0 else 0.0)


# ----------------------------------------------------------------------
# CMA-ES fit
#   CMA-ES works in a normalized [0,1] space; we rescale to real bounds
#   inside the objective so the optimizer sees a well-conditioned problem.
# ----------------------------------------------------------------------
def fit_shape(target, maxiter=300, popsize=20, seed=0, sigma0=0.3, verbose=True):
    assert target.shape == (resolution, resolution)
    span = bounds_hi - bounds_lo

    def denorm(z):
        return bounds_lo + np.clip(z, 0.0, 1.0) * span

    def objective(z):
        return iou_loss(denorm(z), target)

    x0 = np.full(N_PARAMS, 0.5)  # start at the center of every range
    opts = {
        'bounds': [0.0, 1.0],
        'popsize': popsize,
        'maxiter': maxiter,
        'seed': seed,
        'verbose': 1 if verbose else -9,
        'tolfun': 1e-8,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(objective)

    best_z = es.result.xbest
    best_params = denorm(best_z)
    best_score = es.result.fbest
    fitted = params_to_shape(best_params)
    return best_params, best_score, fitted


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


# ----------------------------------------------------------------------
# Run  --  pick ONE target
# ----------------------------------------------------------------------
target_bitmap = make_circle_bitmap(resolution, radius_frac=0.3)
# target_bitmap = make_square_in_square(resolution, outer_half=0.30, inner_half=0.15)

best_params, best_score, fitted = fit_shape(
    target_bitmap, maxiter=300, popsize=20, seed=0
)

print("IoU loss:", best_score, " (IoU =", 1.0 - best_score, ")")
print("Best params:", best_params)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target_bitmap, cmap='gray_r', origin='lower'); axes[0].set_title("Target")
axes[1].imshow(fitted,        cmap='gray_r', origin='lower'); axes[1].set_title(f"Fitted ({best_score:.3f})")
axes[2].imshow(np.abs(fitted.astype(int) - target_bitmap.astype(int)),
               cmap='Reds', origin='lower'); axes[2].set_title("Difference")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()