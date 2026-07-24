#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact SDF shape generator.

A small parameter vector builds a buildable unit-cell pattern from one of a
few primitive families, with optional hollowing (frame) and a 2-fold split.
Everything is an SDF so shapes stay clean and laser-friendly.

Parameter vector (8 numbers, all in [0,1] unless noted):
    shape   : selects primitive family       (circle / square / cross / ring)
    size    : overall scale of the primitive
    aspect  : width/height ratio (1 = symmetric)
    rot     : rotation, 0..1 -> 0..pi/2
    hollow  : 0 = solid, >0 = frame of that wall thickness
    split   : 0 = whole, >0 = central gap of that width
    round   : corner rounding amount
    invert  : <0.5 keep, >=0.5 invert (figure<->ground)
"""

import numpy as np


# ----------------------------------------------------------------------
# SDF helpers
# ----------------------------------------------------------------------
def _sdf_circle(X, Y, size, aspect, round_):
    # ellipse approximated as scaled circle; round_ unused (already smooth)
    return np.sqrt((X / aspect) ** 2 + (Y * aspect) ** 2) - size


def _sdf_box(X, Y, size, aspect, round_):
    bx, by = size * aspect, size / aspect
    qx = np.abs(X) - (bx - round_)
    qy = np.abs(Y) - (by - round_)
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - round_


def _sdf_cross(X, Y, size, aspect, round_):
    arm = size * 0.28          # arm half-length
    th = size * 0.10 * aspect  # arm half-thickness
    v = np.maximum(np.abs(X) - th, np.abs(Y) - arm)
    h = np.maximum(np.abs(X) - arm, np.abs(Y) - th)
    return np.minimum(v, h) - round_


PRIMITIVES = [_sdf_circle, _sdf_box, _sdf_cross]


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------
def generate(params, resolution=209):
    shape, size, aspect, rot, hollow, split, round_, invert = params

    # --- map normalized params to physical ranges ---
    n_prim = len(PRIMITIVES)
    idx = min(int(shape * n_prim), n_prim - 1)   # which primitive
    size = 0.12 + 0.30 * size                      # 0.12 .. 0.42
    aspect = 0.6 + 0.8 * aspect                     # 0.6 .. 1.4
    angle = rot * (np.pi / 2)
    round_ = round_ * 0.08
    wall = hollow * 0.12                            # frame thickness
    gap = split * 0.15                              # central gap half-width

    # --- coordinate grid (rotated) ---
    t = np.linspace(-0.5, 0.5, resolution)
    Xg, Yg = np.meshgrid(t, t)
    c, s = np.cos(angle), np.sin(angle)
    X = Xg * c + Yg * s
    Y = -Xg * s + Yg * c

    # --- base primitive SDF ---
    d = PRIMITIVES[idx](X, Y, size, aspect, round_)

    # --- hollow into a frame: |d| - wall  (only when requested) ---
    if wall > 1e-3:
        d = np.abs(d) - wall

    # --- split: subtract a central vertical slab ---
    if gap > 1e-3:
        slab = gap - np.abs(X)        # >0 inside the slab
        d = np.maximum(d, slab)       # carve it out

    # --- invert figure/ground ---
    if invert >= 0.5:
        d = -d

    return (d <= 0).astype(np.uint8)


# ----------------------------------------------------------------------
# Random sampler for building a dataset
# ----------------------------------------------------------------------
def sample(rng):
    """Draw one random 8-vector with reasonable biases."""
    p = rng.random(8)
    # bias hollow/split/invert toward 'off' so most shapes are simple solids
    p[4] = 0.0 if rng.random() < 0.5 else p[4]   # hollow
    p[5] = 0.0 if rng.random() < 0.6 else p[5]   # split
    p[7] = 1.0 if rng.random() < 0.2 else 0.0    # invert
    return p


def generate_dataset(n, resolution=209, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([generate(sample(rng), resolution) for _ in range(n)])


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    RES = 209

    # hand-picked examples showing each capability
    examples = {
        "circle":        [0.0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ellipse rot":   [0.0, 0.7, 1.4 - 0.6, 0.3, 0.0, 0.0, 0.0, 0.0],
        "square":        [0.4, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rounded sq":    [0.4, 0.6, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
        "frame":         [0.4, 0.8, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0],
        "ring":          [0.0, 0.8, 1.0, 0.0, 0.4, 0.0, 0.0, 0.0],
        "cross":         [0.8, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "split square":  [0.4, 0.7, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
        "inverted ring": [0.0, 0.8, 1.0, 0.0, 0.4, 0.0, 0.0, 1.0],
    }

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, (name, p) in zip(axes.ravel(), examples.items()):
        ax.imshow(generate(np.array(p), RES), cmap='gray_r', origin='lower')
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    plt.suptitle("Compact SDF generator (8 params)")
    plt.tight_layout(); plt.show()

    # random dataset
    batch = generate_dataset(12, RES, seed=42)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for ax, bm in zip(axes.ravel(), batch):
        ax.imshow(bm, cmap='gray_r', origin='lower')
        ax.axis('off')
    plt.suptitle("Random batch")
    plt.tight_layout(); plt.show()