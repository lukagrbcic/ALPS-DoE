import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# GRID
# ============================================================
def make_grid(n):
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs)
    return X, Y

# ============================================================
# SDF PRIMITIVES  (negative = inside)
# ============================================================
def sdf_circle(X, Y, cx, cy, r):
    return np.sqrt((X - cx)**2 + (Y - cy)**2) - r

def sdf_box(X, Y, cx, cy, hw, hh):
    dx = np.abs(X - cx) - hw
    dy = np.abs(Y - cy) - hh
    outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
    inside = np.minimum(np.maximum(dx, dy), 0)
    return outside + inside

# ============================================================
# OPERATORS
# ============================================================
def op_union(a, b):     return np.minimum(a, b)
def op_intersect(a, b): return np.maximum(a, b)
def op_subtract(a, b):  return np.maximum(a, -b)   # a minus b
def op_inverse(a):      return -a

def to_bitmap(sdf, thresh=0.0):
    return (sdf <= thresh).astype(np.uint8)

# ============================================================
# SHAPE DEFINITIONS
# Each returns an SDF over the unit square.
# Params are kept in [0,1]-ish friendly ranges where possible.
# ============================================================
def shape_circle(X, Y, r=0.4, cx=0.5, cy=0.5):
    return sdf_circle(X, Y, cx, cy, r)

def shape_square(X, Y, hw=0.4, cx=0.5, cy=0.5):
    return sdf_box(X, Y, cx, cy, hw, hw)

def shape_rectangle(X, Y, hw=0.4, hh=0.25, cx=0.5, cy=0.5):
    return sdf_box(X, Y, cx, cy, hw, hh)

def shape_hollow_cylinder(X, Y, r_out=0.4, r_in=0.25, cx=0.5, cy=0.5):
    outer = sdf_circle(X, Y, cx, cy, r_out)
    inner = sdf_circle(X, Y, cx, cy, r_in)
    return op_subtract(outer, inner)

def shape_hollow_square(X, Y, hw_out=0.4, hw_in=0.25, cx=0.5, cy=0.5):
    outer = sdf_box(X, Y, cx, cy, hw_out, hw_out)
    inner = sdf_box(X, Y, cx, cy, hw_in, hw_in)
    return op_subtract(outer, inner)

def shape_square_in_square(X, Y, outer_hw=0.42, inner_hw=0.18):
    # solid outer frame + a filled square gap -> "square with another square inside"
    outer = sdf_box(X, Y, 0.5, 0.5, outer_hw, outer_hw)
    inner = sdf_box(X, Y, 0.5, 0.5, inner_hw, inner_hw)
    return op_subtract(outer, inner)

def shape_cross(X, Y, arm=0.42, thick=0.13):
    h = sdf_box(X, Y, 0.5, 0.5, arm, thick)
    v = sdf_box(X, Y, 0.5, 0.5, thick, arm)
    return op_union(h, v)

def shape_two_boxes(X, Y, gap=0.42, hw_a=0.16, hh_a=0.3, hw_b=0.15):
    a = sdf_box(X, Y, 0.5 - gap/2, 0.5, hw_a, hh_a)   # rectangle
    b = sdf_box(X, Y, 0.5 + gap/2, 0.5, hw_b, hw_b)   # square
    return op_union(a, b)

def shape_facing_semicircles(X, Y, r_out=0.22, r_in=0.13, gap=0.12):
    ring_l = op_subtract(sdf_circle(X, Y, 0.5 - gap, 0.5, r_out),
                         sdf_circle(X, Y, 0.5 - gap, 0.5, r_in))
    half_l = op_intersect(ring_l, sdf_box(X, Y, 1.0 - gap, 0.5, 0.5, 1.0))
    ring_r = op_subtract(sdf_circle(X, Y, 0.5 + gap, 0.5, r_out),
                         sdf_circle(X, Y, 0.5 + gap, 0.5, r_in))
    half_r = op_intersect(ring_r, sdf_box(X, Y, gap, 0.5, 0.5, 1.0))
    return op_union(half_l, half_r)

SHAPES = {
    "circle":            shape_circle,
    "square":            shape_square,
    "rectangle":         shape_rectangle,
    "hollow_cylinder":   shape_hollow_cylinder,
    "hollow_square":     shape_hollow_square,
    "square_in_square":  shape_square_in_square,
    "cross":             shape_cross,
    "two_boxes":         shape_two_boxes,
    "semicircles":       shape_facing_semicircles,
}

# ============================================================
# GENERATOR
# ============================================================
def generate(name, n=128, inverse=False, **params):
    X, Y = make_grid(n)
    sdf = SHAPES[name](X, Y, **params)
    if inverse:
        sdf = op_inverse(sdf)
    return to_bitmap(sdf)

# ============================================================
# PLOTTING
# ============================================================
def plot_one(bitmap, title=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(bitmap, origin="lower", cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_grid(samples, ncols=6, figsize=(14, 14)):
    """samples: list of (bitmap, title)"""
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, (bmp, title) in zip(axes, samples):
        ax.imshow(bmp, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ============================================================
# DESIGN OF EXPERIMENTS
# ============================================================
# For each shape we define parameter ranges. The DOE samples
# across these to produce a diverse dataset.
#
# Strategy: Latin Hypercube Sampling (LHS) per shape for good
# coverage of continuous params, plus a 50/50 inverse split.
# ============================================================

PARAM_RANGES = {
    "circle":           {"r": (0.15, 0.47)},
    "square":           {"hw": (0.15, 0.47)},
    "rectangle":        {"hw": (0.2, 0.47), "hh": (0.1, 0.35)},
    "hollow_cylinder":  {"r_out": (0.3, 0.47), "r_in": (0.1, 0.28)},
    "hollow_square":    {"hw_out": (0.3, 0.47), "hw_in": (0.1, 0.28)},
    "square_in_square": {"outer_hw": (0.35, 0.47), "inner_hw": (0.08, 0.25)},
    "cross":            {"arm": (0.3, 0.47), "thick": (0.08, 0.2)},
    "two_boxes":        {"gap": (0.3, 0.5), "hw_a": (0.1, 0.2),
                         "hh_a": (0.2, 0.35), "hw_b": (0.1, 0.2)},
    "semicircles":      {"r_out": (0.15, 0.28), "r_in": (0.06, 0.14),
                         "gap": (0.08, 0.18)},
}

def latin_hypercube(n_samples, ranges, rng):
    """Return list of param dicts via LHS."""
    keys = list(ranges.keys())
    d = len(keys)
    # LHS: stratify each dim into n_samples bins, shuffle independently
    result = np.zeros((n_samples, d))
    for j in range(d):
        perm = rng.permutation(n_samples)
        # one sample uniformly inside each stratum
        u = (perm + rng.random(n_samples)) / n_samples
        lo, hi = ranges[keys[j]]
        result[:, j] = lo + u * (hi - lo)
    return [dict(zip(keys, row)) for row in result]

def build_dataset(n_per_shape=20, n=128, seed=0, inverse_fraction=0.5):
    """
    Returns:
        bitmaps : (N, n, n) uint8 array
        meta    : list of dicts describing each sample
    """
    rng = np.random.default_rng(seed)
    bitmaps, meta = [], []

    for name, ranges in PARAM_RANGES.items():
        param_sets = (latin_hypercube(n_per_shape, ranges, rng)
                      if ranges else [{}] * n_per_shape)

        # enforce inner < outer style constraints where relevant
        param_sets = [_fix_constraints(name, p) for p in param_sets]

        # decide inverse flags: roughly half
        inv_flags = rng.random(n_per_shape) < inverse_fraction

        for p, inv in zip(param_sets, inv_flags):
            bmp = generate(name, n=n, inverse=bool(inv), **p)
            bitmaps.append(bmp)
            meta.append({"shape": name, "inverse": bool(inv), "params": p})

    return np.stack(bitmaps), meta

def _fix_constraints(name, p):
    """Make sure 'inner' dims stay smaller than 'outer' dims."""
    pairs = [("r_out", "r_in"), ("hw_out", "hw_in"),
             ("outer_hw", "inner_hw")]
    for outer, inner in pairs:
        if outer in p and inner in p and p[inner] >= p[outer] - 0.05:
            p[inner] = max(0.05, p[outer] - 0.1)
    return p

# ============================================================
# DEMO (continued)
# ============================================================
if __name__ == "__main__":
    # 1) Quick gallery: one default example of each shape (normal + inverse)
    gallery = []
    for name in SHAPES:
        gallery.append((generate(name, n=128, inverse=False), name))
        gallery.append((generate(name, n=128, inverse=True),  f"{name} (inv)"))
    plot_grid(gallery, ncols=6, figsize=(14, 8))

    # 2) Full DOE dataset
    bitmaps, meta = build_dataset(n_per_shape=12, n=128, seed=42)
    print(f"Dataset shape: {bitmaps.shape}")   # (N, 128, 128)

    # 3) Plot a random diverse subset
    rng = np.random.default_rng(1)
    idx = rng.choice(len(bitmaps), size=min(24, len(bitmaps)), replace=False)
    subset = [(bitmaps[i],
               f"{meta[i]['shape']}{' inv' if meta[i]['inverse'] else ''}")
              for i in idx]
    plot_grid(subset, ncols=6, figsize=(14, 10))

    # 4) Save dataset to disk
    np.save("shapes_bitmaps.npy", bitmaps)
    import json
    with open("shapes_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved shapes_bitmaps.npy and shapes_meta.json")

    # 5) Optional: quick diversity report
    from collections import Counter
    counts = Counter((m["shape"], m["inverse"]) for m in meta)
    print("\nDataset composition:")
    for (shape, inv), c in sorted(counts.items()):
        print(f"  {shape:18s} {'inv' if inv else 'normal':6s}: {c}")