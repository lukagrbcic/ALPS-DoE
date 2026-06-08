"""
Parametric shape generator with a homogeneous primitive-list encoding.
Two interpretations of the SAME vector:
  - constrained: categoricals snapped -> clean, manufacturable shapes
  - free:        categoricals free      -> wider, exotic shapes
Supports: circle, box(square/rect), rings & frames (subtract),
crosses & adjacent boxes (union), facing semicircles (clipped rings),
and the inverse of anything.

Encoding (per primitive, P=9 slots, N=4 primitives):
  [present, type, cx, cy, sx, sy, op, clip_on, clip_angle]
plus 1 global inverse flag.
All slots stored normalized in [0,1].
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
N_PRIM      = 4          # primitives per shape
P_SLOTS     = 9          # params per primitive
VEC_LEN     = N_PRIM * P_SLOTS + 1   # +1 global inverse flag

# slot indices within one primitive block
S_PRESENT, S_TYPE, S_CX, S_CY, S_SX, S_SY, S_OP, S_CLIP_ON, S_CLIP_ANG = range(P_SLOTS)

TYPE_CIRCLE, TYPE_BOX = 0, 1
OP_UNION, OP_SUBTRACT, OP_INTERSECT = 0, 1, 2

# physical ranges (applied via lerp on normalized slots)
CX_RANGE   = (0.2, 0.8)
CY_RANGE   = (0.2, 0.8)
SX_RANGE   = (0.06, 0.45)   # radius / half-width
SY_RANGE   = (0.06, 0.45)   # half-height (box only)

def lerp(u, lo, hi): return lo + u * (hi - lo)

# ----------------------------------------------------------------------
# GRID + SDF PRIMITIVES
# ----------------------------------------------------------------------
def make_grid(n):
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs)
    return X, Y

def sdf_circle(X, Y, cx, cy, r):
    return np.sqrt((X - cx)**2 + (Y - cy)**2) - r

def sdf_box(X, Y, cx, cy, hw, hh):
    dx = np.abs(X - cx) - hw
    dy = np.abs(Y - cy) - hh
    outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
    inside = np.minimum(np.maximum(dx, dy), 0)
    return outside + inside

def sdf_halfplane(X, Y, cx, cy, angle):
    # signed distance to a line through (cx,cy) with given normal angle.
    # negative on the "kept" side. Used to clip rings into semicircles.
    nx, ny = np.cos(angle), np.sin(angle)
    return (X - cx) * nx + (Y - cy) * ny

# ----------------------------------------------------------------------
# CSG OPERATORS
# ----------------------------------------------------------------------
def op_union(a, b):     return np.minimum(a, b)
def op_intersect(a, b): return np.maximum(a, b)
def op_subtract(a, b):  return np.maximum(a, -b)
def to_bitmap(sdf, t=0.0): return (sdf <= t).astype(np.uint8)

# ----------------------------------------------------------------------
# DECODING:  vector -> primitive parameter dicts
# `snap=True` => constrained mode (categoricals rounded/argmax'd)
# `snap=False`=> free mode        (categoricals taken as-is via thresholds)
# ----------------------------------------------------------------------
def _block(v, i):
    return v[i * P_SLOTS:(i + 1) * P_SLOTS]

def decode(v, snap):
    prims = []
    for i in range(N_PRIM):
        b = _block(v, i)

        present = b[S_PRESENT] > 0.5
        if not present:
            continue

        # type: snap -> hard choice; free -> still a choice but threshold is
        # the only nonlinearity (we do not blend geometries between types)
        ptype = TYPE_BOX if b[S_TYPE] > 0.5 else TYPE_CIRCLE

        # op: 3-way categorical from one slot
        if b[S_OP] < 1/3:   op = OP_UNION
        elif b[S_OP] < 2/3: op = OP_SUBTRACT
        else:               op = OP_INTERSECT

        clip_on = b[S_CLIP_ON] > 0.5

        prims.append({
            "type": ptype,
            "op": op,
            "cx": lerp(b[S_CX], *CX_RANGE),
            "cy": lerp(b[S_CY], *CY_RANGE),
            "sx": lerp(b[S_SX], *SX_RANGE),
            "sy": lerp(b[S_SY], *SY_RANGE),
            "clip_on": clip_on,
            "clip_angle": lerp(b[S_CLIP_ANG], 0.0, 2*np.pi),
        })

    if snap:
        prims = _sanitize(prims)

    inverse = v[-1] > 0.5
    return prims, inverse

def _sanitize(prims):
    """Constrained-mode cleanup: ensure first primitive unions (gives a base),
    and avoid all-subtract degenerate shapes."""
    if not prims:
        return prims
    if all(p["op"] != OP_UNION for p in prims):
        prims[0]["op"] = OP_UNION
    return prims

# ----------------------------------------------------------------------
# RENDER:  primitive list -> SDF -> bitmap
# ----------------------------------------------------------------------
def _prim_sdf(X, Y, p):
    if p["type"] == TYPE_CIRCLE:
        d = sdf_circle(X, Y, p["cx"], p["cy"], p["sx"])
    else:
        d = sdf_box(X, Y, p["cx"], p["cy"], p["sx"], p["sy"])
    if p["clip_on"]:
        # intersect with a half-plane through the primitive center -> semicircle / half-box
        hp = sdf_halfplane(X, Y, p["cx"], p["cy"], p["clip_angle"])
        d = op_intersect(d, hp)
    return d

def render_vector(v, snap, n=128):
    prims, inverse = decode(v, snap)
    X, Y = make_grid(n)

    if not prims:
        # empty -> all background (or all foreground if inverse)
        field = np.full_like(X, 1.0)
    else:
        field = None
        for p in prims:
            d = _prim_sdf(X, Y, p)
            if field is None:
                field = d                       # first primitive seeds the field
            elif p["op"] == OP_UNION:
                field = op_union(field, d)
            elif p["op"] == OP_SUBTRACT:
                field = op_subtract(field, d)
            else:
                field = op_intersect(field, d)

    if inverse:
        field = -field
    return to_bitmap(field, n_to_thresh(n))

def n_to_thresh(n):
    # hard threshold at the zero level set
    return 0.0

# ----------------------------------------------------------------------
# HAND-BUILT EXAMPLES (sanity check the encoding can express your shapes)
# helper to set one primitive block
# ----------------------------------------------------------------------
def _set(v, i, present=1, type_=TYPE_CIRCLE, cx=0.5, cy=0.5, sx=0.3, sy=0.3,
         op=OP_UNION, clip_on=0, clip_ang=0.0):
    def inv_lerp(x, lo, hi): return (x - lo) / (hi - lo)
    b = np.zeros(P_SLOTS)
    b[S_PRESENT]  = 1.0 if present else 0.0
    b[S_TYPE]     = 1.0 if type_ == TYPE_BOX else 0.0
    b[S_CX]       = inv_lerp(cx, *CX_RANGE)
    b[S_CY]       = inv_lerp(cy, *CY_RANGE)
    b[S_SX]       = inv_lerp(sx, *SX_RANGE)
    b[S_SY]       = inv_lerp(sy, *SY_RANGE)
    b[S_OP]       = {OP_UNION:0.16, OP_SUBTRACT:0.5, OP_INTERSECT:0.83}[op]
    b[S_CLIP_ON]  = 1.0 if clip_on else 0.0
    b[S_CLIP_ANG] = clip_ang / (2*np.pi)
    v[i*P_SLOTS:(i+1)*P_SLOTS] = b
    return v

def example(name):
    v = np.zeros(VEC_LEN)
    if name == "circle":
        _set(v, 0, type_=TYPE_CIRCLE, sx=0.35)
    elif name == "square":
        _set(v, 0, type_=TYPE_BOX, sx=0.35, sy=0.35)
    elif name == "rectangle":
        _set(v, 0, type_=TYPE_BOX, sx=0.4, sy=0.22)
    elif name == "hollow_cylinder":
        _set(v, 0, type_=TYPE_CIRCLE, sx=0.4, op=OP_UNION)
        _set(v, 1, type_=TYPE_CIRCLE, sx=0.25, op=OP_SUBTRACT)
    elif name == "square_in_square":
        _set(v, 0, type_=TYPE_BOX, sx=0.42, sy=0.42, op=OP_UNION)
        _set(v, 1, type_=TYPE_BOX, sx=0.22, sy=0.22, op=OP_SUBTRACT)
    elif name == "cross":
        _set(v, 0, type_=TYPE_BOX, sx=0.42, sy=0.12, op=OP_UNION)
        _set(v, 1, type_=TYPE_BOX, sx=0.12, sy=0.42, op=OP_UNION)
    elif name == "two_boxes":
        _set(v, 0, type_=TYPE_BOX, cx=0.32, sx=0.13, sy=0.28, op=OP_UNION)
        _set(v, 1, type_=TYPE_BOX, cx=0.7,  sx=0.13, sy=0.13, op=OP_UNION)
    elif name == "semicircles":
        # two clipped rings facing each other = 4 primitives
        _set(v, 0, type_=TYPE_CIRCLE, cx=0.36, sx=0.2,  op=OP_UNION,    clip_on=1, clip_ang=0.0)
        _set(v, 1, type_=TYPE_CIRCLE, cx=0.36, sx=0.12, op=OP_SUBTRACT, clip_on=1, clip_ang=0.0)
        _set(v, 2, type_=TYPE_CIRCLE, cx=0.64, sx=0.2,  op=OP_UNION,    clip_on=1, clip_ang=np.pi)
        _set(v, 3, type_=TYPE_CIRCLE, cx=0.64, sx=0.12, op=OP_SUBTRACT, clip_on=1, clip_ang=np.pi)
    return v

EXAMPLE_NAMES = ["circle", "square", "rectangle", "hollow_cylinder",
                 "square_in_square", "cross", "two_boxes", "semicircles"]

# ----------------------------------------------------------------------
# QMC-LHS DATASET BUILDER  (works for both modes)
# ----------------------------------------------------------------------
def build_dataset(n_samples=64, mode="constrained", n=128, seed=0):
    """
    Samples the FULL VEC_LEN-dimensional unit cube with a QMC Latin
    hypercube, then decodes each point under the chosen mode.

    mode = "constrained" -> categoricals snapped, sanitized, clean shapes
    mode = "free"        -> categoricals taken raw, wilder shapes

    Returns vectors (N, VEC_LEN), bitmaps (N, n, n), meta list.
    """
    snap = (mode == "constrained")
    rng = np.random.default_rng(seed)

    sampler = qmc.LatinHypercube(d=VEC_LEN, seed=rng.integers(1 << 31))
    raw = sampler.random(n=n_samples)            # (n_samples, VEC_LEN) in [0,1]

    if mode == "constrained":
        raw = _bias_constrained(raw, rng)

    vectors, bitmaps, meta = [], [], []
    for v in raw:
        bmp = render_vector(v, snap=snap, n=n)
        # reject empty / near-empty / fully-filled shapes to keep dataset useful
        frac = bmp.mean()
        if frac < 0.01 or frac > 0.99:
            continue
        prims, inverse = decode(v, snap=snap)
        vectors.append(v)
        bitmaps.append(bmp)
        meta.append({"mode": mode, "inverse": bool(inverse),
                     "n_prims": len(prims), "fill_fraction": float(frac)})

    return np.array(vectors), np.array(bitmaps), meta

def _bias_constrained(raw, rng):
    """
    In constrained mode we want clean, recognizable shapes, so we nudge the
    categorical slots toward decisive values and make ~half the primitives
    'present'. This keeps the QMC coverage of the CONTINUOUS geometry slots
    while preventing a mush of overlapping subtractions.
    """
    raw = raw.copy()
    for i in range(N_PRIM):
        base = i * P_SLOTS
        # primitive 0 always present; others present ~50%
        raw[:, base + S_PRESENT] = 1.0 if i == 0 else (raw[:, base + S_PRESENT] > 0.5)
        # first primitive always unions (a base to build on)
        if i == 0:
            raw[:, base + S_OP] = 0.16   # union
        # bias clip off for most primitives (semicircles are special)
        raw[:, base + S_CLIP_ON] = (raw[:, base + S_CLIP_ON] > 0.75).astype(float)
    return raw

# ----------------------------------------------------------------------
# MODE 3: "templated" — canonical shape families, only their real
# parameters vary. Same VEC_LEN encoding out, guaranteed clean shapes.
#
# Each template: (param_ranges dict, builder(params)->VEC_LEN vector).
# params are normalized [0,1]; builder lerps them into meaningful values.
# ----------------------------------------------------------------------

def _tmpl_circle(p):
    v = np.zeros(VEC_LEN)
    r = lerp(p["r"], 0.15, 0.45)
    _set(v, 0, type_=TYPE_CIRCLE, sx=r, op=OP_UNION)
    return v

def _tmpl_square(p):
    v = np.zeros(VEC_LEN)
    hw = lerp(p["hw"], 0.15, 0.45)
    _set(v, 0, type_=TYPE_BOX, sx=hw, sy=hw, op=OP_UNION)
    return v

def _tmpl_rectangle(p):
    v = np.zeros(VEC_LEN)
    hw = lerp(p["hw"], 0.2, 0.45)
    hh = lerp(p["hh"], 0.1, 0.35)
    _set(v, 0, type_=TYPE_BOX, sx=hw, sy=hh, op=OP_UNION)
    return v

def _tmpl_hollow_cylinder(p):
    v = np.zeros(VEC_LEN)
    r_out = lerp(p["r_out"], 0.28, 0.45)
    r_in  = lerp(p["r_in"],  0.1, 0.85) * (r_out - 0.04)
    _set(v, 0, type_=TYPE_CIRCLE, sx=r_out,          op=OP_UNION)
    _set(v, 1, type_=TYPE_CIRCLE, sx=max(r_in,0.04), op=OP_SUBTRACT)
    return v

def _tmpl_hollow_square(p):
    v = np.zeros(VEC_LEN)
    hw_out = lerp(p["hw_out"], 0.28, 0.45)
    hw_in  = lerp(p["hw_in"],  0.1, 0.85) * (hw_out - 0.04)
    hw_in = max(hw_in, 0.04)
    _set(v, 0, type_=TYPE_BOX, sx=hw_out, sy=hw_out, op=OP_UNION)
    _set(v, 1, type_=TYPE_BOX, sx=hw_in,  sy=hw_in,  op=OP_SUBTRACT)
    return v

def _tmpl_cross(p):
    v = np.zeros(VEC_LEN)
    arm   = lerp(p["arm"],   0.28, 0.45)
    thick = lerp(p["thick"], 0.07, 0.2)
    _set(v, 0, type_=TYPE_BOX, sx=arm,   sy=thick, op=OP_UNION)
    _set(v, 1, type_=TYPE_BOX, sx=thick, sy=arm,   op=OP_UNION)
    return v

def _tmpl_two_boxes(p):
    v = np.zeros(VEC_LEN)
    gap  = lerp(p["gap"],  0.28, 0.46)
    hw_a = lerp(p["hw_a"], 0.09, 0.18)
    hh_a = lerp(p["hh_a"], 0.18, 0.32)
    hw_b = lerp(p["hw_b"], 0.09, 0.18)
    _set(v, 0, type_=TYPE_BOX, cx=0.5-gap/2, sx=hw_a, sy=hh_a, op=OP_UNION)
    _set(v, 1, type_=TYPE_BOX, cx=0.5+gap/2, sx=hw_b, sy=hw_b, op=OP_UNION)
    return v

def _tmpl_semicircles(p):
    v = np.zeros(VEC_LEN)
    r_out = lerp(p["r_out"], 0.15, 0.26)
    r_in  = lerp(p["r_in"],  0.1, 0.85) * (r_out - 0.03)
    r_in  = max(r_in, 0.03)
    gap   = lerp(p["gap"],   0.08, 0.18)
    cxl, cxr = 0.5 - gap, 0.5 + gap
    _set(v, 0, type_=TYPE_CIRCLE, cx=cxl, sx=r_out, op=OP_UNION,    clip_on=1, clip_ang=0.0)
    _set(v, 1, type_=TYPE_CIRCLE, cx=cxl, sx=r_in,  op=OP_SUBTRACT, clip_on=1, clip_ang=0.0)
    _set(v, 2, type_=TYPE_CIRCLE, cx=cxr, sx=r_out, op=OP_UNION,    clip_on=1, clip_ang=np.pi)
    _set(v, 3, type_=TYPE_CIRCLE, cx=cxr, sx=r_in,  op=OP_SUBTRACT, clip_on=1, clip_ang=np.pi)
    return v


# ----------------------------------------------------------------------
# Additional canonical families for templated mode.
# ----------------------------------------------------------------------

def _tmpl_pill(p):
    """Stadium / rounded slot: two end caps + connecting box."""
    v = np.zeros(VEC_LEN)
    half_len = lerp(p["len"], 0.15, 0.4)
    r        = lerp(p["r"],   0.07, 0.18)
    _set(v, 0, type_=TYPE_BOX,    sx=half_len, sy=r,           op=OP_UNION)
    _set(v, 1, type_=TYPE_CIRCLE, cx=0.5-half_len, sx=r,       op=OP_UNION)
    _set(v, 2, type_=TYPE_CIRCLE, cx=0.5+half_len, sx=r,       op=OP_UNION)
    return v

def _tmpl_L_shape(p):
    """L bracket: big box minus a corner box."""
    v = np.zeros(VEC_LEN)
    hw   = lerp(p["hw"],   0.28, 0.45)
    cut  = lerp(p["cut"],  0.35, 0.75) * hw      # corner cutout fraction
    _set(v, 0, type_=TYPE_BOX, sx=hw,  sy=hw,  op=OP_UNION)
    _set(v, 1, type_=TYPE_BOX, cx=0.5+hw-cut, cy=0.5+hw-cut,
              sx=cut, sy=cut, op=OP_SUBTRACT)
    return v

def _tmpl_slot(p):
    """Plate with a pill-shaped hole punched through."""
    v = np.zeros(VEC_LEN)
    hw_out   = lerp(p["hw_out"], 0.3, 0.45)
    hh_out   = lerp(p["hh_out"], 0.2, 0.4)
    half_len = lerp(p["len"], 0.1, 0.7) * (hw_out - 0.06)
    r        = lerp(p["r"],   0.3, 0.8) * (hh_out - 0.04)
    r = max(r, 0.03)
    _set(v, 0, type_=TYPE_BOX,    sx=hw_out, sy=hh_out, op=OP_UNION)
    _set(v, 1, type_=TYPE_BOX,    sx=half_len, sy=r,    op=OP_SUBTRACT)
    _set(v, 2, type_=TYPE_CIRCLE, cx=0.5-half_len, sx=r, op=OP_SUBTRACT)
    _set(v, 3, type_=TYPE_CIRCLE, cx=0.5+half_len, sx=r, op=OP_SUBTRACT)
    return v

def _tmpl_chamfered_box(p):
    """Box with one corner sliced off (rotated box subtract)."""
    v = np.zeros(VEC_LEN)
    hw   = lerp(p["hw"],   0.28, 0.45)
    cham = lerp(p["cham"], 0.2, 0.6) * hw
    _set(v, 0, type_=TYPE_BOX, sx=hw, sy=hw, op=OP_UNION)
    _set(v, 1, type_=TYPE_BOX, cx=0.5+hw, cy=0.5+hw,
              sx=cham, sy=cham, ang=np.pi/4, op=OP_SUBTRACT)
    return v

def _tmpl_capsule_ring(p):
    """Annulus but with a chunk removed (C-clip / open ring)."""
    v = np.zeros(VEC_LEN)
    r_out = lerp(p["r_out"], 0.3, 0.45)
    r_in  = lerp(p["r_in"],  0.4, 0.8) * (r_out - 0.04)
    gap_w = lerp(p["gap"],   0.06, 0.16)
    _set(v, 0, type_=TYPE_CIRCLE, sx=r_out,         op=OP_UNION)
    _set(v, 1, type_=TYPE_CIRCLE, sx=max(r_in,0.04),op=OP_SUBTRACT)
    _set(v, 2, type_=TYPE_BOX, cx=0.5+r_out/2, sx=r_out, sy=gap_w,
              op=OP_SUBTRACT)
    return v

def _tmpl_triangle(p):
    """Triangle approximated by clipping a box twice (or use TYPE_TRI if you have it)."""
    v = np.zeros(VEC_LEN)
    hw = lerp(p["hw"], 0.2, 0.42)
    hh = lerp(p["hh"], 0.2, 0.42)
    # box clipped by two angled half-planes -> wedge
    _set(v, 0, type_=TYPE_BOX, sx=hw, sy=hh, op=OP_UNION,
              clip_on=1, clip_ang=lerp(p["a1"], 0.6, 1.0))
    return v

def _tmpl_dumbbell(p):
    """Two discs joined by a thin bar."""
    v = np.zeros(VEC_LEN)
    gap = lerp(p["gap"], 0.18, 0.34)
    r   = lerp(p["r"],   0.1, 0.2)
    bar = lerp(p["bar"], 0.3, 0.7) * r
    _set(v, 0, type_=TYPE_CIRCLE, cx=0.5-gap, sx=r, op=OP_UNION)
    _set(v, 1, type_=TYPE_CIRCLE, cx=0.5+gap, sx=r, op=OP_UNION)
    _set(v, 2, type_=TYPE_BOX, sx=gap, sy=bar,      op=OP_UNION)
    return v

def _tmpl_grid_holes(p):
    """Plate with a 2x2 grid of round holes."""
    v = np.zeros(VEC_LEN)
    hw   = lerp(p["hw"], 0.32, 0.45)
    sp   = lerp(p["sp"], 0.4, 0.7) * hw     # hole offset from center
    rh   = lerp(p["rh"], 0.12, 0.3) * hw    # hole radius
    rh = max(rh, 0.03)
    _set(v, 0, type_=TYPE_BOX, sx=hw, sy=hw, op=OP_UNION)
    for i, (dx, dy) in enumerate([(-1,-1),(1,-1),(-1,1),(1,1)], start=1):
        _set(v, i, type_=TYPE_CIRCLE, cx=0.5+dx*sp, cy=0.5+dy*sp,
                  sx=rh, op=OP_SUBTRACT)
    return v

TEMPLATES = {
    "circle":          ({"r": (0,1)},                                    _tmpl_circle),
    "square":          ({"hw": (0,1)},                                   _tmpl_square),
    "rectangle":       ({"hw": (0,1), "hh": (0,1)},                      _tmpl_rectangle),
    "hollow_cylinder": ({"r_out": (0,1), "r_in": (0,1)},                 _tmpl_hollow_cylinder),
    "hollow_square":   ({"hw_out": (0,1), "hw_in": (0,1)},               _tmpl_hollow_square),
    "cross":           ({"arm": (0,1), "thick": (0,1)},                  _tmpl_cross),
    "two_boxes":       ({"gap": (0,1), "hw_a": (0,1), "hh_a": (0,1),
                         "hw_b": (0,1)},                                 _tmpl_two_boxes),
    "semicircles":     ({"r_out": (0,1), "r_in": (0,1), "gap": (0,1)},   _tmpl_semicircles),
}


def build_dataset_templated(n_per_family=16, n=128, seed=0,
                            inverse_fraction=0.5, families=None):
    """
    Mode 3: only canonical shapes, QMC-sampled over each family's real
    parameters. Returns the SAME VEC_LEN encoding so it's interoperable
    with render_vector(snap=True), interpolate, and the surrogate.
    """
    rng = np.random.default_rng(seed)
    families = families or list(TEMPLATES.keys())
    vectors, bitmaps, meta = [], [], []

    for fam in families:
        ranges, builder = TEMPLATES[fam]
        keys = list(ranges.keys())
        sampler = qmc.LatinHypercube(d=len(keys), seed=rng.integers(1 << 31))
        samples = sampler.random(n=n_per_family)        # (n_per_family, d)
        inv_flags = rng.random(n_per_family) < inverse_fraction

        for row, inv in zip(samples, inv_flags):
            params = dict(zip(keys, row))
            v = builder(params)
            if inv:
                v[-1] = 1.0
            bmp = render_vector(v, snap=True, n=n)   # snap fine; already canonical
            vectors.append(v)
            bitmaps.append(bmp)
            meta.append({"mode": "templated", "family": fam,
                         "inverse": bool(inv), "params": params})

    return np.array(vectors), np.array(bitmaps), meta

# ----------------------------------------------------------------------
# INTERPOLATION  (the whole point: find shapes "in between")
# ----------------------------------------------------------------------
def interpolate(v1, v2, alpha):
    """Linear blend of two encoded shapes. alpha in [0,1]."""
    return (1 - alpha) * v1 + alpha * v2

def interpolation_strip(v1, v2, mode, steps=7, n=128):
    """Return a list of (bitmap, label) morphing v1 -> v2."""
    snap = (mode == "constrained")
    out = []
    for a in np.linspace(0, 1, steps):
        v = interpolate(v1, v2, a)
        out.append((render_vector(v, snap=snap, n=n), f"a={a:.2f}"))
    return out

# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
def plot_grid(samples, ncols=6, figsize=(14, 12), title=None):
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, (bmp, t) in zip(axes, samples):
        ax.imshow(bmp, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def propose_inbetween(gp, v1, v2, mode, steps=21):
    """
    Walk the line between two known-good shapes, predict the response curve at
    each step, and return the encodings + predicted curves. This is the
    simplest 'find shapes in between' routine; swap for Bayesian optimization
    (e.g. expected improvement on a scalar derived from the curve) later.
    """
    snap = (mode == "constrained")
    alphas = np.linspace(0, 1, steps)
    cand_vecs = np.array([interpolate(v1, v2, a) for a in alphas])
    pred_curves, pred_std = gp.predict(cand_vecs, return_std=True)
    bitmaps = [render_vector(v, snap=snap) for v in cand_vecs]
    return cand_vecs, pred_curves, pred_std, bitmaps

# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1) verify the encoding reproduces all your named shapes (constrained decode)
    gallery = []
    for name in EXAMPLE_NAMES:
        v = example(name)
        gallery.append((render_vector(v, snap=True), name))
        gallery.append((render_vector(v, snap=True) ^ 1, name + " (inv)"))
    plot_grid(gallery, ncols=4, figsize=(12, 10), title="Hand-built examples")

    # 2) build BOTH datasets
    vc, bc, mc = build_dataset(n_samples=120, mode="constrained", seed=143)
    vf, bf, mf = build_dataset(n_samples=120, mode="free",        seed=234)
    vt, bt, mt = build_dataset_templated(n_per_family=16, seed=np.random.randint(1,122))

    print(f"constrained dataset: {bc.shape}")
    print(f"free dataset:        {bf.shape}")
    print(f"templated dataset: {bt.shape}")

    # MODE 3: templated / canonical families only
    plot_grid([(b, mt[i]["family"]) for i, b in enumerate(bt[:24])],
              ncols=6, title="Templated samples (canonical families)")

    plot_grid([(b, f"c#{i}") for i, b in enumerate(bc[:24])],
              ncols=6, title="Constrained samples (QMC-LHS)")
    plot_grid([(b, f"f#{i}") for i, b in enumerate(bf[:24])],
              ncols=6, title="Free samples (QMC-LHS)")

    # 3) interpolation demo: morph between two hand-built shapes
    strip = interpolation_strip(example("circle"), example("cross"),
                                mode="free", steps=7)
    plot_grid(strip, ncols=7, figsize=(14, 3),
              title="In-between shapes: circle -> cross (free mode)")

    # 4) save
    np.save("templated_vectors.npy", vt); np.save("templated_bitmaps.npy", bt)
    np.save("constrained_vectors.npy", vc); np.save("constrained_bitmaps.npy", bc)
    np.save("free_vectors.npy", vf);        np.save("free_bitmaps.npy", bf)
    import json
    with open("constrained_meta.json", "w") as f: json.dump(mc, f, indent=2)
    with open("free_meta.json", "w") as f:        json.dump(mf, f, indent=2)
    print("saved datasets")
