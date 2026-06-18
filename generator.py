import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def generate_super_shape(
    w_circle, w_square, w_cross, w_dual, hollow_weight, split_weight, invert_weight,
    r, s, cross_w, cross_l, rect_x, rect_w, rect_h, sq_x, sq_s, thickness, gap_width,
    layer_weight=0.0, num_layers=3, layer_thickness=0.08,
    resolution=64
):
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)

    # Base SDF Primitives with parameterized constants
    d_circle = np.sqrt(X**2 + Y**2) - r
    d_square = np.maximum(np.abs(X) - s, np.abs(Y) - s)

    d_cross_v = np.maximum(np.abs(X) - cross_w, np.abs(Y) - cross_l)
    d_cross_h = np.maximum(np.abs(X) - cross_l, np.abs(Y) - cross_w)
    d_cross = np.minimum(d_cross_v, d_cross_h)

    d_rect = np.maximum(np.abs(X - rect_x) - rect_w, np.abs(Y) - rect_h)
    d_sq = np.maximum(np.abs(X - sq_x) - sq_s, np.abs(Y) - sq_s)
    d_dual = np.minimum(d_rect, d_sq)

    # Normalize base shape weights
    total_base_weight = w_circle + w_square + w_cross + w_dual
    if total_base_weight == 0:
        w_circle = 1.0
        total_base_weight = 1.0

    w_c = w_circle / total_base_weight
    w_s = w_square / total_base_weight
    w_cr = w_cross / total_base_weight
    w_d = w_dual / total_base_weight

    # Blend base shapes
    d_base = (w_c * d_circle) + (w_s * d_square) + (w_cr * d_cross) + (w_d * d_dual)

    # Apply parameterized continuous hollowing
    d_hollowed = np.abs(d_base) - thickness
    d_current = ((1.0 - hollow_weight) * d_base) + (hollow_weight * d_hollowed)

    # Apply concentric layers (shape within a shape within a shape)
    band_index = np.floor((-d_current) / layer_thickness)   # 0,1,2,... going inward
    in_shape = d_current <= 0
    layered_solid = in_shape & (band_index % 2 == 0) & (band_index < 2 * num_layers)
    d_layered = np.where(layered_solid, -1.0, 1.0)
    d_current = ((1.0 - layer_weight) * d_current) + (layer_weight * d_layered)

    # Apply parameterized continuous splitting
    d_gap = gap_width - np.abs(X)
    d_split = np.maximum(d_current, d_gap)
    d_final = ((1.0 - split_weight) * d_current) + (split_weight * d_split)

    # Apply continuous inversion
    inversion_multiplier = 1.0 - (2.0 * invert_weight)
    d_final = d_final * inversion_multiplier

    # Convert to sharp bitmap
    bitmap = (d_final <= 0).astype(int)
    return bitmap
# 1. Setup the Latin Hypercube Sampler for 19 dimensions
num_samples = 1
num_parameters = 19   # 7 weights/modifiers + 1 layer_weight + 11 geometric constants
sampler = qmc.LatinHypercube(d=num_parameters, seed=np.random.randint(1212, 121121))
raw_samples = sampler.random(n=num_samples)

# 2. Define the geometric scaling bounds (min, max) for the 11 spatial constants
bounds = [
    (0.1, 0.45),    # r: circle radius
    (0.1, 0.45),    # s: square half-size
    (0.025, 0.2),   # cross_w: cross arm half-width
    (0.2, 0.475),   # cross_l: cross arm half-length
    (-0.4, -0.05),  # rect_x: dual rectangle X-center
    (0.05, 0.2),    # rect_w: dual rectangle half-width
    (0.1, 0.4),     # rect_h: dual rectangle half-height
    (0.05, 0.4),    # sq_x: dual square X-center
    (0.05, 0.2),    # sq_s: dual square half-size
    (0.025, 0.15),  # thickness: hollowing shell thickness
    (0.025, 0.2)    # gap_width: split gap half-width
]

# 3. Generate the single sample
resolution = 209

# First 8 parameters are weights/modifiers (including layer_weight)
w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv, m_layer = raw_samples[0, :8]

# Next 11 parameters are constants, mapped to their specific physical bounds
mapped_constants = []
for j in range(11):
    raw_val = raw_samples[0, 8 + j]
    min_val, max_val = bounds[j]
    mapped_val = min_val + raw_val * (max_val - min_val)
    mapped_constants.append(mapped_val)

sample = generate_super_shape(
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
    *mapped_constants,
    layer_weight=m_layer,
    resolution=resolution
)

from PIL import Image

img = Image.open("unitCell.jpeg").convert("L")   # load as grayscale
arr = np.array(img)
print(arr.min(), arr.max())
print("fraction white:", (arr == 255).mean())
bitmap = (arr < 128).astype(int)   # cross = 1

plt.imshow(bitmap, cmap='gray_r')
#plt.axis('off')
plt.show()

target_test =  bitmap


# 4. Plot the generated sample
fig, ax = plt.subplots(figsize=(6, 6))
fig.suptitle("19-Dimensional LHS Parameterized Shape (with Layers)", fontsize=14)

ax.imshow(sample, cmap='gray_r', origin='lower')
ax.axis('off')

plt.tight_layout()
plt.show()