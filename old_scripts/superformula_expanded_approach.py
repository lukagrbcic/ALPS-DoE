import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def generate_super_shape(
    w_circle, w_square, w_cross, w_dual, hollow_weight, split_weight, invert_weight,
    r, s, cross_w, cross_l, rect_x, rect_w, rect_h, sq_x, sq_s, thickness, gap_width,
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

# 1. Setup the Latin Hypercube Sampler for 18 dimensions
num_samples = 600
num_parameters = 18
sampler = qmc.LatinHypercube(d=num_parameters, seed=np.random.randint(1212,121121))#42)
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

# 3. Generate the dataset
resolution = 20
dataset = np.zeros((num_samples, resolution, resolution), dtype=int)

for i in range(num_samples):
    # First 7 parameters are weights/modifiers, remaining [0, 1] mapped as is
    w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv = raw_samples[i, :7]
    
    # Next 11 parameters are constants, mapped to their specific physical bounds
    mapped_constants = []
    for j in range(11):
        raw_val = raw_samples[i, 7 + j]
        min_val, max_val = bounds[j]
        mapped_val = min_val + raw_val * (max_val - min_val)
        mapped_constants.append(mapped_val)
        
    dataset[i] = generate_super_shape(
        w_circ, w_sq, w_cross, w_dual, m_hollow, m_split, m_inv,
        *mapped_constants,
        resolution=resolution
    )

# 4. Plot the generated dataset
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("18-Dimensional LHS Parameterized Shapes", fontsize=16)

for i, ax in enumerate(axes.flatten()):
    ax.imshow(dataset[i], cmap='gray_r', origin='lower')
    ax.axis('off')

plt.tight_layout()
plt.show()
