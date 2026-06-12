import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def generate_super_shape(
    w_circle=0.0, 
    w_square=0.0, 
    w_cross=0.0, 
    w_dual=0.0, 
    hollow_weight=0.0, 
    split_weight=0.0, 
    invert_weight=0.0, 
    resolution=10
):
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Base SDF Primitives
    d_circle = np.sqrt(X**2 + Y**2) - 0.7
    d_square = np.maximum(np.abs(X) - 0.6, np.abs(Y) - 0.6)
    
    d_cross_v = np.maximum(np.abs(X) - 0.2, np.abs(Y) - 0.8)
    d_cross_h = np.maximum(np.abs(X) - 0.8, np.abs(Y) - 0.2)
    d_cross = np.minimum(d_cross_v, d_cross_h)
    
    d_rect = np.maximum(np.abs(X + 0.45) - 0.35, np.abs(Y) - 0.5)
    d_sq = np.maximum(np.abs(X - 0.5) - 0.3, np.abs(Y) - 0.3)
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
    
    # Apply continuous hollowing
    thickness = 0.2
    d_hollowed = np.abs(d_base) - thickness
    d_current = ((1.0 - hollow_weight) * d_base) + (hollow_weight * d_hollowed)
    
    # Apply continuous splitting
    gap_width = 0.15
    d_gap = gap_width - np.abs(X) 
    d_split = np.maximum(d_current, d_gap)
    d_final = ((1.0 - split_weight) * d_current) + (split_weight * d_split)
    
    # Apply continuous inversion
    inversion_multiplier = 1.0 - (2.0 * invert_weight)
    d_final = d_final * inversion_multiplier
    
    # Convert to sharp bitmap
    bitmap = (d_final <= 0).astype(int)
    return bitmap

# 1. Setup the Latin Hypercube Sampler
num_samples = 25
num_parameters = 7
sampler = qmc.LatinHypercube(d=num_parameters, seed=np.random.randint(12,2323232))
sample_params = sampler.random(n=num_samples)

# 2. Generate the dataset
resolution = 100
dataset = np.zeros((num_samples, resolution, resolution), dtype=int)

for i in range(num_samples):
    params = sample_params[i]
    dataset[i] = generate_super_shape(
        w_circle=params[0],
        w_square=params[1],
        w_cross=params[2],
        w_dual=params[3],
        hollow_weight=params[4],
        split_weight=params[5],
        invert_weight=params[6],
        resolution=resolution
    )

# 3. Plot the generated dataset
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("Latin Hypercube Sampling of Parameterized SDF Shapes", fontsize=16)

for i, ax in enumerate(axes.flatten()):
    # We use cmap='gray_r' so 1 (solid) is black and 0 (empty) is white
    ax.imshow(dataset[i], cmap='gray_r', origin='lower')
    ax.axis('off')

plt.tight_layout()
plt.show()


"""
Methods

Implicit Surface Representation and Signed Distance Functions

The geometric generation pipeline utilizes implicit surface modeling, relying specifically on Signed Distance Functions to represent boundaries. A Signed Distance Function defines a geometry by evaluating a spatial coordinate grid where each point is assigned a scalar value representing its shortest Euclidean distance to the shape boundary. By convention, points located strictly outside the shape possess positive scalar values, points located inside the shape possess negative scalar values, and the exact boundary contour is located at zero. This continuous spatial mapping allows for fluid shape interpolation and complex topological modifications that are computationally unstable when attempted on discrete pixel grids or explicit vertex meshes.

Base Geometric Primitives

The generation space is anchored by a set of foundational geometric primitives, each defined by an exact Signed Distance Function evaluated over a two-dimensional Cartesian coordinate system spanning from negative one to positive one. The continuous distance field for a circle is defined by taking the Euclidean norm of the coordinates and subtracting the radius. The continuous distance field for a square relies on the maximum norm, calculated by taking the maximum of the absolute values of the spatial coordinates and subtracting the half-scale bound. More complex primitives are constructed using Constructive Solid Geometry principles. A cross shape is mathematically defined by formulating the fields for a tall vertical rectangle and a wide horizontal rectangle, and combining them using a mathematical minimum operator which represents a Boolean union in distance fields. Similarly, a dual-shape primitive consisting of a rectangle and a square separated by a spatial gap is defined by calculating the offset fields for each respective shape and combining them with the union operator.

The Continuous Superformula and Shape Interpolation

To navigate the geometric latent space between the discrete primitives, the system employs a weighted continuous interpolation strategy, referred to as the superformula. The algorithm assigns a non-negative continuous weight variable to each of the base primitives. These variables are subsequently normalized by dividing each weight by the sum of all weights, ensuring the total influence remains at parity. The combined base distance field is computed as the linear combination of the normalized weights and their corresponding primitive fields. Because this blending occurs in the continuous distance domain, the resulting scalar field exhibits a smooth geometric homotopy. This continuous topology ensures that gradient-based search algorithms can smoothly traverse intermediate states and correctly converge upon a pure geometric primitive by maximizing its respective interpolation weight.

Topological Modifiers

Topological variations, including hollowing and symmetry breaking, are introduced to the blended distance field using continuous domain modifiers. The hollowing modifier operates by taking the absolute value of the combined base distance field and subtracting a defined wall thickness. Taking the absolute value forces the field to evaluate to zero precisely at the original boundary while increasing linearly in both inward and outward directions, and subtracting the thickness offsets this new boundary to create a hollowed shell. This shell field is then linearly interpolated with the solid base field using a continuous hollowing weight parameter. The splitting modifier breaks the structural symmetry by defining a secondary distance field representing a vertical gap and utilizing a mathematical maximum operator, which equates to a Boolean intersection, to subtract the gap from the current geometry. A continuous split weight smoothly transitions the field between the whole geometry and the split topology.

Inversion and Discrete Quantization

The final phase of the geometric pipeline handles domain inversion and discrete map generation. Spatial inversion is achieved by mapping an inversion parameter from a standard zero-to-one range into a scalar multiplier ranging from positive one to negative one. Multiplying the entire distance field by this scalar effectively reverses the algebraic sign of all spatial coordinates, instantly converting the solid internal domain into empty space and the empty exterior domain into solid mass. To finalize the generation, a Heaviside step function operates on the continuous field as a binarization threshold. Any coordinate within the matrix evaluating to zero or less is assigned a discrete integer value of one, representing solid geometry, while all positive coordinates are assigned a discrete integer value of zero. This transforms the mathematically rich distance fields into the final sharp, quantifiable bitmap matrices suitable for downstream processing.

"""