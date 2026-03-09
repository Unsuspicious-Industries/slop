"""Physics utilities.

This package is used for diffusion-style latent vector fields and simple
vector calculus operators implemented via finite differences.
"""

from .latent import LatentDiffField, LatentField, LatentFieldConfig
from .operators import curl, divergence, gradient, laplacian
from .types import Field

__all__ = [
    "Field",
    "LatentField",
    "LatentDiffField",
    "LatentFieldConfig",
    "gradient",
    "divergence",
    "curl",
    "laplacian",
]
