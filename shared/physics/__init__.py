"""Physics-based utilities for vector field analysis.

Provides type definitions and operators using standard physics notation.
Refactored from src/physics/ to shared/physics/.
"""

from .types import Vector, VectorField, ScalarField, Tensor
from .operators import (
    divergence, curl, gradient, laplacian,
    divergence_3d, curl_3d, gradient_3d, laplacian_3d
)
from .fields import (
    create_uniform_grid, create_uniform_grid_3d,
    interpolate_field, compute_field_magnitude,
    find_critical_points, classify_critical_point
)

__all__ = [
    # Types
    'Vector',
    'VectorField',
    'ScalarField',
    'Tensor',
    # Differential operators
    'divergence',
    'curl',
    'gradient',
    'laplacian',
    'divergence_3d',
    'curl_3d',
    'gradient_3d',
    'laplacian_3d',
    # Field utilities
    'create_uniform_grid',
    'create_uniform_grid_3d',
    'interpolate_field',
    'compute_field_magnitude',
    'find_critical_points',
    'classify_critical_point',
]
