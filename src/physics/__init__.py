"""Re-export shared physics operators and field utilities.

All computation lives in ``shared/physics/``; this module exists so that
``src.physics`` is a valid import path within the project.
"""
from shared.physics import (  # noqa: F401
    divergence,
    curl,
    gradient,
    laplacian,
    divergence_3d,
    curl_3d,
    gradient_3d,
    laplacian_3d,
    create_uniform_grid,
    create_uniform_grid_3d,
    interpolate_field,
    compute_field_magnitude,
    find_critical_points,
    classify_critical_point,
    Vector,
    VectorField,
    ScalarField,
    Tensor,
)

__all__ = [
    "divergence", "curl", "gradient", "laplacian",
    "divergence_3d", "curl_3d", "gradient_3d", "laplacian_3d",
    "create_uniform_grid", "create_uniform_grid_3d",
    "interpolate_field", "compute_field_magnitude",
    "find_critical_points", "classify_critical_point",
    "Vector", "VectorField", "ScalarField", "Tensor",
]
