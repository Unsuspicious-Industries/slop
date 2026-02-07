"""Type definitions for physics-based computations.

Uses standard physics concepts (ASCII names):
- Vectors: r, v, F
- Scalar fields: phi, rho, T
- Vector fields: V(r), E(r)
- Tensors: sigma_ij, T_ij
"""

from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray

# Scalar types
Scalar = Union[float, np.floating]

# Vector types
# A vector is a 1D array of shape (n,) where n is the dimension (2 or 3)
Vector = NDArray[np.floating]  # Shape: (n,)

# Scalar field types
# A scalar field phi(r) maps positions to scalar values
# 2D: phi(x, y) -> shape (nx, ny)
# 3D: phi(x, y, z) -> shape (nx, ny, nz)
ScalarField = NDArray[np.floating]  # Shape: (nx, ny) or (nx, ny, nz)

# Vector field types
# A vector field V(r) assigns a vector to each position
# 2D: V(x, y) = (Vx, Vy) -> shape (nx, ny, 2)
# 3D: V(x, y, z) = (Vx, Vy, Vz) -> shape (nx, ny, nz, 3)
VectorField = NDArray[np.floating]  # Shape: (nx, ny, 2) or (nx, ny, nz, 3)

# Tensor types
# A tensor field T assigns a matrix/tensor to each position
# 2D rank-2 tensor: T_ij(x, y) -> shape (nx, ny, 2, 2)
# 3D rank-2 tensor: T_ij(x, y, z) -> shape (nx, ny, nz, 3, 3)
Tensor = NDArray[np.floating]  # Shape: (nx, ny, m, n) or (nx, ny, nz, m, n)

# Grid coordinate types
# Grid points for field evaluation
GridCoords2D = Tuple[NDArray[np.floating], NDArray[np.floating]]  # (X, Y) meshgrids
GridCoords3D = Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]  # (X, Y, Z)

# Trajectory types
# A trajectory is a sequence of positions r(t)
Trajectory = NDArray[np.floating]  # Shape: (n_steps, dim)


def is_vector_field_2d(field: VectorField) -> bool:
    """Check if a vector field is 2D."""
    return field.ndim == 3 and field.shape[-1] == 2


def is_vector_field_3d(field: VectorField) -> bool:
    """Check if a vector field is 3D."""
    return field.ndim == 4 and field.shape[-1] == 3


def is_scalar_field_2d(field: ScalarField) -> bool:
    """Check if a scalar field is 2D."""
    return field.ndim == 2


def is_scalar_field_3d(field: ScalarField) -> bool:
    """Check if a scalar field is 3D."""
    return field.ndim == 3


def get_field_dimension(field: Union[ScalarField, VectorField]) -> int:
    """Get the spatial dimension of a field (2 or 3)."""
    if field.ndim == 2:
        return 2
    elif field.ndim == 3:
        if field.shape[-1] == 2:  # 2D vector field
            return 2
        else:  # 3D scalar field
            return 3
    elif field.ndim == 4 and field.shape[-1] == 3:  # 3D vector field
        return 3
    else:
        raise ValueError(f"Cannot determine dimension of field with shape {field.shape}")
