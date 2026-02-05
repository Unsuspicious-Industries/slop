"""Differential operators for vector calculus.

Standard physics notation (ASCII):
- grad(phi): gradient
- div(V): divergence
- curl(V): curl
- laplacian(phi): Laplacian
"""

import numpy as np
from typing import Tuple, cast
from .types import ScalarField, VectorField


def gradient(phi: ScalarField, dx: float = 1.0, dy: float = 1.0) -> VectorField:
    """Compute gradient of a 2D scalar field.

    grad(phi) = (dphi/dx, dphi/dy)

    Args:
        phi: Scalar field, shape (ny, nx)
        dx, dy: Grid spacing

    Returns:
        Gradient vector field, shape (ny, nx, 2)
    """
    dphi_dx = np.gradient(phi, dx, axis=-1)
    dphi_dy = np.gradient(phi, dy, axis=-2)
    return cast(VectorField, np.stack([dphi_dx, dphi_dy], axis=-1))


def divergence(V: VectorField, dx: float = 1.0, dy: float = 1.0) -> ScalarField:
    """Compute divergence of a 2D vector field.

    div(V) = dVx/dx + dVy/dy

    Args:
        V: Vector field, shape (ny, nx, 2) where V[..., 0] = Vx, V[..., 1] = Vy
        dx, dy: Grid spacing

    Returns:
        Divergence scalar field, shape (ny, nx)
    """
    if V.shape[-1] != 2:
        raise ValueError(f"Expected last dimension size 2 for 2D vector field, got {V.shape[-1]}")

    Vx = V[..., 0]
    Vy = V[..., 1]

    dVx_dx = np.gradient(Vx, dx, axis=-1)
    dVy_dy = np.gradient(Vy, dy, axis=-2)

    return cast(ScalarField, dVx_dx + dVy_dy)


def curl(V: VectorField, dx: float = 1.0, dy: float = 1.0) -> ScalarField:
    """Compute curl z-component of a 2D vector field.

    For 2D field V = (Vx, Vy, 0), curl_z = dVy/dx - dVx/dy

    Args:
        V: Vector field, shape (ny, nx, 2)
        dx, dy: Grid spacing

    Returns:
        curl_z: z-component of curl, shape (ny, nx)
    """
    if V.shape[-1] != 2:
        raise ValueError(f"Expected last dimension size 2 for 2D vector field, got {V.shape[-1]}")

    Vx = V[..., 0]
    Vy = V[..., 1]

    dVy_dx = np.gradient(Vy, dx, axis=-1)
    dVx_dy = np.gradient(Vx, dy, axis=-2)

    return cast(ScalarField, dVy_dx - dVx_dy)


def laplacian(phi: ScalarField, dx: float = 1.0, dy: float = 1.0) -> ScalarField:
    """Compute Laplacian of a 2D scalar field.

    laplacian(phi) = d2phi/dx2 + d2phi/dy2

    Args:
        phi: Scalar field, shape (ny, nx)
        dx, dy: Grid spacing

    Returns:
        Laplacian, shape (ny, nx)
    """
    d2phi_dx2 = np.gradient(np.gradient(phi, dx, axis=-1), dx, axis=-1)
    d2phi_dy2 = np.gradient(np.gradient(phi, dy, axis=-2), dy, axis=-2)

    return cast(ScalarField, d2phi_dx2 + d2phi_dy2)


# 3D operators

def gradient_3d(
    phi: ScalarField, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0
) -> VectorField:
    """Compute gradient of a 3D scalar field.

    grad(phi) = (dphi/dx, dphi/dy, dphi/dz)

    Args:
        phi: Scalar field, shape (nz, ny, nx)
        dx, dy, dz: Grid spacing

    Returns:
        Gradient vector field, shape (nz, ny, nx, 3)
    """
    dphi_dx = np.gradient(phi, dx, axis=-1)
    dphi_dy = np.gradient(phi, dy, axis=-2)
    dphi_dz = np.gradient(phi, dz, axis=-3)
    return cast(VectorField, np.stack([dphi_dx, dphi_dy, dphi_dz], axis=-1))


def divergence_3d(
    V: VectorField, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0
) -> ScalarField:
    """Compute divergence of a 3D vector field.

    div(V) = dVx/dx + dVy/dy + dVz/dz

    Args:
        V: Vector field, shape (nz, ny, nx, 3)
        dx, dy, dz: Grid spacing

    Returns:
        Divergence scalar field, shape (nz, ny, nx)
    """
    if V.shape[-1] != 3:
        raise ValueError(f"Expected last dimension size 3 for 3D vector field, got {V.shape[-1]}")

    Vx = V[..., 0]
    Vy = V[..., 1]
    Vz = V[..., 2]

    dVx_dx = np.gradient(Vx, dx, axis=-1)
    dVy_dy = np.gradient(Vy, dy, axis=-2)
    dVz_dz = np.gradient(Vz, dz, axis=-3)

    return cast(ScalarField, dVx_dx + dVy_dy + dVz_dz)


def curl_3d(
    V: VectorField, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0
) -> VectorField:
    """Compute curl of a 3D vector field.

    curl(V) = (
        dVz/dy - dVy/dz,
        dVx/dz - dVz/dx,
        dVy/dx - dVx/dy
    )

    Args:
        V: Vector field, shape (nz, ny, nx, 3)
        dx, dy, dz: Grid spacing

    Returns:
        Curl vector field, shape (nz, ny, nx, 3)
    """
    if V.shape[-1] != 3:
        raise ValueError(f"Expected last dimension size 3 for 3D vector field, got {V.shape[-1]}")

    Vx = V[..., 0]
    Vy = V[..., 1]
    Vz = V[..., 2]

    curl_x = np.gradient(Vz, dy, axis=-2) - np.gradient(Vy, dz, axis=-3)
    curl_y = np.gradient(Vx, dz, axis=-3) - np.gradient(Vz, dx, axis=-1)
    curl_z = np.gradient(Vy, dx, axis=-1) - np.gradient(Vx, dy, axis=-2)

    return cast(VectorField, np.stack([curl_x, curl_y, curl_z], axis=-1))


def laplacian_3d(
    phi: ScalarField, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0
) -> ScalarField:
    """Compute Laplacian of a 3D scalar field.

    laplacian(phi) = d2phi/dx2 + d2phi/dy2 + d2phi/dz2

    Args:
        phi: Scalar field, shape (nz, ny, nx)
        dx, dy, dz: Grid spacing

    Returns:
        Laplacian, shape (nz, ny, nx)
    """
    d2phi_dx2 = np.gradient(np.gradient(phi, dx, axis=-1), dx, axis=-1)
    d2phi_dy2 = np.gradient(np.gradient(phi, dy, axis=-2), dy, axis=-2)
    d2phi_dz2 = np.gradient(np.gradient(phi, dz, axis=-3), dz, axis=-3)

    return cast(ScalarField, d2phi_dx2 + d2phi_dy2 + d2phi_dz2)
