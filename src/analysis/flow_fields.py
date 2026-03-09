"""Flow field construction from trajectory data."""
import numpy as np
from typing import List, Tuple

from shared.physics import (
    create_uniform_grid,
    create_uniform_grid_3d,
    interpolate_field,
)


def compute_flow_field(
    trajectories: List[np.ndarray],
    grid_resolution: int = 50,
    radius: float = 0.5,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Build a 2D velocity field from a list of 2D trajectories.

    Each trajectory should be an array of shape ``(n_steps, 2)``.

    Returns:
        ``((X, Y), V)`` where ``V.shape == (grid_resolution, grid_resolution, 2)``
    """
    all_points = np.concatenate(trajectories, axis=0)
    (X, Y), grid_points = create_uniform_grid(all_points, resolution=grid_resolution)
    V_flat = interpolate_field(trajectories, grid_points, radius=radius)
    V = V_flat.reshape(grid_resolution, grid_resolution, 2)
    return (X, Y), V


def compute_flow_field_3d(
    trajectories: List[np.ndarray],
    grid_resolution: int = 20,
    radius: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D velocity field from a list of 3D trajectories.

    Each trajectory should be an array of shape ``(n_steps, 3)``.

    Returns:
        ``(X, Y, Z, V)`` where ``V.shape == (resolution, resolution, resolution, 3)``
    """
    all_points = np.concatenate(trajectories, axis=0)
    (X, Y, Z), grid_points = create_uniform_grid_3d(all_points, resolution=grid_resolution)
    V_flat = interpolate_field(trajectories, grid_points, radius=radius)
    V = V_flat.reshape(grid_resolution, grid_resolution, grid_resolution, 3)
    return X, Y, Z, V
