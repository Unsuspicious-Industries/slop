"""Vector field construction from trajectory data.

This module constructs vector fields V(r) from trajectory data r(t)
and computes differential properties using physics operators.
"""

from typing import List, Tuple, Optional
from pathlib import Path
import json

import numpy as np

from src.physics.types import VectorField, ScalarField, Trajectory, GridCoords2D, GridCoords3D
from src.physics.operators import divergence, curl, divergence_3d, curl_3d
from src.physics.fields import (
    create_uniform_grid, 
    create_uniform_grid_3d,
    interpolate_field
)


def create_latent_grid(points: np.ndarray, resolution: int = 50) -> np.ndarray:
    """Create a 2D grid covering the extent of points (legacy wrapper).

    Args:
        points: Point cloud r_i, shape (n, 2)
        resolution: Grid resolution per dimension

    Returns:
        Grid points, shape (resolution**2, 2)
    """
    (X, Y), grid_points = create_uniform_grid(points, resolution)
    return grid_points


def create_latent_grid_3d(
    points: np.ndarray, resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 3D grid covering the extent of points (legacy wrapper).

    Args:
        points: Point cloud r_i, shape (n, 3)
        resolution: Grid resolution per dimension

    Returns:
        X, Y, Z: Meshgrid coordinate arrays
    """
    (X, Y, Z), _ = create_uniform_grid_3d(points, resolution)
    return X, Y, Z


def compute_flow_field(
    trajectories: List[Trajectory],
    stereotype_poles: Optional[np.ndarray] = None,
    grid_resolution: int = 50,
    radius: float = 0.5,
    save_dir: Optional[Path] = None
) -> Tuple[np.ndarray, VectorField]:
    """Compute 2D velocity field V(r) from trajectories r(t).

    The velocity field is estimated by averaging local velocities:
    V(r) = <dr/dt> for all r(t) within radius of r

    Args:
        trajectories: List of trajectories r(t), each shape (n_steps, 2)
        stereotype_poles: Optional attractor poles (not used in current impl)
        grid_resolution: Number of grid points per dimension
        radius: Neighborhood radius for velocity averaging
        save_dir: Optional directory to save debug data

    Returns:
        grid: Grid points r_ij, shape (resolution**2, 2)
        V: Velocity field, shape (ny, nx, 2)
    """
    # Combine all trajectory points
    all_points = np.concatenate(trajectories, axis=0)
    
    # Create grid
    (X, Y), grid_flat = create_uniform_grid(all_points, grid_resolution)
    
    # Compute velocity at each grid point
    V_flat = interpolate_field(trajectories, grid_flat, radius, method='velocity')
    
    # Reshape to grid
    V = V_flat.reshape(grid_resolution, grid_resolution, 2)
    
    # Save debug data
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute and save differential quantities
        div_V = divergence(V)
        omega = curl(V)  # vorticity
        
        np.save(save_dir / "flow_field_2d_grid_X.npy", X)
        np.save(save_dir / "flow_field_2d_grid_Y.npy", Y)
        np.save(save_dir / "flow_field_2d_velocity.npy", V)
        np.save(save_dir / "flow_field_2d_divergence.npy", div_V)
        np.save(save_dir / "flow_field_2d_vorticity.npy", omega)
        
        # Save stats
        stats = {
            'grid_resolution': grid_resolution,
            'radius': radius,
            'n_trajectories': len(trajectories),
            'velocity_magnitude_mean': float(np.linalg.norm(V, axis=-1).mean()),
            'divergence_mean': float(div_V.mean()),
            'divergence_std': float(div_V.std()),
            'vorticity_mean': float(omega.mean()),
            'vorticity_std': float(omega.std()),
        }
        
        with open(save_dir / "flow_field_2d_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved 2D flow field debug data to {save_dir}")
    
    return grid_flat, V


def compute_flow_field_3d(
    trajectories: List[Trajectory],
    grid_resolution: int = 20,
    radius: float = 0.5,
    save_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D velocity field V(r) from trajectories r(t).
    
    Returns components separately for legacy compatibility.
    
    Args:
        trajectories: List of trajectories r(t), each shape (n_steps, 3)
        grid_resolution: Grid resolution per dimension
        radius: Neighborhood radius for averaging
        save_dir: Optional directory to save debug data
        
    Returns:
        X, Y, Z: Meshgrid coordinates
        Vx, Vy, Vz: Velocity components
    """
    # Combine all trajectory points
    all_points = np.concatenate(trajectories, axis=0)
    
    # Create grid
    (X, Y, Z), grid_flat = create_uniform_grid_3d(all_points, grid_resolution)
    
    # Compute velocity at each grid point
    V_flat = interpolate_field(trajectories, grid_flat, radius, method='velocity')
    
    # Reshape to 3D grid
    grid_shape = X.shape
    Vx = V_flat[:, 0].reshape(grid_shape)
    Vy = V_flat[:, 1].reshape(grid_shape)
    Vz = V_flat[:, 2].reshape(grid_shape)
    
    # Construct full vector field for differential operations
    V = np.stack([Vx, Vy, Vz], axis=-1)
    
    # Save debug data
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute differential quantities
        div_V = divergence_3d(V)
        curl_V = curl_3d(V)  # vector curl
        
        np.save(save_dir / "flow_field_3d_grid_X.npy", X)
        np.save(save_dir / "flow_field_3d_grid_Y.npy", Y)
        np.save(save_dir / "flow_field_3d_grid_Z.npy", Z)
        np.save(save_dir / "flow_field_3d_velocity.npy", V)
        np.save(save_dir / "flow_field_3d_divergence.npy", div_V)
        np.save(save_dir / "flow_field_3d_curl.npy", curl_V)
        
        # Stats
        curl_mag = np.linalg.norm(curl_V, axis=-1)
        stats = {
            'grid_resolution': grid_resolution,
            'radius': radius,
            'n_trajectories': len(trajectories),
            'velocity_magnitude_mean': float(np.linalg.norm(V, axis=-1).mean()),
            'divergence_mean': float(div_V.mean()),
            'divergence_std': float(div_V.std()),
            'curl_magnitude_mean': float(curl_mag.mean()),
            'curl_magnitude_std': float(curl_mag.std()),
        }
        
        with open(save_dir / "flow_field_3d_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved 3D flow field debug data to {save_dir}")
    
    return X, Y, Z, Vx, Vy, Vz
