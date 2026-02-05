"""Field utilities for vector field analysis.

Provides functions for creating grids, interpolating fields,
and analyzing critical points.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, cast
from .types import (
    Vector, VectorField, ScalarField, Trajectory,
    GridCoords2D, GridCoords3D
)
from .operators import divergence, curl, divergence_3d


def create_uniform_grid(
    points: np.ndarray,
    resolution: int = 50,
    padding: float = 0.1
) -> Tuple[GridCoords2D, np.ndarray]:
    """Create a uniform 2D grid covering point cloud.
    
    Args:
        points: Point cloud, shape (n, 2)
        resolution: Grid resolution per dimension
        padding: Fraction of range to pad on each side
        
    Returns:
        (X, Y): Meshgrid coordinate arrays
        grid_points: Flattened grid points, shape (resolution², 2)
    """
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xs, ys)
    
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    return (X, Y), grid_points


def create_uniform_grid_3d(
    points: np.ndarray,
    resolution: int = 20,
    padding: float = 0.1
) -> Tuple[GridCoords3D, np.ndarray]:
    """Create a uniform 3D grid covering point cloud.
    
    Args:
        points: Point cloud, shape (n, 3)
        resolution: Grid resolution per dimension
        padding: Fraction of range to pad on each side
        
    Returns:
        (X, Y, Z): Meshgrid coordinate arrays
        grid_points: Flattened grid points, shape (resolution³, 3)
    """
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range
    
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    zs = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    return (X, Y, Z), grid_points


def interpolate_field(
    trajectories: List[Trajectory],
    grid_points: np.ndarray,
    radius: float = 0.5,
    method: str = 'velocity'
) -> VectorField:
    """Interpolate vector field from trajectories onto grid.
    
    Args:
        trajectories: List of trajectories
        grid_points: Grid points for interpolation
        radius: Neighborhood radius for averaging
        method: 'velocity' for velocity field, 'displacement' for displacement
        
    Returns:
        V: Interpolated vector field
    """
    dim = trajectories[0].shape[1]
    V = np.zeros((grid_points.shape[0], dim))
    
    for i, r in enumerate(grid_points):
        velocities = []
        
        for traj in trajectories:
            # Find nearest point on trajectory
            distances = np.linalg.norm(traj - r, axis=1)
            idx = np.argmin(distances)
            
            # If within radius and not at trajectory end
            if distances[idx] <= radius and idx + 1 < len(traj):
                if method == 'velocity':
                    v = traj[idx + 1] - traj[idx]
                else:  # displacement
                    v = traj[-1] - traj[idx]
                velocities.append(v)
        
        if velocities:
            V[i] = np.mean(velocities, axis=0)
    
    return V


def compute_field_magnitude(V: VectorField) -> ScalarField:
    """Compute magnitude |V| of vector field.

    Args:
        V: Vector field

    Returns:
        |V|: Magnitude field
    """
    return cast(ScalarField, np.linalg.norm(V, axis=-1))


def find_critical_points(
    div_V: ScalarField,
    attractor_threshold: float = -0.1,
    repeller_threshold: float = 0.1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Find critical points in flow field based on divergence.

    Critical point classification:
    - Attractors (sinks): div(V) < 0 (flow converges)
    - Repellers (sources): div(V) > 0 (flow diverges)
    - Saddles: near zero divergence with mixed stability

    Args:
        div_V: Divergence field
        attractor_threshold: Threshold for attractors (negative)
        repeller_threshold: Threshold for repellers (positive)

    Returns:
        Dictionary with:
            'attractors': (positions, strengths)
            'repellers': (positions, strengths)
            'saddles': (positions, divergence values)
    """
    # Find attractors (sinks): div(V) < 0
    attractor_mask = div_V < attractor_threshold
    attractor_positions = np.argwhere(attractor_mask)
    attractor_strengths = div_V[attractor_mask]

    # Find repellers (sources): div(V) > 0
    repeller_mask = div_V > repeller_threshold
    repeller_positions = np.argwhere(repeller_mask)
    repeller_strengths = div_V[repeller_mask]

    # Find saddles: near-zero divergence
    saddle_mask = (div_V >= attractor_threshold) & (div_V <= repeller_threshold)
    saddle_positions = np.argwhere(saddle_mask)
    saddle_values = div_V[saddle_mask]
    
    return {
        'attractors': (attractor_positions, attractor_strengths),
        'repellers': (repeller_positions, repeller_strengths),
        'saddles': (saddle_positions, saddle_values)
    }


def classify_critical_point(
    V: VectorField,
    position: Tuple[int, ...],
    dx: float = 1.0,
    dy: float = 1.0
) -> Dict[str, Any]:
    """Classify a critical point using local Jacobian analysis.

    At a critical point r0 where V(r0) = 0, the Jacobian matrix
    J = dV/dr determines the local flow behavior:

    J = [dVx/dx  dVx/dy]
        [dVy/dx  dVy/dy]

    Classification based on eigenvalues lambda1, lambda2:
    - Node (attractor): lambda1, lambda2 < 0
    - Node (repeller): lambda1, lambda2 > 0
    - Saddle: lambda1 < 0 < lambda2
    - Spiral/Focus: complex eigenvalues

    Args:
        V: Vector field
        position: Grid indices of critical point
        dx, dy: Grid spacing

    Returns:
        Classification dict with type, eigenvalues, stability
    """
    i, j = position
    
    # Compute Jacobian at point
    dVx_dx = (V[i, j + 1, 0] - V[i, j - 1, 0]) / (2 * dx) if j > 0 and j < V.shape[1] - 1 else 0.0
    dVx_dy = (V[i + 1, j, 0] - V[i - 1, j, 0]) / (2 * dy) if i > 0 and i < V.shape[0] - 1 else 0.0
    dVy_dx = (V[i, j + 1, 1] - V[i, j - 1, 1]) / (2 * dx) if j > 0 and j < V.shape[1] - 1 else 0.0
    dVy_dy = (V[i + 1, j, 1] - V[i - 1, j, 1]) / (2 * dy) if i > 0 and i < V.shape[0] - 1 else 0.0

    J = np.array([[dVx_dx, dVx_dy],
                  [dVy_dx, dVy_dy]])
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    
    # Classify
    if np.all(np.isreal(eigenvalues)):
        lambda1, lambda2 = np.real(eigenvalues)

        if lambda1 < 0 and lambda2 < 0:
            point_type = "stable_node"
            stability = "stable"
        elif lambda1 > 0 and lambda2 > 0:
            point_type = "unstable_node"
            stability = "unstable"
        elif lambda1 * lambda2 < 0:
            point_type = "saddle"
            stability = "unstable"
        else:
            point_type = "degenerate"
            stability = "neutral"
    else:
        # Complex eigenvalues -> spiral/focus
        real_part = np.real(eigenvalues[0])
        if real_part < 0:
            point_type = "stable_spiral"
            stability = "stable"
        elif real_part > 0:
            point_type = "unstable_spiral"
            stability = "unstable"
        else:
            point_type = "center"
            stability = "neutral"
    
    return {
        'type': point_type,
        'stability': stability,
        'eigenvalues': eigenvalues,
        'jacobian': J,
        'divergence': np.trace(J),
        'curl': J[1, 0] - J[0, 1]
    }
