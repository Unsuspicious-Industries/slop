"""Unified physics tools interface for vector field analysis.

Provides easy-to-use interface for:
- Differential operators (divergence, curl, gradient, laplacian)
- Field analysis (critical points, topology)
- Flow visualization
- Grid utilities

Example:
    >>> from src.utils.physics import PhysicsTools
    >>> 
    >>> physics = PhysicsTools()
    >>> 
    >>> # Compute divergence and curl
    >>> div_V = physics.divergence(V)
    >>> curl_V = physics.curl(V)
    >>> 
    >>> # Find attractors and repellers
    >>> attractors, repellers = physics.find_attractors_repellers(V)
    >>> 
    >>> # Analyze topology
    >>> topology = physics.analyze_topology(V)
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from pathlib import Path

# Import physics operators directly
from src.physics import (
    divergence, curl, gradient, laplacian,
    divergence_3d, curl_3d, gradient_3d, laplacian_3d,
    create_uniform_grid, create_uniform_grid_3d,
    interpolate_field, compute_field_magnitude,
    find_critical_points, classify_critical_point
)


class PhysicsTools:
    """Unified interface for physics-based vector field analysis."""
    
    def __init__(self, verbose: bool = True):
        """Initialize physics tools.
        
        Args:
            verbose: Print informative messages
        """
        self.verbose = verbose
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(message)
    
    # ========== Differential Operators ==========
    # Direct access to physics operators (2D/3D auto-dispatch)
    
    @staticmethod
    def divergence(
        V: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: Optional[float] = None
    ) -> np.ndarray:
        """Compute divergence of vector field.
        
        div(V) = dVx/dx + dVy/dy + dVz/dz
        
        Measures expansion/contraction of flow.
        - Positive: Diverging (source)
        - Negative: Converging (sink)
        - Zero: Incompressible flow
        
        Args:
            V: Vector field, shape (..., ny, nx, 2) or (..., nz, ny, nx, 3)
            dx, dy, dz: Grid spacing in each direction
        
        Returns:
            Divergence field, shape (..., ny, nx) or (..., nz, ny, nx)
        """
        ndim = V.shape[-1]
        if ndim == 2:
            return divergence(V, dx, dy)
        elif ndim == 3:
            dz = dz or 1.0
            return divergence_3d(V, dx, dy, dz)
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")
    
    @staticmethod
    def curl(
        V: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: Optional[float] = None
    ) -> np.ndarray:
        """Compute curl (vorticity) of vector field.
        
        2D: curl(V) = dVy/dx - dVx/dy
        3D: curl(V) = [dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy]
        
        Measures rotation of flow.
        - Positive: Counterclockwise rotation
        - Negative: Clockwise rotation
        - Zero: Irrotational flow
        
        Args:
            V: Vector field, shape (..., ny, nx, 2) or (..., nz, ny, nx, 3)
            dx, dy, dz: Grid spacing in each direction
        
        Returns:
            Curl field (scalar for 2D, vector for 3D)
        """
        ndim = V.shape[-1]
        if ndim == 2:
            return curl(V, dx, dy)
        elif ndim == 3:
            dz = dz or 1.0
            return curl_3d(V, dx, dy, dz)
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")
    
    @staticmethod
    def gradient(
        phi: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: Optional[float] = None
    ) -> np.ndarray:
        """Compute gradient of scalar field.
        
        grad(phi) = [dphi/dx, dphi/dy, dphi/dz]
        
        Points in direction of steepest ascent.
        
        Args:
            phi: Scalar field, shape (..., ny, nx) or (..., nz, ny, nx)
            dx, dy, dz: Grid spacing in each direction
        
        Returns:
            Gradient field (vector), shape (..., ny, nx, 2) or (..., nz, ny, nx, 3)
        """
        if phi.ndim == 2 or (phi.ndim > 2 and dz is None):
            return gradient(phi, dx, dy)
        else:
            dz = dz or 1.0
            return gradient_3d(phi, dx, dy, dz)
    
    @staticmethod
    def laplacian(
        phi: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: Optional[float] = None
    ) -> np.ndarray:
        """Compute Laplacian of scalar field.
        
        Δphi = d²phi/dx² + d²phi/dy² + d²phi/dz²
        
        Measures diffusion/smoothness.
        - Positive: Local minimum (diffusion out)
        - Negative: Local maximum (diffusion in)
        
        Args:
            phi: Scalar field, shape (..., ny, nx) or (..., nz, ny, nx)
            dx, dy, dz: Grid spacing in each direction
        
        Returns:
            Laplacian field, same shape as input
        """
        if phi.ndim == 2 or (phi.ndim > 2 and dz is None):
            return laplacian(phi, dx, dy)
        else:
            dz = dz or 1.0
            return laplacian_3d(phi, dx, dy, dz)
    
    # ========== Field Analysis ==========
    
    @staticmethod
    def magnitude(V: np.ndarray) -> np.ndarray:
        """Compute magnitude of vector field.
        
        |V| = sqrt(Vx² + Vy² + Vz²)
        
        Args:
            V: Vector field
        
        Returns:
            Magnitude field (scalar)
        """
        return compute_field_magnitude(V)
    
    @staticmethod
    def find_critical_points(
        V: np.ndarray,
        attractor_threshold: float = -0.1,
        repeller_threshold: float = 0.1
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Find critical points based on divergence.

        Args:
            V: Vector field
            attractor_threshold: Threshold for attractors (negative)
            repeller_threshold: Threshold for repellers (positive)

        Returns:
            Dictionary with positions and strengths for attractors/repellers/saddles
        """
        div_V = PhysicsTools.divergence(V)
        return find_critical_points(div_V, attractor_threshold=attractor_threshold, repeller_threshold=repeller_threshold)
    
    @staticmethod
    def classify_critical_point(
        V: np.ndarray,
        point: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Classify type of critical point using Jacobian eigenvalues.
        
        Classification based on eigenvalues of Jacobian matrix:
        - Attractor (stable node): Both eigenvalues negative
        - Repeller (unstable node): Both eigenvalues positive
        - Saddle: Eigenvalues have opposite signs
        - Center/spiral: Complex eigenvalues
        
        Args:
            V: Vector field
            point: Critical point indices
        
        Returns:
            Classification dict with type, eigenvalues, stability
        """
        return classify_critical_point(V, point)
    
    def find_attractors_repellers(
        self,
        V: np.ndarray,
        div_V: Optional[np.ndarray] = None,
        threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find attractors (sinks) and repellers (sources) in flow field.
        
        Attractors: Points where flow converges (div < 0)
        Repellers: Points where flow diverges (div > 0)
        
        Args:
            V: Vector field
            div_V: Divergence field (computed if None)
            threshold: Threshold for critical points
        
        Returns:
            Tuple of (attractors, repellers)
        """
        from src.analysis.attractors import find_attractors, find_repellers
        
        if div_V is None:
            div_V = self.divergence(V)
        
        attractors, _ = find_attractors(div_V, threshold=threshold, return_values=False)
        repellers, _ = find_repellers(div_V, threshold=threshold, return_values=False)
        
        self._log(f"Found {len(attractors)} attractors, {len(repellers)} repellers")
        return attractors, repellers
    
    def analyze_topology(
        self,
        V: np.ndarray,
        div_V: Optional[np.ndarray] = None,
        curl_V: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Comprehensive flow topology analysis.
        
        Analyzes:
        - Critical points (attractors, repellers, saddles)
        - Flow statistics (mean divergence, vorticity)
        - Stability analysis
        
        Args:
            V: Vector field
            div_V: Divergence field (computed if None)
            curl_V: Curl field (computed if None)
            dx, dy: Grid spacing
        
        Returns:
            Dictionary with topology analysis results
        """
        from src.analysis.attractors import analyze_flow_topology
        
        if div_V is None:
            div_V = self.divergence(V)
        
        if curl_V is None:
            curl_V = self.curl(V)
        
        topology = analyze_flow_topology(div_V, curl_V)
        
        self._log(f"Topology: {len(topology['attractors'])} attractors, "
                 f"{len(topology['repellers'])} repellers, "
                 f"{len(topology['saddles'])} saddles")
        
        return topology
    
    # ========== Grid Utilities ==========
    
    @staticmethod
    def create_grid(
        points: np.ndarray,
        resolution: int = 50,
        padding: float = 0.1
    ) -> Tuple[Any, np.ndarray]:
        """Create uniform grid covering point cloud.

        Args:
            points: Point cloud, shape (n, 2) or (n, 3)
            resolution: Grid resolution per dimension
            padding: Fraction of range to pad on each side

        Returns:
            (grid_coords, grid_points)
        """
        if points.shape[-1] == 2:
            return create_uniform_grid(points, resolution=resolution, padding=padding)
        if points.shape[-1] == 3:
            return create_uniform_grid_3d(points, resolution=resolution, padding=padding)
        raise ValueError(f"Unsupported dimensions: {points.shape[-1]}")
    
    @staticmethod
    def interpolate_field(
        trajectories: List[np.ndarray],
        grid_points: np.ndarray,
        radius: float = 0.5,
        method: str = "velocity"
    ) -> np.ndarray:
        """Interpolate field values at grid points from trajectories.

        Args:
            trajectories: List of trajectories
            grid_points: Points to evaluate
            radius: Neighborhood radius
            method: "velocity" or "displacement"

        Returns:
            Interpolated vector field
        """
        return interpolate_field(trajectories, grid_points, radius=radius, method=method)
    
    # ========== Analysis Helpers ==========
    
    def compute_flow_statistics(
        self,
        V: np.ndarray,
        compute_all: bool = True
    ) -> Dict[str, float]:
        """Compute statistical properties of flow field.
        
        Args:
            V: Vector field
            compute_all: Whether to compute all derivatives
        
        Returns:
            Dictionary with flow statistics
        """
        magnitude = self.magnitude(V)
        
        stats = {
            "velocity_mean": float(magnitude.mean()),
            "velocity_std": float(magnitude.std()),
            "velocity_max": float(magnitude.max()),
            "velocity_min": float(magnitude.min()),
        }
        
        if compute_all:
            div_V = self.divergence(V)
            curl_V = self.curl(V)
            
            stats.update({
                "divergence_mean": float(div_V.mean()),
                "divergence_std": float(div_V.std()),
                "curl_mean": float(np.abs(curl_V).mean()),
                "curl_std": float(np.abs(curl_V).std()),
            })
        
        return stats
    
    def is_incompressible(
        self,
        V: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """Check if flow is approximately incompressible.
        
        A flow is incompressible if div(V) ≈ 0 everywhere.
        
        Args:
            V: Vector field
            tolerance: Tolerance for divergence
        
        Returns:
            True if flow is incompressible
        """
        div_V = self.divergence(V)
        return bool(np.abs(div_V).mean() < tolerance)
    
    def is_irrotational(
        self,
        V: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """Check if flow is approximately irrotational.
        
        A flow is irrotational if curl(V) ≈ 0 everywhere.
        
        Args:
            V: Vector field
            tolerance: Tolerance for curl
        
        Returns:
            True if flow is irrotational
        """
        curl_V = self.curl(V)
        return bool(np.abs(curl_V).mean() < tolerance)
    
    # ========== Visualization Helpers ==========
    
    def save_analysis(
        self,
        V: np.ndarray,
        output_dir: Path,
        compute_topology: bool = True,
        save_fields: bool = True
    ) -> Dict[str, Any]:
        """Save comprehensive flow analysis.
        
        Args:
            V: Vector field
            output_dir: Output directory
            compute_topology: Whether to analyze topology
            save_fields: Whether to save field arrays
        
        Returns:
            Analysis results
        """
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute all fields
        div_V = self.divergence(V)
        curl_V = self.curl(V)
        magnitude = self.magnitude(V)
        
        # Statistics
        stats: Dict[str, Any] = self.compute_flow_statistics(V, compute_all=False)
        stats.update({
            "divergence_mean": float(div_V.mean()),
            "divergence_std": float(div_V.std()),
            "curl_mean": float(np.abs(curl_V).mean()),
            "curl_std": float(np.abs(curl_V).std()),
            "incompressible": self.is_incompressible(V),
            "irrotational": self.is_irrotational(V),
        })
        
        # Topology
        if compute_topology:
            topology = self.analyze_topology(V, div_V, curl_V)
            stats["topology"] = {
                "n_attractors": len(topology["attractors"]),
                "n_repellers": len(topology["repellers"]),
                "n_saddles": len(topology["saddles"]),
            }
            
            # Save topology details
            with open(output_dir / "topology.json", "w") as f:
                topology_json = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in topology.items()
                }
                json.dump(topology_json, f, indent=2)
        
        # Save statistics
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save field arrays
        if save_fields:
            np.save(output_dir / "velocity.npy", V)
            np.save(output_dir / "divergence.npy", div_V)
            np.save(output_dir / "curl.npy", curl_V)
            np.save(output_dir / "magnitude.npy", magnitude)
        
        self._log(f"Saved analysis to {output_dir}")
        return stats


# Convenience function for quick physics operations
def quick_physics(
    V: np.ndarray,
    operation: str = "all",
    **kwargs: Any
) -> Union[np.ndarray, Dict[str, Any]]:
    """Quick physics operations on vector field.
    
    Args:
        V: Vector field
        operation: What to compute ("div", "curl", "grad", "laplacian", "topology", "all")
        **kwargs: Additional arguments
    
    Returns:
        Result of operation
    
    Example:
        >>> div_V = quick_physics(V, "div")
        >>> topology = quick_physics(V, "topology")
    """
    physics = PhysicsTools()
    
    if operation == "div":
        return physics.divergence(V, **kwargs)
    elif operation == "curl":
        return physics.curl(V, **kwargs)
    elif operation == "topology":
        return physics.analyze_topology(V, **kwargs)
    elif operation == "all":
        return {
            "divergence": physics.divergence(V),
            "curl": physics.curl(V),
            "magnitude": physics.magnitude(V),
            "topology": physics.analyze_topology(V),
        }
    else:
        raise ValueError(f"Unknown operation: {operation}")
