"""Flow field topology analysis.

This module analyzes critical points and topological features of vector fields.
Uses standard physics formulas in ASCII (div, curl, grad).
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import json

from src.physics.types import VectorField, ScalarField
from src.physics.operators import divergence, curl, divergence_3d, curl_3d
from src.physics.fields import find_critical_points as find_critical_points_physics

def find_attractors(
    div_V: ScalarField,
    threshold: float = 0.0,
    return_values: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Find attractor locations where div(V) < 0 (sinks).

    Attractors are regions where flow converges (negative divergence).

    Args:
        div_V: Divergence field
        threshold: Divergence threshold (more negative = stronger sink)
        return_values: Whether to return divergence values

    Returns:
        r_attractors: Attractor positions (grid indices)
        strengths: Optional divergence values at attractors
    """
    attractor_mask = div_V < threshold
    r_attractors = np.argwhere(attractor_mask)

    if return_values:
        strengths = div_V[attractor_mask]
        return r_attractors, strengths

    return r_attractors, None


def find_repellers(
    div_V: ScalarField,
    threshold: float = 0.0,
    return_values: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Find repeller locations where div(V) > 0 (sources).

    Repellers are regions where flow diverges (positive divergence).

    Args:
        div_V: Divergence field
        threshold: Divergence threshold (more positive = stronger source)
        return_values: Whether to return divergence values

    Returns:
        r_repellers: Repeller positions (grid indices)
        strengths: Optional divergence values at repellers
    """
    repeller_mask = div_V > threshold
    r_repellers = np.argwhere(repeller_mask)

    if return_values:
        strengths = div_V[repeller_mask]
        return r_repellers, strengths

    return r_repellers, None


def find_critical_points(
    div_V: ScalarField,
    attractor_threshold: float = -0.1,
    repeller_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all critical points in flow field based on divergence.

    Classification:
    - Attractors (sinks): div(V) < threshold_a
    - Repellers (sources): div(V) > threshold_r
    - Saddles: |div(V)| ~= 0

    Args:
        div_V: Divergence field
        attractor_threshold: Threshold for attractors (negative)
        repeller_threshold: Threshold for repellers (positive)

    Returns:
        r_attractors: Attractor positions
        r_repellers: Repeller positions
        r_saddles: Saddle point positions
    """
    r_attractors, _ = find_attractors(div_V, attractor_threshold)
    r_repellers, _ = find_repellers(div_V, repeller_threshold)

    saddle_threshold = min(abs(attractor_threshold), abs(repeller_threshold))
    saddle_mask = np.abs(div_V) < saddle_threshold
    r_saddles = np.argwhere(saddle_mask)

    return r_attractors, r_repellers, r_saddles


def compute_attractor_strength(
    div_V: ScalarField,
    r_positions: np.ndarray,
    radius: int = 1
) -> np.ndarray:
    """Compute attractor strength by integrating |div(V)| in local neighborhood.

    The strength S of an attractor at r0 is:
    S = integral |div(V(r))| dA over neighborhood |r - r0| < radius

    Args:
        div_V: Divergence field
        r_positions: Attractor positions, shape (n, ndim)
        radius: Neighborhood radius for integration

    Returns:
        S: Attractor strengths, shape (n,)
    """
    strengths = []

    for r in r_positions:
        # Create neighborhood slice
        slices = tuple(
            slice(max(0, ri - radius), min(ni, ri + radius + 1))
            for ri, ni in zip(r, div_V.shape)
        )

        neighborhood = div_V[slices]
        strength = float(np.abs(neighborhood.sum()))
        strengths.append(strength)

    return np.array(strengths)


def analyze_flow_topology(
    div_V: ScalarField,
    curl_V: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    prefix: str = "topology",
    attractor_threshold: float = -0.1,
    repeller_threshold: float = 0.1
) -> Dict:
    """Comprehensive topology analysis of vector field V.

    Analyzes critical points and topological features:
    - Attractors (sinks): div(V) < 0
    - Repellers (sources): div(V) > 0
    - Vorticity: |curl(V)|

    Args:
        div_V: Divergence field
        curl_V: Optional curl/vorticity (2D: scalar, 3D: vector)
        save_dir: Directory to save debug data
        prefix: Filename prefix
        attractor_threshold: Threshold for attractors
        repeller_threshold: Threshold for repellers

    Returns:
        Topology metrics dictionary
    """
    # Find critical points
    r_attractors, strengths_a = find_attractors(div_V, attractor_threshold, return_values=True)
    r_repellers, strengths_r = find_repellers(div_V, repeller_threshold, return_values=True)
    
    results = {
        'n_attractors': len(r_attractors),
        'n_repellers': len(r_repellers),
        'attractors': r_attractors,  # For backward compatibility
        'repellers': r_repellers,  # For backward compatibility
        'saddles': np.array([]),  # Placeholder
        'attractor_positions': r_attractors,
        'repeller_positions': r_repellers,
        'attractor_strengths': strengths_a,
        'repeller_strengths': strengths_r,
        'divergence_stats': {
            'mean': float(div_V.mean()),
            'std': float(div_V.std()),
            'min': float(div_V.min()),
            'max': float(div_V.max()),
        }
    }
    
    # Analyze curl/vorticity if provided
    if curl_V is not None:
        if curl_V.ndim == 2:
            # 2D: scalar vorticity
            results['vorticity_stats'] = {
                'mean': float(curl_V.mean()),
                'std': float(curl_V.std()),
                'min': float(curl_V.min()),
                'max': float(curl_V.max()),
            }
        else:
            # 3D: vector curl |curl(V)|
            magnitude = np.linalg.norm(curl_V, axis=-1)
            results['curl_magnitude_stats'] = {
                'mean': float(magnitude.mean()),
                'std': float(magnitude.std()),
                'min': float(magnitude.min()),
                'max': float(magnitude.max()),
            }
    
    # Save debug data
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save fields as .npy
        np.save(save_dir / f"{prefix}_divergence.npy", div_V)
        np.save(save_dir / f"{prefix}_attractors.npy", r_attractors)
        np.save(save_dir / f"{prefix}_repellers.npy", r_repellers)
        
        if strengths_a is not None:
            np.save(save_dir / f"{prefix}_attractor_strengths.npy", strengths_a)
        if strengths_r is not None:
            np.save(save_dir / f"{prefix}_repeller_strengths.npy", strengths_r)
        
        if curl_V is not None:
            np.save(save_dir / f"{prefix}_curl.npy", curl_V)
        
        # Save summary JSON
        summary = {k: v for k, v in results.items() 
                   if not isinstance(v, np.ndarray)}
        
        with open(save_dir / f"{prefix}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        
        print(f"Saved topology analysis to {save_dir}")
    
    return results

