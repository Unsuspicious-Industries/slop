"""Attractor and repeller detection from divergence fields."""
import numpy as np
from typing import Tuple

from shared.physics.fields import find_critical_points as _find_critical_points


def find_attractors(
    div_V: np.ndarray,
    threshold: float = -0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return positions and strengths of attractors (sinks) in ``div_V``.

    Args:
        div_V: Divergence field, shape ``(ny, nx)``
        threshold: Convergence threshold (negative value)

    Returns:
        ``(positions, strengths)`` — ``(N, 2)`` grid indices and ``(N,)`` values
    """
    cp = _find_critical_points(div_V, attractor_threshold=threshold)
    return cp["attractors"]


def find_repellers(
    div_V: np.ndarray,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return positions and strengths of repellers (sources) in ``div_V``.

    Args:
        div_V: Divergence field, shape ``(ny, nx)``
        threshold: Divergence threshold (positive value)

    Returns:
        ``(positions, strengths)`` — ``(N, 2)`` grid indices and ``(N,)`` values
    """
    cp = _find_critical_points(div_V, repeller_threshold=threshold)
    return cp["repellers"]
