"""High-level physics analysis tools backed by shared.physics."""
import numpy as np

from shared.physics import divergence, curl, gradient, find_critical_points


class PhysicsTools:
    """Convenience wrapper for common vector field analyses."""

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_flow_statistics(
        self,
        V: np.ndarray,
        compute_all: bool = False,
    ) -> dict:
        """Compute summary statistics for a 2D vector field.

        Args:
            V: Vector field, shape ``(ny, nx, 2)``
            compute_all: Also compute divergence and curl statistics

        Returns:
            dict with ``velocity_mean``, ``velocity_std``, and — when
            ``compute_all=True`` — ``divergence_mean``, ``curl_mean``
        """
        speed = np.linalg.norm(V, axis=-1)
        stats: dict = {
            "velocity_mean": float(speed.mean()),
            "velocity_std": float(speed.std()),
        }
        if compute_all:
            stats["divergence_mean"] = float(divergence(V).mean())
            stats["curl_mean"] = float(curl(V).mean())
        return stats

    # ------------------------------------------------------------------
    # Qualitative checks
    # ------------------------------------------------------------------

    def is_incompressible(self, V: np.ndarray, tolerance: float = 0.1) -> bool:
        """True if ``|div(V)|`` is everywhere below ``tolerance``."""
        return float(np.abs(divergence(V)).mean()) < tolerance

    def is_irrotational(self, V: np.ndarray, tolerance: float = 0.1) -> bool:
        """True if ``|curl(V)|`` is everywhere below ``tolerance``."""
        return float(np.abs(curl(V)).mean()) < tolerance

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def analyze_topology(self, V: np.ndarray) -> dict:
        """Find attractors, repellers, and saddles in ``V``.

        Returns:
            dict with keys ``attractors``, ``repellers``, ``saddles``
            each holding an ``(N, 2)`` array of grid-index positions.
        """
        div_V = divergence(V)
        cp = find_critical_points(div_V)
        return {
            "attractors": cp["attractors"][0],
            "repellers": cp["repellers"][0],
            "saddles": cp["saddles"][0],
        }
