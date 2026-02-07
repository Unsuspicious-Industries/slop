"""Utility for recording denoising trajectories (latents) during inference.

This is a lightweight implementation used by tests and by the hook adapters.
"""
from typing import List
import numpy as np


class TrajectoryCapture:
    def __init__(self):
        self._steps: List[np.ndarray] = []

    def record(self, latent: np.ndarray):
        """Record a latent snapshot (one timestep)."""
        self._steps.append(np.array(latent))

    def reset(self):
        self._steps = []

    def get_trajectory(self) -> List[np.ndarray]:
        return list(self._steps)

    def get_trajectories(self) -> List[List[np.ndarray]]:
        # For compatibility with batch APIs, return list-of-list
        return [self.get_trajectory()]

    def save(self, path):
        arr = np.stack(self._steps, axis=0)
        np.save(path, arr)

    def get_trajectory_2d(self, method: str = "pca") -> np.ndarray:
        """Project trajectory to 2D. Minimal PCA implementation.

        Returns array of shape (n_steps, 2).
        """
        if len(self._steps) == 0:
            return np.zeros((0, 2))

        X = np.stack([x.reshape(-1) for x in self._steps], axis=0)
        # Center
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # Project onto first two components
        comps = Vt[:2].T
        proj = Xc.dot(comps)
        return proj
