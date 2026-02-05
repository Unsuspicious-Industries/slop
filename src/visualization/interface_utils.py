from pathlib import Path
# mypy: ignore-errors
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .embedding_space import plot_embedding_space
from .trajectories import plot_trajectories
from .flow_fields import plot_flow_field
from .heatmaps import plot_stereotype_density
from src.analysis.flow_fields import compute_flow_field


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def load_labels(path: Path) -> np.ndarray:
    return np.load(path)


def load_poles(path: Path) -> np.ndarray:
    return np.load(path)


def load_trajectories(dir_path: Path) -> List[np.ndarray]:
    trajectories = []
    for npy in sorted(dir_path.glob("*.npy")):
        trajectories.append(np.load(npy))
    return trajectories


def render_embedding_space(historical: np.ndarray, modern: np.ndarray):
    fig = plot_embedding_space(historical, modern)
    return fig


def render_trajectories(trajectories: List[np.ndarray], poles: Optional[np.ndarray]):
    if poles is None:
        poles = np.zeros((0, 2))
    fig = plot_trajectories(trajectories, poles)
    return fig


def render_flow_field(trajectories: List[np.ndarray], poles: Optional[np.ndarray], resolution: int = 40, radius: float = 0.5):
    poles_arr = poles if poles is not None else np.zeros((0, 2))
    grid, flow = compute_flow_field(trajectories, poles_arr if len(poles_arr) > 0 else None, grid_resolution=resolution, radius=radius)
    fig = plot_flow_field(grid, flow, poles_arr)
    return fig


def render_density(embeddings: np.ndarray, mask: np.ndarray):
    fig = plot_stereotype_density(embeddings, mask)
    return fig


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
