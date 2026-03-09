import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


def _flatten(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _reduce(points: np.ndarray) -> np.ndarray:
    flat = _flatten(points)
    if flat.shape[1] == 1:
        return np.concatenate([flat, np.zeros((flat.shape[0], 1), dtype=flat.dtype)], axis=1)
    if flat.shape[1] == 2 or flat.shape[0] < 3:
        return flat[:, :2]
    return np.asarray(UMAP(n_components=2, n_neighbors=min(10, flat.shape[0] - 1)).fit_transform(flat), dtype=np.float32)


def plot_embedding_space(a: np.ndarray, b: np.ndarray | None = None):
    pa = _reduce(a)
    plt.figure(figsize=(12, 8))
    if b is None:
        plt.scatter(pa[:, 0], pa[:, 1], alpha=0.7)
    else:
        pb = _reduce(b)
        plt.scatter(pa[:, 0], pa[:, 1], alpha=0.6, label="a")
        plt.scatter(pb[:, 0], pb[:, 1], alpha=0.6, label="b")
        plt.legend()
    return plt.gcf()


def plot_points(points: np.ndarray, values: np.ndarray | None = None, title: str = "Points"):
    reduced = _reduce(points)
    plt.figure(figsize=(10, 8))
    if values is None:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], c=np.asarray(values), alpha=0.7)
    plt.title(title)
    return plt.gcf()
