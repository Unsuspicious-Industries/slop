import matplotlib.pyplot as plt
import numpy as np

from .embedding_space import _reduce


def plot_trajectories(trajectories: list[np.ndarray]):
    if not trajectories:
        raise ValueError("expected at least one trajectory")
    stacked = np.concatenate([np.asarray(traj) for traj in trajectories], axis=0)
    reduced = _reduce(stacked)
    plt.figure(figsize=(12, 8))
    cursor = 0
    for trajectory in trajectories:
        length = len(trajectory)
        segment = reduced[cursor:cursor + length]
        cursor += length
        plt.plot(segment[:, 0], segment[:, 1], alpha=0.5)
    return plt.gcf()
