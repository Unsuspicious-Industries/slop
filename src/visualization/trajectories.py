import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


def plot_trajectories(trajectories: list[np.ndarray], stereotype_poles: np.ndarray):
    """Plot trajectories in 2D after UMAP reduction.
    
    Args:
        trajectories: List of trajectory arrays, each shape (n_steps, dim)
        stereotype_poles: Array of poles, shape (n_poles, dim_poles)
            If dimensions don't match trajectories, poles are shown separately
    """
    reducer = UMAP(n_components=2)
    
    # First, check if dimensions match
    traj_dim = trajectories[0].shape[1]
    pole_dim = stereotype_poles.shape[1] if len(stereotype_poles.shape) > 1 else 0
    
    # Concatenate all trajectory points
    all_traj_points = np.concatenate(trajectories, axis=0)
    
    # If dimensions match, include poles in UMAP; otherwise reduce separately
    if pole_dim == traj_dim and len(stereotype_poles) > 0:
        all_points = np.concatenate([all_traj_points, stereotype_poles], axis=0)
        reduced = reducer.fit_transform(all_points)
        reduced_traj = reduced[:-len(stereotype_poles)]
        pole_reduced = reduced[-len(stereotype_poles):]
    else:
        # Reduce trajectories only
        reduced_traj = reducer.fit_transform(all_traj_points)
        pole_reduced = None
        if len(stereotype_poles) > 0:
            # Show poles in their own space or skip
            pole_reduced = None  # Can't visualize poles with different dimensionality
    
    plt.figure(figsize=(12, 8))
    cursor = 0
    for traj in trajectories:
        seg = reduced_traj[cursor : cursor + len(traj)]
        cursor += len(traj)
        plt.plot(seg[:, 0], seg[:, 1], alpha=0.3, linewidth=2)
    
    if pole_reduced is not None:
        plt.scatter(pole_reduced[:, 0], pole_reduced[:, 1], s=200, c="red", marker="*", edgecolors="black")
    
    plt.title("Denoising Trajectories")
    return plt.gcf()
