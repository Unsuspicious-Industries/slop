"""Bias metrics for analyzing embedding spaces and trajectories.

Consolidated from src/analysis/metrics.py, distance_metrics.py, and comparison/correlation.py.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, List

NDArray = np.ndarray


def stereotype_concentration(embeddings: NDArray, labels: NDArray) -> float:
    """Measure how tightly clustered embeddings are by label.
    
    Calculated as the difference between mean intra-cluster similarity
    and mean inter-cluster similarity.
    
    Args:
        embeddings: Shape (n_samples, n_features)
        labels: Shape (n_samples,) cluster/class labels
        
    Returns:
        Score where higher values indicate tighter clustering (more stereotype concentration).
    """
    sims = cosine_similarity(embeddings)
    same = sims[labels[:, None] == labels[None, :]]
    diff = sims[labels[:, None] != labels[None, :]]
    return float(np.nanmean(same) - np.nanmean(diff))


def historical_continuity(modern_embeddings: NDArray, historical_embeddings: NDArray) -> float:
    """Measure how well modern embeddings align with historical ones.
    
    Calculated as the mean of the maximum cosine similarity for each modern
    embedding against all historical embeddings.
    
    Args:
        modern_embeddings: Shape (n_modern, n_features)
        historical_embeddings: Shape (n_hist, n_features)
        
    Returns:
        Score in [0, 1] (assuming normalized vectors) indicating alignment.
    """
    sims = cosine_similarity(modern_embeddings, historical_embeddings)
    max_sim = np.max(sims, axis=1)
    return float(np.mean(max_sim))


def trajectory_deviation(trajectory: NDArray, baseline: NDArray) -> float:
    """Measure deviation of a trajectory from a baseline path.
    
    Args:
        trajectory: Shape (n_steps, n_features)
        baseline: Shape (n_steps, n_features)
        
    Returns:
        Mean Euclidean distance between corresponding points.
    """
    n = min(len(trajectory), len(baseline))
    return float(np.mean(np.linalg.norm(trajectory[:n] - baseline[:n], axis=1)))


def drift_strength(trajectory: NDArray, stereotype_poles: NDArray) -> float:
    """Measure how much a trajectory drifts towards or away from stereotype poles.
    
    Calculated as the change in minimum distance to any pole from start to end.
    Positive values indicate drift AWAY from poles (distance increases).
    Negative values indicate drift TOWARDS poles (distance decreases).
    
    Args:
        trajectory: Shape (n_steps, n_features)
        stereotype_poles: Shape (n_poles, n_features)
        
    Returns:
        Change in minimum distance (start_dist - end_dist).
    """
    distances = [np.min(np.linalg.norm(step - stereotype_poles, axis=1)) for step in trajectory]
    return float(distances[0] - distances[-1])


def expected_bias_fit(modern_embeddings: NDArray, expected_vectors: NDArray) -> float:
    """Measure alignment to expected bias directions.
    
    Args:
        modern_embeddings: Shape (n_samples, n_features)
        expected_vectors: Shape (n_expected, n_features)
        
    Returns:
        Mean maximum cosine similarity.
    """
    sims = cosine_similarity(modern_embeddings, expected_vectors)
    return float(np.mean(np.max(sims, axis=1)))


def min_pole_distance(latent: NDArray, poles: NDArray) -> float:
    """Compute the minimum Euclidean distance from a latent vector to any pole.
    
    Args:
        latent: Shape (n_samples, n_features) or (n_features,)
        poles: Shape (n_poles, n_features)
        
    Returns:
        Minimum distance.
    """
    # Ensure latent is 2D
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
        
    return float(np.min(np.linalg.norm(latent - poles, axis=1)))
