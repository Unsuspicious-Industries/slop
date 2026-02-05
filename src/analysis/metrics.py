import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def stereotype_concentration(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette-like: ratio of between/within via cosine similarity mean diff."""
    sims = cosine_similarity(embeddings)
    same = sims[labels[:, None] == labels[None, :]]
    diff = sims[labels[:, None] != labels[None, :]]
    return float(np.nanmean(same) - np.nanmean(diff))


def historical_continuity(modern_embeddings: np.ndarray, historical_embeddings: np.ndarray) -> float:
    sims = cosine_similarity(modern_embeddings, historical_embeddings)
    max_sim = np.max(sims, axis=1)
    return float(np.mean(max_sim))


def trajectory_deviation_score(trajectory: np.ndarray, baseline: np.ndarray) -> float:
    n = min(len(trajectory), len(baseline))
    return float(np.mean(np.linalg.norm(trajectory[:n] - baseline[:n], axis=1)))


def drift_strength(trajectory: np.ndarray, stereotype_poles: np.ndarray) -> float:
    distances = [np.min(np.linalg.norm(step - stereotype_poles, axis=1)) for step in trajectory]
    return float(distances[0] - distances[-1])


def expected_bias_fit(modern_embeddings: np.ndarray, expected_vectors: np.ndarray) -> float:
    """Measure alignment to expected bias directions (cosine)."""
    sims = cosine_similarity(modern_embeddings, expected_vectors)
    return float(np.mean(np.max(sims, axis=1)))
