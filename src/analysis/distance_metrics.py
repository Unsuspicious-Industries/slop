import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def min_distance_to_poles(latent: np.ndarray, poles: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(latent - poles, axis=1)))


def historical_continuity(modern_embeddings: np.ndarray, historical_embeddings: np.ndarray) -> float:
    similarities = cosine_similarity(modern_embeddings, historical_embeddings)
    max_sim = np.max(similarities, axis=1)
    return float(np.mean(max_sim))
