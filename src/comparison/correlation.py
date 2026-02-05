import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def historical_continuity_score(modern_embeddings: np.ndarray, historical_embeddings: np.ndarray) -> float:
    sims = cosine_similarity(modern_embeddings, historical_embeddings)
    max_sim = np.max(sims, axis=1)
    return float(np.mean(max_sim))
