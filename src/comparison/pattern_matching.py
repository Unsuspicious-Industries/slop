import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_historical_analogs(modern_embedding: np.ndarray, historical_embeddings: np.ndarray, k: int = 5):
    sims = cosine_similarity(modern_embedding.reshape(1, -1), historical_embeddings)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    return top_idx, sims[top_idx]
