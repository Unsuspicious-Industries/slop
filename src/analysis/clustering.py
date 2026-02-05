from typing import Tuple, cast

import numpy as np
from sklearn.cluster import DBSCAN
from umap import UMAP


def reduce_embeddings(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    reducer = UMAP(n_components=n_components)
    return cast(np.ndarray, reducer.fit_transform(embeddings))


def dbscan_clusters(embeddings_reduced: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    return cast(np.ndarray, clusterer.fit_predict(embeddings_reduced))


def identify_dense_clusters(embeddings_reduced: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(labels[labels >= 0])
    centers = []
    counts = []
    for lbl in unique:
        pts = embeddings_reduced[labels == lbl]
        centers.append(pts.mean(axis=0))
        counts.append(len(pts))
    return np.array(centers), np.array(counts)
