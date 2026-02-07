import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


def plot_embedding_space(
    historical_embs: np.ndarray,
    modern_embs: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    color_by: str = "groups",
):
    if modern_embs is None:
        all_embs = historical_embs
    else:
        all_embs = np.vstack([historical_embs, modern_embs])

    if len(all_embs) < 3:
        reduced = all_embs[:, :2] if all_embs.shape[1] >= 2 else np.hstack([all_embs, np.zeros((len(all_embs), 1))])
    else:
        reducer = UMAP(n_components=2, n_neighbors=min(10, len(all_embs) - 1))
        reduced = reducer.fit_transform(all_embs)

    plt.figure(figsize=(12, 8))
    if modern_embs is None:
        if color_by == "groups" and groups is not None and len(groups) == len(reduced):
            unique_groups = sorted({g for g in groups if g is not None})
            palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_groups))))
            for idx, group in enumerate(unique_groups):
                mask = np.array([g == group for g in groups])
                plt.scatter(reduced[mask, 0], reduced[mask, 1], alpha=0.7, label=str(group), color=palette[idx])
            plt.title("Embedding Space (Grouped)")
            plt.legend()
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], c="blue", alpha=0.7)
            plt.title("Embedding Space")
        return plt.gcf()

    n_hist = len(historical_embs)
    hist_reduced = reduced[:n_hist]
    modern_reduced = reduced[n_hist:]
    plt.scatter(hist_reduced[:, 0], hist_reduced[:, 1], c="red", alpha=0.6, label="Historical")
    plt.scatter(modern_reduced[:, 0], modern_reduced[:, 1], c="blue", alpha=0.6, label="Modern AI")
    plt.title("Latent Space: Historical vs Modern")
    plt.legend()
    return plt.gcf()
