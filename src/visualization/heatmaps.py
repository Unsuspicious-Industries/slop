import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from umap import UMAP


def plot_stereotype_density(embeddings: np.ndarray, stereotype_mask: np.ndarray):
    reducer = UMAP(n_components=2)
    reduced = reducer.fit_transform(embeddings)
    stereotype_points = reduced[stereotype_mask.astype(bool)]
    kde = gaussian_kde(stereotype_points.T)
    x_min, x_max = reduced[:, 0].min(), reduced[:, 0].max()
    y_min, y_max = reduced[:, 1].min(), reduced[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, density, levels=20, cmap="YlOrRd")
    plt.colorbar(label="Stereotype Density")
    plt.title("Heatmap: Stereotype Concentration")
    return plt.gcf()
