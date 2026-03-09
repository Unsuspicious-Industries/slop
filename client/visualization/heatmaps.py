import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .embedding_space import _reduce


def plot_density(points: np.ndarray):
    reduced = np.asarray(_reduce(points), dtype=np.float32)
    kde = gaussian_kde(reduced.T)
    x = reduced[:, 0]
    y = reduced[:, 1]
    xx, yy = np.meshgrid(np.linspace(float(x.min()), float(x.max()), 100), np.linspace(float(y.min()), float(y.max()), 100))
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, density, levels=20, cmap="YlOrRd")
    return plt.gcf()
