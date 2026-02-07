import matplotlib.pyplot as plt
import numpy as np


def plot_flow_field(grid: np.ndarray, flow_vectors: np.ndarray, stereotype_poles: np.ndarray):
    resolution = int(np.sqrt(len(grid)))
    X, Y = np.meshgrid(
        np.linspace(grid[:, 0].min(), grid[:, 0].max(), resolution),
        np.linspace(grid[:, 1].min(), grid[:, 1].max(), resolution),
    )
    U = flow_vectors[..., 0]
    V = flow_vectors[..., 1]
    magnitude = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(12, 10))
    stream = ax.streamplot(X, Y, U, V, color=magnitude, cmap="Reds", density=2, linewidth=2)
    for pole in stereotype_poles:
        ax.scatter(pole[0], pole[1], s=300, c="black", marker="X", edgecolors="white", linewidths=2)
    plt.colorbar(stream.lines, label="Drift Strength")
    plt.title("Flow Field in Latent Space")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    return fig
