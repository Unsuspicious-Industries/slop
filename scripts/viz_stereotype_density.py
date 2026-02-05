import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.heatmaps import plot_stereotype_density
from src.utils.storage import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path, help="Integer labels; non-negative treated as stereotype clusters")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", type=int, default=0, help="Label to visualize as stereotype density")
    return p.parse_args()


def main():
    args = parse_args()
    embs = np.load(args.embeddings)
    labels = np.load(args.labels)
    mask = labels == args.label
    fig = plot_stereotype_density(embs, mask)
    ensure_dir(args.out.parent)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
