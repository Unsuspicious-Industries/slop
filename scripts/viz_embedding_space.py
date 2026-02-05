import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.embedding_space import plot_embedding_space
from src.utils.storage import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", required=False, type=Path)
    p.add_argument("--modern", required=False, type=Path)
    p.add_argument("--combined", required=False, type=Path, help="NPZ with embeddings + labels (+ optional groups)")
    p.add_argument("--color-by", default="groups", choices=["groups", "labels"], help="Color points by groups or labels if combined")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    if args.combined is not None:
        data = np.load(args.combined, allow_pickle=True)
        fig = plot_embedding_space(data["embeddings"], labels=data.get("labels"), groups=data.get("groups"), color_by=args.color_by)
    else:
        if args.historical is None or args.modern is None:
            raise ValueError("Provide --combined or both --historical and --modern")
        hist = np.load(args.historical)
        modern = np.load(args.modern)
        fig = plot_embedding_space(hist, modern)
    ensure_dir(args.out.parent)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
