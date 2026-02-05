import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.comparison.pattern_matching import find_historical_analogs
from src.utils.storage import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modern", required=True, type=Path)
    p.add_argument("--historical", required=True, type=Path)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    modern = np.load(args.modern)
    hist = np.load(args.historical)
    fig, axes = plt.subplots(args.k, 2, figsize=(6, 3 * args.k))
    for i in range(args.k):
        idx, sims = find_historical_analogs(modern[i], hist, k=1)
        axes[i, 0].plot(modern[i])
        axes[i, 0].set_title(f"Modern {i}")
        axes[i, 1].plot(hist[idx[0]])
        axes[i, 1].set_title(f"Hist match sim={sims[0]:.2f}")
    plt.tight_layout()
    ensure_dir(args.out.parent)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
