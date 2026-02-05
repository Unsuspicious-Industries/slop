import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.trajectories import plot_trajectories
from src.utils.storage import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True, type=Path, help="Directory with trajectory *.npy files")
    p.add_argument("--poles", required=True, type=Path, help="Numpy file of stereotype poles")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def load_trajectories(dir_path: Path):
    trajectories = []
    for npy in sorted(dir_path.glob("*.npy")):
        trajectories.append(np.load(npy))
    return trajectories


def main():
    args = parse_args()
    trajectories = load_trajectories(args.trajectories)
    poles = np.load(args.poles)
    fig = plot_trajectories(trajectories, poles)
    ensure_dir(args.out.parent)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
