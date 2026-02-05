import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.flow_fields import compute_flow_field
from src.visualization.flow_fields import plot_flow_field
from src.utils.storage import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True, type=Path, help="Directory containing *.npy trajectories")
    p.add_argument("--poles", required=True, type=Path, help="Numpy file of stereotype poles")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--resolution", type=int, default=50)
    p.add_argument("--radius", type=float, default=0.5)
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
    grid, flow = compute_flow_field(trajectories, poles, grid_resolution=args.resolution, radius=args.radius)
    fig = plot_flow_field(grid, flow, poles)
    ensure_dir(args.out.parent)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
