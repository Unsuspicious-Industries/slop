"""Run a quick bundle of visualizations from available artifacts."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.visualization.embedding_space import plot_embedding_space
from src.visualization.trajectories import plot_trajectories
from src.visualization.flow_fields import plot_flow_field
from src.analysis.flow_fields import compute_flow_field
from src.utils.storage import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", required=True, type=Path)
    p.add_argument("--modern", required=True, type=Path)
    p.add_argument("--trajectories", required=True, type=Path)
    p.add_argument("--poles", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
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
    ensure_dir(args.outdir)

    hist = np.load(args.historical)
    modern = np.load(args.modern)
    poles = np.load(args.poles)
    trajectories = load_trajectories(args.trajectories)

    fig1 = plot_embedding_space(hist, modern)
    fig1.savefig(args.outdir / "embedding_space.png", dpi=200, bbox_inches="tight")
    fig1.clear()

    fig2 = plot_trajectories(trajectories, poles)
    fig2.savefig(args.outdir / "trajectories.png", dpi=200, bbox_inches="tight")
    fig2.clear()

    grid, flow = compute_flow_field(trajectories, poles, grid_resolution=args.resolution, radius=args.radius)
    fig3 = plot_flow_field(grid, flow, poles)
    fig3.savefig(args.outdir / "flow_field.png", dpi=200, bbox_inches="tight")
    fig3.clear()


if __name__ == "__main__":
    main()
