"""Identify stereotype poles via clustering."""

import argparse
from pathlib import Path

import numpy as np

from src.analysis.clustering import dbscan_clusters, identify_dense_clusters, reduce_embeddings
from src.utils.storage import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--method", default="dbscan")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--components", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    embs = np.load(args.embeddings)
    reduced = reduce_embeddings(embs, n_components=args.components)
    labels = dbscan_clusters(reduced, eps=args.eps, min_samples=args.min_samples)
    centers, counts = identify_dense_clusters(reduced, labels)
    ensure_dir(args.output.parent)
    np.save(args.output, centers)
    np.save(args.output.with_suffix("_counts.npy"), counts)


if __name__ == "__main__":
    main()
