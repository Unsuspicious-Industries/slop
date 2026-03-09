from __future__ import annotations

"""Skeleton evaluation script for distilled models.

Placeholder for FID, CLIP similarity, and bias-field comparisons. This is
intentionally lightweight pending a full training implementation.
"""

import argparse
from pathlib import Path

from distill.dataset import DistillDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled model (skeleton)")
    parser.add_argument("manifest", type=Path, help="Path to manifest.jsonl from collection")
    args = parser.parse_args()

    ds = DistillDataset(args.manifest, use_partials=True)
    print(f"Loaded {len(ds)} samples; evaluation logic not implemented in this stub.")


if __name__ == "__main__":
    main()
