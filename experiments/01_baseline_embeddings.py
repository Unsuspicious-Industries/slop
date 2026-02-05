"""Encode historical images and store embeddings."""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.encoders.multimodal import EmbeddingExtractor
from src.utils.storage import ensure_dir
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strategy", default="concat", choices=["concat", "average", "separate"])
    return parser.parse_args()


def main():
    args = parse_args()
    extractor = EmbeddingExtractor(strategy=args.strategy)
    ensure_dir(args.output)
    embeddings = []
    files = list(args.images.rglob("*.jpg")) + list(args.images.rglob("*.png"))
    for path in tqdm(files, desc="Encoding images"):
        image = Image.open(path).convert("RGB")
        emb = extractor.encode(image)
        embeddings.append(emb)
    np.save(args.output / "embeddings.npy", np.stack(embeddings))


if __name__ == "__main__":
    main()
