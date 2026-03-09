import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from client.config import registry
from client.interface import SlopClient


def provider(name: str | None):
    """Return one provider from the registry."""
    if name is not None:
        cfg = registry.get(name)
        if cfg is None:
            raise ValueError(f"unknown provider: {name}")
        return cfg
    providers = registry.list()
    if not providers:
        raise ValueError("no providers registered")
    return providers[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider")
    args = parser.parse_args()

    latents = np.random.randn(2, 4, 64, 64).astype(np.float32)
    with SlopClient(provider(args.provider)) as client:
        images = client.render(latents)
        assert len(images) == 2
        print(len(images))


if __name__ == "__main__":
    main()
