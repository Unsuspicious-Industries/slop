import argparse
import sys
from pathlib import Path

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

    with SlopClient(provider(args.provider)) as client:
        result = client.sample(prompt="a sphere in a white room", batch_size=2, num_steps=4)
        assert result.points is not None
        assert result.forces is not None
        print(result.points.shape)
        print(result.forces.shape)

        # Render latents to images
        images = client.render(result.points[-1])
        print(f"Rendered {len(images)} images")


if __name__ == "__main__":
    main()
