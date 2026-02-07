"""Loaders and lightweight pipeline adapter for either local mocks or remote Slop client.

By default this returns a lightweight mock pipeline that's safe for unit tests.
Set environment variable `SLOP_REMOTE=1` to attempt to use the real `SlopClient` from
`client.interface` (requires configured ServerConfig and reachable server).
"""
import os
from typing import Any
import numpy as np

try:
    # Optional import for real remote client
    from client.interface import SlopClient
    from client.config import registry
except Exception:
    SlopClient = None


class MockPipeline:
    """A tiny pipeline-like object that simulates inference and returns latents.

    The `generate` method returns an object with `image` and `latents` attributes
    to mimic the remote `InferenceResult` shape used by hooks.
    """
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def generate(self, prompt: str, num_steps: int = 10, height: int = 64, width: int = 64, **kwargs) -> Any:
        # Simulate latents: (steps, batch=1, channels=4, H//8, W//8)
        h_lat = max(1, height // 8)
        w_lat = max(1, width // 8)
        latents = np.random.randn(num_steps, 1, 4, h_lat, w_lat).astype(np.float32)

        class R:
            pass

        r = R()
        r.image = b""  # image bytes not used heavily in tests
        r.latents = latents
        r.payload = {}
        r.arrays = {"latents": latents}
        return r


def _use_remote() -> bool:
    return os.environ.get("SLOP_REMOTE", "0") in ("1", "true", "True") and SlopClient is not None


def load_sd_pipeline(model_name: str = "CompVis/stable-diffusion-v1-4", device: str = "cpu", dtype=None):
    """Return either a remote client wrapper or a mock pipeline for SD.

    Tests can call the object's `generate(...)` method.
    """
    if _use_remote():
        # Use the registry's first server if available
        servers = registry.list()
        if len(servers) > 0:
            cfg = servers[0]
            client = SlopClient(cfg)
            return client
        else:
            # Fallback to mock
            return MockPipeline(model_name=model_name)

    return MockPipeline(model_name=model_name)


def load_flux_pipeline(model_name: str = "black-forest-labs/FLUX.1-dev", device: str = "cpu"):
    if _use_remote():
        servers = registry.list()
        if len(servers) > 0 and SlopClient is not None:
            cfg = servers[0]
            client = SlopClient(cfg)
            return client
        else:
            return MockPipeline(model_name=model_name)

    return MockPipeline(model_name=model_name)
