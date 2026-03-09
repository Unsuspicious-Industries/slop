import numpy as np

from client.interface import _probe_request


def test_probe_request_builds_batched_payload():
    req = _probe_request(model_id="m", points=np.random.randn(5, 4, 64, 64).astype(np.float32), timestep=12, guidance_scale=3.0, prompt="a")
    assert req.score_only is True
    assert req.probe_timestep == 12
    assert req.latent_override is not None
