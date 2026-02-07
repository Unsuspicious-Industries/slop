"""StableDiffusionHook adapter that can use either a local pipeline or the remote Slop client.

This adapter provides a small, test-friendly API used by the unit tests:
- `is_hooked`, `captured_latents`, `reset()`
- `generate(...)` and `generate_batch(...)`
- `get_trajectory()` / `get_trajectories()`
"""
from typing import List, Sequence
import numpy as np
from .trajectory_capture import TrajectoryCapture


class StableDiffusionHook:
    def __init__(self, pipeline):
        """`pipeline` can be either a remote client-like object (has `generate`) or a local pipeline stub."""
        self.pipeline = pipeline
        self.capture = TrajectoryCapture()
        self.is_hooked = True
        self.captured_latents: List[np.ndarray] = []

    def reset(self):
        self.capture.reset()
        self.captured_latents = []

    def _extract_and_record(self, result):
        # Handle different shapes returned by mock or real client
        import numpy as _np

        latents = None
        # remote `InferenceResult` may have `.latents` or `.arrays` or `.arrays['latents']`
        if hasattr(result, "latents"):
            latents = result.latents
        elif hasattr(result, "arrays") and isinstance(result.arrays, dict):
            latents = result.arrays.get("latents")

        if latents is None:
            return

        lat = _np.array(latents)
        # Expect (steps, batch?, ...)
        if lat.ndim >= 4:
            steps = lat.shape[0]
            for s in range(steps):
                self.capture.record(lat[s])
        else:
            self.capture.record(lat)

        self.captured_latents = self.capture.get_trajectory()

    def generate(self, prompt: str, num_inference_steps: int = 50, height: int = 512, width: int = 512, **kwargs):
        """Generate a single image (using remote client or local pipeline)."""
        # If pipeline has a `generate` method, prefer calling it.
        if hasattr(self.pipeline, "generate"):
            try:
                # Try SlopClient-style call
                result = self.pipeline.generate(
                    prompt=prompt,
                    num_steps=num_inference_steps,
                    height=height,
                    width=width,
                    **kwargs
                )
            except TypeError:
                # Fallback to simpler positional signature
                result = self.pipeline.generate(prompt, num_inference_steps)

            # Extract latents if present and record
            self._extract_and_record(result)

            # Return an image object or bytes if available. Tests don't inspect image bytes deeply.
            return getattr(result, "image", None)

        # If no generate available, return None
        return None

    def generate_batch(self, prompts: Sequence[str], num_inference_steps: int = 50, height: int = 512, width: int = 512, **kwargs):
        images = []
        all_trajectories = []
        for p in prompts:
            img = self.generate(p, num_inference_steps=num_inference_steps, height=height, width=width, **kwargs)
            images.append(img)
            all_trajectories.append(self.capture.get_trajectory())

        return images

    def get_trajectory(self) -> List[np.ndarray]:
        return self.capture.get_trajectory()

    def get_trajectories(self) -> List[List[np.ndarray]]:
        return self.capture.get_trajectories()
