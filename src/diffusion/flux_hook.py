"""Simple FLUX hook adapter that mirrors the SD hook behavior."""
from typing import List
import numpy as np
from .trajectory_capture import TrajectoryCapture


class FluxHook:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.capture = TrajectoryCapture()
        self.is_hooked = True

    def reset(self):
        self.capture.reset()

    def generate(self, prompt: str, num_inference_steps: int = 50, height: int = 512, width: int = 512, **kwargs):
        if hasattr(self.pipeline, "generate"):
            try:
                result = self.pipeline.generate(
                    prompt=prompt,
                    num_steps=num_inference_steps,
                    height=height,
                    width=width,
                    **kwargs
                )
            except TypeError:
                result = self.pipeline.generate(prompt, num_inference_steps)

            # record latents if present
            latents = None
            if hasattr(result, "latents"):
                latents = result.latents
            elif hasattr(result, "arrays") and isinstance(result.arrays, dict):
                latents = result.arrays.get("latents")

            if latents is not None:
                lat = np.array(latents)
                if lat.ndim >= 4:
                    for s in range(lat.shape[0]):
                        self.capture.record(lat[s])
                else:
                    self.capture.record(lat)

            return getattr(result, "image", None)

        return None

    def get_trajectory(self) -> List[np.ndarray]:
        return self.capture.get_trajectory()
