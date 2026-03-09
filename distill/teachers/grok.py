from __future__ import annotations

import base64
import os
from typing import Any

import xai_sdk

from distill.env import load_dotenv_if_present
from .base import TeacherClient, TeacherSample


class GrokTeacher(TeacherClient):
    """Grok (xAI) image generation backend.

    Returns only the final image; no intermediate partials are available.
    """

    name = "grok"

    def __init__(self, model: str = "grok-imagine-image", api_key: str | None = None):
        load_dotenv_if_present()
        self.model = model
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for GrokTeacher")
        self.client = xai_sdk.AsyncClient(api_key=self.api_key)

    async def generate(self, prompt: str) -> TeacherSample:
        # Request base64 output to avoid extra HTTP fetch
        resp = await self.client.image.sample(
            prompt=prompt,
            model=self.model,
            image_format="base64",
        )
        # xai_sdk returns .image as bytes when image_format=base64
        if hasattr(resp, "image") and resp.image:
            img_field = resp.image
            if isinstance(img_field, (bytes, bytearray)):
                img_bytes = bytes(img_field)
            else:
                img_bytes = base64.b64decode(str(img_field))
        elif getattr(resp, "url", None):
            # Fallback: download from URL (unlikely when base64 requested)
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(resp.url) as r:
                    r.raise_for_status()
                    img_bytes = await r.read()
        else:
            raise RuntimeError("Grok response missing image data")

        meta: dict[str, Any] = {
            "model": resp.model if hasattr(resp, "model") else self.model,
            "respect_moderation": getattr(resp, "respect_moderation", None),
        }

        return TeacherSample(
            prompt=prompt,
            teacher=self.name,
            final_image=bytes(img_bytes),
            partial_images=[],
            metadata=meta,
        )
