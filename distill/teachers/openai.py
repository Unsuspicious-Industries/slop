from __future__ import annotations

import base64
import os
from typing import Any, List

from openai import AsyncOpenAI

from distill.env import load_dotenv_if_present
from .base import TeacherClient, TeacherSample


class OpenAITeacher(TeacherClient):
    """OpenAI GPT-Image backend with partial image streaming support."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-image-1",
        api_key: str | None = None,
        partial_images: int = 0,
    ):
        load_dotenv_if_present()
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAITeacher")
        self.partial_images = partial_images
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate(self, prompt: str) -> TeacherSample:
        partials: List[bytes] = []

        if self.partial_images > 0:
            # Stream to capture partial images
            stream = await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                response_format="b64_json",
                partial_images=self.partial_images,
            )
            final_b64 = None
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype.endswith("partial_image"):
                    b64 = getattr(event, "b64_json", None) or getattr(event, "partial_image_b64", None)
                    if b64:
                        partials.append(base64.b64decode(b64))
                elif etype.endswith("image") or etype == "image_generation":
                    # final event in some SDK variants
                    b64 = getattr(event, "b64_json", None) or getattr(event, "result", None)
                    if b64:
                        final_b64 = b64
            if final_b64 is None and partials:
                # Some SDKs yield the final image as the last partial; use it.
                final_b64 = base64.b64encode(partials[-1]).decode("ascii")
            if final_b64 is None:
                raise RuntimeError("No final image received from OpenAI stream")
            final_image = base64.b64decode(final_b64)
        else:
            resp = await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                response_format="b64_json",
            )
            data = resp.data[0]
            final_image = base64.b64decode(data.b64_json)

        meta: dict[str, Any] = {
            "model": self.model,
            "partial_images": len(partials),
        }

        return TeacherSample(
            prompt=prompt,
            teacher=self.name,
            final_image=final_image,
            partial_images=partials,
            metadata=meta,
        )
