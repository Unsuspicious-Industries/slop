from __future__ import annotations

import base64
import os
from typing import Any

from openai import AsyncOpenAI

from distill.env import load_dotenv_if_present
from .base import TeacherClient, TeacherSample


class DalleTeacher(TeacherClient):
    """DALL·E 3 backend (final image only)."""

    name = "dalle"

    def __init__(self, model: str = "dall-e-3", api_key: str | None = None):
        load_dotenv_if_present()
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for DalleTeacher")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate(self, prompt: str) -> TeacherSample:
        resp = await self.client.images.generate(
            model=self.model,
            prompt=prompt,
            response_format="b64_json",
        )
        data = resp.data[0]
        img = base64.b64decode(data.b64_json)
        meta: dict[str, Any] = {"model": self.model}

        return TeacherSample(
            prompt=prompt,
            teacher=self.name,
            final_image=img,
            partial_images=[],
            metadata=meta,
        )
