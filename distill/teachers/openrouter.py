from __future__ import annotations

import base64
import os
from typing import Any

import aiohttp

from distill.env import load_dotenv_if_present
from .base import TeacherClient, TeacherSample


class OpenRouterTeacher(TeacherClient):
    """Teacher backend using OpenRouter image generation API.

    Uses the /api/v1/chat/completions endpoint with modalities=["image", "text"].
    """

    name = "openrouter"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        load_dotenv_if_present()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouterTeacher")
        self.base_url = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-image")

    async def generate(self, prompt: str) -> TeacherSample:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "DistillCollection"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"OpenRouter error [{resp.status}]: {text[:500]}")
                data = await resp.json()

        # Parse response - images come in a separate "images" field
        final_image = None
        message = data.get("choices", [{}])[0].get("message", {})
        
        # Check for images field
        images = message.get("images", [])
        if images:
            for img in images:
                if isinstance(img, dict):
                    img_url = img.get("image_url", {}).get("url", "")
                    if img_url:
                        # Handle data URLs
                        if img_url.startswith("data:image"):
                            b64 = img_url.split(",", 1)[1]
                            final_image = base64.b64decode(b64)
                            break
                        elif img_url.startswith("http"):
                            # Download from URL
                            async with aiohttp.ClientSession() as img_session:
                                async with img_session.get(img_url) as img_resp:
                                    final_image = await img_resp.read()
                            break
        
        if final_image is None:
            raise RuntimeError(f"No image in response: {message}")

        meta: dict[str, Any] = {
            "model": self.model,
        }

        return TeacherSample(
            prompt=prompt,
            teacher=self.name,
            final_image=final_image,
            partial_images=[],
            metadata=meta,
        )
