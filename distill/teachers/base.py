from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TeacherSample:
    prompt: str
    teacher: str
    final_image: bytes
    partial_images: List[bytes] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TeacherClient(abc.ABC):
    """Abstract interface for teacher backends."""

    name: str = "teacher"

    @abc.abstractmethod
    async def generate(self, prompt: str) -> TeacherSample:
        raise NotImplementedError
