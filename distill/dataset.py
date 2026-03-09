from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class DistillSample:
    def __init__(self, prompt: str, final_path: Path, partial_paths: List[Path], metadata: Dict[str, Any]):
        self.prompt = prompt
        self.final_path = final_path
        self.partial_paths = partial_paths
        self.metadata = metadata

    def load_final(self) -> Image.Image:
        return Image.open(self.final_path).convert("RGB")

    def load_partials(self) -> List[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.partial_paths]


class DistillDataset(Dataset):
    """Loads teacher samples from a manifest produced by distill.collect."""

    def __init__(self, manifest_path: Path, use_partials: bool = True):
        self.records: List[DistillSample] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                final_path = Path(rec["final_path"])
                partials = [Path(p) for p in rec.get("partial_paths", [])]
                if not final_path.exists():
                    continue
                self.records.append(
                    DistillSample(
                        prompt=rec["prompt"],
                        final_path=final_path,
                        partial_paths=partials if use_partials else [],
                        metadata=rec.get("metadata", {}),
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        final_img = rec.load_final()
        partials = rec.load_partials()
        return {
            "prompt": rec.prompt,
            "final": final_img,
            "partials": partials,
            "metadata": rec.metadata,
        }
