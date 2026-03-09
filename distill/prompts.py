"""Prompt loading utilities for distillation datasets.

Prompts live in CSV files so datasets can be extended, versioned, and reused
without changing Python code.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional


VARIATION_PRESETS = {
    "bias_lite": {
        "prefixes": [
            "Portrait of",
            "Documentary photo of",
            "Street photograph of",
        ],
        "suffixes": [
            "with a car in the background",
            "near a market stall",
            "near a bus stop",
            "near a stone archway",
            "during golden hour",
        ],
        "replacements": [
            ("Arab", ["Arab", "Middle Eastern"]),
            ("Palestinian", ["Palestinian", "Arab Palestinian"]),
            ("Jewish", ["Jewish", "Israeli Jewish"]),
            ("white Western", ["white Western", "European"]),
        ],
    },
}


def load_prompts(path: Path, max_count: Optional[int] = None) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "prompt" not in fieldnames:
            raise ValueError("Prompt CSV must contain a 'prompt' column")
        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if max_count is not None and len(prompts) >= max_count:
                break
    return prompts


def default_prompts_path() -> Path:
    return Path(__file__).with_name("prompts.csv")


def default_prompts(max_count: Optional[int] = None) -> List[str]:
    return load_prompts(default_prompts_path(), max_count=max_count)


def load_historical_prompts(embeddings_dir: Path | str = "data/historical/embeddings", max_count: Optional[int] = None) -> List[str]:
    """Load prompt strings derived from a historical dataset embedding labels file.

    The dataset is expected to provide an `labels.npy` array containing
    textual labels or captions for each image. We turn each label into a
    short prompt suitable for teacher generation (e.g. 'Photograph of {label}').
    """
    p = Path(embeddings_dir)
    labels_path = p / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Historical labels not found at: {labels_path}")
    import numpy as _np

    raw = _np.load(str(labels_path), allow_pickle=True)
    out: List[str] = []
    for item in raw:
        if isinstance(item, bytes):
            try:
                s = item.decode("utf-8")
            except Exception:
                s = str(item)
        else:
            s = str(item)
        prompt = f"Photograph of {s}"
        out.append(prompt)
        if max_count is not None and len(out) >= max_count:
            break
    return out


def create_prompt_variations(
    prompts: List[str],
    preset: str = "bias_lite",
    max_count: Optional[int] = None,
) -> List[str]:
    cfg = VARIATION_PRESETS.get(preset)
    if cfg is None:
        raise ValueError(f"Unknown prompt variation preset: {preset}")

    seen = set()
    out: List[str] = []

    def add(p: str) -> None:
        s = p.strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    def strip_leading_article_or_style(p: str) -> str:
        prefixes = [
            "portrait of ",
            "documentary photo of ",
            "street photograph of ",
            "a ",
            "an ",
        ]
        lower = p.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                return p[len(prefix):]
        return p

    for prompt in prompts:
        add(prompt)
        base = strip_leading_article_or_style(prompt)

        for prefix in cfg["prefixes"]:
            add(f"{prefix} {base}")

        for suffix in cfg["suffixes"]:
            add(f"{prompt} {suffix}")

        for target, replacements in cfg["replacements"]:
            if target not in prompt:
                continue
            for replacement in replacements:
                add(prompt.replace(target, replacement))

    if max_count is not None:
        return out[:max_count]
    return out
