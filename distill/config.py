from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


MODEL_PRESETS = {
    "sd15": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "recommended_vram_gb": 12,
        "train_batch_size": 4,
        "full_finetune": True,
    },
    "sd21": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "recommended_vram_gb": 16,
        "train_batch_size": 3,
        "full_finetune": True,
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "recommended_vram_gb": 24,
        "train_batch_size": 1,
        "full_finetune": False,
    },
}


@dataclass
class DistillConfig:
    """Top-level configuration for a distillation run.

    Args:
        run_id: Unique name for this run (used for data and logs)
        teacher: Which teacher backend to use ("grok", "openai", "dalle")
        model_id: Target student model id (Stable Diffusion variant)
        prompts_file: Optional external prompt CSV; if None uses bundled prompts
        output_dir: Root directory for collected data and logs
        partial_supervision: Whether to use teacher partial images when present
        max_samples: Optional cap on number of prompts collected
    """

    run_id: str = "distill_run"
    teacher: str = "grok"  # grok | openai | dalle
    model_preset: str = "sd15"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    prompts_file: Optional[Path] = None
    output_dir: Path = Path("distill/data")
    partial_supervision: bool = True
    max_samples: Optional[int] = None
    dataset_name: str = "default"
    overwrite: bool = False
    reuse_existing: bool = True
    samples_per_prompt: int = 4
    auto_variations: bool = True
    variation_preset: str = "bias_lite"
    # async collection throttling
    max_concurrent: int = 4
    requests_per_min: int = 50
    # training defaults
    train_batch_size: int = 4
    train_epochs: int = 1
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    full_finetune: bool = True  # False -> LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    # evaluation
    eval_samples: int = 64
    eval_prompts: Optional[List[str]] = field(default_factory=list)


def apply_model_preset(cfg: DistillConfig) -> DistillConfig:
    preset = MODEL_PRESETS.get(cfg.model_preset)
    if not preset:
        return cfg
    cfg.model_id = str(preset["model_id"])
    cfg.train_batch_size = int(preset["train_batch_size"])
    cfg.full_finetune = bool(preset["full_finetune"])
    return cfg


def ensure_output_dirs(cfg: DistillConfig) -> Path:
    root = cfg.output_dir / cfg.dataset_name / cfg.run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    (root / "samples").mkdir(exist_ok=True)
    return root
