"""Distillation utilities for approximating closed-source image generators.

This package contains teacher API adapters, data collection, training, and
evaluation code to produce a distilled Stable Diffusion student that can be
probed with the existing SLOP analysis pipeline.
"""

__all__ = [
    "config",
    "prompts",
    "teachers",
]
