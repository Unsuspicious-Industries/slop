from typing import Optional

import torch

from .clip_encoder import CLIPEncoder
from .dinov2_encoder import DINOv2Encoder
from .multimodal import EmbeddingExtractor


def load_encoder(name: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lname = name.lower()
    
    # Specific aliases
    if lname in {"clip", "openai/clip"}:
         return CLIPEncoder(model_name="openai/clip-vit-large-patch14", device=device)
    if lname in {"dinov2", "facebook/dinov2"}:
         return DINOv2Encoder(model_name="facebook/dinov2-large", device=device)
    if lname == "multi":
        return EmbeddingExtractor()
        
    # General fallback based on name string
    if "clip" in lname:
        return CLIPEncoder(model_name=name, device=device)
    if "dino" in lname:
        return DINOv2Encoder(model_name=name, device=device)
        
    raise ValueError(f"Unknown or unsupported encoder: {name}")
