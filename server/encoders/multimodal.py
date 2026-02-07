from typing import Dict, List, Union, Optional, cast, Any

import numpy as np
from PIL import Image

from .clip_encoder import CLIPEncoder
from .dinov2_encoder import DINOv2Encoder


class EmbeddingExtractor:
    def __init__(self, models: Optional[List[str]] = None, strategy: str = "concat"):
        self.models = models or ["clip", "dinov2"]
        self.strategy = strategy
        self.encoders: Dict[str, Any] = {}
        if "clip" in self.models:
            self.encoders["clip"] = CLIPEncoder()
        if "dinov2" in self.models:
            self.encoders["dinov2"] = DINOv2Encoder()

    def encode_image(self, image: Image.Image, strategy: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        strat = strategy or self.strategy
        embeddings: Dict[str, np.ndarray] = {}
        for name, encoder in self.encoders.items():
            emb = np.asarray(encoder.encode_image(image))
            embeddings[name] = emb.squeeze()

        if strat == "concat":
            return cast(np.ndarray, np.concatenate(list(embeddings.values()), axis=-1))
        if strat == "average":
            return cast(np.ndarray, np.mean(list(embeddings.values()), axis=0))
        return embeddings
    
    def encode(self, image: Image.Image, strategy: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Alias for encode_image for backward compatibility."""
        return self.encode_image(image, strategy)

    def encode_text(self, text: Union[str, List[str]], strategy: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        strat = strategy or self.strategy
        embeddings: Dict[str, np.ndarray] = {}
        for name, encoder in self.encoders.items():
            if hasattr(encoder, "encode_text"):
                emb = np.asarray(encoder.encode_text(text))
                embeddings[name] = emb.squeeze()
        if strat == "concat":
            return cast(np.ndarray, np.concatenate(list(embeddings.values()), axis=-1))
        if strat == "average":
            return cast(np.ndarray, np.mean(list(embeddings.values()), axis=0))
        return embeddings
