from typing import List, Union, Optional, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _to_numpy(tensor: torch.Tensor) -> NDArray[np.float32]:
    return cast(NDArray[np.float32], tensor.detach().cpu().numpy())

class CLIPEncoder:
    """CLIP image/text encoder with dimension hints.
    
    Output embeddings: (batch, 768) for large patch14.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> NDArray[np.float32]:
        """Encode image(s) to embeddings.
        
        Args:
            image: Single PIL Image or list of images
            
        Returns:
            NDArray of shape (batch, 768) or (768,) if single image
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_image_features(**inputs)
        if isinstance(outputs, torch.Tensor):
            return _to_numpy(outputs)
        if hasattr(outputs, "image_embeds") and isinstance(outputs.image_embeds, torch.Tensor):
            return _to_numpy(outputs.image_embeds)
        if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
            return _to_numpy(outputs.pooler_output)
        if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
            return _to_numpy(outputs.last_hidden_state[:, 0])
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return _to_numpy(outputs[0])
        raise TypeError("Unexpected output type from CLIP image encoder")

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> NDArray[np.float32]:
        """Encode text(s) to embeddings.
        
        Args:
            texts: Single string or list of strings
            
        Returns:
            NDArray of shape (batch, 768) or (768,) if single text
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        if isinstance(outputs, torch.Tensor):
            return _to_numpy(outputs)
        if hasattr(outputs, "text_embeds") and isinstance(outputs.text_embeds, torch.Tensor):
            return _to_numpy(outputs.text_embeds)
        if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
            return _to_numpy(outputs.pooler_output)
        if hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
            return _to_numpy(outputs.last_hidden_state[:, 0])
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return _to_numpy(outputs[0])
        raise TypeError("Unexpected output type from CLIP text encoder")
