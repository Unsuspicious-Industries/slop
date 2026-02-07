from typing import List, Union, Optional, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2Encoder:
    """DINOv2 image encoder with dimension hints.
    
    Output embeddings: (batch, 1024) for dinov2-large.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-large", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: AutoModel = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> NDArray[np.float32]:
        """Encode image(s) to embeddings.
        
        Args:
            image: Single PIL Image or list of images
            
        Returns:
            NDArray of shape (batch, 1024) or (1024,) if single image
        """
        processor = cast(Any, self.processor)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model = cast(Any, self.model)
        outputs = model(**inputs)
        # Use CLS token or pooled output
        if hasattr(outputs, "last_hidden_state"):
            cls_token: torch.Tensor = outputs.last_hidden_state[:, 0]
        else:
            cls_token = outputs.pooler_output
        result = cls_token.cpu().numpy()
        return cast(NDArray[np.float32], result)
