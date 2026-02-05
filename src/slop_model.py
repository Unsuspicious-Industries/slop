from typing import Optional, Any, Dict, List, Union
import torch
from PIL import Image

from .diffusion.loaders import load_diffusion_model
from .diffusion.sd_hook import SDTrajectoryHook
from .diffusion.flux_hook import FluxTrajectoryHook
from .encoders.loader import load_encoder

class SlopModel:
    """
    Unified interface for interacting with Diffusion Models and VLMs/Encoders.
    """
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self.model_type = None
        self._backend = None
        self._hook = None

    @classmethod
    def load(cls, model_id: str, device: Optional[str] = None, **kwargs) -> 'SlopModel':
        """
        Load a model (diffusion or encoder) and return the wrapper.
        """
        instance = cls(model_id, device)
        
        # Heuristic to detect model type
        # In a real scenario, this might need to be more robust or explicit
        if any(x in model_id.lower() for x in ["clip", "dino", "vit"]):
            instance.model_type = "encoder"
            instance._backend = load_encoder(model_id, device=device)
        else:
            instance.model_type = "diffusion"
            instance._backend = load_diffusion_model(model_id, device=device, **kwargs)
            
            # Initialize hook based on model type
            if "flux" in model_id.lower():
                instance._hook = FluxTrajectoryHook(instance._backend)
            else:
                instance._hook = SDTrajectoryHook(instance._backend)
                
        return instance

    def diffuse(
        self, 
        prompt: str, 
        num_steps: int = 50, 
        start_image: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run diffusion process and capture internal states.
        """
        if self.model_type != "diffusion":
            raise ValueError(f"Model {self.model_id} is not a diffusion model.")
            
        # Ensure hook is restored/fresh if needed, though generate_with_tracking handles re-hooking
        
        # Prepare arguments
        # Note: Current hooks might not strictly support start_image yet, 
        # but we pass it through. If the underlying pipeline supports 'image', it might work
        # if we adjust the hook.
        
        # We need to update the hooks to accept **kwargs to pass 'image' or other params
        # For now, we assume the hook's generate_with_tracking can be updated or we call it specially.
        
        image, trajectory = self._hook.generate_with_tracking(
            prompt=prompt, 
            num_steps=num_steps, 
            image=start_image, 
            **kwargs
        )
        
        return {
            "image": image,
            "trajectory": trajectory,
            # We could add more analysis here later
        }

    def encode(self, data: Union[str, Image.Image, List[str], List[Image.Image]]) -> Any:
        """
        Encode text or image using the loaded encoder.
        """
        if self.model_type != "encoder":
            raise ValueError(f"Model {self.model_id} is not an encoder.")
            
        if hasattr(self._backend, "encode_image") and (isinstance(data, Image.Image) or (isinstance(data, list) and isinstance(data[0], Image.Image))):
            return self._backend.encode_image(data)
        elif hasattr(self._backend, "encode_text") and (isinstance(data, str) or (isinstance(data, list) and isinstance(data[0], str))):
            return self._backend.encode_text(data)
        else:
            raise ValueError("Unsupported data type for this encoder.")
