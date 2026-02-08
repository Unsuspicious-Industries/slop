from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable
import torch

class BaseTrajectoryHook(ABC):
    """Base class for hooking into diffusion models to capture trajectory data.
    
    Captures:
    - Input latents (position in trajectory)
    - Time embeddings/values
    - Conditioning (prompt embeddings)
    - Model output (noise prediction / velocity)
    """
    
    def __init__(self, pipe: Any):
        self.pipe = pipe
        self.trajectories: List[Dict[str, Any]] = []
        self.original_forward: Optional[Callable[..., Any]] = None

    @abstractmethod
    def _get_target_module(self) -> Any:
        """Return the module (unet/transformer) to be hooked."""
        pass

    @abstractmethod
    def _extract_trajectory_data(self, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        """Extract relevant data (latent, embedding, timestep) from forward arguments."""
        pass
    
    @abstractmethod
    def _extract_output_data(self, output: Any) -> Dict[str, Any]:
        """Extract relevant data (noise pred) from forward output."""
        pass

    def _hook_implementation(self, *args, **kwargs):
        """The generic hook that captures data and calls original forward."""
        # 1. Capture Inputs
        input_data = self._extract_trajectory_data(args, kwargs)
        
        if self.original_forward is None:
             raise RuntimeError("Original forward is missing during execution.")
             
        # 2. Run Forward
        output = self.original_forward(*args, **kwargs)
        
        # 3. Capture Outputs
        output_data = self._extract_output_data(output)
        
        # 4. Store
        # We merge them. Note: We keep tensors on CPU/numpy here to avoid VRAM leaks.
        # The subclass implementations of extract_* should handle detach().cpu().numpy()
        self.trajectories.append({**input_data, **output_data})
        
        return output

    def hook_model(self) -> None:
        if self.original_forward is not None:
            return

        target_module = self._get_target_module()
        self.original_forward = target_module.forward
        target_module.forward = self._hook_implementation

    def restore(self) -> None:
        if self.original_forward is not None:
            target_module = self._get_target_module()
            target_module.forward = self.original_forward
            self.original_forward = None

    def generate_with_tracking(
        self, 
        prompt: Optional[str] = None, 
        num_steps: int = 50, 
        **kwargs
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Run generation and return result + trajectories.
        
        Args:
            prompt: Text prompt. If None, prompt_embeds must be in kwargs
                    (embedding-override mode — bypasses text encoder).
            num_steps: Inference steps
            **kwargs: Passed to pipe (e.g. guidance_scale, prompt_embeds, etc.)
        """
        self.trajectories = []
        self.hook_model()
        try:
            if prompt is not None:
                output = self.pipe(prompt, num_inference_steps=num_steps, **kwargs)
            else:
                # Embedding-override mode: prompt_embeds should be in kwargs
                output = self.pipe(num_inference_steps=num_steps, **kwargs)
            # Handle standard diffusers output
            image = output.images[0] if hasattr(output, 'images') else output
        finally:
            self.restore()
            
        return image, self.trajectories
