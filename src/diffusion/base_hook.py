from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

class BaseTrajectoryHook(ABC):
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

    def _hook_implementation(self, *args, **kwargs):
        """The generic hook that captures data and calls original forward."""
        data = self._extract_trajectory_data(args, kwargs)
        self.trajectories.append(data)
        
        if self.original_forward is None:
             raise RuntimeError("Original forward is missing during execution.")
             
        return self.original_forward(*args, **kwargs)

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
        prompt: str, 
        num_steps: int = 50, 
        **kwargs
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        self.trajectories = []
        self.hook_model()
        try:
            # We assume the pipe call signature matches standard diffusers
            # We pass kwargs to allow for 'image' (img2img), 'negative_prompt', etc.
            output = self.pipe(prompt, num_inference_steps=num_steps, **kwargs)
            image = output.images[0]
        finally:
            self.restore()
            
        return image, self.trajectories
