from typing import Any, Dict, Tuple
from .base import BaseTrajectoryHook

class SDTrajectoryHook(BaseTrajectoryHook):
    def _get_target_module(self) -> Any:
        return getattr(self.pipe, "unet")

    def _extract_trajectory_data(self, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        # signature: forward(sample, timestep, encoder_hidden_states, ...)
        # args mapping based on position or kwargs
        
        sample = kwargs.get("sample") if "sample" in kwargs else (args[0] if len(args) > 0 else None)
        timestep = kwargs.get("timestep") if "timestep" in kwargs else (args[1] if len(args) > 1 else None)
        encoder_hidden_states = kwargs.get("encoder_hidden_states") if "encoder_hidden_states" in kwargs else (args[2] if len(args) > 2 else None)

        # Handle timestep being a tensor or int
        ts_val = timestep.item() if hasattr(timestep, 'item') else timestep

        return {
            "timestep": ts_val,
            "latent": sample.detach().cpu().numpy() if sample is not None else None,
            "prompt_embedding": encoder_hidden_states.detach().cpu().numpy() if encoder_hidden_states is not None else None,
        }
    
    def _extract_output_data(self, output: Any) -> Dict[str, Any]:
        # Diffusers UNet usually returns UNet2DConditionOutput with .sample
        if hasattr(output, "sample"):
            noise_pred = output.sample
        elif isinstance(output, tuple):
            noise_pred = output[0]
        else:
            noise_pred = output
            
        return {
            "noise_pred": noise_pred.detach().cpu().numpy() if noise_pred is not None else None
        }
