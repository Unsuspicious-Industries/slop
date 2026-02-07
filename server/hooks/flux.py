from typing import Any, Dict, Tuple
from .base import BaseTrajectoryHook

class FluxTrajectoryHook(BaseTrajectoryHook):
    def _get_target_module(self) -> Any:
        return getattr(self.pipe, "transformer")

    def _extract_trajectory_data(self, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        # signature: forward(hidden_states, encoder_hidden_states=None, timestep=None, ...)
        
        hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else (args[0] if len(args) > 0 else None)
        
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(args) > 1:
            encoder_hidden_states = args[1]
            
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 2:
            timestep = args[2]

        ts_val = timestep.item() if hasattr(timestep, 'item') else timestep

        return {
            "timestep": ts_val,
            "latent": hidden_states.detach().cpu().numpy() if hidden_states is not None else None,
            "prompt_embedding": encoder_hidden_states.detach().cpu().numpy() if encoder_hidden_states is not None else None,
        }

    def _extract_output_data(self, output: Any) -> Dict[str, Any]:
        # Flux transformer output
        if hasattr(output, "sample"):
            noise_pred = output.sample
        elif isinstance(output, tuple):
            noise_pred = output[0]
        else:
            noise_pred = output
            
        return {
            "noise_pred": noise_pred.detach().cpu().numpy() if noise_pred is not None else None
        }
