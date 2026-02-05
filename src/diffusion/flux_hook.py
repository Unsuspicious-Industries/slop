from typing import Any, Dict, Tuple
from .base_hook import BaseTrajectoryHook

class FluxTrajectoryHook(BaseTrajectoryHook):
    def _get_target_module(self) -> Any:
        return getattr(self.pipe, "transformer")

    def _extract_trajectory_data(self, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        # signature: forward(hidden_states, encoder_hidden_states=None, timestep=None, ...)
        
        hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else (args[0] if len(args) > 0 else None)
        # Position 1 is encoder_hidden_states, Position 2 is timestep in typical transformer signature if positional
        # But Flux implementation might vary, relying on kwargs is safer if possible.
        
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(args) > 1:
            encoder_hidden_states = args[1]
            
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 2:
            timestep = args[2]

        return {
            "timestep": timestep,
            "latent": hidden_states.detach().cpu().numpy() if hidden_states is not None else None,
            "prompt_embedding": encoder_hidden_states.detach().cpu().numpy() if encoder_hidden_states is not None else None,
        }
