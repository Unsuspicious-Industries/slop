from typing import Any, Dict, Tuple
from .base import BaseTrajectoryHook

class FluxTrajectoryHook(BaseTrajectoryHook):
    def _get_target_module(self) -> Any:
        return getattr(self.pipe, "transformer")

    def _extract_trajectory_data(self, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        # Flux Transformer signature usually:
        # forward(hidden_states, encoder_hidden_states=..., pooled_projections=..., timestep=..., img_ids=..., txt_ids=..., guidance=...)
        
        # 1. Hidden States (Latents)
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
            
        # 2. Encoder Hidden States (Prompt Embedding)
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(args) > 1:
            encoder_hidden_states = args[1]
            
        # 3. Timestep
        timestep = kwargs.get("timestep")
        # In some Flux versions, timestep might be at a different positional index, but usually it's passed as kwarg by the pipeline.
        # If it's positional, it's typically later (index 3 or 4), but let's rely on kwargs mostly.
        
        # 4. Guidance
        guidance = kwargs.get("guidance")

        # Extraction logic with safety checks
        ts_val = None
        if timestep is not None:
             if hasattr(timestep, 'item') and timestep.numel() == 1:
                 ts_val = timestep.item()
             else:
                 # It might be a tensor of shape (batch,) -> take first
                 ts_val = timestep[0].item() if hasattr(timestep, 'item') else timestep

        guidance_val = None
        if guidance is not None:
             if hasattr(guidance, 'item') and guidance.numel() == 1:
                 guidance_val = guidance.item()
             elif hasattr(guidance, 'detach'):
                 guidance_val = guidance.detach().cpu().numpy()

        return {
            "timestep": ts_val,
            "guidance": guidance_val,
            "latent": hidden_states.detach().float().cpu().numpy() if hidden_states is not None else None,
            "prompt_embedding": encoder_hidden_states.detach().float().cpu().numpy() if encoder_hidden_states is not None else None,
        }

    def _extract_output_data(self, output: Any) -> Dict[str, Any]:
        # Flux transformer output is usually Transformer2DModelOutput with .sample
        if hasattr(output, "sample"):
            noise_pred = output.sample
        elif isinstance(output, tuple) and len(output) > 0:
            noise_pred = output[0]
        else:
            noise_pred = output
            
        return {
            "noise_pred": noise_pred.detach().float().cpu().numpy() if noise_pred is not None else None
        }
