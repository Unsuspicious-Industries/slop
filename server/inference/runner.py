import io
import time
import torch
import numpy as np
import base64
from PIL import Image
from typing import Optional, Any

from shared.protocol.messages import InferenceRequest, JobResult
from shared.protocol.serialization import pack_array
from server.inference.loaders import load_diffusion_model
from server.hooks.sd import SDTrajectoryHook
from server.hooks.flux import FluxTrajectoryHook

class InferenceRunner:
    """Stateful runner for processing inference requests."""
    
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.pipe: Any = None
        
    def run(self, req: InferenceRequest) -> JobResult:
        start_time = time.time()
        
        # 1. Load Model (Lazy / Switch)
        if req.model_id != self.current_model_id:
            # Free memory if switching
            if self.pipe is not None:
                del self.pipe
                torch.cuda.empty_cache()
            
            # Load new model
            self.pipe = load_diffusion_model(req.model_id)
            self.current_model_id = req.model_id

        # 2. Setup Hook
        if "flux" in req.model_id.lower():
            hook = FluxTrajectoryHook(self.pipe)
        else:
            hook = SDTrajectoryHook(self.pipe)

        # 3. Run Inference
        # Convert seed to generator
        generator = None
        if req.seed != -1:
            generator = torch.Generator(device=self.pipe.device).manual_seed(req.seed)

        # Prepare kwargs
        kwargs = {
            "guidance_scale": req.guidance_scale,
            "height": req.height,
            "width": req.width,
            "generator": generator,
        }
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt

        # Run!
        image, trajectories = hook.generate_with_tracking(
            prompt=req.prompt,
            num_steps=req.num_steps,
            **kwargs
        )
        
        # 4. Process Data
        arrays = {}
        
        # Stack latents
        if req.capture_latents and trajectories:
             # trajectories is list of dicts. Stack 'latent' keys.
             latents = np.stack([t['latent'] for t in trajectories])
             arrays['latents'] = pack_array(latents, compress=req.compress_latents, half=True)
             
        if req.capture_noise_pred and trajectories:
             noise = np.stack([t['noise_pred'] for t in trajectories])
             arrays['noise_preds'] = pack_array(noise, compress=req.compress_latents, half=True)

        if req.capture_prompt_embeds and trajectories:
             # Assume constant prompt embedding for now
             embed = trajectories[0]['prompt_embedding']
             arrays['prompt_embedding'] = pack_array(embed, compress=req.compress_latents, half=True)
        
        if req.capture_timesteps and trajectories:
             # Extract timestep values from each trajectory entry
             timesteps = np.array([t['timestep'] for t in trajectories], dtype=np.int32)
             arrays['timesteps'] = pack_array(timesteps, compress=False, half=False)

        # Convert image to bytes then base64
        img_b64 = ""
        if isinstance(image, Image.Image):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
            
        elapsed = time.time() - start_time
            
        # 5. Build Result
        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            prompt=req.prompt,
            elapsed_s=elapsed,
            payload={
                "image": img_b64,
                "width": req.width,
                "height": req.height,
                "steps_completed": len(trajectories)
            },
            arrays=arrays
        )
