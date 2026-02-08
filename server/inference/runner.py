import io
import time
import torch
import numpy as np
import base64
from PIL import Image
from typing import Optional, Any

from shared.protocol.messages import InferenceRequest, EncodeRequest, JobResult
from shared.protocol.serialization import pack_array, unpack_array
from server.inference.loaders import load_diffusion_model
from server.hooks.sd import SDTrajectoryHook
from server.hooks.flux import FluxTrajectoryHook

class InferenceRunner:
    """Stateful runner for processing inference requests."""
    
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.pipe: Any = None
        
    def _ensure_model(self, model_id: str):
        """Ensure the requested model is loaded, switching if needed."""
        if model_id != self.current_model_id:
            if self.pipe is not None:
                del self.pipe
                torch.cuda.empty_cache()
            self.pipe = load_diffusion_model(model_id)
            self.current_model_id = model_id

    def _get_device(self):
        """Get the device of the loaded pipeline."""
        if hasattr(self.pipe, '_execution_device'):
            return self.pipe._execution_device
        if hasattr(self.pipe, 'device'):
            return self.pipe.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode_prompt(self, req: EncodeRequest) -> JobResult:
        """Encode text prompts through the diffusion model's text encoder.

        Returns the prompt embeddings that would be fed to the UNet,
        without running any diffusion steps. Used for computing identity
        vectors and embedding-override generation.
        """
        start_time = time.time()
        self._ensure_model(req.model_id)

        device = self._get_device()

        embeddings = []
        for text in req.inputs:
            text_inputs = self.pipe.tokenizer(
                text,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                encoder_output = self.pipe.text_encoder(
                    text_inputs.input_ids.to(device)
                )
                embed = encoder_output[0]  # last_hidden_state: (1, seq_len, hidden_dim)
            embeddings.append(embed.cpu().float().numpy())

        # Stack into (n_texts, seq_len, hidden_dim)
        stacked = np.concatenate(embeddings, axis=0)

        elapsed = time.time() - start_time

        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            elapsed_s=elapsed,
            payload={"n_texts": len(req.inputs)},
            arrays={"prompt_embeds": pack_array(stacked, compress=True, half=False)},
        )

    def run(self, req: InferenceRequest) -> JobResult:
        start_time = time.time()
        
        # 1. Load Model (Lazy / Switch)
        self._ensure_model(req.model_id)

        # 2. Setup Hook
        if "flux" in req.model_id.lower():
            hook = FluxTrajectoryHook(self.pipe)
        else:
            hook = SDTrajectoryHook(self.pipe)

        # 3. Run Inference
        # Convert seed to generator
        device = self._get_device()
        generator = None
        if req.seed != -1:
            generator = torch.Generator(device=device).manual_seed(req.seed)

        # Prepare kwargs
        kwargs = {
            "guidance_scale": req.guidance_scale,
            "height": req.height,
            "width": req.width,
            "generator": generator,
        }
        
        # FLUX does not support negative_prompt natively in the standard pipeline
        if "flux" not in req.model_id.lower() and req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt

        # 3b. Handle prompt embedding overrides
        # When set, we bypass the text encoder entirely and inject embeddings directly.
        # This is used for identity-vector experiments where the ONLY difference
        # between categories is a computed direction vector in embedding space.
        use_embed_override = req.prompt_embeds_override is not None

        if use_embed_override:
            pe_np = unpack_array(req.prompt_embeds_override)
            pe_tensor = torch.from_numpy(pe_np).to(device=device)
            # Let the pipeline handle dtype conversion internally
            kwargs["prompt_embeds"] = pe_tensor

            if req.negative_prompt_embeds_override is not None:
                npe_np = unpack_array(req.negative_prompt_embeds_override)
                npe_tensor = torch.from_numpy(npe_np).to(device=device)
                kwargs["negative_prompt_embeds"] = npe_tensor
            else:
                # Default: encode empty string for CFG unconditional
                empty_inputs = self.pipe.tokenizer(
                    "",
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    neg_embed = self.pipe.text_encoder(
                        empty_inputs.input_ids.to(device)
                    )[0]
                kwargs["negative_prompt_embeds"] = neg_embed

            # Don't pass negative_prompt when using embeds
            kwargs.pop("negative_prompt", None)

        # Run!
        image, trajectories = hook.generate_with_tracking(
            prompt=req.prompt if not use_embed_override else None,
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
             # Use float32 to support both SD (int-like floats) and FLUX (0.0-1.0 floats)
             timesteps = np.array([t['timestep'] for t in trajectories], dtype=np.float32)
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
