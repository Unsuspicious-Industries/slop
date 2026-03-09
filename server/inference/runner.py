import time
from typing import cast
from typing import Any, Optional

import numpy as np
import torch

from shared.protocol.messages import DecodeRequest, EmbedRequest, EncodeRequest, InferenceRequest, JobResult
from shared.protocol.serialization import pack_array
from server.hooks.flux import FluxTrajectoryHook
from server.hooks.sd import SDTrajectoryHook
from server.inference.loaders import load_diffusion_model
from server.inference.primitives import (
    encode_text,
    latent_batch,
    model_device,
    module_dtype,
    png_b64,
    prepare_conditioning,
    render,
    score,
)


class InferenceRunner:
    def __init__(self) -> None:
        self.current_model_id: Optional[str] = None
        self.pipe: Any = None

    def _ensure_model(self, model_id: str) -> None:
        """Load a model on first use or when the model id changes."""
        if model_id != self.current_model_id:
            self.clear()
            self.pipe = load_diffusion_model(model_id)
            self.current_model_id = model_id

    def clear(self) -> None:
        """Release loaded model state and clear CUDA cache."""
        if self.pipe is not None:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
            del self.pipe
            self.pipe = None
            self.current_model_id = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def cleanup(self, clear_model: bool = False) -> dict:
        """Proactive memory cleanup. Returns memory stats."""
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        freed_model = False
        if clear_model and self.pipe is not None:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
            del self.pipe
            self.pipe = None
            self.current_model_id = None
            freed_model = True
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        stats: dict = {"freed_model": freed_model}
        if torch.cuda.is_available():
            stats["gpu_memory_mb"] = int(torch.cuda.memory_allocated() // 1024 // 1024)
            stats["gpu_memory_reserved_mb"] = int(torch.cuda.memory_reserved() // 1024 // 1024)
        return stats

    def _hook(self, model_id: str) -> Any:
        """Select the capture hook for the loaded pipeline."""
        if "flux" in model_id.lower():
            return FluxTrajectoryHook(self.pipe)
        return SDTrajectoryHook(self.pipe)

    def _conditioning(self, req: InferenceRequest, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the conditioning tensors used by probe code paths."""
        prompt_embeds, negative_prompt_embeds, _ = prepare_conditioning(self.pipe, req, device, dtype)
        return prompt_embeds, negative_prompt_embeds

    def encode_prompt(self, req: EncodeRequest) -> JobResult:
        """Encode text inputs with the model text encoder."""
        start = time.time()
        self._ensure_model(req.model_id)
        device = model_device(self.pipe)
        dtype = module_dtype(self.pipe.text_encoder)
        prompt_embeds = encode_text(self.pipe, req.inputs, device, dtype).cpu().float().numpy()
        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            elapsed_s=time.time() - start,
            payload={"n_texts": len(req.inputs)},
            arrays={"prompt_embeds": pack_array(prompt_embeds, compress=True, half=False)},
        )

    def embed_prompt(self, req: EmbedRequest) -> JobResult:
        """Encode prompt and negative prompt into embeddings for later use."""
        start = time.time()
        self._ensure_model(req.model_id)
        device = model_device(self.pipe)
        dtype = module_dtype(self.pipe.text_encoder)

        arrays = {}
        if req.return_prompt_embeds:
            prompt_embeds = encode_text(self.pipe, req.prompt, device, dtype)
            arrays["prompt_embeds"] = pack_array(prompt_embeds.cpu().float().numpy(), compress=True, half=False)
        if req.return_negative_prompt_embeds:
            neg_embeds = encode_text(self.pipe, req.negative_prompt, device, dtype)
            arrays["negative_prompt_embeds"] = pack_array(neg_embeds.cpu().float().numpy(), compress=True, half=False)

        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            elapsed_s=time.time() - start,
            payload={"prompt": req.prompt, "negative_prompt": req.negative_prompt},
            arrays=arrays,
        )

    def decode_latents(self, req: DecodeRequest) -> JobResult:
        """Decode a batch of latents into images."""
        start = time.time()
        self._ensure_model(req.model_id)
        device = model_device(self.pipe)
        images = [png_b64(image) for image in render(self.pipe, latent_batch(req.latents), device)]
        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            elapsed_s=time.time() - start,
            payload={"image": images[0] if images else "", "images": images, "n": len(images)},
        )

    def _run_probe(self, req: InferenceRequest) -> JobResult:
        """Run one batched probe. When requested, also return a base-conditioned probe."""
        if req.latent_override is None:
            raise ValueError("score_only requires latent_override")

        start = time.time()
        self._ensure_model(req.model_id)
        device = model_device(self.pipe)
        dtype = module_dtype(self.pipe.unet)
        points = torch.from_numpy(latent_batch(req.latent_override)).to(device=device, dtype=dtype)
        prompt_embeds, negative_prompt_embeds = self._conditioning(req, device, dtype)

        self.pipe.scheduler.set_timesteps(req.num_steps, device=device)
        timestep = torch.tensor(req.probe_timestep, device=device, dtype=torch.long)
        forces = score(self.pipe, points, timestep, prompt_embeds, negative_prompt_embeds, req.guidance_scale)

        arrays = {
            "latents": pack_array(points.detach().cpu().float().numpy(), compress=req.compress_latents, half=True),
            "noise_preds": pack_array(forces.detach().cpu().float().numpy(), compress=req.compress_latents, half=True),
            "timesteps": pack_array(np.array([req.probe_timestep], dtype=np.float32), compress=False, half=False),
        }

        if req.delta_probe:
            base_req = cast(
                InferenceRequest,
                InferenceRequest(
                **{
                    **req.to_dict(),
                    "prompt": req.base_prompt,
                    "prompt_embeds_override": req.base_prompt_embeds_override,
                }
                )
            )
            base_prompt_embeds, base_negative_prompt_embeds = self._conditioning(base_req, device, dtype)
            base_forces = score(self.pipe, points, timestep, base_prompt_embeds, base_negative_prompt_embeds, req.guidance_scale)
            arrays["base_noise_preds"] = pack_array(base_forces.detach().cpu().float().numpy(), compress=req.compress_latents, half=True)
            arrays["delta_noise_preds"] = pack_array((forces - base_forces).detach().cpu().float().numpy(), compress=req.compress_latents, half=True)

        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            prompt=req.prompt,
            elapsed_s=time.time() - start,
            payload={"probe_timestep": req.probe_timestep},
            arrays=arrays,
        )

    def _run_delta_sample(self, req: InferenceRequest) -> JobResult:
        """Run sampling where the drift is the difference between two score functions."""
        start = time.time()
        self._ensure_model(req.model_id)
        device = model_device(self.pipe)
        dtype = module_dtype(self.pipe.unet)

        # 1. Prepare conditioning for both fields
        prompt_embeds, negative_prompt_embeds, _ = prepare_conditioning(self.pipe, req, device, dtype)
        
        base_req = InferenceRequest(**{**req.to_dict(), "prompt": req.base_prompt, "prompt_embeds_override": req.base_prompt_embeds_override})
        base_embeds, base_neg_embeds, _ = prepare_conditioning(self.pipe, base_req, device, dtype)

        # 2. Initialize latents
        batch_size = req.batch_size
        latents_shape = (batch_size, 4, req.height // 8, req.width // 8)
        if req.latent_override is not None:
            latents = torch.from_numpy(latent_batch(req.latent_override)).to(device=device, dtype=dtype)
        else:
            generator = torch.Generator(device=device).manual_seed(req.seed) if req.seed != -1 else None
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

        # 3. Sampling loop
        self.pipe.scheduler.set_timesteps(req.num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        records = []
        for i, t in enumerate(timesteps):
            # Compute both score functions
            f_target = score(self.pipe, latents, t, prompt_embeds, negative_prompt_embeds, req.guidance_scale)
            f_base = score(self.pipe, latents, t, base_embeds, base_neg_embeds, req.guidance_scale)
            
            # Differential force
            f_diff = f_target - f_base
            
            # Record state
            if req.capture_latents or req.capture_noise_pred:
                records.append({
                    "latent": latents.detach().cpu().float().numpy(),
                    "noise_pred": f_diff.detach().cpu().float().numpy(),
                    "timestep": t.item(),
                })

            # Step
            latents = self.pipe.scheduler.step(f_diff, t, latents).prev_sample

        # 4. Final state
        records.append({
            "latent": latents.detach().cpu().float().numpy(),
            "noise_pred": np.zeros_like(records[-1]["noise_pred"]),
            "timestep": 0,
        })

        # 5. Build arrays
        arrays = {}
        if req.capture_latents:
            arrays["latents"] = pack_array(np.stack([r["latent"] for r in records]), compress=req.compress_latents, half=True)
        if req.capture_noise_pred:
            arrays["noise_preds"] = pack_array(np.stack([r["noise_pred"] for r in records]), compress=req.compress_latents, half=True)
        
        # 6. Render if requested
        encoded_images = []
        if req.decode_latents:
            images = render(self.pipe, latents, device)
            encoded_images = [png_b64(img) for img in images]

        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            elapsed_s=time.time() - start,
            payload={
                "images": encoded_images,
                "n": len(encoded_images),
                "steps_completed": len(timesteps),
            },
            arrays=arrays,
        )

    def run(self, req: InferenceRequest) -> JobResult:
        """Dispatch either a probe or a sampled generation."""
        if req.score_only:
            return self._run_probe(req)
        
        if req.delta_sample:
            return self._run_delta_sample(req)

        start = time.time()
        self._ensure_model(req.model_id)
        device = model_device(self.pipe)
        dtype = module_dtype(self.pipe.unet)
        hook = self._hook(req.model_id)

        generator = None
        if req.seed != -1:
            generator = torch.Generator(device=device).manual_seed(req.seed)

        prompt_embeds, negative_prompt_embeds, use_override = prepare_conditioning(self.pipe, req, device, dtype)
        kwargs: dict[str, Any] = {
            "guidance_scale": req.guidance_scale,
            "height": req.height,
            "width": req.width,
            "num_images_per_prompt": req.batch_size,
            "generator": generator,
        }

        if req.latent_override is not None:
            kwargs["latents"] = torch.from_numpy(latent_batch(req.latent_override)).to(device=device, dtype=dtype)

        if use_override:
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        elif "flux" not in req.model_id.lower() and req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt

        images, records = hook.generate_with_tracking(
            prompt=None if use_override else req.prompt,
            num_steps=req.num_steps,
            **kwargs,
        )

        arrays: dict[str, Any] = {}
        if records and req.capture_latents:
            arrays["latents"] = pack_array(
                np.stack([record["latent"] for record in records]),
                compress=req.compress_latents,
                half=True,
            )
        if records and req.capture_noise_pred:
            arrays["noise_preds"] = pack_array(
                np.stack([record["noise_pred"] for record in records]),
                compress=req.compress_latents,
                half=True,
            )
        if records and req.capture_prompt_embeds:
            arrays["prompt_embeds"] = pack_array(
                records[0]["prompt_embedding"],
                compress=req.compress_latents,
                half=True,
            )
        if records and req.capture_timesteps:
            arrays["timesteps"] = pack_array(
                np.array([record["timestep"] for record in records], dtype=np.float32),
                compress=False,
                half=False,
            )

        encoded_images = []
        if req.decode_latents and images:
            encoded_images = [png_b64(image) for image in images]
        return JobResult(
            job_id=req.job_id,
            request_kind=req.kind,
            model_id=req.model_id,
            prompt=req.prompt,
            elapsed_s=time.time() - start,
            payload={
                "image": encoded_images[0] if encoded_images else "",
                "images": encoded_images,
                "n": len(encoded_images),
                "width": req.width,
                "height": req.height,
                "steps_completed": len(records),
            },
            arrays=arrays,
        )
