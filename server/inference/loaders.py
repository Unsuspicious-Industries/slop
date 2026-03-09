from typing import Optional
import torch

def _default_dtype():
    """Choose the tensor dtype used when loading a model."""
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_diffusion_model(
    model_id: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    sequential_offload: bool = False,
):
    """Load a diffusion pipeline."""
    dtype = dtype or _default_dtype()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    from diffusers import StableDiffusionPipeline, FluxPipeline

    pipe = None
    if "flux" in model_id.lower():
        try:
            if dtype == torch.float16 and torch.cuda.is_bf16_supported():
                  dtype = torch.bfloat16
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                if vram_gb < 16:
                     raise RuntimeError(f"Insufficient VRAM for FLUX: {vram_gb:.1f}GB detected. ~24GB recommended.")
            
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
            
        except Exception as exc:
            raise RuntimeError(f"FluxPipeline unavailable or failed to load {model_id}: {exc}")
    else:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        except Exception as exc:
             raise RuntimeError(f"StableDiffusionPipeline failed to load {model_id}: {exc}")

    if "flux" not in model_id.lower():
        if sequential_offload and torch.cuda.is_available():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

    return pipe
