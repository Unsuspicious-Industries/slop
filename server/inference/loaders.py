from typing import Optional
import torch

def _default_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_diffusion_model(
    model_id: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    sequential_offload: bool = False,
):
    """Load a diffusion pipeline (Stable Diffusion or Flux) with safe defaults.
    
    Args:
        model_id: HuggingFace model ID or path
        dtype: torch.dtype (defaults to float16 if cuda available)
        device: Device to load onto ("cuda", "cpu")
        sequential_offload: Use CPU offloading for low VRAM
        
    Returns:
        The loaded pipeline
    """
    dtype = dtype or _default_dtype()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Lazy imports to avoid hard deps at module import time
    from diffusers import StableDiffusionPipeline, FluxPipeline

    pipe = None
    if "flux" in model_id.lower():
        try:
            # FLUX generally requires bfloat16 for best performance/quality
            if dtype == torch.float16 and torch.cuda.is_bf16_supported():
                 dtype = torch.bfloat16
            
            # Hardware capability check for FLUX
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                # FLUX.1-dev is very heavy. ~24GB recommended for comfortable inference.
                # With aggressive offloading it might run on less, but we want to ensure stability.
                if vram_gb < 16:
                     raise RuntimeError(f"Insufficient VRAM for FLUX: {vram_gb:.1f}GB detected. ~24GB recommended.")
            
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
            
            # For FLUX on consumer hardware (like 3090/4090), model cpu offload is often necessary
            # or vastly more efficient than keeping it all in VRAM.
            # We default to model_cpu_offload if not specified otherwise for FLUX.
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
            
        except Exception as exc:
            raise RuntimeError(f"FluxPipeline unavailable or failed to load {model_id}: {exc}")
    else:
        # Assume Stable Diffusion architecture for others
        try:
            # Load with safety checker disabled to avoid false positives and black images during research
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        except Exception as exc:
             raise RuntimeError(f"StableDiffusionPipeline failed to load {model_id}: {exc}")

    # If we are using FLUX, we have already handled device placement via enable_model_cpu_offload
    # For SD, we proceed with standard logic
    if "flux" not in model_id.lower():
        if sequential_offload and torch.cuda.is_available():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

    return pipe
