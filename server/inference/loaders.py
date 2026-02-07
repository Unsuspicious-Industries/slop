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
    from diffusers import StableDiffusionPipeline

    pipe = None
    if "flux" in model_id.lower():
        try:
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
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

    if sequential_offload and torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    return pipe
