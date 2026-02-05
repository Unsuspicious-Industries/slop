"""Unified AI model loader for diffusion and encoder models.

Provides simple, consistent interface for loading:
- Diffusion models (Stable Diffusion, FLUX)
- Encoders (CLIP, DINOv2)
- Multimodal extractors

Example:
    >>> ai = AILoader()
    >>> model = ai.load_diffusion("stabilityai/stable-diffusion-2-1")
    >>> encoder = ai.load_encoder("clip")
    >>> extractor = ai.load_extractor(strategy="concat")
"""

from typing import Optional, Any, List
from pathlib import Path
import torch


class AILoader:
    """Unified loader for AI models."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize AI loader.
        
        Args:
            device: Device to load models on ("cuda", "cpu", or None for auto)
            dtype: Default dtype for models (None for auto)
            cache_dir: Directory to cache models (None for default HF cache)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.cache_dir = cache_dir
        
        print(f"AI Loader initialized (device={self.device}, dtype={self.dtype})")
    
    def load_diffusion(
        self,
        model_id: str,
        enable_tracking: bool = False,
        **kwargs: Any
    ) -> Any:
        """Load diffusion model for image generation.
        
        Args:
            model_id: Model identifier (HF path or shorthand)
                - "sd21" -> stabilityai/stable-diffusion-2-1
                - "sdxl" -> stabilityai/stable-diffusion-xl-base-1.0
                - "flux-dev" -> black-forest-labs/FLUX.1-dev
                - Or any HF diffusion model path
            enable_tracking: If True, return model with trajectory hooks enabled
            **kwargs: Additional arguments passed to model loader
        
        Returns:
            Diffusion pipeline or hooked pipeline for tracking
        """
        from src.diffusion.loaders import load_diffusion_model
        from src.diffusion.sd_hook import SDTrajectoryHook
        from src.diffusion.flux_hook import FluxTrajectoryHook
        
        # Resolve shorthand names
        model_map = {
            "sd21": "stabilityai/stable-diffusion-2-1",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "flux-dev": "black-forest-labs/FLUX.1-dev",
            "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        }
        model_id = model_map.get(model_id.lower(), model_id)
        
        # Load base pipeline
        pipe = load_diffusion_model(model_id, **kwargs)
        
        if enable_tracking:
            # Wrap with trajectory capture hook
            if "flux" in model_id.lower():
                pipe = FluxTrajectoryHook(pipe)
                print("Loaded FLUX diffusion model with trajectory tracking")
            else:
                pipe = SDTrajectoryHook(pipe)
                print("Loaded SD diffusion model with trajectory tracking")
        else:
            print(f"Loaded diffusion model: {model_id}")
        
        return pipe
    
    def load_encoder(
        self,
        encoder_type: str = "clip",
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """Load image/text encoder.
        
        Args:
            encoder_type: Type of encoder ("clip", "dinov2", "auto")
            model_name: Specific model name (None for default)
            **kwargs: Additional arguments for encoder
        
        Returns:
            Encoder model
        """
        from src.encoders.loader import load_encoder
        
        if model_name is None:
            # Use defaults
            if encoder_type == "clip":
                model_name = "openai/clip-vit-large-patch14"
            elif encoder_type == "dinov2":
                model_name = "facebook/dinov2-large"
            else:
                model_name = encoder_type
        
        encoder = load_encoder(model_name, device=self.device)
        print(f"Loaded encoder: {encoder_type} ({model_name})")
        return encoder
    
    def load_extractor(
        self,
        strategy: str = "concat",
        models: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Any:
        """Load multimodal embedding extractor.
        
        Args:
            strategy: How to combine CLIP and DINOv2 embeddings
                - "concat": Concatenate embeddings
                - "average": Average embeddings
                - "separate": Keep separate
            clip_model: CLIP model name (None for default)
            dino_model: DINOv2 model name (None for default)
            **kwargs: Additional arguments for extractor
        
        Returns:
            Multimodal embedding extractor
        """
        from src.encoders.multimodal import EmbeddingExtractor
        
        extractor = EmbeddingExtractor(models=models, strategy=strategy)
        print(f"Loaded multimodal extractor (strategy={strategy})")
        return extractor
    
    def load_hf_bridge(
        self,
        model_id: str,
        **kwargs: Any
    ) -> Any:
        """Load arbitrary HuggingFace model via unified bridge.
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional arguments for bridge
        
        Returns:
            HFModelBridge instance
        """
        from src.encoders.hf_bridge import HFModelBridge
        
        bridge = HFModelBridge(
            model_id=model_id,
            device=self.device,
            dtype=self.dtype,
            **kwargs
        )
        print(f"Loaded HF model via bridge: {model_id}")
        return bridge


# Convenience function for quick loading
def load_ai(
    diffusion: Optional[str] = None,
    encoder: Optional[str] = None,
    extractor_strategy: Optional[str] = None,
    device: Optional[str] = None,
    enable_tracking: bool = False
) -> dict[str, Any]:
    """Quick load AI components.
    
    Args:
        diffusion: Diffusion model ID (None to skip)
        encoder: Encoder type (None to skip)
        extractor_strategy: Extractor strategy (None to skip)
        device: Device to use
        enable_tracking: Enable trajectory tracking for diffusion
    
    Returns:
        Dictionary with loaded components
    
    Example:
        >>> models = load_ai(diffusion="sd21", encoder="clip", enable_tracking=True)
        >>> pipe = models["diffusion"]
        >>> encoder = models["encoder"]
    """
    loader = AILoader(device=device)
    components = {}
    
    if diffusion:
        components["diffusion"] = loader.load_diffusion(diffusion, enable_tracking=enable_tracking)
    
    if encoder:
        components["encoder"] = loader.load_encoder(encoder)
    
    if extractor_strategy:
        components["extractor"] = loader.load_extractor(strategy=extractor_strategy)
    
    return components
