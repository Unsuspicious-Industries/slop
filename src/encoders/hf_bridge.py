"""Unified Hugging Face model bridge for embeddings and generation.

Provides a single interface to work with different model types:
- Text encoders (CLIP, BERT, GPT-2, etc.)
- Image encoders (CLIP, DINOv2, etc.)
- Diffusion models (Stable Diffusion, FLUX, etc.)
"""

from typing import Optional, List, Union, Dict, Any, Tuple, cast
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from numpy.typing import NDArray


class HFModelBridge:
    """Unified interface for Hugging Face models."""
    
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize model from Hugging Face.
        
        Args:
            model_id: HF model identifier (e.g., "openai/clip-vit-base-patch32")
            device: Device to load model on ("cuda", "cpu", or None for auto)
            dtype: Torch dtype for model weights
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        # Detect model type and load
        self.model_type = self._detect_model_type(model_id)
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.pipe: Optional[Any] = None
        
        self._load_model()
    
    def _detect_model_type(self, model_id: str) -> str:
        """Detect what type of model this is."""
        model_lower = model_id.lower()
        
        if "clip" in model_lower:
            return "clip"
        elif "dinov2" in model_lower or "dino" in model_lower:
            return "dinov2"
        elif "bert" in model_lower:
            return "bert"
        elif "gpt" in model_lower:
            return "gpt"
        elif "stable-diffusion" in model_lower or "sd-" in model_lower:
            return "stable-diffusion"
        elif "flux" in model_lower:
            return "flux"
        elif "llama" in model_lower:
            return "llama"
        elif "t5" in model_lower:
            return "t5"
        else:
            # Try to auto-detect from model config
            return "auto"
    
    def _load_model(self):
        """Load the appropriate model based on type."""
        if self.model_type == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            
        elif self.model_type == "dinov2":
            from transformers import AutoModel, AutoImageProcessor
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            
        elif self.model_type in ["bert", "gpt", "llama", "t5"]:
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
        elif self.model_type == "stable-diffusion":
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            )
            self.pipe.to(self.device)
            
        elif self.model_type == "flux":
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            )
            self.pipe.to(self.device)
            
        else:
            # Auto-detect
            try:
                from transformers import AutoModel, AutoTokenizer
                self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model_type = "auto-encoder"
            except Exception as exc:
                raise ValueError(f"Could not auto-detect model type for {self.model_id}") from exc
    
    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> NDArray[np.float32]:
        """Encode text to embedding vector(s).
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Array of shape (dim,) for single text or (batch, dim) for list
        """
        if self.model_type == "clip":
            assert self.processor is not None
            assert self.model is not None
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.get_text_features(**inputs)
            return cast(NDArray[np.float32], embeddings.cpu().numpy())
            
        elif self.model_type in ["bert", "gpt", "llama", "t5", "auto-encoder"]:
            assert self.tokenizer is not None
            assert self.model is not None
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Use [CLS] token or mean pooling
            if hasattr(outputs, "last_hidden_state"):
                # Mean pooling over sequence
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                # Fallback: use last hidden state
                embeddings = outputs[0].mean(dim=1)
            
            return cast(NDArray[np.float32], embeddings.cpu().numpy())
        else:
            raise NotImplementedError(f"Text encoding not supported for {self.model_type}")
    
    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> NDArray[np.float32]:
        """Encode image(s) to embedding vector(s).
        
        Args:
            image: Single PIL Image or list of images
            
        Returns:
            Array of shape (dim,) for single image or (batch, dim) for list
        """
        if self.model_type == "clip":
            assert self.processor is not None
            assert self.model is not None
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.get_image_features(**inputs)
            return cast(NDArray[np.float32], embeddings.cpu().numpy())
            
        elif self.model_type == "dinov2":
            assert self.processor is not None
            assert self.model is not None
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Use CLS token
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                embeddings = outputs.pooler_output
            
            return cast(NDArray[np.float32], embeddings.cpu().numpy())
        else:
            raise NotImplementedError(f"Image encoding not supported for {self.model_type}")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate image(s) from text prompt(s).
        
        Args:
            prompt: Text prompt or list of prompts
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            height: Image height
            width: Image width
            **kwargs: Additional arguments passed to pipeline
            
        Returns:
            Generated image(s)
        """
        if self.model_type not in ["stable-diffusion", "flux"]:
            raise NotImplementedError(f"Generation not supported for {self.model_type}")
        
        assert self.pipe is not None
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            **kwargs
        )
        
        images = result.images
        if isinstance(prompt, str):
            return cast(Image.Image, images[0])
        return cast(List[Image.Image], images)
    
    def generate_with_trajectory(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        sample_rate: int = 5,
        height: int = 512,
        width: int = 512,
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Generate image and capture intermediate latents.
        
        Args:
            prompt: Text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            sample_rate: Capture latent every N steps
            height: Image height
            width: Image width
            
        Returns:
            Tuple of (final_image, trajectory)
            trajectory is list of dicts with keys: step, timestep, latent
        """
        if self.model_type not in ["stable-diffusion", "flux"]:
            raise NotImplementedError(f"Trajectory capture not supported for {self.model_type}")
        
        assert self.pipe is not None
        trajectory: List[Dict[str, Any]] = []
        
        def capture_callback(step: int, timestep: int, latents: torch.Tensor):
            """Callback to capture intermediate latents."""
            if step % sample_rate == 0:
                trajectory.append({
                    "step": step,
                    "timestep": int(timestep),
                    "latent": latents.cpu().numpy().copy()
                })
        
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            callback=capture_callback,
            callback_steps=1,
        )
        
        return result.images[0], trajectory
    
    def get_embedding_dim(self) -> Optional[int]:
        """Get dimensionality of embeddings produced by this model."""
        if self.model_type == "clip":
            # Try with dummy input
            dummy_emb = self.encode_text("test")
            return dummy_emb.shape[-1]
        elif self.model_type == "dinov2":
            dummy_img = Image.new("RGB", (224, 224))
            dummy_emb = self.encode_image(dummy_img)
            return dummy_emb.shape[-1]
        elif self.model_type in ["bert", "gpt", "llama", "t5", "auto-encoder"]:
            dummy_emb = self.encode_text("test")
            return dummy_emb.shape[-1]
        else:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "embedding_dim": self.get_embedding_dim(),
            "capabilities": {
                "text_encoding": self.model_type in ["clip", "bert", "gpt", "llama", "t5", "auto-encoder"],
                "image_encoding": self.model_type in ["clip", "dinov2"],
                "generation": self.model_type in ["stable-diffusion", "flux"],
                "trajectory_capture": self.model_type in ["stable-diffusion", "flux"],
            }
        }
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return f"HFModelBridge({info['model_id']}, type={info['model_type']}, dim={info['embedding_dim']})"


def load_hf_model(model_id: str, **kwargs) -> HFModelBridge:
    """Convenience function to load a Hugging Face model.
    
    Args:
        model_id: HF model identifier
        **kwargs: Additional arguments for HFModelBridge
        
    Returns:
        HFModelBridge instance
    """
    return HFModelBridge(model_id, **kwargs)
