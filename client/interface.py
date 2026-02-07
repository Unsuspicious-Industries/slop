"""High-level API for interacting with the SLOP server and running experiments."""
from typing import Dict, Any, Optional, List, Union, Iterator
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from client.config import ServerConfig
from client.transport import SSHTransport, TransportError
from shared.protocol.messages import (
    Request, InferenceRequest, JobResult, ServerInfo, MessageKind,
    ErrorResponse
)
from shared.protocol.serialization import unpack_array
import shared.metrics.bias as bias_metrics

# Optional visualization imports
try:
    from client.visualization.trajectories import plot_trajectories
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class TrajectoryStep:
    """Represents a single denoising step in the diffusion process.
    
    This provides access to the internal state of the model at each timestep.
    All arrays are NumPy arrays with the following shapes:
    
    Attributes:
        step_index: Which step this is (0 to num_steps)
        timestep: The scheduler timestep value (int, usually 0-1000)
        latent: The latent representation at this step. Shape: (batch, channels, height, width)
        noise_pred: The model's predicted noise. Shape: (batch, channels, height, width)
        prompt_embedding: Text conditioning embeddings. Shape: (batch, seq_len, hidden_dim)
    """
    step_index: int
    timestep: int
    latent: np.ndarray
    noise_pred: np.ndarray
    prompt_embedding: np.ndarray
    
    @property
    def latent_flat(self) -> np.ndarray:
        """Flattened latent vector for analysis. Shape: (flattened_dim,)"""
        return self.latent.flatten()
    
    @property
    def noise_pred_flat(self) -> np.ndarray:
        """Flattened noise prediction. Shape: (flattened_dim,)"""
        return self.noise_pred.flatten()


@dataclass  
class InferenceResult:
    """Complete result from a generation job.
    
    This contains:
    - The final generated image
    - The full trajectory (every step from noise to image)
    - Metadata about the generation
    
    Example:
        >>> with SlopClient(config) as client:
        ...     result = client.generate("a cat", num_steps=20)
        ...     
        ...     # Access final image
        ...     print(f"Image size: {len(result.image)} bytes")
        ...     
        ...     # Access trajectory
        ...     print(f"Captured {len(result.trajectory)} steps")
        ...     
        ...     # Access specific step
        ...     step_10 = result.trajectory[10]
        ...     print(f"Step 10 timestep: {step_10.timestep}")
        ...     print(f"Step 10 latent shape: {step_10.latent.shape}")
    """
    
    # Final output
    image: Optional[bytes] = None
    """PNG image bytes (None if generation failed)"""
    
    # Trajectory data - ALL steps are captured
    latents: Optional[np.ndarray] = None
    """ALL latents from every denoising step. 
    Shape: (num_steps+1, batch, channels, height, width)
    Index 0 = pure noise, Index -1 = final image latent"""
    
    noise_preds: Optional[np.ndarray] = None
    """Model's noise predictions at each step.
    Shape: (num_steps, batch, channels, height, width)"""
    
    prompt_embeds: Optional[np.ndarray] = None
    """Text encoder embeddings (constant across steps).
    Shape: (batch, seq_len, hidden_dim)"""
    
    timesteps: Optional[np.ndarray] = None
    """Scheduler timesteps for each step.
    Shape: (num_steps,) - integer values like 999, 949, ..., 0"""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Generation metadata: elapsed_s, model_id, width, height, etc."""
    
    @classmethod
    def from_job_result(cls, job: JobResult) -> 'InferenceResult':
        """Unpack a JobResult into a friendly InferenceResult."""
        res = cls()
        
        # 1. Image (PNG bytes)
        if "image" in job.payload:
            import base64
            img_data = job.payload["image"]
            if isinstance(img_data, str):
                res.image = base64.b64decode(img_data)
            else:
                res.image = img_data

        # 2. Unpack arrays from wire format
        arrays = job.arrays
        
        if "latents" in arrays:
            res.latents = unpack_array(arrays["latents"])
            
        if "noise_preds" in arrays:
            res.noise_preds = unpack_array(arrays["noise_preds"])
            
        if "prompt_embeds" in arrays:
            res.prompt_embeds = unpack_array(arrays["prompt_embeds"])
            
        if "timesteps" in arrays:
            res.timesteps = unpack_array(arrays["timesteps"])
        
        # 3. Metadata
        res.metadata = {
            **job.payload.get("metadata", {}),
            "job_id": job.job_id,
            "model_id": job.model_id,
            "prompt": job.prompt,
            "elapsed_s": job.elapsed_s,
            "request_kind": job.request_kind,
        }
        
        return res
    
    @property
    def num_steps(self) -> int:
        """Number of denoising steps performed (excluding initial noise)."""
        if self.latents is not None:
            # latents has num_steps+1 (initial noise + each denoising step)
            return self.latents.shape[0] - 1
        return 0
    
    @property
    def latent_shape(self) -> Optional[tuple]:
        """Shape of individual latent tensors: (batch, channels, height, width)"""
        if self.latents is not None and self.latents.ndim >= 4:
            return self.latents.shape[1:]
        return None
    
    @property
    def trajectory(self) -> List[TrajectoryStep]:
        """Get the full trajectory as a list of TrajectoryStep objects.
        
        This provides easy access to step-by-step data with named attributes.
        
        Returns:
            List of TrajectoryStep, one per captured step
        """
        steps = []
        if self.latents is None:
            return steps
            
        num_steps = self.latents.shape[0]
        
        for i in range(num_steps):
            step = TrajectoryStep(
                step_index=i,
                timestep=int(self.timesteps[i]) if self.timesteps is not None and i < len(self.timesteps) else 0,
                latent=self.latents[i],
                noise_pred=self.noise_preds[i] if self.noise_preds is not None and i < len(self.noise_preds) else np.array([]),
                prompt_embedding=self.prompt_embeds[0] if self.prompt_embeds is not None else np.array([])
            )
            steps.append(step)
            
        return steps
    
    def get_step(self, index: int) -> Optional[TrajectoryStep]:
        """Get a specific step by index.
        
        Args:
            index: Step index (0 = initial noise, -1 = final latent)
            
        Returns:
            TrajectoryStep or None if data unavailable
        """
        steps = self.trajectory
        if not steps or index >= len(steps) or index < -len(steps):
            return None
        return steps[index]
    
    def __len__(self) -> int:
        """Number of steps in trajectory."""
        return self.num_steps
    
    def __iter__(self) -> Iterator[TrajectoryStep]:
        """Iterate over trajectory steps."""
        return iter(self.trajectory)
    
    def __repr__(self) -> str:
        return (f"InferenceResult(num_steps={self.num_steps}, "
                f"latent_shape={self.latent_shape}, "
                f"image={'present' if self.image else 'missing'})")


class SlopClient:
    """Main entry point for running remote diffusion inference.
    
    Example:
        >>> from client.config import registry
        >>> from client.interface import SlopClient
        >>> 
        >>> config = registry.get("vast-auto-test")
        >>> with SlopClient(config) as client:
        ...     result = client.generate(
        ...         prompt="a serene lake at sunset",
        ...         num_steps=50,
        ...         capture_latents=True  # Capture EVERY step
        ...     )
        ...     
        ...     # Full trajectory access
        ...     print(f"Generated in {len(result)} steps")
        ...     
        ...     # Access specific step
        ...     mid_step = result.get_step(25)
        ...     print(f"Step 25 timestep value: {mid_step.timestep}")
        ...     
        ...     # Iterate all steps
        ...     for step in result:
        ...         print(f"Step {step.step_index}: latent mean = {step.latent.mean():.4f}")
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.transport = SSHTransport(config)
        self._connected = False

    def connect(self):
        """Connect to the remote server."""
        if not self._connected:
            self.transport.connect()
            self._connected = True

    def get_server_info(self) -> ServerInfo:
        """Query the server for capabilities and hardware info."""
        self.connect()
        req = Request(kind=MessageKind.SERVER_INFO.value)
        resp = self.transport.send_request(req)
        
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(f"Server error: {resp.error}")
        if not isinstance(resp, ServerInfo):
            raise RuntimeError(f"Unexpected response type: {type(resp)}")
            
        return resp

    def generate(
        self,
        prompt: str,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = -1,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        capture_latents: bool = True,
        capture_noise: bool = True,
        capture_timesteps: bool = True
    ) -> InferenceResult:
        """Run a generation job on the remote server.
        
        This method triggers inference and can capture the FULL trajectory
        of the diffusion process - every step from initial noise to final image.
        
        Args:
            prompt: Text description of what to generate
            num_steps: Number of denoising steps (default: 50)
            guidance_scale: CFG scale for prompt adherence (default: 7.5)
            seed: Random seed for reproducibility (-1 = random)
            model_id: HuggingFace model ID to use
            capture_latents: If True, captures latent at EVERY step (default: True)
            capture_noise: If True, captures noise predictions at every step (default: True)
            capture_timesteps: If True, captures scheduler timestep values (default: True)
            
        Returns:
            InferenceResult containing:
            - image: Final generated PNG image
            - latents: Array of shape (num_steps+1, batch, c, h, w) with ALL step latents
            - noise_preds: Array of shape (num_steps, batch, c, h, w) with model predictions
            - trajectory: List of TrajectoryStep objects for easy step-by-step access
            - metadata: Generation info (timing, model, etc.)
            
        Example:
            >>> result = client.generate("a cat", num_steps=20)
            >>> 
            >>> # Check we got all steps
            >>> assert len(result) == 20
            >>> 
            >>> # Access final image
            >>> with open("output.png", "wb") as f:
            ...     f.write(result.image)
            >>> 
            >>> # Analyze trajectory
            >>> step_0 = result.get_step(0)  # Initial noise
            >>> step_10 = result.get_step(10)  # Mid-generation
            >>> step_20 = result.get_step(-1)  # Final latent
            >>> 
            >>> print(f"Step 0 latent mean: {step_0.latent.mean():.4f}")
            >>> print(f"Step 20 latent mean: {step_20.latent.mean():.4f}")
        """
        self.connect()
        
        req = InferenceRequest(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model_id=model_id,
            capture_latents=capture_latents,
            capture_noise_pred=capture_noise,
            capture_timesteps=capture_timesteps
        )
        
        resp = self.transport.send_request(req)
        
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(f"Inference failed: {resp.error}\n{resp.traceback}")
            
        if not isinstance(resp, JobResult):
            raise RuntimeError(f"Unexpected response: {resp}")
            
        return InferenceResult.from_job_result(resp)

    def close(self):
        """Shutdown the connection."""
        self.transport.close()
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --------------------------------------------------------------------------
    # Batch Generation
    # --------------------------------------------------------------------------

    def generate_batch(
        self,
        prompt: str,
        n_variations: int = 20,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        seed_start: int = 0,
        capture_latents: bool = True,
        capture_noise: bool = True,
        progress: bool = True,
    ) -> List[InferenceResult]:
        """Generate multiple variations of the same prompt with different seeds.
        
        Used for averaging out noise to find systematic patterns.
        Each variation uses seed_start + i as seed.
        
        Args:
            prompt: Text prompt
            n_variations: Number of variations to generate
            num_steps: Denoising steps per generation
            guidance_scale: CFG scale
            model_id: HuggingFace model ID
            seed_start: Starting seed (seeds will be seed_start, seed_start+1, ...)
            capture_latents: Capture full trajectory
            capture_noise: Capture noise predictions
            progress: Print progress
            
        Returns:
            List of InferenceResult, one per variation
        """
        results = []
        for i in range(n_variations):
            if progress:
                print(f"  [{i+1}/{n_variations}] seed={seed_start + i}", end="... ", flush=True)
            
            result = self.generate(
                prompt=prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed_start + i,
                model_id=model_id,
                capture_latents=capture_latents,
                capture_noise=capture_noise,
            )
            results.append(result)
            
            if progress:
                elapsed = result.metadata.get('elapsed_s', 0)
                print(f"{elapsed:.1f}s")
        
        return results

    def encode_images(
        self,
        images: List[bytes],
        model_id: str = "openai/clip-vit-large-patch14",
    ) -> np.ndarray:
        """Encode images through a remote encoder (CLIP, DINOv2, etc).
        
        Args:
            images: List of PNG/JPEG image bytes
            model_id: Encoder model ID
            
        Returns:
            Embeddings array of shape (n_images, embed_dim)
        """
        import base64
        from shared.protocol.messages import EncodeRequest, MessageKind, ErrorResponse, JobResult
        from shared.protocol.serialization import unpack_array
        
        self.connect()
        
        encoded = [base64.b64encode(img).decode('ascii') for img in images]
        
        req = EncodeRequest(
            model_id=model_id,
            modality="image",
            inputs=encoded,
        )
        
        resp = self.transport.send_request(req)
        
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(f"Encode failed: {resp.error}")
        if not isinstance(resp, JobResult):
            raise RuntimeError(f"Unexpected response: {type(resp)}")
        
        if "embeddings" in resp.arrays:
            return unpack_array(resp.arrays["embeddings"])
        
        raise RuntimeError("No embeddings in response")

    # --------------------------------------------------------------------------
    # Analysis Helpers
    # --------------------------------------------------------------------------

    def analyze_drift(self, result: InferenceResult, stereotype_poles: np.ndarray) -> float:
        """Calculate drift strength for the generated trajectory against stereotype poles.
        
        Args:
            result: The result from `generate()`
            stereotype_poles: Array of shape (n_poles, n_features) defining the bias directions
            
        Returns:
            Drift strength score (positive = drift away, negative = drift towards)
        """
        if result.latents is None:
            raise ValueError("Result does not contain latents. Did you run with capture_latents=True?")
            
        # Reshape latents if necessary: (steps, batch, c, h, w) -> (steps, flattened)
        # Or usually we analyze the embedding space. If latents are spatial, we might need to average.
        # For this example, let's assume we flatten per step.
        traj = result.latents
        if traj.ndim > 2:
            # Average spatial dims and channel dims to get a single vector per step?
            # Or just flatten. Let's flatten for now as a naive baseline.
            traj = traj.reshape(traj.shape[0], -1)
            
        return bias_metrics.drift_strength(traj, stereotype_poles)

    def analyze_deviation(self, result: InferenceResult, baseline_latents: np.ndarray) -> float:
        """Calculate deviation from a baseline trajectory."""
        if result.latents is None:
            raise ValueError("Result does not contain latents.")
            
        traj = result.latents.reshape(result.latents.shape[0], -1)
        base = baseline_latents.reshape(baseline_latents.shape[0], -1)
        
        return bias_metrics.trajectory_deviation(traj, base)

    def visualize_trajectory(self, result: InferenceResult, poles: Optional[np.ndarray] = None):
        """Visualize the denoising trajectory using UMAP.
        
        Args:
            result: The inference result containing latents.
            poles: Optional stereotype poles to plot alongside the trajectory.
            
        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization dependencies (umap-learn, matplotlib) not installed.")
            
        if result.latents is None:
            raise ValueError("Result does not contain latents.")
            
        # Flatten latents: (steps, batch, c, h, w) -> (steps, flattened)
        # We assume batch size 1 for now, or just take the first batch item
        latents = result.latents
        if latents.ndim > 2:
            # Take first batch item if present
            if latents.shape[1] == 1: # (steps, 1, ...)
                latents = latents[:, 0]
            
            # Flatten remaining dims
            latents = latents.reshape(latents.shape[0], -1)
            
        return plot_trajectories([latents], poles if poles is not None else np.array([]))
