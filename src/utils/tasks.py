"""Task runners for latent extraction, flow analysis, and physics computations.

Provides high-level interfaces for common workflows:
- Latent space extraction from images
- Trajectory capture from diffusion
- Flow field analysis
- Physics-based analysis

Example:
    >>> from src.utils.tasks import TaskRunner
    >>> runner = TaskRunner()
    >>> 
    >>> # Extract embeddings from images
    >>> embeddings = runner.extract_latents(
    ...     images_dir="outputs/images",
    ...     encoder="clip",
    ...     output_path="outputs/embeddings.npy"
    ... )
    >>>
    >>> # Analyze flow from trajectories
    >>> flow_data = runner.analyze_flow(
    ...     trajectories_dir="data/trajectories",
    ...     resolution=50,
    ...     output_dir="outputs/flow"
    ... )
"""

from typing import Optional, Literal, List, Union, Dict, Any, Tuple, cast
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


class TaskRunner:
    """High-level task runner for common workflows."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize task runner.
        
        Args:
            device: Device for computations ("cuda", "cpu", or None for auto)
            verbose: Print progress messages
        """
        self.device = device
        self.verbose = verbose
        self._encoder = None
        self._extractor = None
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(message)
    
    def extract_latents(
        self,
        images: Union[Path, List[Path], List[np.ndarray]],
        encoder: Literal["clip", "dinov2", "multi"] = "multi",
        strategy: Literal["concat", "average", "separate"] = "concat",
        output_path: Optional[Path] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """Extract latent embeddings from images.
        
        Args:
            images: Path to directory, list of paths, or list of image arrays
            encoder: Which encoder to use
            strategy: Strategy for multimodal extraction
            output_path: Where to save embeddings (None to skip saving)
            batch_size: Batch size for processing
        
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        
        Example:
            >>> embeddings = runner.extract_latents(
            ...     images="outputs/images",
            ...     encoder="multi",
            ...     output_path="outputs/embeddings.npy"
            ... )
        """
        from src.utils.ai import AILoader
        
        # Load encoder
        loader = AILoader(device=self.device)
        if encoder == "multi":
            model = loader.load_extractor(strategy=strategy)
        else:
            model = loader.load_encoder(encoder)
        
        # Collect images
        image_paths: Optional[List[Path]] = None
        image_arrays: Optional[List[np.ndarray]] = None
        if isinstance(images, Path):
            image_paths = sorted(images.glob("*.png")) + sorted(images.glob("*.jpg"))
            self._log(f"Found {len(image_paths)} images in {images}")
        elif isinstance(images, list) and images and isinstance(images[0], Path):
            image_paths = cast(List[Path], images)
        else:
            image_arrays = cast(List[np.ndarray], images)
        
        # Extract embeddings
        embeddings: List[np.ndarray] = []
        
        if image_paths is not None:
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting", disable=not self.verbose):
                batch_paths = image_paths[i:i+batch_size]
                image_batch = [Image.open(p).convert("RGB") for p in batch_paths]
                
                batch_emb = [np.asarray(model.encode_image(img)) for img in image_batch]
                
                embeddings.extend(batch_emb)
        else:
            image_arrays = image_arrays or []
            for i in tqdm(range(0, len(image_arrays), batch_size), desc="Extracting", disable=not self.verbose):
                array_batch = image_arrays[i:i+batch_size]
                batch_emb = [np.asarray(model.encode_image(img)) for img in array_batch]
                embeddings.extend(batch_emb)
        
        embeddings_array = np.array(embeddings)
        self._log(f"Extracted embeddings: {embeddings_array.shape}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embeddings_array)
            self._log(f"Saved embeddings to {output_path}")
        
        return cast(np.ndarray, embeddings_array)
    
    def capture_trajectories(
        self,
        prompts: Union[List[str], Path],
        model: str = "sd21",
        output_dir: Optional[Path] = None,
        steps: int = 50,
        sample_rate: int = 5,
        save_images: bool = True
    ) -> List[Dict[str, Any]]:
        """Capture diffusion trajectories for prompts.
        
        Args:
            prompts: List of prompts or path to prompts file
            model: Diffusion model to use
            output_dir: Where to save outputs
            steps: Number of diffusion steps
            sample_rate: Sample every N steps
            save_images: Whether to save final images
        
        Returns:
            List of trajectory data dicts with keys: prompt, image, trajectory, metadata
        
        Example:
            >>> trajectories = runner.capture_trajectories(
            ...     prompts=["a doctor", "a nurse"],
            ...     model="sd21",
            ...     output_dir="outputs/trajectories"
            ... )
        """
        from src.utils.ai import AILoader
        from src.diffusion.trajectory_capture import compress_trajectory
        
        # Load prompts
        if isinstance(prompts, Path):
            with open(prompts) as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        # Load model with tracking
        loader = AILoader(device=self.device)
        pipe = loader.load_diffusion(model, enable_tracking=True)
        
        results = []
        
        for prompt in tqdm(prompts, desc="Generating", disable=not self.verbose):
            # Generate with tracking
            image, trajectory = pipe.generate_with_tracking(prompt, num_steps=steps)
            traj_compressed = compress_trajectory(trajectory, sample_rate=sample_rate)
            
            result = {
                "prompt": prompt,
                "image": np.array(image),
                "trajectory": traj_compressed,
                "metadata": {
                    "timesteps": [step["timestep"] for step in traj_compressed],
                    "steps": steps,
                    "sample_rate": sample_rate
                }
            }
            results.append(result)
            
            # Save if output_dir provided
            if output_dir:
                output_dir = Path(output_dir)
                filename = prompt.replace(" ", "_")[:50]
                
                if save_images:
                    img_dir = output_dir / "images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(result["image"]).save(img_dir / f"{filename}.png")
                
                # Save trajectory
                traj_dir = output_dir / "trajectories"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_latents = np.array([step["latent"] for step in traj_compressed])
                np.save(traj_dir / f"{filename}.npy", traj_latents)
                
                with open(traj_dir / f"{filename}.json", "w") as f:
                    json.dump(result["metadata"], f, indent=2)
        
        self._log(f"Captured {len(results)} trajectories")
        return results
    
    def analyze_flow(
        self,
        trajectories: Union[Path, List[np.ndarray]],
        resolution: int = 50,
        radius: float = 0.5,
        output_dir: Optional[Path] = None,
        compute_topology: bool = True
    ) -> Dict[str, Any]:
        """Analyze flow field from trajectories.
        
        Args:
            trajectories: Path to trajectories directory or list of trajectory arrays
            resolution: Grid resolution for flow field
            radius: Radius for interpolation
            output_dir: Where to save outputs
            compute_topology: Whether to analyze flow topology
        
        Returns:
            Dictionary with flow field data and analysis results
        
        Example:
            >>> flow = runner.analyze_flow(
            ...     trajectories="data/trajectories",
            ...     resolution=50,
            ...     output_dir="outputs/flow"
            ... )
            >>> print(flow["attractors"])
        """
        from src.analysis.flow_fields import compute_flow_field, compute_flow_field_3d
        from src.analysis.attractors import analyze_flow_topology
        from src.physics import divergence, curl
        
        # Load trajectories
        if isinstance(trajectories, Path):
            traj_files = sorted(trajectories.glob("*.npy"))
            trajectories = [np.load(f) for f in traj_files]
            self._log(f"Loaded {len(trajectories)} trajectories from {trajectories}")
        
        # Determine dimensionality
        ndim = trajectories[0].shape[-1]
        self._log(f"Analyzing {ndim}D flow field (resolution={resolution})")
        
        # Compute flow field
        if ndim == 2:
            grid, V = compute_flow_field(
                trajectories,
                grid_resolution=resolution,
                radius=radius
            )
            
            # Compute physics quantities
            div_V = divergence(V)
            curl_V = curl(V)
            
        else:  # 3D
            X, Y, Z, Vx, Vy, Vz = compute_flow_field_3d(
                trajectories,
                grid_resolution=resolution,
                radius=radius
            )
            grid = np.stack([X, Y, Z], axis=-1)
            V = np.stack([Vx, Vy, Vz], axis=-1)
            
            # Compute physics quantities
            from src.physics import divergence_3d, curl_3d
            div_V = divergence_3d(V)
            curl_V = curl_3d(V)
        
        result = {
            "grid": grid,
            "velocity": V,
            "divergence": div_V,
            "curl": curl_V,
            "dimensions": ndim,
            "resolution": resolution
        }
        
        # Analyze topology
        if compute_topology:
            topology = analyze_flow_topology(div_V, curl_V)
            result["topology"] = topology
            self._log(f"Found {topology['n_attractors']} attractors, {topology['n_repellers']} repellers")
        
        # Save outputs
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / "grid.npy", grid)
            np.save(output_dir / "velocity.npy", V)
            np.save(output_dir / "divergence.npy", div_V)
            np.save(output_dir / "curl.npy", curl_V)
            
            if compute_topology:
                with open(output_dir / "topology.json", "w") as f:
                    # Convert numpy arrays to lists for JSON
                    topology_json = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in topology.items()
                    }
                    json.dump(topology_json, f, indent=2)
            
            self._log(f"Saved flow analysis to {output_dir}")
        
        return result
    
    def run_physics_analysis(
        self,
        vector_field: np.ndarray,
        spacing: Union[float, Tuple[float, ...]] = 1.0,
        find_critical_points: bool = True,
        classify_points: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive physics analysis on vector field.
        
        Args:
            vector_field: Vector field array, shape (..., ny, nx, ndim)
            spacing: Grid spacing (scalar or tuple)
            find_critical_points: Whether to find critical points
            classify_points: Whether to classify critical points
        
        Returns:
            Dictionary with physics analysis results
        
        Example:
            >>> physics = runner.run_physics_analysis(
            ...     vector_field=V,
            ...     spacing=0.02,
            ...     find_critical_points=True
            ... )
            >>> print(physics["divergence"].mean())
        """
        from src.physics import (
            divergence, curl, gradient, laplacian,
            find_critical_points as find_crit_pts,
            classify_critical_point,
            compute_field_magnitude
        )
        
        ndim = vector_field.shape[-1]
        
        # Basic differential operators
        if ndim == 2:
            if isinstance(spacing, tuple):
                dx, dy = spacing[0], spacing[1]
            else:
                dx, dy = spacing, spacing
            div_V = divergence(vector_field, dx, dy)
            curl_V = curl(vector_field, dx, dy)
        else:
            from src.physics import divergence_3d, curl_3d
            if isinstance(spacing, tuple):
                dx, dy, dz = spacing[0], spacing[1], spacing[2]
            else:
                dx, dy, dz = spacing, spacing, spacing
            div_V = divergence_3d(vector_field, dx, dy, dz)
            curl_V = curl_3d(vector_field, dx, dy, dz)
        
        magnitude = compute_field_magnitude(vector_field)
        
        result = {
            "divergence": div_V,
            "curl": curl_V,
            "magnitude": magnitude,
            "dimensions": ndim
        }
        
        # Find and classify critical points
        if find_critical_points:
            critical_pts = find_crit_pts(div_V, attractor_threshold=-0.01, repeller_threshold=0.01)
            result["critical_points"] = critical_pts
            
            if classify_points:
                classifications = []
                for positions, _ in critical_pts.values():
                    for pos in positions:
                        classification = classify_critical_point(vector_field, tuple(pos))
                        classifications.append(classification)
                result["classifications"] = classifications
                self._log(f"Found {len(classifications)} critical points")
        
        self._log(f"Physics analysis complete (div: {div_V.mean():.3e}, curl: {np.abs(curl_V).mean():.3e})")
        return result


# Convenience function for quick task execution
def run_task(
    task: Literal["extract", "trajectories", "flow", "physics"],
    **kwargs: Any
) -> Any:
    """Quick task execution.
    
    Args:
        task: Task type to run
        **kwargs: Arguments for the task
    
    Returns:
        Task results
    
    Example:
        >>> embeddings = run_task("extract", images="outputs/images", encoder="clip")
        >>> flow = run_task("flow", trajectories="data/trajectories", resolution=50)
    """
    runner = TaskRunner()
    
    if task == "extract":
        return runner.extract_latents(**kwargs)
    elif task == "trajectories":
        return runner.capture_trajectories(**kwargs)
    elif task == "flow":
        return runner.analyze_flow(**kwargs)
    elif task == "physics":
        return runner.run_physics_analysis(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
