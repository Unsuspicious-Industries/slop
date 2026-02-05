"""Test diffusion models and trajectory capture.

Tests:
- Stable Diffusion hook and trajectory capture
- FLUX model hook and trajectory capture
- Latent space flow extraction
- Trajectory storage and analysis
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.diffusion.sd_hook import StableDiffusionHook
from src.diffusion.flux_hook import FluxHook
from src.diffusion.trajectory_capture import TrajectoryCapture
from src.diffusion.loaders import load_sd_pipeline, load_flux_pipeline


class TestStableDiffusionHook:
    """Test Stable Diffusion trajectory capture."""
    
    @pytest.fixture
    def pipeline(self):
        """Load SD pipeline (using small/fast model)."""
        return load_sd_pipeline(
            model_name="CompVis/stable-diffusion-v1-4",
            device="cpu",
            dtype=torch.float32
        )
    
    @pytest.fixture
    def hook(self, pipeline):
        """Create SD hook."""
        return StableDiffusionHook(pipeline)
    
    def test_hook_installation(self, hook):
        """Test that hooks are properly installed."""
        assert hook.is_hooked
        assert len(hook.captured_latents) == 0
    
    def test_trajectory_capture(self, hook):
        """Test capturing latent trajectories during generation."""
        prompt = "a simple test image"
        
        # Generate with trajectory capture
        hook.reset()
        image = hook.generate(
            prompt=prompt,
            num_inference_steps=5,  # Small for testing
            height=64,  # Small for speed
            width=64
        )
        
        # Check trajectory
        trajectory = hook.get_trajectory()
        assert len(trajectory) > 0
        assert all(isinstance(t, np.ndarray) for t in trajectory)
        
        # Check trajectory shape consistency
        shapes = [t.shape for t in trajectory]
        assert len(set(shapes)) == 1  # All same shape
    
    def test_batch_generation(self, hook):
        """Test batch trajectory capture."""
        prompts = ["test 1", "test 2"]
        
        hook.reset()
        images = hook.generate_batch(
            prompts=prompts,
            num_inference_steps=5,
            height=64,
            width=64
        )
        
        assert len(images) == 2
        trajectories = hook.get_trajectories()
        assert len(trajectories) == 2


class TestFluxHook:
    """Test FLUX model trajectory capture."""
    
    @pytest.fixture
    def pipeline(self):
        """Load FLUX pipeline."""
        # Use dev version for testing
        return load_flux_pipeline(
            model_name="black-forest-labs/FLUX.1-dev",
            device="cpu"
        )
    
    @pytest.fixture
    def hook(self, pipeline):
        """Create FLUX hook."""
        return FluxHook(pipeline)
    
    def test_hook_installation(self, hook):
        """Test FLUX hook installation."""
        assert hook.is_hooked
    
    def test_trajectory_capture(self, hook):
        """Test FLUX trajectory capture."""
        prompt = "a test image"
        
        hook.reset()
        image = hook.generate(
            prompt=prompt,
            num_inference_steps=4,
            height=64,
            width=64
        )
        
        trajectory = hook.get_trajectory()
        assert len(trajectory) > 0


class TestTrajectoryCapture:
    """Test generic trajectory capture utility."""
    
    @pytest.fixture
    def capture(self):
        """Create trajectory capture."""
        return TrajectoryCapture()
    
    def test_capture_storage(self, capture):
        """Test trajectory storage."""
        # Simulate captured latents
        latents = [np.random.randn(1, 4, 8, 8) for _ in range(10)]
        
        for latent in latents:
            capture.record(latent)
        
        trajectory = capture.get_trajectory()
        assert len(trajectory) == 10
    
    def test_trajectory_downsampling(self, capture):
        """Test trajectory downsampling for analysis."""
        # High-res latents
        latents = [np.random.randn(1, 4, 64, 64) for _ in range(50)]
        
        for latent in latents:
            capture.record(latent)
        
        # Downsample to 2D for flow analysis
        trajectory_2d = capture.get_trajectory_2d(method="pca")
        
        assert trajectory_2d.shape[0] == 50
        assert trajectory_2d.shape[1] == 2  # 2D projection
    
    def test_save_load_trajectory(self, capture, tmp_path):
        """Test saving and loading trajectories."""
        latents = [np.random.randn(1, 4, 8, 8) for _ in range(10)]
        for latent in latents:
            capture.record(latent)
        
        # Save
        save_path = tmp_path / "trajectory.npy"
        capture.save(save_path)
        
        # Load
        loaded = np.load(save_path)
        assert loaded.shape[0] == 10


class TestFlowExtraction:
    """Test flow field extraction from trajectories."""
    
    def test_velocity_computation(self):
        """Test velocity field computation from trajectories."""
        from src.analysis.flow_fields import compute_flow_field
        
        # Create synthetic trajectories
        trajectories = [
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[0, 1], [1, 2], [2, 3]]),
            np.array([[1, 0], [2, 1], [3, 2]])
        ]
        
        grid, V = compute_flow_field(
            trajectories,
            grid_resolution=20,
            radius=1.0
        )
        
        assert V.shape == (20, 20, 2)
        assert not np.isnan(V).any()
    
    def test_3d_flow_extraction(self):
        """Test 3D flow field extraction."""
        from src.analysis.flow_fields import compute_flow_field_3d
        
        # 3D trajectories
        trajectories = [
            np.random.randn(10, 3) for _ in range(5)
        ]
        
        X, Y, Z, V = compute_flow_field_3d(
            trajectories,
            grid_resolution=10
        )
        
        assert V.shape == (10, 10, 10, 3)


class TestLatentSpaceAnalysis:
    """Test analysis of latent space dynamics."""
    
    def test_trajectory_statistics(self):
        """Test computing trajectory statistics."""
        trajectory = np.random.randn(100, 2)
        
        # Compute statistics
        mean_pos = trajectory.mean(axis=0)
        std_pos = trajectory.std(axis=0)
        
        # Velocity
        velocity = np.diff(trajectory, axis=0)
        mean_vel = velocity.mean(axis=0)
        
        assert mean_pos.shape == (2,)
        assert velocity.shape == (99, 2)
    
    def test_trajectory_smoothness(self):
        """Test trajectory smoothness metrics."""
        # Smooth trajectory
        t = np.linspace(0, 10, 100)
        smooth_traj = np.column_stack([np.sin(t), np.cos(t)])
        
        # Compute acceleration (smoothness measure)
        velocity = np.diff(smooth_traj, axis=0)
        acceleration = np.diff(velocity, axis=0)
        smoothness = np.linalg.norm(acceleration, axis=1).mean()
        
        # Noisy trajectory
        noisy_traj = smooth_traj + np.random.randn(*smooth_traj.shape) * 0.5
        velocity_noisy = np.diff(noisy_traj, axis=0)
        acceleration_noisy = np.diff(velocity_noisy, axis=0)
        smoothness_noisy = np.linalg.norm(acceleration_noisy, axis=1).mean()
        
        # Smooth should have less acceleration
        assert smoothness < smoothness_noisy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
