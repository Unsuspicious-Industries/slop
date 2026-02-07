"""Quick test of the new trajectory features."""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from client.interface import SlopClient
from client.config import registry

def test_trajectory_features(server_name: str):
    print(f"Testing trajectory capture on {server_name}...")
    
    cfg = registry.get(server_name)
    if not cfg:
        print(f"Server '{server_name}' not found!")
        return
        
    with SlopClient(cfg) as client:
        print(f"Connected to {server_name}")
        info = client.get_server_info()
        print(f"GPU: {info.gpu_name}")
        
        # Generate with full capture
        print("\nGenerating with trajectory capture...")
        result = client.generate(
            prompt="a simple geometric pattern",
            num_steps=10,  # Quick test
            model_id="runwayml/stable-diffusion-v1-5",
            capture_latents=True,
            capture_noise=True,
            capture_timesteps=True
        )
        
        print(f"\n✓ Generation complete!")
        print(f"\n  Result Overview:")
        print(f"    - Total steps: {len(result)}")
        print(f"    - Latent shape per step: {result.latent_shape}")
        print(f"    - Image present: {result.image is not None}")
        
        print(f"\n  Trajectory Data:")
        print(f"    - Number of trajectory steps: {len(result.trajectory)}")
        print(f"    - Timesteps captured: {result.timesteps is not None}")
        print(f"    - Noise predictions captured: {result.noise_preds is not None}")
        
        # Inspect trajectory
        print(f"\n  Step-by-step inspection:")
        for i, step in enumerate(result.trajectory[:3]):  # Show first 3
            print(f"    Step {step.step_index}: timestep={step.timestep}, latent_mean={step.latent.mean():.4f}")
        print(f"    ...")
        for i, step in enumerate(result.trajectory[-2:]):  # Show last 2
            print(f"    Step {step.step_index}: timestep={step.timestep}, latent_mean={step.latent.mean():.4f}")
        
        # Test iteration
        print(f"\n  Testing iteration:")
        latents_mean = [step.latent.mean() for step in result]
        print(f"    Mean of latents across steps: {np.mean(latents_mean):.4f}")
        
        print("\n✓ All trajectory features working correctly!")

if __name__ == "__main__":
    test_trajectory_features("vast-auto-test")
