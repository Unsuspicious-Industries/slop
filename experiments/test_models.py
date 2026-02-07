"""Test inference with different models to verify trajectory capture."""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from client.interface import SlopClient
from client.config import registry

def test_model(server_name: str, model_id: str, description: str):
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")
    
    cfg = registry.get(server_name)
    if not cfg:
        print(f"Server '{server_name}' not found!")
        return False
        
    try:
        with SlopClient(cfg) as client:
            print(f"Connecting to {server_name}...")
            info = client.get_server_info()
            print(f"Server: {info.gpu_name}")
            
            print(f"\nGenerating with {description}...")
            result = client.generate(
                prompt="a professional headshot of a person",
                num_steps=20,
                model_id=model_id,
                capture_latents=True,
                capture_noise=True,
                capture_timesteps=True
            )
            
            print(f"\n✓ Generation complete!")
            print(f"  Steps captured: {len(result)}")
            print(f"  Latent shape: {result.latent_shape}")
            print(f"  Image size: {len(result.image) if result.image else 0} bytes")
            
            # Check trajectory
            if result.trajectory:
                step_0 = result.get_step(0)
                step_mid = result.get_step(len(result)//2)
                step_final = result.get_step(-1)
                
                print(f"\n  Trajectory sample:")
                print(f"    Step 0 (noise): timestep={step_0.timestep}, latent_mean={step_0.latent.mean():.4f}")
                print(f"    Step {len(result)//2}: timestep={step_mid.timestep}, latent_mean={step_mid.latent.mean():.4f}")
                print(f"    Step {len(result)-1} (final): timestep={step_final.timestep}, latent_mean={step_final.latent.mean():.4f}")
            
            return True
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    server = "vast-auto-test"
    
    # Test different models
    models = [
        ("runwayml/stable-diffusion-v1-5", "Stable Diffusion 1.5"),
        ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL"),
    ]
    
    results = []
    for model_id, description in models:
        success = test_model(server, model_id, description)
        results.append((description, success))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {desc}")
