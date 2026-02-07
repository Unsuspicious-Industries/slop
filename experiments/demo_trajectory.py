"""
Example: Working with Trajectory Data

This script demonstrates how to use the full trajectory capture feature.
"""
import sys
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from client.interface import SlopClient
from client.config import registry

def main():
    print("="*70)
    print("SLOP Trajectory Capture Demo")
    print("="*70)
    
    cfg = registry.get("vast-auto-test")
    if not cfg:
        print("Error: Server 'vast-auto-test' not found!")
        print("Run: python3 client/vastai.py --provision")
        return
    
    with SlopClient(cfg) as client:
        # Get server info
        info = client.get_server_info()
        print(f"\nConnected to: {info.gpu_name}")
        print(f"GPU Memory: {info.gpu_memory_mb} MB")
        
        # Generate with full trajectory
        print("\n" + "-"*70)
        print("Running inference with trajectory capture...")
        print("-"*70)
        
        result = client.generate(
            prompt="a futuristic robot assistant in a lab",
            num_steps=15,  # Quick demo
            model_id="runwayml/stable-diffusion-v1-5",
            seed=42,
            capture_latents=True,
            capture_noise=True,
            capture_timesteps=True
        )
        
        # Display results
        print(f"\n✓ Generation Complete!")
        print(f"  Elapsed: {result.metadata.get('elapsed_s', 0):.2f}s")
        print(f"  Steps captured: {len(result)}")
        print(f"  Latent shape per step: {result.latent_shape}")
        print(f"  Image size: {len(result.image)/1024:.1f} KB")
        
        # Save image
        output_dir = Path("results/trajectory_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "output.png", "wb") as f:
            f.write(result.image)
        print(f"\n✓ Saved image to {output_dir/'output.png'}")
        
        # Analyze trajectory
        print("\n" + "-"*70)
        print("Trajectory Analysis")
        print("-"*70)
        
        print(f"\nArray shapes:")
        print(f"  latents:      {result.latents.shape} (steps, batch, channels, h, w)")
        print(f"  noise_preds:  {result.noise_preds.shape} (steps, batch, channels, h, w)")
        print(f"  timesteps:    {result.timesteps.shape} (steps,)")
        
        print(f"\nTimestep schedule:")
        print(f"  {result.timesteps}")
        
        print(f"\nStep-by-step latent statistics:")
        print(f"  {'Step':<6} {'Timestep':<10} {'Mean':<12} {'Std':<12}")
        print(f"  {'-'*42}")
        
        for step in result.trajectory[:5]:  # Show first 5
            latent = step.latent
            print(f"  {step.step_index:<6} {step.timestep:<10} {latent.mean():<12.4f} {latent.std():<12.4f}")
        print(f"  ...")
        for step in result.trajectory[-3:]:  # Show last 3
            latent = step.latent
            print(f"  {step.step_index:<6} {step.timestep:<10} {latent.mean():<12.4f} {latent.std():<12.4f}")
        
        # Calculate interesting metrics
        print(f"\nInteresting metrics:")
        initial = result.get_step(0).latent[0]  # First batch item
        final = result.get_step(-1).latent[0]
        
        change = np.abs(final - initial)
        print(f"  Mean absolute change: {change.mean():.4f}")
        print(f"  Max change in any pixel: {change.max():.4f}")
        
        # Show trajectory length
        print(f"\n✓ Successfully captured {len(result.trajectory)} trajectory steps")
        print(f"\nOutput saved to: {output_dir}/")
        print("\n" + "="*70)

if __name__ == "__main__":
    main()
