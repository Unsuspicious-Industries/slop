"""
Test script for FLUX.1-dev trajectory capture.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from client.interface import SlopClient
from client.config import registry

def main():
    print("="*70)
    print("SLOP FLUX.1-dev Trajectory Test")
    print("="*70)
    
    cfg = registry.get("vast-auto-test")
    if not cfg:
        print("Error: Server 'vast-auto-test' not found!")
        return
    
    with SlopClient(cfg) as client:
        # Get server info
        try:
            info = client.get_server_info()
            print(f"\nConnected to: {info.gpu_name}")
            print(f"GPU Memory: {info.gpu_memory_mb} MB")
        except Exception as e:
            print(f"Failed to get server info: {e}")
            # Continue anyway, maybe server is up but info fails
        
        # Generate with FLUX
        print("\n" + "-"*70)
        print("Running FLUX inference...")
        print("Model: black-forest-labs/FLUX.1-dev")
        print("-"*70)
        
        try:
            result = client.generate(
                prompt="A highly detailed cinematic shot of a futuristic cyberpunk city street at night, neon lights, rain reflections, 8k resolution",
                num_steps=10,  # Short run for testing
                model_id="black-forest-labs/FLUX.1-dev",
                seed=42,
                guidance_scale=3.5, # Recommended for FLUX
                capture_latents=True,
                capture_noise=True,
                capture_timesteps=True
            )
            
            # Display results
            print(f"\n✓ Generation Complete!")
            print(f"  Elapsed: {result.metadata.get('elapsed_s', 0):.2f}s")
            print(f"  Steps captured: {len(result)}")
            
            if result.latents is not None:
                print(f"  Latent shape (all steps): {result.latents.shape}")
                # Check if packed or spatial
                if result.latents.ndim == 4:
                    # (steps, batch, seq_len, dim) -> Packed
                    print("  Type: Packed Latents (Transformer sequence)")
                elif result.latents.ndim == 5:
                    # (steps, batch, c, h, w) -> Spatial
                    print("  Type: Spatial Latents (UNet feature map)")
            
            # Save image
            output_dir = Path("results/flux_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if result.image:
                with open(output_dir / "flux_output.png", "wb") as f:
                    f.write(result.image)
                print(f"\n✓ Saved image to {output_dir/'flux_output.png'}")
            else:
                print("\n! No image returned (generation failed?)")
            
            # Inspect first and last latent stats
            if len(result.trajectory) > 0:
                first = result.trajectory[0]
                last = result.trajectory[-1]
                print(f"\nFirst Step Timestep: {first.timestep}")
                print(f"Last Step Timestep: {last.timestep}")
                print(f"First Latent Mean: {first.latent.mean():.4f}")
                print(f"Last Latent Mean: {last.latent.mean():.4f}")

        except Exception as e:
            print(f"\n❌ Inference Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
