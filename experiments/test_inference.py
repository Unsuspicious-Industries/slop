"""Simple inference test script to verify server functionality."""
import argparse
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from client.interface import SlopClient
from client.config import registry

def test_inference(server_name: str):
    print(f"Testing server: {server_name}")
    
    cfg = registry.get(server_name)
    if not cfg:
        print(f"Server '{server_name}' not found in registry.")
        sys.exit(1)
        
    print(f"Connecting to {cfg.host}...")
    
    try:
        with SlopClient(cfg) as client:
            # 1. Info
            print("Getting server info...")
            info = client.get_server_info()
            print(f"Server Info: {info}")
            
            if not info.loaded_models:
                print("No models loaded initially (expected).")
            
            # 2. Generate
            print("\nRunning test generation (this triggers model load)...")
            prompt = "A photo of a cat astronaut"
            # Using SD v1.5 as it is generally open. 
            # Note: SD 2.1 (default) requires accepting license terms on HF Hub.
            res = client.generate(
                prompt=prompt,
                num_steps=20, 
                model_id="runwayml/stable-diffusion-v1-5",
                capture_latents=True
            )
            
            print(f"Generation complete in {res.metadata.get('elapsed_s', 0):.2f}s")
            
            if res.image:
                print(f"  [OK] Image received ({len(res.image)} bytes)")
            else:
                print("  [FAIL] No image received!")
                
            if res.latents is not None:
                print(f"  [OK] Latents received (shape: {res.latents.shape})")
            else:
                print("  [FAIL] No latents received!")
                
            print("\nTest passed!")
            
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("server", help="Server name in registry")
    args = parser.parse_args()
    
    test_inference(args.server)
