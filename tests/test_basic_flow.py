import sys
import os
import torch
import numpy as np
from PIL import Image

# Ensure root is in path
sys.path.append(os.getcwd())

from src.slop_model import SlopModel

def test_encoder():
    print("Testing Encoder (CLIP)...")
    try:
        # Use a smaller model if possible, or the default but check if it runs
        # We'll use the default for now but catch errors if download fails/too slow
        # Actually, let's try to mock or use a very small one if we can found one, 
        # but standard clip-vit-base-patch32 is ~600MB.
        # Maybe 'openai/clip-vit-base-patch32' is smaller than large-patch14
        model_id = "openai/clip-vit-base-patch32" 
        model = SlopModel.load(model_id, device="cpu")
        
        text = "a test prompt"
        emb_text = model.encode(text)
        print(f"Text embedding shape: {emb_text.shape}")
        
        img = Image.new('RGB', (224, 224), color='red')
        emb_img = model.encode(img)
        print(f"Image embedding shape: {emb_img.shape}")
        print("Encoder test passed.")
    except Exception as e:
        print(f"Encoder test failed: {e}")

def test_diffusion():
    print("Testing Diffusion (Tiny)...")
    try:
        # Use a tiny model for testing
        model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        # We need to make sure the loader handles this generic ID or we pass kwargs?
        # The loader uses StableDiffusionPipeline.from_pretrained(model_id)
        # tiny-stable-diffusion-torch is a valid model ID for SD pipeline usually.
        
        model = SlopModel.load(model_id, device="cpu")
        
        prompt = "a tiny prompt"
        # Generate just 1 step for speed
        result = model.diffuse(prompt, num_steps=1)
        
        image = result["image"]
        trajectory = result["trajectory"]
        
        print(f"Generated image size: {image.size}")
        print(f"Trajectory length: {len(trajectory)}")
        if len(trajectory) > 0:
            print(f"First step keys: {trajectory[0].keys()}")
            
        print("Diffusion test passed.")
    except Exception as e:
        print(f"Diffusion test failed: {e}")

if __name__ == "__main__":
    test_encoder()
    test_diffusion()
