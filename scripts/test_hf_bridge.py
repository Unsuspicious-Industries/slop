
#!/usr/bin/env python3
"""Test the HF bridge with different model types."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders.hf_bridge import load_hf_model
import numpy as np

def test_model(model_id: str):
    """Test a model through the bridge."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print('='*60)
    
    try:
        # Load model
        print("Loading model...")
        bridge = load_hf_model(model_id)
        
        # Print info
        info = bridge.get_model_info()
        print(f"\nModel Info:")
        print(f"  Type: {info['model_type']}")
        print(f"  Device: {info['device']}")
        print(f"  Embedding dim: {info['embedding_dim']}")
        print(f"  Capabilities:")
        for cap, enabled in info['capabilities'].items():
            print(f"    - {cap}: {'enabled' if enabled else 'disabled'}")
        
        # Test text encoding if available
        if info['capabilities']['text_encoding']:
            print("\nTesting text encoding...")
            test_texts = ["a person", "a dog", "a car"]
            embeddings = bridge.encode_text(test_texts)
            print(f"  Encoded {len(test_texts)} texts -> shape {embeddings.shape}")
            
            # Check similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(embeddings)
            print(f"  Cosine similarities:")
            for i, text_i in enumerate(test_texts):
                for j, text_j in enumerate(test_texts):
                    if i < j:
                        print(f"    '{text_i}' <-> '{text_j}': {sims[i,j]:.3f}")
        
        # Test image encoding if available
        if info['capabilities']['image_encoding']:
            print("\nTesting image encoding...")
            from PIL import Image
            # Create dummy images
            img = Image.new('RGB', (224, 224), color='red')
            emb = bridge.encode_image(img)
            print(f"  Encoded image -> shape {emb.shape}")
        
        print("\nTest passed!")
        return bridge
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("HF Model Bridge Test Suite")
    print("="*60)
    
    # Test different model types
    models_to_test = [
        "openai/clip-vit-base-patch32",  # CLIP
        # "facebook/dinov2-base",  # DINOv2 (uncomment if needed)
        # "bert-base-uncased",  # BERT (uncomment if needed)
        # "gpt2",  # GPT-2 (uncomment if needed)
    ]
    
    bridges = {}
    for model_id in models_to_test:
        bridge = test_model(model_id)
        if bridge:
            bridges[model_id] = bridge
    
    # Comparison test
    if len(bridges) > 1:
        print(f"\n{'='*60}")
        print("Cross-model comparison")
        print('='*60)
        
        test_prompt = "a person walking"
        print(f"\nPrompt: '{test_prompt}'")
        
        for model_id, bridge in bridges.items():
            if bridge.get_model_info()['capabilities']['text_encoding']:
                emb = bridge.encode_text(test_prompt)
                print(f"\n{model_id}:")
                print(f"  Embedding shape: {emb.shape}")
                print(f"  L2 norm: {np.linalg.norm(emb):.3f}")
                print(f"  Mean: {emb.mean():.6f}")
                print(f"  Std: {emb.std():.6f}")
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)

if __name__ == "__main__":
    main()
