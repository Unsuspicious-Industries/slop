"""Test latent space extraction from models.

Tests:
- CLIP encoder for text/image embeddings
- DINOv2 encoder for vision
- Embedding extraction and storage
"""

import pytest
import numpy as np
from pathlib import Path
import shutil

from src.encoders.clip_encoder import CLIPEncoder
from src.encoders.dinov2_encoder import DINOv2Encoder
from src.encoders.hf_bridge import HFBridge


class TestCLIPEncoder:
    """Test CLIP encoder for latent space extraction."""
    
    @pytest.fixture
    def encoder(self):
        """Create CLIP encoder."""
        return CLIPEncoder(device="cpu")
    
    def test_text_encoding(self, encoder):
        """Test text encoding to latent space."""
        texts = ["a photo of a cat", "a photo of a dog"]
        embeddings = encoder.encode_text(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Feature dimension
        assert embeddings.dtype == np.float32
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_image_encoding(self, encoder):
        """Test image encoding to latent space."""
        # Create dummy images (RGB)
        images = np.random.randint(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)
        embeddings = encoder.encode_image(images)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        assert embeddings.dtype == np.float32
    
    def test_similarity(self, encoder):
        """Test text-image similarity in latent space."""
        texts = ["a cat", "a dog"]
        images = np.random.randint(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)
        
        text_emb = encoder.encode_text(texts)
        image_emb = encoder.encode_image(images)
        
        # Compute cosine similarity
        similarity = text_emb @ image_emb.T
        
        assert similarity.shape == (2, 2)
        assert similarity.min() >= -1.0
        assert similarity.max() <= 1.0


class TestDINOv2Encoder:
    """Test DINOv2 encoder for vision features."""
    
    @pytest.fixture
    def encoder(self):
        """Create DINOv2 encoder."""
        return DINOv2Encoder(model_size="small", device="cpu")
    
    def test_image_encoding(self, encoder):
        """Test image encoding with DINOv2."""
        images = np.random.randint(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)
        embeddings = encoder.encode(images)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        assert embeddings.dtype == np.float32
    
    def test_batch_encoding(self, encoder):
        """Test batch encoding efficiency."""
        images = np.random.randint(0, 255, size=(8, 224, 224, 3), dtype=np.uint8)
        embeddings = encoder.encode(images, batch_size=4)
        
        assert embeddings.shape[0] == 8
        assert embeddings.shape[1] > 0


class TestHFBridge:
    """Test Hugging Face model bridge."""
    
    def test_clip_loading(self):
        """Test loading CLIP via HF bridge."""
        bridge = HFBridge()
        model, processor = bridge.load_clip(model_name="openai/clip-vit-base-patch32")
        
        assert model is not None
        assert processor is not None
    
    def test_text_encoding(self):
        """Test text encoding via HF bridge."""
        bridge = HFBridge()
        texts = ["hello world", "test prompt"]
        embeddings = bridge.encode_text_clip(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0


class TestEmbeddingStorage:
    """Test embedding storage and retrieval."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for storage."""
        return tmp_path / "embeddings"
    
    def test_save_load_embeddings(self, temp_dir):
        """Test saving and loading embeddings."""
        temp_dir.mkdir()
        
        # Create test embeddings
        embeddings = np.random.randn(100, 512).astype(np.float32)
        labels = np.array(["test"] * 100)
        
        # Save
        np.save(temp_dir / "embeddings.npy", embeddings)
        np.save(temp_dir / "labels.npy", labels)
        
        # Load
        loaded_emb = np.load(temp_dir / "embeddings.npy")
        loaded_labels = np.load(temp_dir / "labels.npy")
        
        assert np.allclose(embeddings, loaded_emb)
        assert np.array_equal(labels, loaded_labels)
    
    def test_embedding_dimensionality(self):
        """Test consistency of embedding dimensions."""
        encoder = CLIPEncoder(device="cpu")
        
        emb1 = encoder.encode_text(["test 1"])
        emb2 = encoder.encode_text(["test 2", "test 3"])
        
        assert emb1.shape[1] == emb2.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
