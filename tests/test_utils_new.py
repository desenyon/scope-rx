"""
Tests for utility functions.
"""

import numpy as np
import torch
import tempfile
from pathlib import Path


class TestPreprocessing:
    """Tests for preprocessing utilities."""
    
    def test_normalize_image(self):
        """Test image normalization."""
        from scope_rx.utils import normalize_image
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        normalized = normalize_image(image)
        
        # Should change the values
        assert not np.allclose(image, normalized)
    
    def test_denormalize_image(self):
        """Test image denormalization."""
        from scope_rx.utils.preprocessing import normalize_image, denormalize_image
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        normalized = normalize_image(image)
        denormalized = denormalize_image(normalized)
        
        # Should recover approximately the original
        assert np.allclose(image, denormalized, atol=1e-5)
    
    def test_preprocess_image(self):
        """Test full preprocessing pipeline."""
        from scope_rx.utils import preprocess_image
        
        # Create a dummy image file
        import cv2
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
            cv2.imwrite(path, image)
        
        tensor = preprocess_image(path, size=(224, 224))
        
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(tensor, torch.Tensor)
        
        Path(path).unlink()


class TestPostprocessing:
    """Tests for postprocessing utilities."""
    
    def test_normalize_attribution_minmax(self):
        """Test minmax normalization."""
        from scope_rx.utils import normalize_attribution
        
        attr = np.array([[0, 10], [5, 20]])
        normalized = normalize_attribution(attr, method="minmax")
        
        assert normalized.min() == 0
        assert normalized.max() == 1
    
    def test_normalize_attribution_percentile(self):
        """Test percentile normalization."""
        from scope_rx.utils import normalize_attribution
        
        attr = np.random.randn(100, 100)
        normalized = normalize_attribution(attr, method="percentile")
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_smooth_attribution(self):
        """Test smoothing."""
        from scope_rx.utils.postprocessing import smooth_attribution
        
        attr = np.random.rand(32, 32).astype(np.float32)
        smoothed = smooth_attribution(attr, kernel_size=5)
        
        assert smoothed.shape == attr.shape
        # Smoothed should have less variance
        assert smoothed.var() <= attr.var()
    
    def test_threshold_attribution(self):
        """Test thresholding."""
        from scope_rx.utils.postprocessing import threshold_attribution
        
        attr = np.array([[0.1, 0.5], [0.3, 0.9]])
        thresholded = threshold_attribution(attr, threshold=0.4)
        
        assert thresholded[0, 0] == 0
        assert thresholded[0, 1] == 0.5


class TestTensorUtils:
    """Tests for tensor utilities."""
    
    def test_to_numpy(self):
        """Test tensor to numpy conversion."""
        from scope_rx.utils import to_numpy
        
        tensor = torch.randn(3, 32, 32)
        array = to_numpy(tensor)
        
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 32, 32)
    
    def test_to_tensor(self):
        """Test numpy to tensor conversion."""
        from scope_rx.utils import to_tensor
        
        array = np.random.rand(3, 32, 32)
        tensor = to_tensor(array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
    
    def test_ensure_4d(self):
        """Test 4D tensor ensuring."""
        from scope_rx.utils.tensor import ensure_4d
        
        # 2D -> 4D
        t2 = torch.randn(32, 32)
        t4 = ensure_4d(t2)
        assert t4.shape == (1, 1, 32, 32)
        
        # 3D -> 4D
        t3 = torch.randn(3, 32, 32)
        t4 = ensure_4d(t3)
        assert t4.shape == (1, 3, 32, 32)
        
        # 4D unchanged
        t4_orig = torch.randn(2, 3, 32, 32)
        t4 = ensure_4d(t4_orig)
        assert t4.shape == (2, 3, 32, 32)
