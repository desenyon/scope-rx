"""
Tests for visualization module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def attribution():
    """Create a sample attribution."""
    return np.random.rand(224, 224)


@pytest.fixture
def image():
    """Create a sample image."""
    return (np.random.rand(224, 224, 3) * 255).astype(np.uint8)


class TestPlots:
    """Tests for plotting functions."""
    
    def test_overlay_attribution(self, attribution, image):
        """Test overlay creation."""
        from scope_rx.visualization import overlay_attribution
        
        overlay = overlay_attribution(attribution, image, alpha=0.5)
        
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8
    
    def test_plot_attribution(self, attribution, image):
        """Test plot creation."""
        from scope_rx.visualization import plot_attribution
        
        fig = plot_attribution(attribution, image, show=False)
        
        assert fig is not None
    
    def test_plot_comparison(self, attribution, image):
        """Test comparison plot."""
        from scope_rx.visualization import plot_comparison
        
        attributions = {
            "method1": attribution,
            "method2": np.random.rand(224, 224),
        }
        
        fig = plot_comparison(attributions, image, show=False)
        
        assert fig is not None


class TestExport:
    """Tests for export functions."""
    
    def test_export_png(self, attribution):
        """Test PNG export."""
        from scope_rx.visualization import export_visualization
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        
        export_visualization(attribution, path)
        
        assert Path(path).exists()
        Path(path).unlink()
    
    def test_export_npy(self, attribution):
        """Test NPY export."""
        from scope_rx.visualization import export_visualization
        
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        
        export_visualization(attribution, path)
        
        assert Path(path).exists()
        
        # Verify content
        loaded = np.load(path)
        assert np.allclose(loaded, attribution)
        
        Path(path).unlink()
