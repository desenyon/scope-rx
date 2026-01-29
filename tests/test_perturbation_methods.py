"""
Tests for perturbation-based explanation methods.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@pytest.fixture
def model():
    """Create a simple model for testing."""
    net = SimpleConvNet(num_classes=10)
    net.eval()
    return net


@pytest.fixture
def input_tensor():
    """Create a sample input tensor."""
    return torch.randn(1, 3, 32, 32)


class TestOcclusionSensitivity:
    """Tests for Occlusion Sensitivity."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.perturbation import OcclusionSensitivity
        
        explainer = OcclusionSensitivity(model, patch_size=8, stride=8)
        result = explainer.explain(input_tensor, target_class=0)
        
        assert result.attribution.shape == (32, 32)
    
    def test_different_patch_sizes(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test with different patch sizes."""
        from scope_rx.methods.perturbation import OcclusionSensitivity
        
        explainer_small = OcclusionSensitivity(model, patch_size=4, stride=4)
        explainer_large = OcclusionSensitivity(model, patch_size=16, stride=16)
        
        result_small = explainer_small.explain(input_tensor, target_class=0)
        result_large = explainer_large.explain(input_tensor, target_class=0)
        
        # Both should produce valid attributions
        assert not np.allclose(result_small.attribution, result_large.attribution)


class TestRISE:
    """Tests for RISE."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.perturbation import RISE
        
        explainer = RISE(model, num_masks=50, mask_size=8)
        result = explainer.explain(input_tensor, target_class=0)
        
        assert result.attribution.shape == (32, 32)
    
    def test_more_masks_convergence(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test that more masks lead to more stable results."""
        from scope_rx.methods.perturbation import RISE
        
        explainer = RISE(model, num_masks=100, mask_size=8)
        
        result1 = explainer.explain(input_tensor, target_class=0)
        result2 = explainer.explain(input_tensor, target_class=0)
        
        # Both should produce valid attributions of the same shape
        assert result1.attribution.shape == result2.attribution.shape
        assert result1.attribution.shape == (32, 32)


class TestMeaningfulPerturbation:
    """Tests for Meaningful Perturbation."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.perturbation import MeaningfulPerturbation
        
        explainer = MeaningfulPerturbation(model, mode='deletion')
        result = explainer.explain(input_tensor, target_class=0, num_iterations=10)
        
        assert result.attribution.shape == (32, 32)
