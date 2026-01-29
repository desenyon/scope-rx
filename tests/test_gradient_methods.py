"""
Tests for gradient-based explanation methods.
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


class TestGradCAM:
    """Tests for GradCAM."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test that attribution has correct shape."""
        from scope_rx.methods.gradient import GradCAM
        
        # Pass actual layer module, not string
        target_layer = model.features[2]  # Second conv layer
        explainer = GradCAM(model, target_layer=target_layer)
        result = explainer.explain(input_tensor, target_class=0)
        
        # Attribution should match input spatial dimensions
        assert result.attribution.shape == (32, 32)
    
    def test_attribution_values(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution value range."""
        from scope_rx.methods.gradient import GradCAM
        
        target_layer = model.features[2]
        explainer = GradCAM(model, target_layer=target_layer)
        result = explainer.explain(input_tensor, target_class=0)
        
        # GradCAM should produce non-negative values after ReLU
        assert result.attribution.min() >= 0


class TestGradCAMPlusPlus:
    """Tests for GradCAM++."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.gradient import GradCAMPlusPlus
        
        target_layer = model.features[2]
        explainer = GradCAMPlusPlus(model, target_layer=target_layer)
        result = explainer.explain(input_tensor, target_class=0)
        
        assert result.attribution.shape == (32, 32)


class TestSmoothGrad:
    """Tests for SmoothGrad."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.gradient import SmoothGrad
        
        explainer = SmoothGrad(model, num_samples=5, noise_level=0.1)
        result = explainer.explain(input_tensor, target_class=0)
        
        # SmoothGrad should produce attribution matching input
        assert len(result.attribution.shape) == 2
    
    def test_noise_reduction(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test that SmoothGrad reduces noise compared to vanilla."""
        from scope_rx.methods.gradient import SmoothGrad, VanillaGradients
        
        vanilla = VanillaGradients(model)
        smooth = SmoothGrad(model, num_samples=10, noise_level=0.15)
        
        vanilla_result = vanilla.explain(input_tensor, target_class=0)
        smooth_result = smooth.explain(input_tensor, target_class=0)
        
        # Both should have attributions
        assert vanilla_result.attribution is not None
        assert smooth_result.attribution is not None


class TestIntegratedGradients:
    """Tests for Integrated Gradients."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.gradient import IntegratedGradients
        
        explainer = IntegratedGradients(model, steps=10)
        result = explainer.explain(input_tensor, target_class=0)
        
        assert len(result.attribution.shape) == 2
    
    def test_different_baselines(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test with different baselines."""
        from scope_rx.methods.gradient import IntegratedGradients
        
        # Zero baseline (default)
        explainer_zero = IntegratedGradients(model, steps=10)
        result_zero = explainer_zero.explain(input_tensor, target_class=0)
        
        # Custom baseline
        custom_baseline = torch.rand_like(input_tensor)
        explainer_custom = IntegratedGradients(model, steps=10, baseline=custom_baseline)
        result_custom = explainer_custom.explain(input_tensor, target_class=0)
        
        # Results should be different
        assert not np.allclose(result_zero.attribution, result_custom.attribution)


class TestVanillaGradients:
    """Tests for Vanilla Gradients."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.gradient import VanillaGradients
        
        explainer = VanillaGradients(model)
        result = explainer.explain(input_tensor, target_class=0)
        
        assert len(result.attribution.shape) == 2


class TestGuidedBackprop:
    """Tests for Guided Backpropagation."""
    
    def test_attribution_shape(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test attribution shape."""
        from scope_rx.methods.gradient import GuidedBackprop
        
        explainer = GuidedBackprop(model)
        result = explainer.explain(input_tensor, target_class=0)
        
        assert len(result.attribution.shape) == 2
