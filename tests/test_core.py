"""
Tests for ScopeRX core module.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from scope_rx.core.base import ExplanationResult
from scope_rx.core.wrapper import ModelWrapper, auto_detect_target_layer
from scope_rx.core.scope import ScopeRX


class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


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


class TestExplanationResult:
    """Tests for ExplanationResult."""
    
    def test_creation(self):
        """Test basic creation."""
        attribution = np.random.rand(32, 32)
        result = ExplanationResult(
            attribution=attribution,
            method="test",
            target_class=5
        )
        
        assert result.attribution.shape == (32, 32)
        assert result.method == "test"
        assert result.target_class == 5
    
    def test_normalized_attribution(self):
        """Test normalization property."""
        attribution = np.array([[0, 10], [5, 20]])
        result = ExplanationResult(
            attribution=attribution,
            method="test",
            target_class=0
        )
        
        normalized = result.normalized_attribution
        assert normalized.min() == 0
        assert normalized.max() == 1
    
    def test_shape_property(self):
        """Test shape property."""
        attribution = np.array([[0.1, 0.5], [0.3, 0.9]])
        result = ExplanationResult(
            attribution=attribution,
            method="test",
            target_class=0
        )
        
        assert result.shape == (2, 2)


class TestModelWrapper:
    """Tests for ModelWrapper."""
    
    def test_wrapping(self, model: SimpleConvNet) -> None:
        """Test model wrapping."""
        wrapper = ModelWrapper(model)
        assert wrapper.model is model
    
    def test_get_layer(self, model: SimpleConvNet) -> None:
        """Test get_layer method."""
        wrapper = ModelWrapper(model)
        layer = wrapper.get_layer("conv2")
        
        assert isinstance(layer, nn.Conv2d)
    
    def test_auto_detect_layer(self, model: SimpleConvNet) -> None:
        """Test layer auto-detection."""
        layer_name = auto_detect_target_layer(model)
        # Should detect conv2 as last conv layer
        assert layer_name is not None


class TestScopeRX:
    """Tests for ScopeRX main class."""
    
    def test_creation(self, model: SimpleConvNet) -> None:
        """Test ScopeRX creation."""
        scope = ScopeRX(model)
        assert scope.model is model
    
    def test_available_methods(self, model: SimpleConvNet) -> None:
        """Test listing available methods."""
        scope = ScopeRX(model)
        methods = scope.available_methods()
        
        assert "gradcam" in methods
        assert "smoothgrad" in methods
    
    def test_explain(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test basic explanation."""
        scope = ScopeRX(model)
        
        result = scope.explain(
            input_tensor,
            method="gradcam",
            target_class=0,
            target_layer="conv2"
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.target_class == 0
    
    def test_compare_methods(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test method comparison."""
        scope = ScopeRX(model)
        
        comparison = scope.compare_methods(
            input_tensor,
            methods=["vanilla"],
            target_class=0
        )
        
        # ComparisonResult has a results dict
        assert hasattr(comparison, 'results')
        assert "vanilla" in comparison.results
        assert isinstance(comparison.results["vanilla"], ExplanationResult)
