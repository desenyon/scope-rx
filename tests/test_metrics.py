"""
Tests for metrics module.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from numpy.typing import NDArray


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


@pytest.fixture
def attribution() -> NDArray[np.float32]:
    """Create a sample attribution."""
    attr = np.random.rand(32, 32).astype(np.float32)
    return attr


class TestFaithfulness:
    """Tests for faithfulness metrics."""
    
    def test_insertion_deletion_auc(
        self, model: SimpleConvNet, input_tensor: Tensor, attribution: NDArray[np.float32]
    ) -> None:
        """Test insertion/deletion AUC."""
        from scope_rx.metrics import insertion_deletion_auc
        
        # Use contiguous copy of attribution to avoid negative stride issues
        attr_copy = np.ascontiguousarray(attribution)
        
        insertion_auc, deletion_auc = insertion_deletion_auc(
            model, input_tensor, attr_copy,
            target_class=0, num_steps=10
        )
        
        assert 0 <= insertion_auc <= 1
        assert 0 <= deletion_auc <= 1
    
    def test_faithfulness_score(
        self, model: SimpleConvNet, input_tensor: Tensor, attribution: NDArray[np.float32]
    ) -> None:
        """Test faithfulness score."""
        from scope_rx.metrics import faithfulness_score
        
        # Use contiguous copy
        attr_copy = np.ascontiguousarray(attribution)
        
        score = faithfulness_score(
            model, input_tensor, attr_copy,
            target_class=0
        )
        
        assert isinstance(score, float)


class TestSensitivity:
    """Tests for sensitivity metrics."""
    
    def test_sensitivity_score(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test sensitivity score."""
        from scope_rx.metrics import sensitivity_score
        from scope_rx import VanillaGradients
        
        # Use VanillaGradients which doesn't need a target layer
        explainer = VanillaGradients(model)
        
        score = sensitivity_score(
            explainer, input_tensor,
            target_class=0, n_perturbations=5
        )
        
        assert isinstance(score, (float, np.floating))
        assert score >= 0


class TestStability:
    """Tests for stability metrics."""
    
    def test_stability_score(self, model: SimpleConvNet, input_tensor: Tensor) -> None:
        """Test stability score."""
        from scope_rx.metrics import stability_score
        from scope_rx import VanillaGradients
        
        # Use VanillaGradients which doesn't need a target layer
        explainer = VanillaGradients(model)
        
        # stability_score expects a list of inputs
        inputs = [input_tensor, input_tensor.clone()]
        score = stability_score(
            explainer, inputs,
            target_class=0
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
