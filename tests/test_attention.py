
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn as nn

from scope_rx.methods.attention.flow import AttentionFlow


class MockTransformer(nn.Module):
    def __init__(self, num_layers=2, embed_dim=32, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, 10)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        b, s, e = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            # output is a tuple
            # We want per-head weights for testing
            output = layer(x, x, x, average_attn_weights=False)
            x = output[0]

        # Use CLS token for prediction
        x = x[:, 0]
        return self.fc(x)

class MockViT(MockTransformer):
    def forward(self, x):
        # x: (N, C, H, W)
        b, c, h, w = x.shape
        # Flatten patches: (N, C, H*W) -> (N, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        return super().forward(x)

@pytest.fixture
def transformer_model():
    return MockTransformer()

@pytest.fixture
def vit_model():
    model = MockViT()
    model.eval()
    return model

@pytest.fixture
def input_tensor():
    # Batch size 1, 196 patches (14x14), 32 dim
    return torch.randn(1, 196, 32)

@pytest.fixture
def img_tensor():
    # Batch size 1, 32 channels, 14x14 spatial
    return torch.randn(1, 32, 14, 14)

def test_attention_flow_init(vit_model):
    explainer = AttentionFlow(vit_model)
    assert explainer.model == vit_model
    # Check if hooks are registered
    # Note: hooks are stored in _hooks list in BaseExplainer
    assert len(explainer._hooks) > 0

def test_attention_flow_explain(vit_model, img_tensor):
    explainer = AttentionFlow(vit_model)

    result = explainer.explain(img_tensor)

    assert result.attribution is not None
    # Output shape should match input spatial dims (14, 14)
    assert result.attribution.shape == (14, 14)
    assert result.method == "AttentionFlow"

    # Check if attribution is normalized (0-1) - implied by _flow_to_spatial logic
    assert result.attribution.min() >= 0
    assert result.attribution.max() <= 1

def test_compute_flow_logic(vit_model, img_tensor):
    explainer = AttentionFlow(vit_model)

    # Manually run parts of explain to verify internal state
    explainer._attention_maps.clear()

    with torch.no_grad():
        _ = vit_model(img_tensor)

    assert len(explainer._attention_maps) == 2 # 2 layers

    # Check shape of captured maps
    # (1, num_heads, seq_len, seq_len)
    # seq_len = 196 + 1 (CLS) = 197
    assert explainer._attention_maps[0].shape == (1, 4, 197, 197)

    flow = explainer._compute_flow()
    assert flow.shape == (197,)
    assert np.isclose(flow.sum(), 1.0)
