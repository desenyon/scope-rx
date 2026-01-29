"""
Explanation methods for ScopeRX.

This module provides various explanation methods organized by category:
- gradient: Gradient-based methods (GradCAM, SmoothGrad, etc.)
- perturbation: Perturbation-based methods (Occlusion, RISE, etc.)
- model_agnostic: Model-agnostic methods (SHAP, LIME)
- attention: Attention-based methods for transformers
"""

from scope_rx.methods.gradient import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    LayerCAM,
    SmoothGrad,
    IntegratedGradients,
    VanillaGradients,
    GuidedBackprop,
)

from scope_rx.methods.perturbation import (
    OcclusionSensitivity,
    RISE,
    MeaningfulPerturbation,
)

from scope_rx.methods.model_agnostic import (
    KernelSHAP,
    LIME,
)

from scope_rx.methods.attention import (
    AttentionRollout,
    AttentionFlow,
    RawAttention,
)

__all__ = [
    # Gradient-based
    "GradCAM",
    "GradCAMPlusPlus",
    "ScoreCAM",
    "LayerCAM",
    "SmoothGrad",
    "IntegratedGradients",
    "VanillaGradients",
    "GuidedBackprop",
    
    # Perturbation-based
    "OcclusionSensitivity",
    "RISE",
    "MeaningfulPerturbation",
    
    # Model-agnostic
    "KernelSHAP",
    "LIME",
    
    # Attention-based
    "AttentionRollout",
    "AttentionFlow",
    "RawAttention",
]
