"""
ScopeRX - Neural Network Explainability and Interpretability Library
====================================================================

ScopeRX is a comprehensive Python library for explaining and interpreting
neural network predictions. It provides state-of-the-art attribution methods,
evaluation metrics, and visualization tools.

Quick Start
-----------
>>> from scope_rx import ScopeRX
>>> import torch
>>> import torchvision.models as models
>>>
>>> # Load a model
>>> model = models.resnet50(pretrained=True)
>>> model.eval()
>>>
>>> # Create explainer
>>> explainer = ScopeRX(model)
>>>
>>> # Generate explanation
>>> result = explainer.explain(
...     input_tensor,
...     method='gradcam',
...     target_layer='layer4',
...     target_class=predicted_class
... )
>>>
>>> # Visualize
>>> result.visualize()

Available Methods
-----------------
Gradient-based:
    - GradCAM, GradCAM++, ScoreCAM, LayerCAM
    - SmoothGrad, VanillaGradients
    - IntegratedGradients

Perturbation-based:
    - OcclusionSensitivity
    - RISE (Randomized Input Sampling)
    - MeaningfulPerturbation

Model-agnostic:
    - KernelSHAP
    - LIME

Attention-based:
    - AttentionRollout
    - AttentionFlow
    - RawAttention
"""

__version__ = "2.0.1"
__author__ = "desenyon"
__email__ = "desenyon@gmail.com"
__license__ = "MIT"

# Core classes
from scope_rx.core.base import BaseExplainer, ExplanationResult
from scope_rx.core.scope import ScopeRX
from scope_rx.core.wrapper import ModelWrapper

# Attention-based methods
from scope_rx.methods.attention import (
    AttentionFlow,
    AttentionRollout,
    RawAttention,
)

# Gradient-based methods
from scope_rx.methods.gradient import (
    GradCAM,
    GradCAMPlusPlus,
    GuidedBackprop,
    IntegratedGradients,
    LayerCAM,
    ScoreCAM,
    SmoothGrad,
    VanillaGradients,
)

# Model-agnostic methods
from scope_rx.methods.model_agnostic import (
    LIME,
    KernelSHAP,
)

# Perturbation-based methods
from scope_rx.methods.perturbation import (
    RISE,
    MeaningfulPerturbation,
    OcclusionSensitivity,
)

# Metrics
from scope_rx.metrics import (
    faithfulness_score,
    insertion_deletion_auc,
    sensitivity_score,
    stability_score,
)

# Utilities
from scope_rx.utils import (
    load_image,
    normalize_attribution,
    preprocess_image,
    to_numpy,
    to_tensor,
)

# Visualization
from scope_rx.visualization import (
    create_interactive_plot,
    export_visualization,
    overlay_attribution,
    plot_attribution,
    plot_comparison,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Core
    "ScopeRX",
    "BaseExplainer",
    "ExplanationResult",
    "ModelWrapper",

    # Gradient methods
    "GradCAM",
    "GradCAMPlusPlus",
    "ScoreCAM",
    "LayerCAM",
    "SmoothGrad",
    "IntegratedGradients",
    "VanillaGradients",
    "GuidedBackprop",

    # Perturbation methods
    "OcclusionSensitivity",
    "RISE",
    "MeaningfulPerturbation",

    # Model-agnostic
    "KernelSHAP",
    "LIME",

    # Attention
    "AttentionRollout",
    "AttentionFlow",
    "RawAttention",

    # Visualization
    "plot_attribution",
    "plot_comparison",
    "overlay_attribution",
    "create_interactive_plot",
    "export_visualization",

    # Metrics
    "faithfulness_score",
    "sensitivity_score",
    "stability_score",
    "insertion_deletion_auc",

    # Utilities
    "preprocess_image",
    "load_image",
    "normalize_attribution",
    "to_numpy",
    "to_tensor",
]
