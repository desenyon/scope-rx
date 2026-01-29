"""
Gradient-based explanation methods.

This module provides various gradient-based attribution methods including:
- GradCAM and variants (GradCAM++, ScoreCAM, LayerCAM)
- SmoothGrad
- Integrated Gradients
- Vanilla Gradients
- Guided Backpropagation
"""

from scope_rx.methods.gradient.gradcam import GradCAM
from scope_rx.methods.gradient.gradcam_plusplus import GradCAMPlusPlus
from scope_rx.methods.gradient.scorecam import ScoreCAM
from scope_rx.methods.gradient.layercam import LayerCAM
from scope_rx.methods.gradient.smoothgrad import SmoothGrad
from scope_rx.methods.gradient.integrated_gradients import IntegratedGradients
from scope_rx.methods.gradient.vanilla import VanillaGradients
from scope_rx.methods.gradient.guided_backprop import GuidedBackprop

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "ScoreCAM",
    "LayerCAM",
    "SmoothGrad",
    "IntegratedGradients",
    "VanillaGradients",
    "GuidedBackprop",
]
