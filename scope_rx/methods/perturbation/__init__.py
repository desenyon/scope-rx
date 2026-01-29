"""
Perturbation-based explanation methods.

This module provides perturbation-based attribution methods including:
- Occlusion Sensitivity
- RISE (Randomized Input Sampling)
- Meaningful Perturbation
"""

from scope_rx.methods.perturbation.occlusion import OcclusionSensitivity
from scope_rx.methods.perturbation.rise import RISE
from scope_rx.methods.perturbation.meaningful_perturbation import MeaningfulPerturbation

__all__ = [
    "OcclusionSensitivity",
    "RISE",
    "MeaningfulPerturbation",
]
