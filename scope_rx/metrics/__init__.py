"""
Evaluation metrics for explanation quality.

This module provides metrics to evaluate the quality of explanations:
- Faithfulness metrics (insertion, deletion, AOPC)
- Sensitivity metrics
- Stability metrics
- Localization metrics
"""

from scope_rx.metrics.faithfulness import (
    aopc_score,
    faithfulness_score,
    insertion_deletion_auc,
    sufficiency_score,
)
from scope_rx.metrics.sensitivity import (
    avg_sensitivity,
    max_sensitivity,
    sensitivity_score,
)
from scope_rx.metrics.stability import (
    explanation_consistency,
    stability_score,
)

__all__ = [
    # Faithfulness
    "insertion_deletion_auc",
    "faithfulness_score",
    "aopc_score",
    "sufficiency_score",

    # Sensitivity
    "sensitivity_score",
    "max_sensitivity",
    "avg_sensitivity",

    # Stability
    "stability_score",
    "explanation_consistency",
]
