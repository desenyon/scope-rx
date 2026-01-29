"""
Evaluation metrics for explanation quality.

This module provides metrics to evaluate the quality of explanations:
- Faithfulness metrics (insertion, deletion, AOPC)
- Sensitivity metrics
- Stability metrics
- Localization metrics
"""

from scope_rx.metrics.faithfulness import (
    insertion_deletion_auc,
    faithfulness_score,
    aopc_score,
    sufficiency_score,
)

from scope_rx.metrics.sensitivity import (
    sensitivity_score,
    max_sensitivity,
    avg_sensitivity,
)

from scope_rx.metrics.stability import (
    stability_score,
    explanation_consistency,
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
