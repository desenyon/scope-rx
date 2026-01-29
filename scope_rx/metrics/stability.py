"""
Stability metrics for explanation evaluation.

These metrics measure the stability and consistency of explanations.
"""

from typing import Any, List

import numpy as np
import torch

from scope_rx.core.base import BaseExplainer


def stability_score(
    explainer: BaseExplainer,
    inputs: List[torch.Tensor],
    target_class: int,
    **explainer_kwargs: Any
) -> float:
    """Compute stability of explanations across similar inputs.
    
    Higher score = more consistent explanations for similar inputs.
    
    Args:
        explainer: Explanation method instance.
        inputs: List of similar input tensors.
        target_class: Target class index.
        **explainer_kwargs: Additional arguments for explainer.
        
    Returns:
        Stability score in [0, 1].
    """
    if len(inputs) < 2:
        return 1.0
    
    attributions = []
    
    for inp in inputs:
        result = explainer.explain(
            inp, target_class=target_class, **explainer_kwargs
        )
        attributions.append(result.normalized_attribution)
    
    # Compute pairwise similarities
    similarities = []
    
    for i in range(len(attributions)):
        for j in range(i + 1, len(attributions)):
            # Pearson correlation
            attr_i = attributions[i].flatten()
            attr_j = attributions[j].flatten()
            
            corr = np.corrcoef(attr_i, attr_j)[0, 1]
            
            if not np.isnan(corr):
                similarities.append((corr + 1) / 2)  # Map to [0, 1]
    
    return float(np.mean(similarities)) if similarities else 0.0


def explanation_consistency(
    explainer: BaseExplainer,
    input_tensor: torch.Tensor,
    target_class: int,
    num_runs: int = 5,
    **explainer_kwargs: Any
) -> float:
    """Compute consistency of explanations across multiple runs.
    
    Measures if the explanation method is deterministic.
    
    Args:
        explainer: Explanation method instance.
        input_tensor: Input tensor.
        target_class: Target class index.
        num_runs: Number of runs to compare.
        **explainer_kwargs: Additional arguments for explainer.
        
    Returns:
        Consistency score in [0, 1] (1 = perfectly deterministic).
    """
    if num_runs < 2:
        return 1.0
    
    attributions = []
    
    for _ in range(num_runs):
        result = explainer.explain(
            input_tensor, target_class=target_class, **explainer_kwargs
        )
        attributions.append(result.attribution)
    
    # Compute variance across runs
    stacked = np.stack(attributions, axis=0)
    variance = np.var(stacked, axis=0)
    
    # Average variance
    avg_variance = np.mean(variance)
    
    # Convert to consistency score
    # Lower variance = higher consistency
    consistency = np.exp(-avg_variance * 10)
    
    return consistency
