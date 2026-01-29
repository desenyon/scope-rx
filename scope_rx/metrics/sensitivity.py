"""
Sensitivity metrics for explanation evaluation.

These metrics measure how sensitive explanations are to input perturbations.
"""

from typing import Any

import numpy as np
import torch

from scope_rx.core.base import BaseExplainer


def sensitivity_score(
    explainer: BaseExplainer,
    input_tensor: torch.Tensor,
    target_class: int,
    num_samples: int = 10,
    noise_level: float = 0.1,
    **explainer_kwargs: Any
) -> float:
    """Compute average sensitivity of explanations to input noise.
    
    Lower sensitivity = more stable explanations.
    
    Args:
        explainer: Explanation method instance.
        input_tensor: Input tensor.
        target_class: Target class index.
        num_samples: Number of noisy samples to test.
        noise_level: Standard deviation of noise.
        **explainer_kwargs: Additional arguments for explainer.
        
    Returns:
        Average sensitivity score (lower = more stable).
    """
    # Get original explanation
    original_result = explainer.explain(
        input_tensor, target_class=target_class, **explainer_kwargs
    )
    original_attr = original_result.normalized_attribution
    
    sensitivities = []
    
    for _ in range(num_samples):
        # Add noise
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        
        # Get noisy explanation
        noisy_result = explainer.explain(
            noisy_input, target_class=target_class, **explainer_kwargs
        )
        noisy_attr = noisy_result.normalized_attribution
        
        # Compute L2 distance
        distance = np.sqrt(np.mean((original_attr - noisy_attr) ** 2))
        sensitivities.append(distance)
    
    return float(np.mean(sensitivities))


def max_sensitivity(
    explainer: BaseExplainer,
    input_tensor: torch.Tensor,
    target_class: int,
    num_samples: int = 10,
    noise_level: float = 0.1,
    **explainer_kwargs: Any
) -> float:
    """Compute maximum sensitivity of explanations to input noise.
    
    Args:
        explainer: Explanation method instance.
        input_tensor: Input tensor.
        target_class: Target class index.
        num_samples: Number of noisy samples to test.
        noise_level: Standard deviation of noise.
        **explainer_kwargs: Additional arguments for explainer.
        
    Returns:
        Maximum sensitivity score.
    """
    original_result = explainer.explain(
        input_tensor, target_class=target_class, **explainer_kwargs
    )
    original_attr = original_result.normalized_attribution
    
    max_distance = 0.0
    
    for _ in range(num_samples):
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        
        noisy_result = explainer.explain(
            noisy_input, target_class=target_class, **explainer_kwargs
        )
        noisy_attr = noisy_result.normalized_attribution
        
        distance = np.sqrt(np.mean((original_attr - noisy_attr) ** 2))
        max_distance = max(max_distance, distance)
    
    return max_distance


def avg_sensitivity(
    explainer: BaseExplainer,
    input_tensor: torch.Tensor,
    target_class: int,
    **kwargs: Any
) -> float:
    """Alias for sensitivity_score."""
    return sensitivity_score(explainer, input_tensor, target_class, **kwargs)
