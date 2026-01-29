"""
Faithfulness metrics for explanation evaluation.

These metrics measure how faithfully an explanation reflects
the model's decision process.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def insertion_deletion_auc(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    num_steps: int = 100,
    baseline: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """Compute insertion and deletion AUC scores.
    
    Insertion: Start from baseline, gradually insert most important pixels.
               Higher AUC = better explanation.
    
    Deletion: Start from input, gradually delete most important pixels.
              Lower AUC = better explanation.
    
    Args:
        model: PyTorch model.
        input_tensor: Input tensor (N, C, H, W).
        attribution: Attribution map (H, W).
        target_class: Target class index.
        num_steps: Number of steps for the curve.
        baseline: Baseline for insertion/deletion (default: blurred).
        device: Device to use.
        
    Returns:
        Tuple of (insertion_auc, deletion_auc).
    """
    if device is None:
        device = next(model.parameters()).device
    
    input_tensor = input_tensor.to(device)
    _, channels, _, _ = input_tensor.shape
    
    # Create baseline if not provided
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)
    
    # Flatten and sort attribution (make contiguous copy to avoid negative stride)
    attr_flat = np.ascontiguousarray(attribution.flatten())
    sorted_indices = np.argsort(attr_flat)[::-1].copy()  # Descending order, copy for contiguous
    
    # Number of pixels per step
    total_pixels = len(attr_flat)
    pixels_per_step = max(1, total_pixels // num_steps)
    
    insertion_scores = []
    deletion_scores = []
    
    # Current states
    insertion_img = baseline.clone()
    deletion_img = input_tensor.clone()
    
    input_flat = input_tensor.view(channels, -1).clone()
    baseline_flat = baseline.view(channels, -1).clone()
    
    with torch.no_grad():
        for step in range(num_steps + 1):
            # Compute scores
            insertion_out = model(insertion_img)
            deletion_out = model(deletion_img)
            
            insertion_prob = F.softmax(insertion_out, dim=1)[0, target_class].item()
            deletion_prob = F.softmax(deletion_out, dim=1)[0, target_class].item()
            
            insertion_scores.append(insertion_prob)
            deletion_scores.append(deletion_prob)
            
            if step < num_steps:
                # Update images
                start_idx = step * pixels_per_step
                end_idx = min((step + 1) * pixels_per_step, total_pixels)
                pixel_indices = sorted_indices[start_idx:end_idx]
                
                # Insert pixels into insertion image
                insertion_img_flat = insertion_img.view(channels, -1).clone()
                insertion_img_flat[:, pixel_indices] = input_flat[:, pixel_indices]
                insertion_img = insertion_img_flat.view_as(input_tensor)
                
                # Delete pixels from deletion image
                deletion_img_flat = deletion_img.view(channels, -1).clone()
                deletion_img_flat[:, pixel_indices] = baseline_flat[:, pixel_indices]
                deletion_img = deletion_img_flat.view_as(input_tensor)
    
    # Compute AUC using trapezoidal rule
    x = np.linspace(0, 1, len(insertion_scores))
    insertion_auc = float(np.trapezoid(insertion_scores, x))
    deletion_auc = float(np.trapezoid(deletion_scores, x))
    
    return insertion_auc, deletion_auc


def faithfulness_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    **kwargs: Any
) -> float:
    """Compute overall faithfulness score.
    
    Combines insertion and deletion scores into a single metric.
    Higher is better.
    
    Args:
        model: PyTorch model.
        input_tensor: Input tensor.
        attribution: Attribution map.
        target_class: Target class index.
        **kwargs: Additional arguments for insertion_deletion_auc.
        
    Returns:
        Faithfulness score in [0, 1].
    """
    insertion_auc, deletion_auc = insertion_deletion_auc(
        model, input_tensor, attribution, target_class, **kwargs
    )
    
    # Higher insertion and lower deletion is better
    # Normalize to [0, 1]
    faithfulness = (insertion_auc + (1 - deletion_auc)) / 2
    
    return faithfulness


def aopc_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    percentages: List[int] = [10, 20, 30, 40, 50],
    device: Optional[torch.device] = None
) -> float:
    """Compute Area Over Perturbation Curve (AOPC).
    
    Measures the average drop in confidence when removing
    the top-k% most important pixels.
    
    Args:
        model: PyTorch model.
        input_tensor: Input tensor.
        attribution: Attribution map.
        target_class: Target class index.
        percentages: Percentages of pixels to remove.
        device: Device to use.
        
    Returns:
        AOPC score (higher = better explanation).
    """
    if device is None:
        device = next(model.parameters()).device
    
    input_tensor = input_tensor.to(device)
    
    # Get baseline confidence
    with torch.no_grad():
        output = model(input_tensor)
        baseline_prob = F.softmax(output, dim=1)[0, target_class].item()
    
    # Flatten and sort attribution
    attr_flat = attribution.flatten()
    sorted_indices = np.argsort(attr_flat)[::-1]
    
    drops = []
    
    for pct in percentages:
        # Remove top pct% pixels
        num_remove = int(len(attr_flat) * pct / 100)
        remove_indices = sorted_indices[:num_remove]
        
        # Create perturbed input
        perturbed = input_tensor.clone()
        perturbed_flat = perturbed.view(perturbed.shape[1], -1)
        perturbed_flat[:, remove_indices] = 0
        perturbed = perturbed_flat.view_as(input_tensor)
        
        # Get perturbed confidence
        with torch.no_grad():
            output = model(perturbed)
            perturbed_prob = F.softmax(output, dim=1)[0, target_class].item()
        
        drops.append(baseline_prob - perturbed_prob)
    
    return float(np.mean(drops))


def sufficiency_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    percentage: float = 20,
    device: Optional[torch.device] = None
) -> float:
    """Compute sufficiency score.
    
    Measures how much of the original prediction is preserved
    when keeping only the top-k% most important pixels.
    
    Args:
        model: PyTorch model.
        input_tensor: Input tensor.
        attribution: Attribution map.
        target_class: Target class index.
        percentage: Percentage of pixels to keep.
        device: Device to use.
        
    Returns:
        Sufficiency score (higher = better).
    """
    if device is None:
        device = next(model.parameters()).device
    
    input_tensor = input_tensor.to(device)
    
    # Get baseline confidence
    with torch.no_grad():
        output = model(input_tensor)
        baseline_prob = F.softmax(output, dim=1)[0, target_class].item()
    
    # Keep only top percentage pixels
    attr_flat = attribution.flatten()
    sorted_indices = np.argsort(attr_flat)[::-1]
    num_keep = int(len(attr_flat) * percentage / 100)
    keep_indices = sorted_indices[:num_keep]
    
    # Create masked input
    mask = np.zeros_like(attr_flat)
    mask[keep_indices] = 1
    mask = mask.reshape(attribution.shape)
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    
    masked_input = input_tensor * mask_tensor
    
    # Get masked confidence
    with torch.no_grad():
        output = model(masked_input)
        masked_prob = F.softmax(output, dim=1)[0, target_class].item()
    
    # Sufficiency = preserved probability ratio
    sufficiency = masked_prob / (baseline_prob + 1e-8)
    
    return min(sufficiency, 1.0)
