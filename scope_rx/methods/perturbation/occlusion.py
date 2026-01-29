"""
Occlusion Sensitivity - Sliding window occlusion analysis.

Reference: "Visualizing and Understanding Convolutional Networks"
https://arxiv.org/abs/1311.2901
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class OcclusionSensitivity(BaseExplainer):
    """Occlusion Sensitivity - Sliding window occlusion analysis.
    
    Systematically occludes regions of the input and measures the
    effect on the model's output. Regions where occlusion causes
    the largest drop in confidence are most important.
    
    Attributes:
        model: PyTorch model to explain.
        patch_size: Size of the occlusion patch.
        stride: Stride for sliding the patch.
        occlusion_value: Value to use for occlusion.
        
    Example:
        >>> from scope_rx.methods.perturbation import OcclusionSensitivity
        >>> occlusion = OcclusionSensitivity(model, patch_size=16, stride=8)
        >>> result = occlusion.explain(input_tensor, target_class=243)
        
    References:
        Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks", 
        ECCV 2014.
    """
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 16,
        stride: int = 8,
        occlusion_value: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """Initialize Occlusion Sensitivity.
        
        Args:
            model: PyTorch model to explain.
            patch_size: Size of occlusion patch (square).
            stride: Stride for sliding window.
            occlusion_value: Value to fill occluded regions.
            device: Device to use.
        """
        super().__init__(model, device)
        self.patch_size = patch_size
        self.stride = stride
        self.occlusion_value = occlusion_value
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate Occlusion Sensitivity explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            batch_size: Batch size for processing occluded inputs.
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the occlusion sensitivity map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        _, _, height, width = input_shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            baseline_score = F.softmax(output, dim=1)[0, target_class].item()
        
        # Generate all patch positions
        positions = []
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                positions.append((y, x))
        
        # Compute sensitivity for each position
        sensitivities = np.zeros((height, width))
        counts = np.zeros((height, width))
        
        iterator = range(0, len(positions), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Occlusion")
        
        with torch.no_grad():
            for batch_start in iterator:
                batch_end = min(batch_start + batch_size, len(positions))
                batch_positions = positions[batch_start:batch_end]
                
                # Create batch of occluded inputs
                occluded_batch = input_tensor.repeat(len(batch_positions), 1, 1, 1)
                
                for i, (y, x) in enumerate(batch_positions):
                    occluded_batch[i, :, y:y+self.patch_size, x:x+self.patch_size] = self.occlusion_value
                
                # Get predictions
                outputs = self.model(occluded_batch)
                scores = F.softmax(outputs, dim=1)[:, target_class]
                
                # Update sensitivity map
                for i, (y, x) in enumerate(batch_positions):
                    sensitivity = baseline_score - scores[i].item()
                    sensitivities[y:y+self.patch_size, x:x+self.patch_size] += sensitivity
                    counts[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Average overlapping regions
        counts = np.maximum(counts, 1)
        sensitivity_map = sensitivities / counts
        
        # Normalize
        sens_min, sens_max = sensitivity_map.min(), sensitivity_map.max()
        if sens_max - sens_min > 1e-8:
            sensitivity_map = (sensitivity_map - sens_min) / (sens_max - sens_min)
        else:
            sensitivity_map = np.zeros_like(sensitivity_map)
        
        return ExplanationResult(
            attribution=sensitivity_map,
            method="OcclusionSensitivity",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "patch_size": self.patch_size,
                "stride": self.stride,
                "occlusion_value": self.occlusion_value,
                "baseline_score": baseline_score,
                "num_positions": len(positions)
            }
        )
