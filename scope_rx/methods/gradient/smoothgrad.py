"""
SmoothGrad - Noise-smoothed gradient visualization.

Implementation of SmoothGrad for reducing noise in gradient-based visualizations.
Reference: "SmoothGrad: removing noise by adding noise"
https://arxiv.org/abs/1706.03825
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class SmoothGrad(BaseExplainer):
    """SmoothGrad - Noise-smoothed gradient visualization.
    
    SmoothGrad reduces noise in gradient-based saliency maps by averaging
    gradients computed on multiple noisy versions of the input.
    
    Attributes:
        model: PyTorch model to explain.
        num_samples: Number of noisy samples to average.
        noise_level: Standard deviation of Gaussian noise (as fraction of input range).
        
    Example:
        >>> from scope_rx.methods.gradient import SmoothGrad
        >>> smoothgrad = SmoothGrad(model, num_samples=50, noise_level=0.15)
        >>> result = smoothgrad.explain(input_tensor, target_class=243)
        
    References:
        Smilkov et al., "SmoothGrad: removing noise by adding noise", 
        ICML Workshop 2017.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 50,
        noise_level: float = 0.15,
        device: Optional[torch.device] = None
    ):
        """Initialize SmoothGrad.
        
        Args:
            model: PyTorch model to explain.
            num_samples: Number of noisy samples to average.
            noise_level: Standard deviation of noise as fraction of input range.
            device: Device to use.
        """
        super().__init__(model, device)
        self.num_samples = num_samples
        self.noise_level = noise_level
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        show_progress: bool = False,
        magnitude: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate SmoothGrad explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            show_progress: Whether to show progress bar.
            magnitude: If True, use squared gradients (SmoothGrad²).
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the SmoothGrad attribution map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Compute smoothed gradients
        smooth_grad = self._compute_smooth_grad(
            input_tensor,
            target_class,
            show_progress,
            magnitude
        )
        
        # Convert to 2D saliency map
        if smooth_grad.ndim == 3:
            # Take max across channels
            saliency = np.max(np.abs(smooth_grad), axis=0)
        else:
            saliency = smooth_grad
        
        # Normalize
        saliency_min, saliency_max = saliency.min(), saliency.max()
        if saliency_max - saliency_min > 1e-8:
            saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
        else:
            saliency = np.zeros_like(saliency)
        
        method_name = "SmoothGrad²" if magnitude else "SmoothGrad"
        
        return ExplanationResult(
            attribution=saliency,
            method=method_name,
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "num_samples": self.num_samples,
                "noise_level": self.noise_level,
                "magnitude": magnitude,
                "raw_gradients": smooth_grad  # Store full gradients
            }
        )
    
    def _compute_smooth_grad(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        show_progress: bool = False,
        magnitude: bool = False
    ) -> np.ndarray:
        """Compute smoothed gradients."""
        # Compute noise standard deviation based on input range
        input_range = input_tensor.max() - input_tensor.min()
        stdev = self.noise_level * input_range.item()
        
        accumulated_gradients = torch.zeros_like(input_tensor)
        
        iterator = range(self.num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="SmoothGrad")
        
        for _ in iterator:
            # Add Gaussian noise
            noise = torch.randn_like(input_tensor) * stdev
            noisy_input = (input_tensor + noise).detach().clone()
            noisy_input.requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(noisy_input)
            
            # Backward pass
            score = output[:, target_class]
            score.backward()
            
            if noisy_input.grad is not None:
                grad = noisy_input.grad.detach()
                
                if magnitude:
                    grad = grad ** 2
                
                accumulated_gradients += grad
        
        # Average
        smooth_grad = accumulated_gradients / self.num_samples
        
        return smooth_grad.squeeze().cpu().numpy()
