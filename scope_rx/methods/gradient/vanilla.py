"""
Vanilla Gradients - Simple input gradient saliency.

The most basic gradient-based attribution method.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from scope_rx.core.base import BaseExplainer, ExplanationResult


class VanillaGradients(BaseExplainer):
    """Vanilla Gradients - Simple input gradient saliency.
    
    Computes the gradient of the output with respect to the input,
    which highlights input features that have the largest effect
    on the output.
    
    Example:
        >>> from scope_rx.methods.gradient import VanillaGradients
        >>> vanilla = VanillaGradients(model)
        >>> result = vanilla.explain(input_tensor, target_class=243)
        
    References:
        Simonyan et al., "Deep Inside Convolutional Networks: Visualising 
        Image Classification Models and Saliency Maps", ICLR Workshop 2014.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize Vanilla Gradients.
        
        Args:
            model: PyTorch model to explain.
            device: Device to use.
        """
        super().__init__(model, device)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        absolute: bool = True,
        **kwargs
    ) -> ExplanationResult:
        """Generate Vanilla Gradients explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            absolute: Whether to take absolute value of gradients.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the gradient saliency map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Compute gradients
        input_tensor = input_tensor.detach().clone()
        input_tensor.requires_grad_(True)
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        score = output[:, target_class]
        score.backward()
        
        assert input_tensor.grad is not None, "Gradients not computed"
        gradients = input_tensor.grad.detach().squeeze().cpu().numpy()
        
        # Process gradients
        if absolute:
            gradients = np.abs(gradients)
        
        # Convert to 2D saliency
        if gradients.ndim == 3:
            saliency = np.max(gradients, axis=0)
        else:
            saliency = gradients
        
        # Normalize
        saliency_min, saliency_max = saliency.min(), saliency.max()
        if saliency_max - saliency_min > 1e-8:
            saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return ExplanationResult(
            attribution=saliency,
            method="VanillaGradients",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "absolute": absolute,
                "raw_gradients": gradients
            }
        )
