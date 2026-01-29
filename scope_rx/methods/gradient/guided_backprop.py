"""
Guided Backpropagation - Modified backprop for cleaner visualizations.

Reference: "Striving for Simplicity: The All Convolutional Net"
https://arxiv.org/abs/1412.6806
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from scope_rx.core.base import BaseExplainer, ExplanationResult


class GuidedBackprop(BaseExplainer):
    """Guided Backpropagation for cleaner gradient visualizations.
    
    Guided Backpropagation modifies the backward pass through ReLU layers
    to only propagate gradients that are both positive in the forward
    pass and positive in the backward pass.
    
    This produces sharper, more interpretable visualizations.
    
    Example:
        >>> from scope_rx.methods.gradient import GuidedBackprop
        >>> guided = GuidedBackprop(model)
        >>> result = guided.explain(input_tensor, target_class=243)
        
    References:
        Springenberg et al., "Striving for Simplicity: The All Convolutional Net",
        ICLR Workshop 2015.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize Guided Backpropagation.
        
        Args:
            model: PyTorch model to explain.
            device: Device to use.
        """
        super().__init__(model, device)
        self._relu_outputs: List[torch.Tensor] = []
        self._original_relu_backward = None
    
    def _guided_relu_hook(self, module, grad_input, grad_output):
        """Hook to implement guided backprop through ReLU."""
        return (torch.clamp(grad_output[0], min=0),)
    
    def _register_guided_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """Register guided backprop hooks on all ReLU layers."""
        handles = []
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_backward_hook(self._guided_relu_hook)
                handles.append(handle)
        
        return handles
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """Generate Guided Backpropagation explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the guided backprop saliency map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Register guided backprop hooks
        handles = self._register_guided_hooks()
        
        try:
            # Compute gradients with guided backprop
            input_tensor.requires_grad = True
            self.model.zero_grad()
            
            output = self.model(input_tensor)
            score = output[:, target_class]
            score.backward()
            
            assert input_tensor.grad is not None, "Gradients not computed"
            gradients = input_tensor.grad.detach().squeeze().cpu().numpy()
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        # Process gradients
        gradients = np.abs(gradients)
        
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
            method="GuidedBackprop",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={"raw_gradients": gradients}
        )
