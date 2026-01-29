"""
GradCAM++ - Improved Gradient-weighted Class Activation Mapping.

Implementation of GradCAM++ with better localization using pixel-wise gradient weighting.
Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
https://arxiv.org/abs/1710.11063
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scope_rx.core.base import BaseExplainer, ExplanationResult


class GradCAMPlusPlus(BaseExplainer):
    """GradCAM++ - Improved Gradient-weighted Class Activation Mapping.
    
    GradCAM++ extends GradCAM by using a weighted combination of the positive
    partial derivatives of the last convolutional layer feature maps with
    respect to a specific class score as weights.
    
    This results in better localization, especially for multiple instances
    of the same class in an image.
    
    Attributes:
        model: PyTorch model to explain.
        target_layer: The convolutional layer to compute CAM from.
        
    Example:
        >>> from scope_rx.methods.gradient import GradCAMPlusPlus
        >>> gradcam_pp = GradCAMPlusPlus(model, model.layer4)
        >>> result = gradcam_pp.explain(input_tensor, target_class=243)
        
    References:
        Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for 
        Deep Convolutional Networks", WACV 2018.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize GradCAM++.
        
        Args:
            model: PyTorch model to explain.
            target_layer: Target convolutional layer.
            device: Device to use.
        """
        super().__init__(model, device)
        self.target_layer = target_layer
        
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self._register_hook(self.target_layer, forward_hook, "forward")
        self._register_hook(self.target_layer, backward_hook, "full_backward")
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """Generate GradCAM++ explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the GradCAM++ attribution map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass
        score = output[:, target_class]
        score.backward(retain_graph=True)
        
        # Compute CAM using GradCAM++ weighting
        cam = self._compute_cam(tuple(input_tensor.shape[2:]))  # type: ignore[arg-type]
        
        return ExplanationResult(
            attribution=cam,
            method="GradCAM++",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={"target_layer": str(self.target_layer)}
        )
    
    def _compute_cam(self, input_size: Tuple[int, int]) -> np.ndarray:
        """Compute GradCAM++ activation map."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured.")
        
        # GradCAM++ weighting formula
        # α_k^c = (∂²Y^c / ∂A_k²) / (2 * ∂²Y^c / ∂A_k² + Σ_a,b A_k^{a,b} * ∂³Y^c / ∂A_k³)
        
        grads = self.gradients
        acts = self.activations
        
        # Compute second and third order gradients approximation
        grads_2 = grads ** 2
        grads_3 = grads ** 3
        
        # Compute alpha weights
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)
        
        # Avoid division by zero
        eps = 1e-8
        
        alpha_num = grads_2
        alpha_denom = 2 * grads_2 + sum_acts * grads_3 + eps
        
        alpha = alpha_num / alpha_denom
        
        # Only consider positive gradients
        alpha = alpha * F.relu(grads)
        
        # Compute weights
        weights = torch.sum(alpha, dim=(2, 3), keepdim=True)
        
        # Weighted sum
        cam = torch.sum(weights * acts, dim=1)
        cam = F.relu(cam)
        
        # Convert and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
