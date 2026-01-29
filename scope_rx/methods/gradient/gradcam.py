"""
GradCAM - Gradient-weighted Class Activation Mapping.

Implementation of the GradCAM method for visualizing CNN predictions.
Reference: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
https://arxiv.org/abs/1610.02391
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scope_rx.core.base import BaseExplainer, ExplanationResult


class GradCAM(BaseExplainer):
    """Gradient-weighted Class Activation Mapping (GradCAM).
    
    GradCAM produces coarse localization maps highlighting important regions
    in the image for predicting a concept. It uses gradients flowing into
    the final convolutional layer to produce a coarse localization map.
    
    Attributes:
        model: PyTorch model to explain.
        target_layer: The convolutional layer to compute CAM from.
        
    Example:
        >>> from scope_rx.methods.gradient import GradCAM
        >>> gradcam = GradCAM(model, model.layer4)
        >>> result = gradcam.explain(input_tensor, target_class=243)
        >>> result.visualize()
        
    References:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
        via Gradient-based Localization", ICCV 2017.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_relu: bool = True,
        device: Optional[torch.device] = None
    ):
        """Initialize GradCAM.
        
        Args:
            model: PyTorch model to explain.
            target_layer: Target convolutional layer.
            use_relu: Whether to apply ReLU to the CAM.
            device: Device to use.
        """
        super().__init__(model, device)
        self.target_layer = target_layer
        self.use_relu = use_relu
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Register hooks
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up forward and backward hooks on the target layer."""
        
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
        """Generate GradCAM explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index. If None, uses predicted class.
            **kwargs: Additional arguments (unused).
            
        Returns:
            ExplanationResult with the GradCAM attribution map.
        """
        # Validate input
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        # Get target class and confidence
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Forward pass (hooks capture activations)
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass for target class
        score = output[:, target_class]
        score.backward(retain_graph=True)
        
        # Compute CAM
        cam = self._compute_cam(tuple(input_tensor.shape[2:]))  # type: ignore[arg-type]
        
        return ExplanationResult(
            attribution=cam,
            method="GradCAM",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "target_layer": str(self.target_layer),
                "use_relu": self.use_relu,
            }
        )
    
    def _compute_cam(
        self,
        input_size: Tuple[int, int]
    ) -> np.ndarray:
        """Compute the class activation map.
        
        Args:
            input_size: Original input (H, W) for resizing.
            
        Returns:
            CAM as numpy array.
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Check hooks.")
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1)
        
        # Apply ReLU if specified
        if self.use_relu:
            cam = F.relu(cam)
        
        # Convert to numpy and process
        cam = cam.squeeze().cpu().numpy()
        
        # Handle negative values if ReLU not applied
        if not self.use_relu:
            cam = np.maximum(cam, 0)
        
        # Resize to input size
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def __repr__(self) -> str:
        return f"GradCAM(target_layer={self.target_layer.__class__.__name__})"
