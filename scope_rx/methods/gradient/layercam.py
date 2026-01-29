"""
LayerCAM - Layer-wise Class Activation Mapping.

Implementation of LayerCAM for pixel-wise weighted CAMs.
Reference: "LayerCAM: Exploring Hierarchical Class Activation Maps for Localization"
https://ieeexplore.ieee.org/document/9462463
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scope_rx.core.base import BaseExplainer, ExplanationResult


class LayerCAM(BaseExplainer):
    """Layer-wise Class Activation Mapping (LayerCAM).
    
    LayerCAM generates class activation maps by computing element-wise
    products between gradients and activations, providing finer-grained
    spatial weighting compared to GradCAM.
    
    This method can be applied to any layer in the network, making it
    useful for understanding hierarchical representations.
    
    Example:
        >>> from scope_rx.methods.gradient import LayerCAM
        >>> layercam = LayerCAM(model, model.layer4)
        >>> result = layercam.explain(input_tensor, target_class=243)
        
    References:
        Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps 
        for Localization", IEEE TIP 2021.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize LayerCAM.
        
        Args:
            model: PyTorch model to explain.
            target_layer: Target layer for activation maps.
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
        """Generate LayerCAM explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the LayerCAM attribution map.
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
        
        # Compute LayerCAM
        cam = self._compute_cam(tuple(input_tensor.shape[2:]))  # type: ignore[arg-type]
        
        return ExplanationResult(
            attribution=cam,
            method="LayerCAM",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={"target_layer": str(self.target_layer)}
        )
    
    def _compute_cam(self, input_size: Tuple[int, int]) -> np.ndarray:
        """Compute LayerCAM activation map."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured.")
        
        # Element-wise multiplication of positive gradients and activations
        # Then sum across channels
        positive_grads = F.relu(self.gradients)
        cam = torch.sum(positive_grads * self.activations, dim=1)
        cam = F.relu(cam)
        
        # Convert to numpy and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
