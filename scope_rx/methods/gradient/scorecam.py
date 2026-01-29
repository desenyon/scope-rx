"""
ScoreCAM - Score-weighted Class Activation Mapping.

A gradient-free CAM method that uses the score increase as weights.
Reference: "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"
https://arxiv.org/abs/1910.01279
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class ScoreCAM(BaseExplainer):
    """Score-weighted Class Activation Mapping (ScoreCAM).
    
    ScoreCAM is a gradient-free CAM method that uses the increase in
    confidence score when an activation map is applied to the input
    as the weight for that activation map.
    
    This makes it more robust to gradient saturation issues.
    
    Attributes:
        model: PyTorch model to explain.
        target_layer: The convolutional layer to compute CAM from.
        batch_size: Batch size for processing activation maps.
        
    Example:
        >>> from scope_rx.methods.gradient import ScoreCAM
        >>> scorecam = ScoreCAM(model, model.layer4)
        >>> result = scorecam.explain(input_tensor, target_class=243)
        
    References:
        Wang et al., "Score-CAM: Score-Weighted Visual Explanations for 
        Convolutional Neural Networks", CVPRW 2020.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        """Initialize ScoreCAM.
        
        Args:
            model: PyTorch model to explain.
            target_layer: Target convolutional layer.
            batch_size: Batch size for processing.
            device: Device to use.
        """
        super().__init__(model, device)
        self.target_layer = target_layer
        self.batch_size = batch_size
        
        self.activations: Optional[torch.Tensor] = None
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up forward hook to capture activations."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        self._register_hook(self.target_layer, forward_hook, "forward")
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        show_progress: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate ScoreCAM explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the ScoreCAM attribution map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Get activations
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if self.activations is None:
            raise RuntimeError("Activations not captured.")
        
        # Compute ScoreCAM
        cam = self._compute_cam(
            input_tensor, 
            target_class, 
            show_progress
        )
        
        return ExplanationResult(
            attribution=cam,
            method="ScoreCAM",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "target_layer": str(self.target_layer),
                "batch_size": self.batch_size
            }
        )
    
    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        show_progress: bool = False
    ) -> np.ndarray:
        """Compute ScoreCAM activation map."""
        activations = self.activations
        
        if activations is None:
            raise RuntimeError("No activations captured")
        
        _, num_channels, _, _ = activations.shape
        input_h, input_w = input_tensor.shape[2:]
        
        # Upsample activations to input size
        upsampled_acts = F.interpolate(
            activations,
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize each activation map to [0, 1]
        upsampled_acts = upsampled_acts.squeeze(0)  # (C, H, W)
        
        acts_min = upsampled_acts.view(num_channels, -1).min(dim=1)[0].view(-1, 1, 1)
        acts_max = upsampled_acts.view(num_channels, -1).max(dim=1)[0].view(-1, 1, 1)
        
        eps = 1e-8
        normalized_acts = (upsampled_acts - acts_min) / (acts_max - acts_min + eps)
        
        # Compute scores for each masked input
        scores = torch.zeros(num_channels, device=self.device)
        
        iterator = range(0, num_channels, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="ScoreCAM")
        
        with torch.no_grad():
            for i in iterator:
                batch_end = min(i + self.batch_size, num_channels)
                masks = normalized_acts[i:batch_end].unsqueeze(1)  # (B, 1, H, W)
                
                # Apply masks to input
                masked_inputs = input_tensor * masks  # (B, C, H, W)
                
                # Get predictions
                outputs = self.model(masked_inputs)
                
                # Get softmax scores for target class
                probs = F.softmax(outputs, dim=1)
                scores[i:batch_end] = probs[:, target_class]
        
        # Normalize scores
        scores = scores - scores.min()
        scores = scores / (scores.max() + eps)
        
        # Weighted sum of activations
        weights = scores.view(-1, 1, 1)
        cam = torch.sum(weights * normalized_acts, dim=0)
        cam = F.relu(cam)
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > eps:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
