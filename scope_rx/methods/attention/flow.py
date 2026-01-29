"""
Attention Flow - Graph-based attention attribution.

Uses maximum flow algorithms to compute attention attribution.
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from scope_rx.core.base import BaseExplainer, ExplanationResult


class AttentionFlow(BaseExplainer):
    """Attention Flow - Graph-based attention attribution.
    
    Computes attention flow using graph-based methods to determine
    how information flows from input tokens to the output.
    
    Example:
        >>> from scope_rx.methods.attention import AttentionFlow
        >>> flow = AttentionFlow(vit_model)
        >>> result = flow.explain(input_tensor)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize Attention Flow.
        
        Args:
            model: PyTorch transformer model.
            device: Device to use.
        """
        super().__init__(model, device)
        
        self._attention_maps: List[torch.Tensor] = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up hooks to capture attention weights."""
        
        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attention_maps.append(output[1].detach())
            
        for _, module in self.model.named_modules():
            module_type = type(module).__name__.lower()
            if 'attention' in module_type:
                self._register_hook(module, attention_hook, "forward")
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """Generate Attention Flow explanation.
        
        Args:
            input_tensor: Input tensor.
            target_class: Target class index.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with attention flow map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        self._attention_maps.clear()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self._attention_maps:
            raise RuntimeError("No attention maps captured.")
        
        # Compute attention flow
        flow = self._compute_flow()
        
        # Convert to spatial map
        saliency = self._flow_to_spatial(flow, input_shape)
        
        return ExplanationResult(
            attribution=saliency,
            method="AttentionFlow",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={"num_layers": len(self._attention_maps)}
        )
    
    def _compute_flow(self) -> np.ndarray:
        """Compute attention flow using iterative propagation."""
        # Start with uniform attention to input tokens
        num_tokens = self._attention_maps[0].shape[-1]
        flow = np.ones(num_tokens) / num_tokens
        
        for attention in self._attention_maps:
            # Average across heads
            attn_matrix = attention.mean(dim=1).squeeze().cpu().numpy()
            
            # Propagate flow
            flow = np.dot(attn_matrix.T, flow)
            
            # Normalize
            flow = flow / (flow.sum() + 1e-8)
        
        return flow
    
    def _flow_to_spatial(
        self,
        flow: np.ndarray,
        input_shape: torch.Size
    ) -> np.ndarray:
        """Convert flow to spatial map."""
        import cv2
        
        # Exclude CLS token
        spatial_flow = flow[1:]
        
        # Reshape to grid
        num_patches = len(spatial_flow)
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            patch_size = int(np.ceil(np.sqrt(num_patches)))
            spatial_flow = np.pad(spatial_flow, (0, patch_size * patch_size - num_patches))
        
        saliency = spatial_flow.reshape(patch_size, patch_size)
        
        # Resize
        saliency = cv2.resize(
            saliency,
            (input_shape[3], input_shape[2]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        sal_min, sal_max = saliency.min(), saliency.max()
        if sal_max - sal_min > 1e-8:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return saliency
