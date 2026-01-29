"""
Attention Rollout - Attention flow through transformer layers.

Reference: "Quantifying Attention Flow in Transformers"
https://arxiv.org/abs/2005.00928
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from scope_rx.core.base import BaseExplainer, ExplanationResult


class AttentionRollout(BaseExplainer):
    """Attention Rollout - Combining attention across transformer layers.
    
    Attention Rollout computes how attention flows through a transformer
    by multiplying attention matrices across layers, accounting for
    residual connections.
    
    Attributes:
        model: PyTorch transformer model to explain.
        head_fusion: How to combine attention heads ('mean', 'max', 'min').
        discard_ratio: Ratio of lowest attentions to discard.
        
    Example:
        >>> from scope_rx.methods.attention import AttentionRollout
        >>> rollout = AttentionRollout(vit_model)
        >>> result = rollout.explain(input_tensor, target_class=243)
        
    References:
        Abnar & Zuidema, "Quantifying Attention Flow in Transformers", 
        ACL 2020.
    """
    
    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """Initialize Attention Rollout.
        
        Args:
            model: PyTorch transformer model.
            head_fusion: Method to combine heads ('mean', 'max', 'min').
            discard_ratio: Ratio of lowest attention values to discard.
            device: Device to use.
        """
        super().__init__(model, device)
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        self._attention_maps: List[torch.Tensor] = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up hooks to capture attention weights."""
        
        def attention_hook(module, input, output):
            # Handle different attention output formats
            if isinstance(output, tuple):
                # (output, attention_weights)
                if len(output) > 1 and output[1] is not None:
                    self._attention_maps.append(output[1].detach())
            
        # Find and hook attention modules
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
        """Generate Attention Rollout explanation.
        
        Args:
            input_tensor: Input tensor.
            target_class: Target class (optional, for metadata).
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with attention rollout map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Clear previous attention maps
        self._attention_maps.clear()
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self._attention_maps:
            raise RuntimeError(
                "No attention maps captured. Make sure the model has "
                "attention modules that return attention weights."
            )
        
        # Compute rollout
        rollout = self._compute_rollout()
        
        # Convert to 2D spatial map
        saliency = self._attention_to_spatial(rollout, input_shape)
        
        return ExplanationResult(
            attribution=saliency,
            method="AttentionRollout",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "head_fusion": self.head_fusion,
                "discard_ratio": self.discard_ratio,
                "num_layers": len(self._attention_maps)
            }
        )
    
    def _compute_rollout(self) -> np.ndarray:
        """Compute attention rollout across layers."""
        result = None
        
        for attention in self._attention_maps:
            # attention shape: (batch, heads, seq_len, seq_len)
            
            # Fuse heads
            if self.head_fusion == "mean":
                attention_fused = attention.mean(dim=1)
            elif self.head_fusion == "max":
                attention_fused = attention.max(dim=1)[0]
            elif self.head_fusion == "min":
                attention_fused = attention.min(dim=1)[0]
            else:
                attention_fused = attention.mean(dim=1)
            
            # Discard low attention values
            if self.discard_ratio > 0:
                flat = attention_fused.view(-1)
                threshold = torch.quantile(flat, self.discard_ratio)
                attention_fused[attention_fused < threshold] = 0
            
            # Add residual connection (identity matrix)
            eye = torch.eye(attention_fused.shape[-1], device=attention_fused.device)
            attention_fused = 0.5 * attention_fused + 0.5 * eye
            
            # Re-normalize
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True)
            
            # Accumulate rollout
            if result is None:
                result = attention_fused.cpu().numpy()
            else:
                result = np.matmul(result, attention_fused.cpu().numpy())
        
        assert result is not None, "No attention matrices found"
        return result
    
    def _attention_to_spatial(
        self,
        rollout: np.ndarray,
        input_shape: torch.Size
    ) -> np.ndarray:
        """Convert attention rollout to spatial map."""
        # Get attention to CLS token (first token)
        # rollout shape: (batch, seq_len, seq_len)
        cls_attention = rollout[0, 0, 1:]  # Exclude CLS token itself
        
        # Determine spatial dimensions
        # Assume square patch grid
        num_patches = len(cls_attention)
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            # Not a perfect square, use closest
            patch_size = int(np.ceil(np.sqrt(num_patches)))
            cls_attention = np.pad(
                cls_attention,
                (0, patch_size * patch_size - num_patches)
            )
        
        # Reshape to spatial grid
        saliency = cls_attention.reshape(patch_size, patch_size)
        
        # Resize to input spatial dimensions
        import cv2
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
