"""
Raw Attention - Direct attention weight extraction.

Simple extraction of attention weights from transformer models.
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from scope_rx.core.base import BaseExplainer, ExplanationResult


class RawAttention(BaseExplainer):
    """Raw Attention - Direct attention weight visualization.
    
    Extracts and visualizes raw attention weights from a specific
    layer and head of a transformer model.
    
    Example:
        >>> from scope_rx.methods.attention import RawAttention
        >>> raw_attn = RawAttention(vit_model, layer_index=-1)
        >>> result = raw_attn.explain(input_tensor)
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_index: int = -1,
        head_index: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize Raw Attention.
        
        Args:
            model: PyTorch transformer model.
            layer_index: Which layer's attention to extract (-1 for last).
            head_index: Which head to visualize (None for average).
            device: Device to use.
        """
        super().__init__(model, device)
        self.layer_index = layer_index
        self.head_index = head_index
        
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
        """Generate Raw Attention explanation.
        
        Args:
            input_tensor: Input tensor.
            target_class: Target class index.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with raw attention map.
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
        
        # Get specific layer
        attention = self._attention_maps[self.layer_index]
        
        # Get specific head or average
        if self.head_index is not None:
            attention = attention[:, self.head_index:self.head_index+1, :, :]
        
        attention = attention.mean(dim=1).squeeze()  # (seq_len, seq_len)
        
        # Get attention to CLS token
        cls_attention = attention[0, 1:].cpu().numpy()
        
        # Convert to spatial map
        saliency = self._attention_to_spatial(cls_attention, input_shape)
        
        return ExplanationResult(
            attribution=saliency,
            method="RawAttention",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "layer_index": self.layer_index,
                "head_index": self.head_index,
                "num_layers": len(self._attention_maps)
            }
        )
    
    def _attention_to_spatial(
        self,
        attention: np.ndarray,
        input_shape: torch.Size
    ) -> np.ndarray:
        """Convert attention vector to spatial map."""
        import cv2
        
        num_patches = len(attention)
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            patch_size = int(np.ceil(np.sqrt(num_patches)))
            attention = np.pad(attention, (0, patch_size * patch_size - num_patches))
        
        saliency = attention.reshape(patch_size, patch_size)
        
        saliency = cv2.resize(
            saliency,
            (input_shape[3], input_shape[2]),
            interpolation=cv2.INTER_LINEAR
        )
        
        sal_min, sal_max = saliency.min(), saliency.max()
        if sal_max - sal_min > 1e-8:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return saliency
