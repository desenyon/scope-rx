"""
RISE - Randomized Input Sampling for Explanation.

Reference: "RISE: Randomized Input Sampling for Explanation of Black-box Models"
https://arxiv.org/abs/1806.07421
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class RISE(BaseExplainer):
    """RISE - Randomized Input Sampling for Explanation.
    
    RISE generates explanations by probing the model with randomly
    masked versions of the input and weighting the masks by the
    model's output probability.
    
    This is a black-box method that doesn't require access to
    model gradients or internal structure.
    
    Attributes:
        model: PyTorch model to explain.
        num_masks: Number of random masks to use.
        mask_size: Size of the mask before upsampling.
        probability: Probability of each cell being 1 (unmasked).
        
    Example:
        >>> from scope_rx.methods.perturbation import RISE
        >>> rise = RISE(model, num_masks=4000, mask_size=7)
        >>> result = rise.explain(input_tensor, target_class=243)
        
    References:
        Petsiuk et al., "RISE: Randomized Input Sampling for Explanation 
        of Black-box Models", BMVC 2018.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_masks: int = 4000,
        mask_size: int = 7,
        probability: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """Initialize RISE.
        
        Args:
            model: PyTorch model to explain.
            num_masks: Number of random masks to generate.
            mask_size: Grid size for mask generation.
            probability: Probability of cell being unmasked.
            device: Device to use.
        """
        super().__init__(model, device)
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.probability = probability
        
        self._masks: Optional[torch.Tensor] = None
    
    def _generate_masks(
        self,
        input_size: Tuple[int, int],
        batch_size: int = 100
    ) -> torch.Tensor:
        """Generate random masks.
        
        Args:
            input_size: Size of input (H, W).
            batch_size: Batch size for generation.
            
        Returns:
            Tensor of masks of shape (num_masks, 1, H, W).
        """
        h, w = input_size
        
        # Cell size with upsampling
        cell_h = np.ceil(h / self.mask_size)
        cell_w = np.ceil(w / self.mask_size)
        
        # Upsampled size (slightly larger for random shifts)
        up_h = int((self.mask_size + 1) * cell_h)
        up_w = int((self.mask_size + 1) * cell_w)
        
        masks = []
        
        for _ in range(0, self.num_masks, batch_size):
            batch_count = min(batch_size, self.num_masks - len(masks))
            
            # Generate binary grid
            grid = (torch.rand(batch_count, 1, self.mask_size, self.mask_size) 
                    < self.probability).float()
            
            # Upsample
            grid = F.interpolate(grid, size=(up_h, up_w), mode='bilinear', align_corners=False)
            
            # Random crop to input size
            for i in range(batch_count):
                y = np.random.randint(0, up_h - h + 1)
                x = np.random.randint(0, up_w - w + 1)
                mask = grid[i:i+1, :, y:y+h, x:x+w]
                masks.append(mask)
        
        return torch.cat(masks, dim=0).to(self.device)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 100,
        show_progress: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate RISE explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the RISE saliency map.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        _, _, height, width = input_shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Generate masks
        masks = self._generate_masks((height, width), batch_size)
        
        # Compute saliency
        saliency = torch.zeros(height, width, device=self.device)
        
        iterator = range(0, self.num_masks, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RISE")
        
        with torch.no_grad():
            for i in iterator:
                batch_end = min(i + batch_size, self.num_masks)
                batch_masks = masks[i:batch_end]
                
                # Apply masks to input
                masked_inputs = input_tensor * batch_masks
                
                # Get predictions
                outputs = self.model(masked_inputs)
                scores = F.softmax(outputs, dim=1)[:, target_class]
                
                # Weight masks by scores
                for j, score in enumerate(scores):
                    saliency += score * batch_masks[j, 0]
        
        # Normalize
        saliency = saliency / self.num_masks
        saliency = saliency.cpu().numpy()
        
        sal_min, sal_max = saliency.min(), saliency.max()
        if sal_max - sal_min > 1e-8:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return ExplanationResult(
            attribution=saliency,
            method="RISE",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "num_masks": self.num_masks,
                "mask_size": self.mask_size,
                "probability": self.probability
            }
        )
