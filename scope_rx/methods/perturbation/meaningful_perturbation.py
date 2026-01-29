"""
Meaningful Perturbation - Optimization-based perturbation explanations.

Reference: "Interpretable Explanations of Black Boxes by Meaningful Perturbation"
https://arxiv.org/abs/1704.03296
"""

from typing import Optional, Tuple, Literal

# numpy imported below via torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class MeaningfulPerturbation(BaseExplainer):
    """Meaningful Perturbation - Optimization-based explanations.
    
    Finds the smallest region of the input that, when removed or preserved,
    maximally affects the model's output. Uses optimization to find a
    smooth mask that best explains the prediction.
    
    Attributes:
        model: PyTorch model to explain.
        mode: 'deletion' (find what to remove) or 'preservation' (find what to keep).
        
    Example:
        >>> from scope_rx.methods.perturbation import MeaningfulPerturbation
        >>> mp = MeaningfulPerturbation(model, mode='deletion')
        >>> result = mp.explain(input_tensor, target_class=243)
        
    References:
        Fong & Vedaldi, "Interpretable Explanations of Black Boxes by 
        Meaningful Perturbation", ICCV 2017.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mode: Literal["deletion", "preservation"] = "deletion",
        device: Optional[torch.device] = None
    ):
        """Initialize Meaningful Perturbation.
        
        Args:
            model: PyTorch model to explain.
            mode: 'deletion' or 'preservation'.
            device: Device to use.
        """
        super().__init__(model, device)
        self.mode = mode
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        num_iterations: int = 300,
        learning_rate: float = 0.1,
        l1_coeff: float = 0.01,
        tv_coeff: float = 0.2,
        blur_sigma: float = 10.0,
        mask_size: Optional[Tuple[int, int]] = None,
        show_progress: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate Meaningful Perturbation explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            num_iterations: Number of optimization iterations.
            learning_rate: Learning rate for optimization.
            l1_coeff: Coefficient for L1 regularization (mask sparsity).
            tv_coeff: Coefficient for total variation regularization.
            blur_sigma: Sigma for Gaussian blur perturbation.
            mask_size: Size for learnable mask (None = full resolution).
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the optimized mask.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        _, _, height, width = input_shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Create blurred version as perturbation
        blurred = self._gaussian_blur(input_tensor, blur_sigma)
        
        # Initialize mask
        if mask_size is None:
            mask_size = (height, width)
        
        mask = torch.ones(1, 1, *mask_size, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([mask], lr=learning_rate)
        
        iterator = range(num_iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="MeaningfulPerturbation")
        
        losses = []
        
        for _ in iterator:
            optimizer.zero_grad()
            
            # Upsample mask if needed
            if mask_size != (height, width):
                upsampled_mask = F.interpolate(
                    mask, size=(height, width), mode='bilinear', align_corners=False
                )
            else:
                upsampled_mask = mask
            
            # Clamp mask to [0, 1]
            clamped_mask = torch.sigmoid(mask)
            if mask_size != (height, width):
                upsampled_mask = F.interpolate(
                    clamped_mask, size=(height, width), mode='bilinear', align_corners=False
                )
            else:
                upsampled_mask = clamped_mask
            
            # Apply mask
            if self.mode == "deletion":
                perturbed = input_tensor * (1 - upsampled_mask) + blurred * upsampled_mask
            else:  # preservation
                perturbed = input_tensor * upsampled_mask + blurred * (1 - upsampled_mask)
            
            # Compute output
            output = self.model(perturbed)
            
            # Classification loss
            if self.mode == "deletion":
                # Maximize confidence drop
                class_loss = F.softmax(output, dim=1)[0, target_class]
            else:
                # Minimize confidence drop
                class_loss = -F.softmax(output, dim=1)[0, target_class]
            
            # Regularization
            l1_loss = l1_coeff * torch.mean(torch.abs(clamped_mask))
            tv_loss = tv_coeff * self._total_variation(clamped_mask)
            
            # Total loss
            loss = class_loss + l1_loss + tv_loss
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        # Get final mask
        with torch.no_grad():
            final_mask = torch.sigmoid(mask)
            if mask_size != (height, width):
                final_mask = F.interpolate(
                    final_mask, size=(height, width), mode='bilinear', align_corners=False
                )
            
            saliency = final_mask.squeeze().cpu().numpy()
        
        # For deletion mode, invert so high values = important
        if self.mode == "deletion":
            saliency = 1 - saliency
        
        return ExplanationResult(
            attribution=saliency,
            method=f"MeaningfulPerturbation ({self.mode})",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "mode": self.mode,
                "num_iterations": num_iterations,
                "l1_coeff": l1_coeff,
                "tv_coeff": tv_coeff,
                "losses": losses
            }
        )
    
    def _gaussian_blur(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to tensor."""
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(tensor.shape[1], 1, 1, 1)
        
        # Apply padding
        padding = kernel_size // 2
        
        blurred = F.conv2d(tensor, kernel_2d, padding=padding, groups=tensor.shape[1])
        
        return blurred
    
    def _total_variation(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute total variation of mask."""
        tv_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
        tv_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
        return tv_h + tv_w
