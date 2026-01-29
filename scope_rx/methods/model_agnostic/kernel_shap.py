"""
KernelSHAP - SHAP values via weighted linear regression.

A model-agnostic method for computing SHAP values using a weighted
linear regression approach.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class KernelSHAP(BaseExplainer):
    """KernelSHAP - SHAP values via weighted linear regression.
    
    KernelSHAP approximates SHAP values using a weighted linear regression
    approach. It works by sampling coalition subsets of features and
    fitting a linear model weighted by the SHAP kernel.
    
    For images, features are typically superpixels/segments.
    
    Attributes:
        model: PyTorch model to explain.
        num_samples: Number of samples for approximation.
        
    Example:
        >>> from scope_rx.methods.model_agnostic import KernelSHAP
        >>> shap = KernelSHAP(model, num_samples=1000)
        >>> result = shap.explain(input_tensor, target_class=243)
        
    References:
        Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", 
        NeurIPS 2017.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_segments: int = 50,
        device: Optional[torch.device] = None
    ):
        """Initialize KernelSHAP.
        
        Args:
            model: PyTorch model to explain.
            num_samples: Number of coalition samples.
            num_segments: Number of superpixels/segments.
            device: Device to use.
        """
        super().__init__(model, device)
        self.num_samples = num_samples
        self.num_segments = num_segments
    
    def _segment_image(
        self,
        input_tensor: torch.Tensor
    ) -> np.ndarray:
        """Segment image into superpixels using SLIC."""
        from skimage.segmentation import slic  # type: ignore[import-not-found]
        
        # Convert to numpy image
        image = input_tensor.squeeze().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize to [0, 1] for SLIC
        image_min, image_max = image.min(), image.max()
        if image_max - image_min > 1e-8:
            image = (image - image_min) / (image_max - image_min)
        
        # SLIC segmentation
        segments = slic(
            image,
            n_segments=self.num_segments,
            compactness=10,
            start_label=0
        )
        
        return segments
    
    def _shap_kernel(self, M: int, z: np.ndarray) -> np.ndarray:
        """Compute SHAP kernel weights for coalition z.
        
        Args:
            M: Total number of features.
            z: Binary coalition vector.
            
        Returns:
            Kernel weight.
        """
        s = np.sum(z)
        
        if s == 0 or s == M:
            return np.array(1e6)  # Large weight for full/empty coalitions
        
        # SHAP kernel: M-1 / (C(M,s) * s * (M-s))
        from scipy.special import comb
        weight = (M - 1) / (comb(M, s) * s * (M - s))
        
        return weight
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        show_progress: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate KernelSHAP explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            baseline: Baseline input (default: blurred).
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with SHAP values.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        _, _, height, width = input_shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Create baseline (blurred image)
        if baseline is None:
            baseline = self._create_baseline(input_tensor)
        else:
            baseline = baseline.to(self.device)
        
        # Segment image
        segments = self._segment_image(input_tensor)
        num_segments = segments.max() + 1
        
        # Sample coalitions
        coalitions, weights, predictions = self._sample_coalitions(
            input_tensor,
            baseline,
            segments,
            num_segments,
            target_class,
            show_progress
        )
        
        # Fit weighted linear regression
        shap_values = self._fit_linear_model(
            coalitions,
            weights,
            predictions,
            num_segments
        )
        
        # Create saliency map from SHAP values
        saliency = np.zeros((height, width))
        for i in range(num_segments):
            saliency[segments == i] = shap_values[i]
        
        # Normalize
        sal_min, sal_max = saliency.min(), saliency.max()
        if sal_max - sal_min > 1e-8:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return ExplanationResult(
            attribution=saliency,
            method="KernelSHAP",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "num_samples": self.num_samples,
                "num_segments": num_segments,
                "shap_values": shap_values
            }
        )
    
    def _create_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Create baseline by blurring the input."""
        kernel_size = 31
        sigma = 10.0
        
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)
        
        padding = kernel_size // 2
        blurred = F.conv2d(input_tensor, kernel_2d, padding=padding, groups=input_tensor.shape[1])
        
        return blurred
    
    def _sample_coalitions(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        segments: np.ndarray,
        num_segments: int,
        target_class: int,
        show_progress: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample coalitions and compute predictions."""
        coalitions = []
        weights = []
        predictions = []
        
        # Always include empty and full coalitions
        coalitions.append(np.zeros(num_segments))
        weights.append(1e6)
        
        coalitions.append(np.ones(num_segments))
        weights.append(1e6)
        
        # Sample random coalitions
        iterator = range(self.num_samples - 2)
        if show_progress:
            iterator = tqdm(iterator, desc="KernelSHAP")
        
        for _ in iterator:
            # Sample coalition
            z = np.random.binomial(1, 0.5, num_segments)
            coalitions.append(z)
            weights.append(self._shap_kernel(num_segments, z))
        
        coalitions = np.array(coalitions)
        weights = np.array(weights)
        
        # Compute predictions for all coalitions
        with torch.no_grad():
            for z in coalitions:
                # Create masked input
                mask = np.zeros_like(segments, dtype=np.float32)
                for i in range(num_segments):
                    if z[i] == 1:
                        mask[segments == i] = 1
                
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
                masked_input = input_tensor * mask_tensor + baseline * (1 - mask_tensor)
                
                output = self.model(masked_input)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                predictions.append(prob)
        
        return coalitions, weights, np.array(predictions)
    
    def _fit_linear_model(
        self,
        coalitions: np.ndarray,
        weights: np.ndarray,
        predictions: np.ndarray,
        num_segments: int
    ) -> np.ndarray:
        """Fit weighted linear regression to get SHAP values."""
        # Normalize weights
        weights = weights / weights.max()
        
        # Fit ridge regression
        model = Ridge(alpha=0.01, fit_intercept=True)
        model.fit(coalitions, predictions, sample_weight=weights)
        
        return model.coef_
