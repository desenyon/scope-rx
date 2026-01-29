"""
LIME - Local Interpretable Model-agnostic Explanations.

Reference: "Why Should I Trust You?": Explaining the Predictions of Any Classifier
https://arxiv.org/abs/1602.04938
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class LIME(BaseExplainer):
    """LIME - Local Interpretable Model-agnostic Explanations.
    
    LIME explains predictions by learning a simple interpretable model
    (linear) locally around the instance being explained. For images,
    it perturbs the image by turning superpixels on/off.
    
    Attributes:
        model: PyTorch model to explain.
        num_samples: Number of perturbed samples.
        num_segments: Number of superpixels.
        
    Example:
        >>> from scope_rx.methods.model_agnostic import LIME
        >>> lime = LIME(model, num_samples=1000, num_segments=50)
        >>> result = lime.explain(input_tensor, target_class=243)
        
    References:
        Ribeiro et al., "Why Should I Trust You?": Explaining the Predictions 
        of Any Classifier", KDD 2016.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_segments: int = 50,
        kernel_width: float = 0.25,
        device: Optional[torch.device] = None
    ):
        """Initialize LIME.
        
        Args:
            model: PyTorch model to explain.
            num_samples: Number of perturbed samples.
            num_segments: Target number of superpixels.
            kernel_width: Width of exponential kernel.
            device: Device to use.
        """
        super().__init__(model, device)
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.kernel_width = kernel_width
    
    def _segment_image(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Segment image into superpixels."""
        from skimage.segmentation import slic  # type: ignore[import-not-found]
        
        image = input_tensor.squeeze().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        image_min, image_max = image.min(), image.max()
        if image_max - image_min > 1e-8:
            image = (image - image_min) / (image_max - image_min)
        
        segments = slic(
            image,
            n_segments=self.num_segments,
            compactness=10,
            start_label=0
        )
        
        return segments
    
    def _exponential_kernel(self, distance: float) -> float:
        """Compute exponential kernel weight."""
        return np.exp(-(distance ** 2) / (self.kernel_width ** 2))
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        hide_color: Optional[float] = None,
        show_progress: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """Generate LIME explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            hide_color: Color to use for hidden superpixels (default: mean).
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with LIME feature importance.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        _, _, height, width = input_shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Segment image
        segments = self._segment_image(input_tensor)
        num_segments = segments.max() + 1
        
        # Determine hide color
        if hide_color is None:
            hide_color = input_tensor.mean().item()
        
        # Generate perturbed samples
        data, labels, weights = self._generate_samples(
            input_tensor,
            segments,
            num_segments,
            target_class,
            hide_color,
            show_progress
        )
        
        # Fit interpretable model
        coefficients = self._fit_interpretable_model(data, labels, weights)
        
        # Create saliency map
        saliency = np.zeros((height, width))
        for i in range(num_segments):
            saliency[segments == i] = coefficients[i]
        
        # Normalize
        sal_min, sal_max = saliency.min(), saliency.max()
        if sal_max - sal_min > 1e-8:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return ExplanationResult(
            attribution=saliency,
            method="LIME",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "num_samples": self.num_samples,
                "num_segments": num_segments,
                "coefficients": coefficients,
                "kernel_width": self.kernel_width
            }
        )
    
    def _generate_samples(
        self,
        input_tensor: torch.Tensor,
        segments: np.ndarray,
        num_segments: int,
        target_class: int,
        hide_color: float,
        show_progress: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate perturbed samples and collect predictions."""
        data = []
        labels = []
        weights = []
        
        # Always include original
        data.append(np.ones(num_segments))
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = F.softmax(output, dim=1)[0, target_class].item()
            labels.append(prob)
        
        weights.append(1.0)
        
        # Generate random perturbations
        iterator = range(self.num_samples - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="LIME")
        
        for _ in iterator:
            # Random binary mask
            num_active = np.random.randint(1, num_segments)
            active_segments = np.random.choice(
                num_segments, num_active, replace=False
            )
            
            z = np.zeros(num_segments)
            z[active_segments] = 1
            data.append(z)
            
            # Create perturbed input
            perturbed = self._apply_perturbation(
                input_tensor, segments, z, hide_color
            )
            
            # Get prediction
            with torch.no_grad():
                output = self.model(perturbed)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                labels.append(prob)
            
            # Compute weight (distance in binary space)
            distance = np.sqrt(num_segments - num_active)
            weights.append(self._exponential_kernel(distance / num_segments))
        
        return np.array(data), np.array(labels), np.array(weights)
    
    def _apply_perturbation(
        self,
        input_tensor: torch.Tensor,
        segments: np.ndarray,
        z: np.ndarray,
        hide_color: float
    ) -> torch.Tensor:
        """Apply perturbation mask to input."""
        perturbed = input_tensor.clone()
        
        for i in range(len(z)):
            if z[i] == 0:
                mask = torch.from_numpy(segments == i).to(self.device)
                perturbed[:, :, mask] = hide_color
        
        return perturbed
    
    def _fit_interpretable_model(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """Fit weighted ridge regression."""
        # Normalize weights
        weights = weights / weights.max()
        
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(data, labels, sample_weight=weights)
        
        return model.coef_
