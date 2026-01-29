"""
Integrated Gradients - Attribution via integral approximation.

Implementation of Integrated Gradients for feature attribution.
Reference: "Axiomatic Attribution for Deep Networks"
https://arxiv.org/abs/1703.01365
"""

from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scope_rx.core.base import BaseExplainer, ExplanationResult


class IntegratedGradients(BaseExplainer):
    """Integrated Gradients - Attribution via integral approximation.
    
    Integrated Gradients computes the attribution of each input feature
    by integrating the gradients of the model's output with respect to
    the input along a path from a baseline to the input.
    
    This method satisfies important axioms: Sensitivity and Implementation
    Invariance.
    
    Attributes:
        model: PyTorch model to explain.
        steps: Number of steps for the integral approximation.
        baseline: Baseline input for integration path.
        
    Example:
        >>> from scope_rx.methods.gradient import IntegratedGradients
        >>> ig = IntegratedGradients(model, steps=100)
        >>> result = ig.explain(input_tensor, target_class=243)
        
    References:
        Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
    """
    
    def __init__(
        self,
        model: nn.Module,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
        method: Literal["riemann", "gauss_legendre"] = "riemann",
        device: Optional[torch.device] = None
    ):
        """Initialize Integrated Gradients.
        
        Args:
            model: PyTorch model to explain.
            steps: Number of steps for integral approximation.
            baseline: Baseline input. If None, uses zeros.
            method: Integration method ('riemann' or 'gauss_legendre').
            device: Device to use.
        """
        super().__init__(model, device)
        self.steps = steps
        self.baseline = baseline
        self.method = method
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        show_progress: bool = False,
        internal_batch_size: int = 1,
        **kwargs
    ) -> ExplanationResult:
        """Generate Integrated Gradients explanation.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class index.
            baseline: Override baseline for this explanation.
            show_progress: Whether to show progress bar.
            internal_batch_size: Batch size for processing steps.
            **kwargs: Additional arguments.
            
        Returns:
            ExplanationResult with the Integrated Gradients attribution.
        """
        input_tensor = self._validate_input(input_tensor)
        input_shape = input_tensor.shape
        
        target_class, confidence = self._get_target_class(input_tensor, target_class)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Determine baseline
        if baseline is not None:
            use_baseline = baseline.to(self.device)
        elif self.baseline is not None:
            use_baseline = self.baseline.to(self.device)
        else:
            use_baseline = torch.zeros_like(input_tensor)
        
        # Compute integrated gradients
        attributions = self._compute_integrated_gradients(
            input_tensor,
            use_baseline,
            target_class,
            show_progress,
            internal_batch_size
        )
        
        # Convert to 2D saliency map
        if attributions.ndim == 3:
            saliency = np.sum(np.abs(attributions), axis=0)
        else:
            saliency = attributions
        
        # Normalize
        saliency_min, saliency_max = saliency.min(), saliency.max()
        if saliency_max - saliency_min > 1e-8:
            saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
        else:
            saliency = np.zeros_like(saliency)
        
        return ExplanationResult(
            attribution=saliency,
            method="IntegratedGradients",
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            input_shape=input_shape,
            metadata={
                "steps": self.steps,
                "method": self.method,
                "raw_attributions": attributions,
                "convergence_delta": self._compute_convergence_delta(
                    input_tensor, use_baseline, attributions, target_class
                )
            }
        )
    
    def _compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        show_progress: bool = False,
        internal_batch_size: int = 1
    ) -> np.ndarray:
        """Compute integrated gradients using specified method."""
        if self.method == "gauss_legendre":
            return self._compute_gauss_legendre(
                input_tensor, baseline, target_class, show_progress
            )
        else:
            return self._compute_riemann(
                input_tensor, baseline, target_class, show_progress, internal_batch_size
            )
    
    def _compute_riemann(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        show_progress: bool = False,
        internal_batch_size: int = 1
    ) -> np.ndarray:
        """Compute IG using Riemann sum approximation."""
        # Generate alphas
        alphas = torch.linspace(0, 1, self.steps + 1, device=self.device)[1:]
        
        # Compute difference
        diff = input_tensor - baseline
        
        accumulated_gradients = torch.zeros_like(input_tensor)
        
        iterator = range(0, self.steps, internal_batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="IntegratedGradients")
        
        for i in iterator:
            batch_end = min(i + internal_batch_size, self.steps)
            batch_alphas = alphas[i:batch_end]
            
            # Create interpolated inputs
            interpolated = (baseline + batch_alphas.view(-1, 1, 1, 1) * diff).detach().clone()
            interpolated.requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(interpolated)
            
            # Sum scores for target class
            scores = outputs[:, target_class].sum()
            scores.backward()
            
            if interpolated.grad is not None:
                accumulated_gradients += interpolated.grad.sum(dim=0, keepdim=True)
        
        # Average gradients and multiply by difference
        avg_gradients = accumulated_gradients / self.steps
        integrated_gradients = (diff * avg_gradients).squeeze()
        
        return integrated_gradients.cpu().numpy()
    
    def _compute_gauss_legendre(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        show_progress: bool = False
    ) -> np.ndarray:
        """Compute IG using Gauss-Legendre quadrature."""
        # Get Gauss-Legendre points and weights
        points, weights = np.polynomial.legendre.leggauss(self.steps)
        
        # Transform from [-1, 1] to [0, 1]
        points = (points + 1) / 2
        weights = weights / 2
        
        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        diff = input_tensor - baseline
        
        accumulated_gradients = torch.zeros_like(input_tensor)
        
        iterator = range(self.steps)
        if show_progress:
            iterator = tqdm(iterator, desc="IntegratedGradients (GL)")
        
        for i in iterator:
            alpha = points[i]
            weight = weights[i]
            
            interpolated = baseline + alpha * diff
            interpolated.requires_grad = True
            
            self.model.zero_grad()
            output = self.model(interpolated)
            
            score = output[:, target_class]
            score.backward()
            
            if interpolated.grad is not None:
                accumulated_gradients += weight * interpolated.grad
        
        integrated_gradients = (diff * accumulated_gradients).squeeze()
        
        return integrated_gradients.cpu().numpy()
    
    def _compute_convergence_delta(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        attributions: np.ndarray,
        target_class: int
    ) -> float:
        """Compute convergence delta (should be close to 0 for good approximation)."""
        with torch.no_grad():
            input_score = self.model(input_tensor)[:, target_class].item()
            baseline_score = self.model(baseline)[:, target_class].item()
        
        score_diff = input_score - baseline_score
        attribution_sum = np.sum(attributions)
        
        return abs(score_diff - attribution_sum)
