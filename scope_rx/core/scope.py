"""
ScopeRX - Main high-level interface.

Provides a unified API for all explanation methods.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-not-found]

import numpy as np
import torch
import torch.nn as nn

from scope_rx.core.base import BaseExplainer, ExplanationResult
from scope_rx.core.wrapper import ModelWrapper, auto_detect_target_layer


class ScopeRX:
    """High-level interface for neural network explanations.
    
    ScopeRX provides a unified API for generating explanations using
    various methods. It handles model wrapping, method selection, and
    provides convenient utilities for comparison and analysis.
    
    Attributes:
        model: The wrapped PyTorch model.
        device: Device being used.
        
    Example:
        >>> from scope_rx import ScopeRX
        >>> import torchvision.models as models
        >>>
        >>> model = models.resnet50(pretrained=True)
        >>> scope = ScopeRX(model)
        >>>
        >>> # Single explanation
        >>> result = scope.explain(image, method='gradcam', target_layer='layer4')
        >>> result.visualize()
        >>>
        >>> # Compare methods
        >>> comparison = scope.compare_methods(
        ...     image,
        ...     methods=['gradcam', 'integrated_gradients', 'lime'],
        ...     target_class=predicted_class
        ... )
        >>> comparison.visualize_all()
    """
    
    # Registry of available methods
    _methods: Dict[str, type] = {}
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        auto_layer: bool = True
    ):
        """Initialize ScopeRX.
        
        Args:
            model: PyTorch model to explain.
            device: Device to use. If None, inferred from model.
            auto_layer: Whether to auto-detect target layer for CAM methods.
        """
        self.wrapper = ModelWrapper(model, device)
        self.model = self.wrapper.model
        self.device = self.wrapper.device
        
        self._default_layer: Optional[str] = None
        if auto_layer:
            self._default_layer = auto_detect_target_layer(model)
        
        # Initialize method registry
        self._init_methods()
        
        # Cache for explainer instances
        self._explainers: Dict[str, BaseExplainer] = {}
    
    def _init_methods(self):
        """Initialize the method registry with lazy imports."""
        self._method_imports = {
            # Gradient-based
            'gradcam': ('scope_rx.methods.gradient', 'GradCAM'),
            'gradcam++': ('scope_rx.methods.gradient', 'GradCAMPlusPlus'),
            'gradcamplusplus': ('scope_rx.methods.gradient', 'GradCAMPlusPlus'),
            'scorecam': ('scope_rx.methods.gradient', 'ScoreCAM'),
            'layercam': ('scope_rx.methods.gradient', 'LayerCAM'),
            'smoothgrad': ('scope_rx.methods.gradient', 'SmoothGrad'),
            'integrated_gradients': ('scope_rx.methods.gradient', 'IntegratedGradients'),
            'ig': ('scope_rx.methods.gradient', 'IntegratedGradients'),
            'vanilla': ('scope_rx.methods.gradient', 'VanillaGradients'),
            'vanilla_gradients': ('scope_rx.methods.gradient', 'VanillaGradients'),
            'guided_backprop': ('scope_rx.methods.gradient', 'GuidedBackprop'),
            
            # Perturbation-based
            'occlusion': ('scope_rx.methods.perturbation', 'OcclusionSensitivity'),
            'rise': ('scope_rx.methods.perturbation', 'RISE'),
            'meaningful_perturbation': ('scope_rx.methods.perturbation', 'MeaningfulPerturbation'),
            
            # Model-agnostic
            'shap': ('scope_rx.methods.model_agnostic', 'KernelSHAP'),
            'kernel_shap': ('scope_rx.methods.model_agnostic', 'KernelSHAP'),
            'lime': ('scope_rx.methods.model_agnostic', 'LIME'),
            
            # Attention-based
            'attention_rollout': ('scope_rx.methods.attention', 'AttentionRollout'),
            'attention_flow': ('scope_rx.methods.attention', 'AttentionFlow'),
            'raw_attention': ('scope_rx.methods.attention', 'RawAttention'),
        }
    
    def _get_explainer_class(self, method: str) -> type:
        """Get the explainer class for a method name."""
        method = method.lower().replace('-', '_').replace(' ', '_')
        
        if method not in self._method_imports:
            available = ', '.join(sorted(set(self._method_imports.keys())))
            raise ValueError(
                f"Unknown method '{method}'. Available methods: {available}"
            )
        
        module_path, class_name = self._method_imports[method]
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Could not import {class_name} from {module_path}. "
                f"Make sure the required dependencies are installed. Error: {e}"
            )
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        method: str = "gradcam",
        target_class: Optional[int] = None,
        target_layer: Optional[str] = None,
        **kwargs: Any
    ) -> ExplanationResult:
        """Generate an explanation using the specified method.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W) or (C, H, W).
            method: Explanation method to use.
            target_class: Target class to explain. If None, uses predicted class.
            target_layer: Target layer for CAM methods. If None, auto-detected.
            **kwargs: Method-specific arguments.
            
        Returns:
            ExplanationResult containing the attribution and metadata.
            
        Example:
            >>> result = scope.explain(
            ...     image,
            ...     method='gradcam',
            ...     target_layer='layer4',
            ...     target_class=243  # Golden retriever
            ... )
        """
        # Validate input
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Get explainer class
        explainer_class = self._get_explainer_class(method)
        
        # Determine if this method needs a target layer
        layer_methods = {'gradcam', 'gradcam++', 'gradcamplusplus', 'scorecam', 'layercam'}
        method_key = method.lower().replace('-', '_').replace(' ', '_')
        
        # Build kwargs for explainer
        explainer_kwargs = {}
        
        if method_key in layer_methods:
            layer = target_layer or self._default_layer
            if layer is None:
                raise ValueError(
                    f"Method '{method}' requires a target_layer. "
                    f"Either provide target_layer or use a model with convolutional layers."
                )
            explainer_kwargs['target_layer'] = self.wrapper.get_layer(layer)
        
        # Add any method-specific kwargs
        explainer_kwargs.update(kwargs.get('explainer_kwargs', {}))
        
        # Create explainer
        explainer = explainer_class(self.model, **explainer_kwargs)
        
        # Generate explanation
        result = explainer.explain(input_tensor, target_class=target_class, **kwargs)
        
        return result
    
    def compare_methods(
        self,
        input_tensor: torch.Tensor,
        methods: List[str],
        target_class: Optional[int] = None,
        target_layer: Optional[str] = None,
        **kwargs: Any
    ) -> 'ComparisonResult':
        """Compare multiple explanation methods on the same input.
        
        Args:
            input_tensor: Input tensor.
            methods: List of method names to compare.
            target_class: Target class to explain.
            target_layer: Target layer for CAM methods.
            **kwargs: Method-specific arguments.
            
        Returns:
            ComparisonResult containing all explanations.
        """
        results = {}
        
        for method in methods:
            try:
                result = self.explain(
                    input_tensor,
                    method=method,
                    target_class=target_class,
                    target_layer=target_layer,
                    **kwargs
                )
                results[method] = result
            except Exception as e:
                print(f"Warning: Failed to generate {method} explanation: {e}")
        
        return ComparisonResult(results, input_tensor)
    
    def available_methods(self) -> List[str]:
        """Get list of available explanation methods.
        
        Returns:
            List of method names.
        """
        # Return unique method names (excluding aliases)
        return sorted(set([
            'gradcam', 'gradcam++', 'scorecam', 'layercam',
            'smoothgrad', 'integrated_gradients', 'vanilla_gradients',
            'guided_backprop', 'occlusion', 'rise', 'meaningful_perturbation',
            'kernel_shap', 'lime', 'attention_rollout', 'attention_flow'
        ]))
    
    def get_predictions(
        self,
        input_tensor: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get model predictions for an input.
        
        Args:
            input_tensor: Input tensor.
            top_k: Number of top predictions to return.
            
        Returns:
            Tuple of (probabilities, class_indices).
        """
        return self.wrapper.predict(input_tensor, top_k)
    
    def __repr__(self) -> str:
        return f"ScopeRX(model={type(self.model).__name__}, device={self.device})"


class ComparisonResult:
    """Container for multi-method comparison results."""
    
    def __init__(
        self,
        results: Dict[str, ExplanationResult],
        input_tensor: torch.Tensor
    ):
        self.results = results
        self.input_tensor = input_tensor
        self.methods = list(results.keys())
    
    def get(self, method: str) -> ExplanationResult:
        """Get result for a specific method."""
        return self.results[method]
    
    def visualize_all(
        self,
        image: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None
    ):
        """Visualize all comparison results.
        
        Args:
            image: Optional original image.
            figsize: Figure size.
            save_path: Optional path to save the figure.
        """
        import matplotlib.pyplot as plt
        
        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)
        
        if image is not None:
            axes[0].imshow(image if image.ndim == 3 else image, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis("off")
            start_idx = 1
        else:
            start_idx = 0
            axes = axes[1:]
        
        for i, (method, result) in enumerate(self.results.items()):
            ax = axes[start_idx + i] if image is not None else axes[i]
            ax.imshow(result.normalized_attribution, cmap='jet')
            ax.set_title(method.replace('_', ' ').title())
            ax.axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert results to a pandas DataFrame."""
        import pandas as pd  # type: ignore[import-not-found]
        
        data = []
        for method, result in self.results.items():
            data.append({
                'method': method,
                'target_class': result.target_class,
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'attribution_mean': result.attribution.mean(),
                'attribution_std': result.attribution.std(),
                'attribution_max': result.attribution.max(),
            })
        
        return pd.DataFrame(data)
    
    def __repr__(self) -> str:
        return f"ComparisonResult(methods={self.methods})"
