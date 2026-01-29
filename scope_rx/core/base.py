"""
Core base classes for ScopeRX explainers.

This module provides the abstract base class that all explanation methods
inherit from, ensuring a consistent interface across the library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


@dataclass
class ExplanationResult:
    """Container for explanation results with rich functionality.
    
    This dataclass holds the attribution map and associated metadata,
    providing convenient methods for visualization and analysis.
    
    Attributes:
        attribution: The attribution map as a numpy array.
        method: Name of the explanation method used.
        target_class: The target class index explained.
        predicted_class: The model's predicted class.
        confidence: Model's confidence for the target class.
        input_shape: Shape of the input tensor.
        metadata: Additional method-specific metadata.
    
    Example:
        >>> result = explainer.explain(input_tensor, target_class=5)
        >>> print(f"Method: {result.method}")
        >>> print(f"Attribution shape: {result.attribution.shape}")
        >>> result.visualize()  # Display the attribution
        >>> result.save("explanation.png")  # Save to file
    """
    
    attribution: np.ndarray
    method: str
    target_class: Optional[int] = None
    predicted_class: Optional[int] = None
    confidence: Optional[float] = None
    input_shape: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {})
    
    # Private fields for caching
    _normalized: Optional[np.ndarray] = field(default=None, repr=False)
    _original_input: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Validate and normalize attribution after initialization."""
        if not hasattr(self.attribution, 'shape'):
            self.attribution = np.array(self.attribution)
    
    @property
    def normalized_attribution(self) -> NDArray[np.floating[Any]]:
        """Get attribution normalized to [0, 1] range.
        
        Returns:
            Normalized attribution map.
        """
        if self._normalized is None:
            attr = self.attribution
            attr_min, attr_max = attr.min(), attr.max()
            if attr_max - attr_min > 1e-8:
                self._normalized = (attr - attr_min) / (attr_max - attr_min)
            else:
                self._normalized = np.zeros_like(attr)
        return self._normalized  # type: ignore[return-value]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the attribution map."""
        return self.attribution.shape
    
    def to_heatmap(
        self,
        colormap: str = "jet",
        normalize: bool = True
    ) -> np.ndarray:
        """Convert attribution to a colored heatmap.
        
        Args:
            colormap: Matplotlib colormap name.
            normalize: Whether to normalize before applying colormap.
            
        Returns:
            RGB heatmap as uint8 numpy array.
        """
        import cv2
        
        attr = self.normalized_attribution if normalize else self.attribution
        
        # Get colormap
        colormap_cv2: int = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
        
        # Apply colormap
        heatmap_arr = np.uint8(255 * attr)
        heatmap: NDArray[np.uint8] = cv2.applyColorMap(heatmap_arr, colormap_cv2)  # type: ignore[assignment]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # type: ignore[assignment]
        
        return heatmap
    
    def overlay(
        self,
        image: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> np.ndarray:
        """Overlay attribution heatmap on the original image.
        
        Args:
            image: Original image as numpy array (H, W, C) or (H, W).
            alpha: Blending factor for the overlay.
            colormap: Colormap for the heatmap.
            
        Returns:
            Overlaid image as numpy array.
        """
        import cv2
        
        # Ensure image is 3-channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Get heatmap
        heatmap = self.to_heatmap(colormap)
        
        # Resize heatmap to match image
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def visualize(
        self,
        image: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 5)
    ):
        """Visualize the attribution map.
        
        Args:
            image: Optional original image for overlay.
            title: Optional title for the plot.
            show: Whether to display the plot.
            save_path: Optional path to save the figure.
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt
        
        if image is not None:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Original image
            axes[0].imshow(image if image.ndim == 3 else image, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis("off")
            
            # Attribution
            axes[1].imshow(self.normalized_attribution, cmap='jet')
            axes[1].set_title(f"{self.method} Attribution")
            axes[1].axis("off")
            
            # Overlay
            overlay = self.overlay(image)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
        else:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(self.normalized_attribution, cmap='jet')
            ax.set_title(title or f"{self.method} Attribution")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save(self, path: str, format: str = "auto"):
        """Save the attribution to a file.
        
        Args:
            path: Output file path.
            format: Output format ('png', 'npy', 'npz', 'auto').
        """
        import os
        
        if format == "auto":
            format = os.path.splitext(path)[1].lower().strip('.')
        
        if format in ['png', 'jpg', 'jpeg']:
            import cv2
            heatmap = self.to_heatmap()
            cv2.imwrite(path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        elif format == 'npy':
            np.save(path, self.attribution)
        elif format == 'npz':
            np.savez(path, attribution=self.attribution)  # type: ignore[call-overload]
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def __repr__(self) -> str:
        return (
            f"ExplanationResult(method='{self.method}', "
            f"shape={self.shape}, "
            f"target_class={self.target_class})"
        )


class BaseExplainer(ABC):
    """Abstract base class for all explanation methods.
    
    All explanation methods in ScopeRX inherit from this class, ensuring
    a consistent interface and shared functionality.
    
    Attributes:
        model: The PyTorch model to explain.
        device: Device the model is on.
        
    Example:
        >>> class MyExplainer(BaseExplainer):
        ...     def explain(self, input_tensor, target_class=None, **kwargs):
        ...         # Implementation
        ...         return ExplanationResult(attribution=attr, method="MyMethod")
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize the base explainer.
        
        Args:
            model: PyTorch model to explain.
            device: Device to use. If None, inferred from model.
        """
        self.model = model
        self.model.eval()
        
        # Infer device
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Tracking
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    @abstractmethod
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs: Any
    ) -> ExplanationResult:
        """Generate an explanation for the input.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W).
            target_class: Target class to explain. If None, uses predicted class.
            **kwargs: Method-specific arguments.
            
        Returns:
            ExplanationResult containing the attribution map and metadata.
        """
        pass
    
    def _validate_input(
        self,
        input_tensor: torch.Tensor,
        expected_dims: int = 4
    ) -> torch.Tensor:
        """Validate and prepare input tensor.
        
        Args:
            input_tensor: Input tensor to validate.
            expected_dims: Expected number of dimensions.
            
        Returns:
            Validated tensor on the correct device.
            
        Raises:
            ValueError: If input has wrong dimensions.
        """
        if input_tensor.dim() != expected_dims:
            if input_tensor.dim() == expected_dims - 1:
                input_tensor = input_tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected {expected_dims}D tensor, got {input_tensor.dim()}D"
                )
        
        return input_tensor.to(self.device)
    
    def _get_target_class(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[int, float]:
        """Get target class and confidence.
        
        Args:
            input_tensor: Input tensor.
            target_class: Optional specified target class.
            
        Returns:
            Tuple of (target_class, confidence).
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            
            if target_class is None:
                target_class = int(output.argmax(dim=1).item())
            
            confidence = float(probs[0, target_class].item())
        
        return target_class, confidence
    
    def _register_hook(
        self,
        module: nn.Module,
        hook_fn: Any,
        hook_type: str = "forward"
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a hook on a module.
        
        Args:
            module: Module to hook.
            hook_fn: Hook function.
            hook_type: "forward", "backward", or "full_backward".
            
        Returns:
            Hook handle for later removal.
        """
        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "backward":
            handle = module.register_backward_hook(hook_fn)
        elif hook_type == "full_backward":
            handle = module.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
        
        self._hooks.append(handle)
        return handle
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
    
    def _find_layer(
        self,
        layer_name: str
    ) -> nn.Module:
        """Find a layer by name.
        
        Args:
            layer_name: Name of the layer (e.g., "layer4", "features.28").
            
        Returns:
            The requested module.
            
        Raises:
            ValueError: If layer not found.
        """
        # Try direct attribute access
        parts = layer_name.split('.')
        module: Any = self.model
        
        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, KeyError):
            pass
        
        # Try named_modules
        for name, mod in self.model.named_modules():  # type: ignore[assignment]
            if name == layer_name:
                return mod  # type: ignore[return-value]
        
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    def explain_batch(
        self,
        inputs: torch.Tensor,
        target_classes: Optional[List[Optional[int]]] = None,
        **kwargs: Any
    ) -> List[ExplanationResult]:
        """Generate explanations for a batch of inputs.
        
        Args:
            inputs: Batch of input tensors (N, C, H, W).
            target_classes: Optional list of target classes.
            **kwargs: Method-specific arguments.
            
        Returns:
            List of ExplanationResult objects.
        """
        results: List[ExplanationResult] = []
        batch_size = inputs.shape[0]
        
        tc_list: List[Optional[int]] = target_classes if target_classes is not None else [None] * batch_size
        
        for i in range(batch_size):
            inp = inputs[i]
            tc = tc_list[i] if i < len(tc_list) else None
            result = self.explain(inp.unsqueeze(0), target_class=tc, **kwargs)
            results.append(result)
        
        return results
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})"


class AttributionContext:
    """Context manager for safe gradient computation.
    
    Ensures gradients are properly enabled and model is in eval mode
    during attribution computation, then restores original state.
    
    Example:
        >>> with AttributionContext(model, input_tensor) as (model, tensor):
        ...     output = model(tensor)
        ...     output.backward()
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        requires_grad: bool = True
    ):
        self.model = model
        self.input_tensor = input_tensor
        self.requires_grad = requires_grad
        
        # Store original states
        self._original_training = model.training
        self._original_requires_grad = input_tensor.requires_grad
    
    def __enter__(self) -> Tuple[nn.Module, torch.Tensor]:
        self.model.eval()
        self.input_tensor.requires_grad = self.requires_grad
        return self.model, self.input_tensor
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> bool:
        self.model.train(self._original_training)
        self.input_tensor.requires_grad = self._original_requires_grad
        return False
