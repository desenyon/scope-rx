"""
Model wrapper utilities for ScopeRX.

Provides unified interface for different model architectures and frameworks.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ModelWrapper:
    """Unified wrapper for PyTorch models with layer inspection.
    
    Provides convenient methods for accessing layers, getting intermediate
    activations, and handling different model architectures.
    
    Attributes:
        model: The wrapped PyTorch model.
        device: Device the model is on.
        
    Example:
        >>> wrapper = ModelWrapper(resnet50)
        >>> layers = wrapper.get_conv_layers()
        >>> activations = wrapper.get_activations(input_tensor, "layer4")
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """Initialize the model wrapper.
        
        Args:
            model: PyTorch model to wrap.
            device: Device to use. If None, inferred from model.
        """
        self.model = model
        self.model.eval()
        
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Cache for layer info
        self._layer_cache: Dict[str, nn.Module] = {}
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def get_layer(self, layer_name: str) -> nn.Module:
        """Get a layer by name.
        
        Args:
            layer_name: Name of the layer (e.g., "layer4", "features.28").
            
        Returns:
            The requested module.
        """
        if layer_name in self._layer_cache:
            return self._layer_cache[layer_name]
        
        # Try direct attribute access
        parts = layer_name.split('.')
        module: Any = self.model
        
        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            self._layer_cache[layer_name] = module
            return module
        except (AttributeError, IndexError, KeyError):
            pass
        
        # Try named_modules
        for name, mod in self.model.named_modules():  # type: ignore[assignment]
            if name == layer_name:
                self._layer_cache[layer_name] = mod
                return mod  # type: ignore[return-value]
        
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    def get_all_layers(self) -> Dict[str, nn.Module]:
        """Get all named layers in the model.
        
        Returns:
            Dictionary mapping layer names to modules.
        """
        return dict(self.model.named_modules())  # type: ignore[arg-type]
    
    def get_conv_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get all convolutional layers.
        
        Returns:
            List of (name, module) tuples for conv layers.
        """
        conv_layers: List[Tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():  # type: ignore[assignment]
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_layers.append((name, module))
        return conv_layers
    
    def get_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get all attention layers (for transformers).
        
        Returns:
            List of (name, module) tuples for attention layers.
        """
        attention_layers: List[Tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():  # type: ignore[assignment]
            # Check for common attention module names
            module_type = type(module).__name__.lower()
            if 'attention' in module_type or 'mha' in module_type:
                attention_layers.append((name, module))
            # Also check for MultiheadAttention
            if isinstance(module, nn.MultiheadAttention):
                attention_layers.append((name, module))
        return attention_layers
    
    def get_activations(
        self,
        input_tensor: torch.Tensor,
        layer_names: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Get activations from specified layers.
        
        Args:
            input_tensor: Input tensor.
            layer_names: Single layer name or list of layer names.
            
        Returns:
            Dictionary mapping layer names to activation tensors.
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        self._activations.clear()
        hooks = []
        
        def make_hook(name: str) -> Callable[[nn.Module, Any, Any], None]:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                self._activations[name] = output.detach()
            return hook
        
        # Register hooks
        for name in layer_names:
            layer = self.get_layer(name)
            handle = layer.register_forward_hook(make_hook(name))
            hooks.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        return dict(self._activations)
    
    def get_gradients(
        self,
        input_tensor: torch.Tensor,
        layer_names: Union[str, List[str]],
        target_class: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get gradients at specified layers.
        
        Args:
            input_tensor: Input tensor.
            layer_names: Single layer name or list of layer names.
            target_class: Target class for backprop.
            
        Returns:
            Dictionary mapping layer names to gradient tensors.
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        self._gradients.clear()
        self._activations.clear()
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        def make_forward_hook(name: str) -> Callable[[nn.Module, Any, Any], None]:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                self._activations[name] = output
                output.retain_grad()
            return hook
        
        def make_backward_hook(name: str) -> Callable[[nn.Module, Any, Any], None]:
            def hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
                self._gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks
        for name in layer_names:
            layer = self.get_layer(name)
            hooks.append(layer.register_forward_hook(make_forward_hook(name)))
            hooks.append(layer.register_full_backward_hook(make_backward_hook(name)))
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        # Determine target
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        score = output[:, target_class]
        score.backward()
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        return dict(self._gradients)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_tensor: Input tensor.
            
        Returns:
            Model output.
        """
        return self.model(input_tensor.to(self.device))
    
    def predict(
        self,
        input_tensor: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k predictions.
        
        Args:
            input_tensor: Input tensor.
            top_k: Number of top predictions to return.
            
        Returns:
            Tuple of (probabilities, class_indices).
        """
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probs = torch.softmax(output, dim=1)
            top_probs, top_indices = probs.topk(top_k, dim=1)
        
        return top_probs, top_indices
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get information about a specific layer.
        
        Args:
            layer_name: Name of the layer.
            
        Returns:
            Dictionary with layer information.
        """
        layer = self.get_layer(layer_name)
        
        info = {
            "name": layer_name,
            "type": type(layer).__name__,
            "trainable_params": sum(p.numel() for p in layer.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in layer.parameters()),
        }
        
        # Add layer-specific info
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            info.update({
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding,
            })
        elif isinstance(layer, nn.Linear):
            info.update({
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            })
        
        return info
    
    def summary(self) -> str:
        """Get a summary of the model architecture.
        
        Returns:
            String summary of the model.
        """
        lines = [
            f"Model: {type(self.model).__name__}",
            f"Device: {self.device}",
            "-" * 60,
        ]
        
        total_params = 0
        trainable_params = 0
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                if params > 0:
                    total_params += params
                    trainable_params += trainable
                    lines.append(f"{name}: {type(module).__name__} ({params:,} params)")
        
        lines.extend([
            "-" * 60,
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
        ])
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ModelWrapper({type(self.model).__name__})"


def auto_detect_target_layer(model: nn.Module) -> Optional[str]:
    """Auto-detect the best target layer for CAM methods.
    
    Tries to find the last convolutional layer before the classifier.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Layer name or None if not found.
    """
    wrapper = ModelWrapper(model)
    conv_layers = wrapper.get_conv_layers()
    
    if conv_layers:
        # Return the last conv layer
        return conv_layers[-1][0]
    
    return None


def get_model_input_size(model: nn.Module) -> Optional[Tuple[int, int]]:
    """Try to infer the expected input size for a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Tuple of (height, width) or None.
    """
    # Common sizes for known architectures
    model_name = type(model).__name__.lower()
    
    known_sizes = {
        'resnet': (224, 224),
        'vgg': (224, 224),
        'alexnet': (224, 224),
        'densenet': (224, 224),
        'mobilenet': (224, 224),
        'efficientnet': (224, 224),
        'inception': (299, 299),
        'vit': (224, 224),
    }
    
    for name, size in known_sizes.items():
        if name in model_name:
            return size
    
    return (224, 224)  # Default
