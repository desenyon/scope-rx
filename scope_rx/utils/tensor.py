"""
Tensor conversion utilities.
"""

from typing import Union, Optional

import numpy as np
import torch


def to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor or numpy array.
        
    Returns:
        Numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def to_tensor(
    array: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert array to PyTorch tensor.
    
    Args:
        array: Numpy array or tensor.
        device: Target device.
        
    Returns:
        PyTorch tensor.
    """
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.from_numpy(np.array(array))
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 4D (N, C, H, W).
    
    Args:
        tensor: Input tensor.
        
    Returns:
        4D tensor.
    """
    if tensor.dim() == 2:
        # H, W -> 1, 1, H, W
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        # C, H, W -> 1, C, H, W
        return tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        return tensor
    else:
        raise ValueError(f"Cannot convert {tensor.dim()}D tensor to 4D")
