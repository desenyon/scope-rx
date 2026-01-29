"""
Image preprocessing utilities.
"""

from typing import Tuple, Optional, Union
from pathlib import Path

import numpy as np
import torch
import cv2


def load_image(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb"
) -> np.ndarray:
    """Load an image from disk.
    
    Args:
        path: Path to the image file.
        size: Optional (width, height) to resize to.
        color_mode: 'rgb', 'bgr', or 'gray'.
        
    Returns:
        Image as numpy array.
    """
    image = cv2.imread(str(path))
    
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    
    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 'bgr' keeps original format
    
    if size is not None:
        image = cv2.resize(image, size)
    
    return image


def preprocess_image(
    image: Union[str, Path, np.ndarray],
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """Preprocess an image for model input.
    
    Args:
        image: Image path or numpy array.
        size: Target size (width, height).
        mean: Normalization mean (per channel).
        std: Normalization std (per channel).
        to_tensor: Whether to convert to PyTorch tensor.
        
    Returns:
        Preprocessed image.
    """
    # Load if path
    if isinstance(image, (str, Path)):
        image = load_image(image, size=size, color_mode="rgb")
    
    # Resize if needed
    if image.shape[:2] != (size[1], size[0]):
        image = cv2.resize(image, size)
    
    # Convert to float [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize
    image = normalize_image(image, mean, std)
    
    # Convert to tensor
    if to_tensor:
        # HWC -> CHW
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        tensor_result = torch.from_numpy(image).float().unsqueeze(0)
        return tensor_result
    
    return image


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalize image with mean and std.
    
    Args:
        image: Image array (H, W, C) with values in [0, 1].
        mean: Per-channel mean.
        std: Per-channel std.
        
    Returns:
        Normalized image.
    """
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    
    if image.ndim == 3:
        result = (image - mean_arr) / std_arr
    else:
        result = (image - mean_arr[0]) / std_arr[0]
    
    return result


def denormalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Denormalize image to [0, 1] range.
    
    Args:
        image: Normalized image (C, H, W) or (H, W, C).
        mean: Per-channel mean used for normalization.
        std: Per-channel std used for normalization.
        
    Returns:
        Denormalized image in [0, 1].
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Handle batch dimension
    if image.ndim == 4:
        image = image[0]
    
    # CHW -> HWC
    if image.shape[0] in [1, 3] and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    
    result = image * std_arr + mean_arr
    result = np.clip(result, 0, 1)
    
    return result
