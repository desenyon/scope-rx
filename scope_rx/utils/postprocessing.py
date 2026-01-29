"""
Attribution postprocessing utilities.
"""

import numpy as np
import cv2


def normalize_attribution(
    attribution: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """Normalize attribution to [0, 1] range.
    
    Args:
        attribution: Attribution map.
        method: Normalization method ('minmax', 'abs_max', 'percentile').
        
    Returns:
        Normalized attribution.
    """
    if method == "minmax":
        attr_min, attr_max = attribution.min(), attribution.max()
        if attr_max - attr_min > 1e-8:
            return (attribution - attr_min) / (attr_max - attr_min)
        else:
            return np.zeros_like(attribution)
    
    elif method == "abs_max":
        abs_max = np.abs(attribution).max()
        if abs_max > 1e-8:
            return (attribution / abs_max + 1) / 2
        else:
            return np.ones_like(attribution) * 0.5
    
    elif method == "percentile":
        p_low, p_high = np.percentile(attribution, [2, 98])
        attribution = np.clip(attribution, p_low, p_high)
        if p_high - p_low > 1e-8:
            return (attribution - p_low) / (p_high - p_low)
        else:
            return np.zeros_like(attribution)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_attribution(
    attribution: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian smoothing to attribution.
    
    Args:
        attribution: Attribution map.
        kernel_size: Gaussian kernel size.
        sigma: Gaussian sigma.
        
    Returns:
        Smoothed attribution.
    """
    return cv2.GaussianBlur(
        attribution.astype(np.float32),
        (kernel_size, kernel_size),
        sigma
    )


def threshold_attribution(
    attribution: np.ndarray,
    threshold: float = 0.5,
    method: str = "value"
) -> np.ndarray:
    """Threshold attribution map.
    
    Args:
        attribution: Attribution map (should be normalized to [0, 1]).
        threshold: Threshold value or percentile.
        method: 'value' (absolute threshold) or 'percentile'.
        
    Returns:
        Thresholded attribution (binary or clipped).
    """
    if method == "percentile":
        threshold = np.percentile(attribution, threshold * 100)
    
    return np.where(attribution >= threshold, attribution, 0)


def upsample_attribution(
    attribution: np.ndarray,
    target_size: tuple,
    interpolation: str = "bilinear"
) -> np.ndarray:
    """Upsample attribution to target size.
    
    Args:
        attribution: Attribution map.
        target_size: Target (height, width).
        interpolation: Interpolation method.
        
    Returns:
        Upsampled attribution.
    """
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    
    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
    
    return cv2.resize(
        attribution.astype(np.float32),
        (target_size[1], target_size[0]),
        interpolation=interp
    )
