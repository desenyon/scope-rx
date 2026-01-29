"""
Plotting utilities for explanations.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import cv2


def plot_attribution(
    attribution: np.ndarray,
    image: Optional[np.ndarray] = None,
    title: str = "Attribution",
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None
) -> "Figure":
    """Plot an attribution map.
    
    Args:
        attribution: Attribution map (H, W).
        image: Optional original image for overlay.
        title: Plot title.
        colormap: Matplotlib colormap name.
        alpha: Alpha for overlay.
        figsize: Figure size.
        show: Whether to display the plot.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    if image is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        if image.ndim == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Attribution
        im = axes[1].imshow(attribution, cmap=colormap)
        axes[1].set_title(title)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = overlay_attribution(attribution, image, alpha, colormap)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attribution, cmap=colormap)
        ax.set_title(title)
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


def plot_comparison(
    attributions: Dict[str, np.ndarray],
    image: Optional[np.ndarray] = None,
    colormap: str = "jet",
    figsize: Tuple[int, int] = (15, 5),
    show: bool = True,
    save_path: Optional[str] = None
) -> Figure:
    """Plot comparison of multiple attribution methods.
    
    Args:
        attributions: Dictionary mapping method names to attribution maps.
        image: Optional original image.
        colormap: Matplotlib colormap name.
        figsize: Figure size.
        show: Whether to display.
        save_path: Optional path to save.
        
    Returns:
        Matplotlib figure.
    """
    n_methods = len(attributions)
    cols = n_methods + (1 if image is not None else 0)
    
    fig, axes = plt.subplots(1, cols, figsize=figsize)
    
    if cols == 1:
        axes = [axes]
    
    start_idx = 0
    if image is not None:
        if image.ndim == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        start_idx = 1
    
    for i, (method, attr) in enumerate(attributions.items()):
        ax = axes[start_idx + i]
        ax.imshow(attr, cmap=colormap)  # Store in ax, not as unused variable
        ax.set_title(method.replace('_', ' ').title())
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def overlay_attribution(
    attribution: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """Overlay attribution heatmap on image.
    
    Args:
        attribution: Attribution map (H, W), values in [0, 1].
        image: Original image (H, W, C) or (H, W).
        alpha: Blending factor.
        colormap: Colormap name.
        
    Returns:
        Overlaid image as numpy array.
    """
    # Normalize attribution
    attr_min, attr_max = attribution.min(), attribution.max()
    if attr_max - attr_min > 1e-8:
        attribution = (attribution - attr_min) / (attr_max - attr_min)
    
    # Convert to heatmap
    colormap_cv2 = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(np.uint8(255 * attribution).astype(np.uint8), colormap_cv2)  # type: ignore[call-overload]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is proper format
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def create_interactive_plot(
    attribution: np.ndarray,
    image: Optional[np.ndarray] = None,
    title: str = "Attribution",
    colorscale: str = "jet"
) -> Any:
    """Create an interactive Plotly plot.
    
    Args:
        attribution: Attribution map (H, W).
        image: Optional original image.
        title: Plot title.
        colorscale: Plotly colorscale name.
        
    Returns:
        Plotly figure object.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        from plotly.subplots import make_subplots  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "Plotly is required for interactive plots. "
            "Install with: pip install plotly"
        )
    
    if image is not None:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Original", title, "Overlay")
        )
        
        # Original image
        fig.add_trace(
            go.Image(z=image),
            row=1, col=1
        )
        
        # Attribution heatmap
        fig.add_trace(
            go.Heatmap(z=attribution, colorscale=colorscale, showscale=True),
            row=1, col=2
        )
        
        # Overlay
        overlay = overlay_attribution(attribution, image)
        fig.add_trace(
            go.Image(z=overlay),
            row=1, col=3
        )
    else:
        fig = go.Figure(data=go.Heatmap(
            z=attribution,
            colorscale=colorscale,
            showscale=True
        ))
        fig.update_layout(title=title)
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig
