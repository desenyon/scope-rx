# Changelog

All notable changes to ScopeRX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-28

### Major Release - Complete Rewrite as ScopeRX

This release marks a complete transformation from the original `gradientvis` library into `ScopeRX` - a comprehensive neural network explainability toolkit.

### Added

#### Core Architecture
- **`BaseExplainer`**: Abstract base class providing unified interface for all explanation methods
- **`ExplanationResult`**: Structured data class for consistent attribution outputs
- **`AttributionContext`**: Context manager for safe gradient computation
- **`ModelWrapper`**: Unified wrapper supporting PyTorch models with layer inspection

#### Gradient-Based Methods
- **`GradCAM`**: Enhanced implementation with proper hook management
- **`GradCAMPlusPlus`**: Improved gradient weighting for better localization
- **`ScoreCAM`**: Gradient-free CAM using activation channel scores
- **`LayerCAM`**: Per-pixel weighted class activation mapping
- **`SmoothGrad`**: Noise-averaged gradient visualization
- **`IntegratedGradients`**: Attribution via integral approximation with Riemann/Gauss-Legendre methods
- **`VanillaGradients`**: Simple input gradient saliency maps

#### Perturbation-Based Methods
- **`OcclusionSensitivity`**: Sliding window occlusion analysis with multiple strategies
- **`RISE`**: Randomized Input Sampling for Explanation
- **`MeaningfulPerturbation`**: Optimization-based perturbation explanations

#### Model-Agnostic Methods
- **`KernelSHAP`**: SHAP values via weighted linear regression with multiple sampling strategies
- **`LIME`**: Local Interpretable Model-agnostic Explanations with superpixel segmentation

#### Attention-Based Methods
- **`AttentionRollout`**: Attention flow through transformer layers
- **`AttentionFlow`**: Graph-based attention attribution
- **`RawAttention`**: Direct attention weight extraction

#### Evaluation Metrics
- **Faithfulness Metrics**: Insertion/deletion curves, AOPC, sufficiency scores
- **Sensitivity Metrics**: Max/average sensitivity, input perturbation sensitivity
- **Stability Metrics**: Explanation consistency across similar inputs
- **Localization Metrics**: Pointing game, energy inside bounding box

#### Visualization
- **Interactive Visualizations**: Plotly-based interactive heatmaps and comparisons
- **Multi-Method Comparison**: Side-by-side comparison of explanation methods
- **Overlay Tools**: Advanced heatmap overlays with customizable colormaps
- **Animation Support**: Animated attribution evolution
- **Export Utilities**: PNG, SVG, HTML, PDF export options

#### Utilities
- **Preprocessing Pipeline**: Image loading, normalization, augmentation
- **Postprocessing**: Attribution smoothing, thresholding, upsampling
- **Batch Processing**: Efficient batch explanation generation
- **Caching**: Result caching for expensive computations

#### CLI Tool
- Command-line interface for quick explanations without coding

### Changed
- Complete project restructure with modern Python packaging (pyproject.toml)
- Type hints throughout the codebase
- Comprehensive docstrings following Google style
- Improved error handling and validation

### Removed
- Legacy `gradientvis` package structure
- Deprecated setuptools-only configuration

### Migration Guide

If you're migrating from `gradientvis`:

```python
# Old import
from gradientvis.methods.gradcam import GradCAM

# New import
from scope_rx.methods import GradCAM

# Old usage
gradcam = GradCAM(model, model.layer4)
cam = gradcam.generate(image, class_idx=0)

# New usage (more features)
from scope_rx import ScopeRX
scope = ScopeRX(model)
result = scope.explain(image, method='gradcam', target_layer='layer4', target_class=0)
attribution = result.attribution
```

## [0.5.0] - Previous Release (gradientvis)

- Initial release as gradientvis
- Basic GradCAM, SmoothGrad, and Integrated Gradients
- Simple visualization utilities
