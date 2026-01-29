# ScopeRX

**Neural Network Explainability and Interpretability Library**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/scope-rx.svg)](https://badge.fury.io/py/scope-rx)

ScopeRX is a comprehensive Python library for explaining and interpreting neural network predictions. It provides state-of-the-art attribution methods, evaluation metrics, and visualization tools - all unified under a simple, intuitive API.

## Features

- **15+ Explanation Methods**: From classic GradCAM to cutting-edge RISE and attention methods
- **Unified API**: One interface to rule them all - switch between methods with a single parameter
- **Evaluation Metrics**: Faithfulness, sensitivity, and stability metrics to quantify explanation quality
- **Beautiful Visualizations**: Publication-ready plots with minimal code
- **Model Agnostic**: Works with any PyTorch model architecture
- **Transformer Support**: Dedicated attention visualization for Vision Transformers
- **CLI Tool**: Generate explanations from the command line

## Installation

```bash
pip install scope-rx
```

**With optional dependencies:**

```bash
# For interactive Plotly visualizations
pip install scope-rx[interactive]

# For development
pip install scope-rx[dev]

# Full installation with all extras
pip install scope-rx[full]
```

## Quick Start

```python
from scope_rx import ScopeRX
import torch
import torchvision.models as models

# Load your model
model = models.resnet50(pretrained=True)
model.eval()

# Create explainer
explainer = ScopeRX(model)

# Generate explanation
result = explainer.explain(
    input_tensor,
    method='gradcam',
    target_class=predicted_class
)

# Visualize
result.visualize()

# Or save to file
result.save("explanation.png")
```

## Available Methods

### Gradient-Based Methods

| Method                   | Description                                | Use Case                            |
| ------------------------ | ------------------------------------------ | ----------------------------------- |
| `gradcam`              | Gradient-weighted Class Activation Mapping | General CNN visualization           |
| `gradcam++`            | Improved GradCAM with better localization  | Multiple object instances           |
| `scorecam`             | Score-based CAM (gradient-free)            | When gradients are unstable         |
| `layercam`             | Layer-wise CAM                             | Fine-grained attribution            |
| `smoothgrad`           | Noise-smoothed gradients                   | Reducing gradient noise             |
| `integrated_gradients` | Axiomatic attribution method               | Theoretically grounded explanations |
| `vanilla`              | Simple input gradients                     | Quick baseline                      |
| `guided_backprop`      | Guided backpropagation                     | High-resolution visualization       |

### Perturbation-Based Methods

| Method                      | Description                    | Use Case                         |
| --------------------------- | ------------------------------ | -------------------------------- |
| `occlusion`               | Sliding window occlusion       | Understanding spatial importance |
| `rise`                    | Randomized Input Sampling      | Black-box models                 |
| `meaningful_perturbation` | Optimized minimal perturbation | Finding minimal explanations     |

### Model-Agnostic Methods

| Method          | Description                      | Use Case                       |
| --------------- | -------------------------------- | ------------------------------ |
| `kernel_shap` | Kernel SHAP approximation        | Shapley value estimation       |
| `lime`        | Local Interpretable Explanations | Interpretable local surrogates |

### Attention-Based Methods (for Transformers)

| Method                | Description                  | Use Case                      |
| --------------------- | ---------------------------- | ----------------------------- |
| `attention_rollout` | Attention weight aggregation | Vision Transformers           |
| `attention_flow`    | Attention flow propagation   | Understanding attention paths |
| `raw_attention`     | Raw attention weights        | Quick attention inspection    |

## Compare Methods

```python
from scope_rx import ScopeRX

explainer = ScopeRX(model)

# Compare multiple methods at once
results = explainer.compare_methods(
    input_tensor,
    methods=['gradcam', 'smoothgrad', 'integrated_gradients', 'rise'],
    target_class=predicted_class
)

# Visualize comparison
from scope_rx.visualization import plot_comparison
plot_comparison({name: r.attribution for name, r in results.items()})
```

## Evaluate Explanations

```python
from scope_rx.metrics import (
    faithfulness_score,
    insertion_deletion_auc,
    sensitivity_score,
    stability_score
)

# Faithfulness: Does the explanation reflect model behavior?
faith = faithfulness_score(model, input_tensor, attribution, target_class=0)

# Insertion/Deletion: How does model output change as we add/remove important features?
scores = insertion_deletion_auc(model, input_tensor, attribution, target_class=0)
print(f"Insertion AUC: {scores['insertion_auc']:.3f}")
print(f"Deletion AUC: {scores['deletion_auc']:.3f}")

# Sensitivity: Are explanations sensitive to meaningful changes?
sens = sensitivity_score(explainer, input_tensor, target_class=0)

# Stability: Are explanations stable across similar inputs?
stab = stability_score(explainer, input_tensor, target_class=0)
```

## Visualization

```python
from scope_rx.visualization import (
    plot_attribution,
    plot_comparison,
    overlay_attribution,
    create_interactive_plot,
    export_visualization
)

# Simple plot
plot_attribution(attribution, image=original_image)

# Interactive Plotly plot
fig = create_interactive_plot(attribution, image=original_image)
fig.show()

# Export to various formats
export_visualization(attribution, "output.png", colormap="jet")
export_visualization(attribution, "output.npy")  # Raw numpy array
```

## Command Line Interface

```bash
# Generate explanation
scope-rx explain image.jpg --model resnet50 --method gradcam --output heatmap.png

# Compare methods
scope-rx compare image.jpg --model resnet50 --methods gradcam,smoothgrad,rise

# List available methods
scope-rx list-methods

# Show model layers (for layer selection)
scope-rx show-layers --model resnet50
```

## Advanced Usage

### Custom Target Layers

```python
from scope_rx import GradCAM

# Specify exact layer
explainer = GradCAM(model, target_layer="layer4.1.conv2")
```

### Custom Baselines for Integrated Gradients

```python
from scope_rx import IntegratedGradients

# Use different baselines
explainer = IntegratedGradients(
    model,
    n_steps=50,
    baseline="blur"  # Options: "zero", "random", "blur"
)
```

### Batch Processing

```python
from scope_rx import ScopeRX
from scope_rx.utils import preprocess_image
from pathlib import Path

explainer = ScopeRX(model)

# Process multiple images
for image_path in image_paths:
    input_tensor = preprocess_image(image_path)
    result = explainer.explain(input_tensor, method='gradcam')
    result.save(f"explanations/{Path(image_path).stem}.png")
```

### Using Individual Explainers

```python
from scope_rx import GradCAM, SmoothGrad, RISE

# Use specific explainer directly
gradcam = GradCAM(model, target_layer="layer4")
result = gradcam.explain(input_tensor, target_class=0)

# SmoothGrad with custom parameters
smoothgrad = SmoothGrad(model, n_samples=50, noise_level=0.2)
result = smoothgrad.explain(input_tensor, target_class=0)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=scope_rx --cov-report=html

# Run specific test module
pytest tests/test_gradient_methods.py -v
```

## Documentation

For full documentation, visit our [documentation site](https://github.com/xcalen/scope-rx/docs).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ScopeRX in your research, please cite:

```bibtex
@software{scoperx2024,
  title = {ScopeRX: Neural Network Explainability Library},
  author = {XCALEN},
  year = {2024},
  url = {https://github.com/xcalen/scope-rx}
}
```

## Acknowledgments

ScopeRX builds upon the excellent work of the interpretability research community. Special thanks to the authors of:

- GradCAM, GradCAM++, ScoreCAM, LayerCAM
- SHAP and LIME
- Integrated Gradients
- RISE
- And many others who have contributed to the field of explainable AI

---

**Made with love by Desenyon**
