# Contributing to ScopeRX

Thank you for your interest in contributing to ScopeRX! This document provides guidelines and instructions for contributing.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/scope-rx.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev,full]"`
5. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- Git

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/xcalen/scope-rx.git
cd scope-rx

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,full,interactive,docs]"

# Install pre-commit hooks
pre-commit install
```

## Testing

We use pytest for testing. Please ensure all tests pass before submitting a PR.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scope_rx --cov-report=html

# Run specific test file
pytest tests/test_gradcam.py

# Run tests matching a pattern
pytest -k "test_gradcam"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_gradcam_returns_correct_shape`
- Include both positive and edge case tests
- Mock external dependencies when appropriate

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black scope_rx tests
isort scope_rx tests

# Check linting
flake8 scope_rx tests

# Type checking
mypy scope_rx
```

### Style Guidelines

- Use type hints for all function signatures
- Write docstrings in Google style format
- Keep functions focused and under 50 lines when possible
- Use meaningful variable names

Example:
```python
def compute_attribution(
    self,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    *,
    normalize: bool = True,
) -> ExplanationResult:
    """Compute attribution map for the input.
    
    Args:
        input_tensor: Input image tensor of shape (N, C, H, W).
        target_class: Target class index. If None, uses predicted class.
        normalize: Whether to normalize the output to [0, 1].
        
    Returns:
        ExplanationResult containing the attribution map and metadata.
        
    Raises:
        ValueError: If input_tensor has incorrect dimensions.
    """
    ...
```

## Pull Request Process

1. **Create an Issue First**: For significant changes, create an issue to discuss
2. **Branch Naming**: Use descriptive names like `feature/add-scorecam` or `fix/gradcam-memory-leak`
3. **Commit Messages**: Use conventional commits format:
   - `feat: add ScoreCAM implementation`
   - `fix: resolve memory leak in GradCAM hooks`
   - `docs: update README with new examples`
   - `test: add tests for batch processing`
4. **Keep PRs Focused**: One feature or fix per PR
5. **Update Documentation**: Include docstrings and update README if needed
6. **Add Tests**: New features must include tests

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] No breaking changes (or clearly documented if necessary)

## Project Structure

```
scope_rx/
├── __init__.py           # Public API exports
├── core/                 # Core abstractions
│   ├── base.py          # BaseExplainer, ExplanationResult
│   ├── registry.py      # Method registry
│   └── wrapper.py       # ModelWrapper
├── methods/             # Explanation methods
│   ├── gradient/        # Gradient-based (GradCAM, IG, etc.)
│   ├── perturbation/    # Perturbation-based (Occlusion, RISE)
│   ├── attention/       # Attention-based methods
│   └── model_agnostic/  # SHAP, LIME
├── metrics/             # Evaluation metrics
├── visualization/       # Visualization tools
├── utils/               # Utilities
└── cli.py              # Command-line interface
```

## Adding a New Explanation Method

1. Create a new file in the appropriate `methods/` subdirectory
2. Inherit from `BaseExplainer`
3. Implement required methods: `explain()`, `_validate_input()`
4. Add tests in `tests/`
5. Export in `__init__.py` files
6. Add documentation

Example:
```python
from scope_rx.core.base import BaseExplainer, ExplanationResult

class MyNewMethod(BaseExplainer):
    """My new explanation method.
    
    This method does X by computing Y.
    
    Args:
        model: PyTorch model to explain.
        param1: Description of param1.
        
    Example:
        >>> explainer = MyNewMethod(model, param1=value)
        >>> result = explainer.explain(input_tensor, target_class=5)
    """
    
    def __init__(self, model: nn.Module, param1: float = 1.0):
        super().__init__(model)
        self.param1 = param1
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        # Implementation here
        ...
```

## Documentation

- Use Google-style docstrings
- Include examples in docstrings
- Update README.md for significant features
- Consider adding tutorials for complex features

## Reporting Bugs

When reporting bugs, please include:

1. ScopeRX version (`pip show scope-rx`)
2. Python version
3. PyTorch version
4. Operating system
5. Minimal reproducible example
6. Full error traceback

## Feature Requests

We welcome feature requests! Please:

1. Check existing issues first
2. Describe the use case clearly
3. Provide examples if possible
4. Consider if you'd like to implement it yourself

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open a GitHub Discussion for general questions
- Open an Issue for bugs or feature requests
- Reach out to maintainers for sensitive matters

Thank you for contributing to ScopeRX!
