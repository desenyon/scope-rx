"""
ScopeRX Command Line Interface.

Usage:
    scope-rx explain IMAGE --model MODEL [--method METHOD] [--output OUTPUT]
    scope-rx compare IMAGE --model MODEL --methods METHODS [--output OUTPUT]
    scope-rx --version
    scope-rx --help
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="scope-rx",
        description="ScopeRX: Neural Network Explainability Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a GradCAM explanation
  scope-rx explain image.jpg --model resnet50 --method gradcam --output heatmap.png
  
  # Compare multiple methods
  scope-rx compare image.jpg --model resnet50 --methods gradcam,smoothgrad,ig
  
  # List available methods
  scope-rx list-methods
  
  # Show model layers (for layer selection)
  scope-rx show-layers --model resnet50
"""
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version"
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Generate explanation for an image"
    )
    explain_parser.add_argument("image", help="Path to input image")
    explain_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name (e.g., resnet50, vgg16) or path to saved model"
    )
    explain_parser.add_argument(
        "--method",
        default="gradcam",
        help="Explanation method (default: gradcam)"
    )
    explain_parser.add_argument(
        "--target-class", "-c",
        type=int,
        default=None,
        help="Target class index (default: predicted class)"
    )
    explain_parser.add_argument(
        "--layer", "-l",
        default=None,
        help="Target layer name (default: auto-detect)"
    )
    explain_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path"
    )
    explain_parser.add_argument(
        "--colormap",
        default="jet",
        help="Colormap for visualization (default: jet)"
    )
    explain_parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the result"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple explanation methods"
    )
    compare_parser.add_argument("image", help="Path to input image")
    compare_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name or path"
    )
    compare_parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated list of methods"
    )
    compare_parser.add_argument(
        "--target-class", "-c",
        type=int,
        default=None,
        help="Target class index"
    )
    compare_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path"
    )
    compare_parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the result"
    )
    
    # List methods
    subparsers.add_parser(
        "list-methods",
        help="List available explanation methods"
    )
    
    # Show layers
    layers_parser = subparsers.add_parser(
        "show-layers",
        help="Show model layers"
    )
    layers_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name"
    )
    
    args = parser.parse_args()
    
    if args.version:
        from scope_rx import __version__
        print(f"ScopeRX v{__version__}")
        return 0
    
    if args.command == "explain":
        return cmd_explain(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "list-methods":
        return cmd_list_methods(args)
    elif args.command == "show-layers":
        return cmd_show_layers(args)
    else:
        parser.print_help()
        return 0


def cmd_explain(args):
    """Handle explain command."""
    try:
        import torch
        from scope_rx import ScopeRX
        from scope_rx.utils import preprocess_image, load_image
        from scope_rx.visualization import plot_attribution, export_visualization
        
        # Load model
        model = load_model(args.model)
        
        # Load and preprocess image
        original_image = load_image(args.image, size=(224, 224))
        input_tensor = preprocess_image(args.image)
        
        # Ensure tensor type
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.from_numpy(input_tensor).float()
        
        # Create explainer
        scope = ScopeRX(model)
        
        # Generate explanation
        result = scope.explain(
            input_tensor,
            method=args.method,
            target_class=args.target_class,
            target_layer=args.layer
        )
        
        print(f"Method: {args.method}")
        print(f"Target class: {result.target_class}")
        print(f"Attribution shape: {result.attribution.shape}")
        
        # Visualize
        if args.output:
            export_visualization(
                result.attribution,
                args.output,
                colormap=args.colormap,
                image=original_image
            )
            print(f"Saved to: {args.output}")
        
        if not args.no_display:
            plot_attribution(
                result.attribution,
                image=original_image,
                title=args.method.replace('_', ' ').title(),
                colormap=args.colormap
            )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_compare(args):
    """Handle compare command."""
    try:
        import torch
        from scope_rx import ScopeRX
        from scope_rx.utils import preprocess_image, load_image
        from scope_rx.visualization import plot_comparison
        
        # Load model
        model = load_model(args.model)
        
        # Load and preprocess image
        original_image = load_image(args.image, size=(224, 224))
        input_tensor = preprocess_image(args.image)
        
        # Ensure tensor type
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.from_numpy(input_tensor).float()
        
        # Create explainer
        scope = ScopeRX(model)
        
        # Parse methods
        methods = [m.strip() for m in args.methods.split(',')]
        
        # Generate explanations
        results = scope.compare_methods(
            input_tensor,
            methods=methods,
            target_class=args.target_class
        )
        
        # Extract attributions
        attributions = {name: r.attribution for name, r in results.results.items()}
        
        print(f"Compared methods: {', '.join(methods)}")
        
        # Visualize
        if not args.no_display:
            plot_comparison(
                attributions,
                image=original_image,
                save_path=args.output
            )
        elif args.output:
            plot_comparison(
                attributions,
                image=original_image,
                save_path=args.output,
                show=False
            )
            print(f"Saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_methods(args):
    """List available methods."""
    print("Available explanation methods:")
    print("-" * 40)
    
    # Use static list of available methods
    methods = [
        'gradcam', 'gradcam++', 'scorecam', 'layercam',
        'smoothgrad', 'integrated_gradients', 'vanilla_gradients',
        'guided_backprop', 'occlusion', 'rise', 'meaningful_perturbation',
        'kernel_shap', 'lime', 'attention_rollout', 'attention_flow'
    ]
    
    categories = {
        "Gradient-based": ["gradcam", "gradcam++", "scorecam", "layercam", 
                          "smoothgrad", "integrated_gradients", "vanilla", 
                          "guided_backprop"],
        "Perturbation-based": ["occlusion", "rise", "meaningful_perturbation"],
        "Model-agnostic": ["kernel_shap", "lime"],
        "Attention-based": ["attention_rollout", "attention_flow", "raw_attention"],
    }
    
    for category, method_list in categories.items():
        print(f"\n{category}:")
        for method in method_list:
            if method in methods:
                print(f"  - {method}")
    
    return 0


def cmd_show_layers(args):
    """Show model layers."""
    model = load_model(args.model)
    
    print(f"Layers in {args.model}:")
    print("-" * 40)
    
    for name, module in model.named_modules():
        if name:
            print(f"  {name}: {module.__class__.__name__}")
    
    return 0


def load_model(model_name: str):
    """Load a model by name or path."""
    import torch
    import torchvision.models as models  # type: ignore[import-not-found]
    
    # Check if it's a file path
    if Path(model_name).exists():
        model = torch.load(model_name, weights_only=False)
        model.eval()
        return model
    
    # Check if it's a torchvision model
    model_fn = getattr(models, model_name, None)
    if model_fn is not None:
        try:
            # Try with weights parameter (newer torchvision)
            model = model_fn(weights="DEFAULT")
        except TypeError:
            # Fall back to pretrained parameter (older torchvision)
            model = model_fn(pretrained=True)
        model.eval()
        return model
    
    raise ValueError(
        f"Unknown model: {model_name}. "
        "Use a torchvision model name (e.g., resnet50) or path to saved model."
    )


if __name__ == "__main__":
    sys.exit(main())
