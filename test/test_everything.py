"""
test_everything.py
==================
Hands-on test of every scope-rx command and API call,
written as a random user who just installed the library and wants to
kick the tyres on everything it offers.

Run from the ``test/`` directory:

    cd test
    python test_everything.py
"""

import os
import subprocess
import sys
import traceback

# ── Always resolve paths relative to this script, regardless of launch cwd ───
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Force headless matplotlib so no GUI windows pop up ───────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
_results: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    line = f"  {tag}  {label}"
    if detail:
        line += f"  ·  {detail}"
    print(line)
    _results.append((label, ok))
    return ok


def _attr_ok(arr: np.ndarray) -> tuple[bool, str]:
    """Verify an attribution map is non-trivial: finite, non-constant, non-all-zero."""
    if arr is None:
        return False, "None"
    flat = arr.flatten()
    if not np.all(np.isfinite(arr)):
        bad = arr[~np.isfinite(arr)]
        return False, f"non-finite values present: {bad[:3]}"
    if not np.any(arr != 0):
        return False, "all-zero attribution (method produced no signal)"
    if np.std(flat) < 1e-8:
        return False, f"constant output (std={np.std(flat):.2e}) – all pixels identical"
    return True, f"shape={arr.shape}  std={np.std(flat):.4f}  range=[{flat.min():.4f}, {flat.max():.4f}]"


# Use the scope-rx binary that lives alongside the current Python interpreter
# (i.e. the same virtual-environment), so the subprocess always has the same
# packages available as this script.
_SCOPE_RX_BIN = os.path.join(os.path.dirname(sys.executable), "scope-rx")


def run_cli(*args: str, timeout: int = 240) -> tuple[str, str, int]:
    """Run the ``scope-rx`` CLI and return (stdout, stderr, returncode)."""
    proc = subprocess.run(
        [_SCOPE_RX_BIN, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        # Inherit the chdir above so CLI sees sample.jpg in the right place
        cwd=os.getcwd(),
    )
    return proc.stdout, proc.stderr, proc.returncode


# ---------------------------------------------------------------------------
# 0. Setup – create a sample image and output directory
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("  scope-rx  ·  end-to-end user test")
print("=" * 65 + "\n")

os.makedirs("outputs", exist_ok=True)

try:
    import urllib.request
    import cv2

    # Remove any stale sample.jpg from a previous run (e.g. the noise image).
    for _stale in ("sample.jpg", "sample_raw.jpg"):
        if os.path.exists(_stale):
            os.remove(_stale)

    # Try multiple public-domain dog photos; use the first that succeeds.
    _URLS = [
        # Golden retriever – Wikimedia Commons (public domain)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/"
        "Golden_Retriever_Obedience.jpg/320px-Golden_Retriever_Obedience.jpg",
        # Labrador – Wikimedia Commons (public domain)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/"
        "Labrador_on_Quantock_%282175262184%29.jpg/320px-Labrador_on_Quantock_%282175262184%29.jpg",
    ]

    # _raw always ends up as a concrete ndarray; initialise here so Pylance
    # can track the type through the conditional branches below.
    _raw: np.ndarray = np.zeros((224, 224, 3), dtype=np.uint8)
    _src_label: str = "uninitialised"

    # Build an SSL context that skips certificate verification — avoids
    # SSL: CERTIFICATE_VERIFY_FAILED errors on macOS / minimal Linux images.
    import ssl as _ssl
    _ssl_ctx = _ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = _ssl.CERT_NONE

    _downloaded = False
    for _url in _URLS:
        try:
            with urllib.request.urlopen(_url, context=_ssl_ctx, timeout=10) as _resp:
                with open("sample_raw.jpg", "wb") as _fout:
                    _fout.write(_resp.read())
            _downloaded = True
            _src_label = _url.split("/")[-1]
            break
        except Exception:
            continue

    if _downloaded:
        _imread_result = cv2.imread("sample_raw.jpg")
        if _imread_result is None:
            _downloaded = False
        else:
            _raw = _imread_result

    if not _downloaded:
        # Offline fallback: a structured sinusoidal colour image that has real
        # spatial gradients so saliency maps are still informative.
        _h, _w = 224, 224
        _y, _x = np.mgrid[0:_h, 0:_w].astype(np.float32)
        _r = (np.sin(2 * np.pi * _x / 56) * 127 + 128).astype(np.uint8)
        _g = (np.sin(2 * np.pi * _y / 56) * 127 + 128).astype(np.uint8)
        _b = (np.sin(2 * np.pi * (_x + _y) / 80) * 127 + 128).astype(np.uint8)
        _raw = np.stack([_b, _g, _r], axis=2)          # OpenCV is BGR
        _src_label = "synthetic sinusoidal (offline fallback)"

    _resized: np.ndarray = cv2.resize(_raw, (224, 224))  # type: ignore[call-overload]
    cv2.imwrite("sample.jpg", _resized)
    check("Setup: write sample.jpg", os.path.exists("sample.jpg"),
          f"source: {_src_label}")
except Exception as exc:
    check("Setup: write sample.jpg", False, str(exc))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. CLI commands
# ---------------------------------------------------------------------------

print("\n--- CLI ---\n")

# 1-a. --version
out, err, code = run_cli("--version")
check("CLI: --version", code == 0, out.strip())

# 1-b. list-methods
out, err, code = run_cli("list-methods")
check("CLI: list-methods", code == 0 and "gradcam" in out.lower())

# 1-c. show-layers resnet50
out, err, code = run_cli("show-layers", "--model", "resnet50")
check("CLI: show-layers --model resnet50", code == 0 and "layer4" in out)

# 1-d. explain – GradCAM
out, err, code = run_cli(
    "explain", "sample.jpg",
    "--model", "resnet50",
    "--method", "gradcam",
    "--no-display",
    "--output", "outputs/cli_gradcam.png",
)
_gcam_ok = code == 0 and os.path.exists("outputs/cli_gradcam.png") and os.path.getsize("outputs/cli_gradcam.png") > 2000
check(
    "CLI: explain (gradcam)",
    _gcam_ok,
    err.strip() if code != 0 else f"outputs/cli_gradcam.png written ({os.path.getsize('outputs/cli_gradcam.png') if os.path.exists('outputs/cli_gradcam.png') else 0} bytes)",
)

# 1-e. explain – SmoothGrad
out, err, code = run_cli(
    "explain", "sample.jpg",
    "--model", "resnet50",
    "--method", "smoothgrad",
    "--no-display",
    "--output", "outputs/cli_smoothgrad.png",
)
_sg_exists = os.path.exists("outputs/cli_smoothgrad.png")
check(
    "CLI: explain (smoothgrad)",
    code == 0 and _sg_exists and os.path.getsize("outputs/cli_smoothgrad.png") > 2000,
    err.strip() if code != 0 else f"{os.path.getsize('outputs/cli_smoothgrad.png') if _sg_exists else 0} bytes",
)

# 1-f. explain – Integrated Gradients
out, err, code = run_cli(
    "explain", "sample.jpg",
    "--model", "resnet50",
    "--method", "integrated_gradients",
    "--no-display",
    "--output", "outputs/cli_ig.png",
)
_ig_exists = os.path.exists("outputs/cli_ig.png")
check(
    "CLI: explain (integrated_gradients)",
    code == 0 and _ig_exists and os.path.getsize("outputs/cli_ig.png") > 2000,
    err.strip() if code != 0 else f"{os.path.getsize('outputs/cli_ig.png') if _ig_exists else 0} bytes",
)

# 1-g. explain – GradCAM++ with explicit target layer
out, err, code = run_cli(
    "explain", "sample.jpg",
    "--model", "resnet50",
    "--method", "gradcam++",
    "--layer", "layer4",
    "--no-display",
    "--output", "outputs/cli_gradcam_plus.png",
)
_gpp_exists = os.path.exists("outputs/cli_gradcam_plus.png")
check(
    "CLI: explain (gradcam++ --layer layer4)",
    code == 0 and _gpp_exists and os.path.getsize("outputs/cli_gradcam_plus.png") > 2000,
    err.strip() if code != 0 else f"{os.path.getsize('outputs/cli_gradcam_plus.png') if _gpp_exists else 0} bytes",
)

# 1-h. compare – multiple methods
out, err, code = run_cli(
    "compare", "sample.jpg",
    "--model", "resnet50",
    "--methods", "gradcam,smoothgrad,integrated_gradients",
    "--no-display",
    "--output", "outputs/cli_compare.png",
)
_cmp_exists = os.path.exists("outputs/cli_compare.png")
check(
    "CLI: compare (3 methods)",
    code == 0 and _cmp_exists and os.path.getsize("outputs/cli_compare.png") > 5000,
    err.strip() if code != 0 else f"{os.path.getsize('outputs/cli_compare.png') if _cmp_exists else 0} bytes",
)

# ---------------------------------------------------------------------------
# 2. Python API – setup
# ---------------------------------------------------------------------------

print("\n--- Python API: setup ---\n")

import torch
import torchvision.models as models
from scope_rx import ScopeRX
from scope_rx.utils import load_image, preprocess_image

try:
    model = models.resnet50(weights="DEFAULT")
    model.eval()
    scope = ScopeRX(model)
    check("API: ScopeRX(resnet50)", True, repr(scope))
except Exception as exc:
    check("API: ScopeRX(resnet50)", False, str(exc))
    sys.exit(1)

# Initialise with safe defaults so Pylance knows the type is always defined
# even if the try blocks below raise an exception.
img_np: np.ndarray = np.zeros((224, 224, 3), dtype=np.uint8)
input_tensor: torch.Tensor = torch.zeros(1, 3, 224, 224)

try:
    img_np = load_image("sample.jpg", size=(224, 224))
    check("Utils: load_image()", img_np.shape == (224, 224, 3), str(img_np.shape))
except Exception as exc:
    check("Utils: load_image()", False, str(exc))

try:
    _preprocessed = preprocess_image("sample.jpg")
    if not isinstance(_preprocessed, torch.Tensor):
        _preprocessed = torch.from_numpy(_preprocessed)
    input_tensor = _preprocessed.detach()  # detach to avoid requires_grad issues
    check("Utils: preprocess_image()", input_tensor.shape == (1, 3, 224, 224),
          str(input_tensor.shape))
except Exception as exc:
    check("Utils: preprocess_image()", False, str(exc))

# ---------------------------------------------------------------------------
# 3. available_methods() / get_predictions()
# ---------------------------------------------------------------------------

print("\n--- Python API: discovery ---\n")

try:
    methods = scope.available_methods()
    _expected_methods = {"gradcam", "gradcam++", "smoothgrad", "integrated_gradients",
                         "vanilla_gradients", "guided_backprop", "occlusion"}
    _missing = _expected_methods - set(methods)
    check("API: available_methods()",
          len(methods) > 0 and not _missing,
          f"{len(methods)} methods" if not _missing else f"missing: {_missing}")
except Exception as exc:
    check("API: available_methods()", False, str(exc))
    methods = []

try:
    probs, classes = scope.get_predictions(input_tensor, top_k=5)
    predicted_class = int(classes[0, 0].item())
    top_prob = float(probs[0, 0].item())
    _probs_np = probs.cpu().numpy().flatten()
    _probs_valid = (
        probs.shape == (1, 5)
        and bool(np.all(_probs_np >= 0)) and bool(np.all(_probs_np <= 1))
        and top_prob > 1e-4                  # model is confident about something
        and float(_probs_np.sum()) <= 1.01   # top-5 can't exceed total probability
    )
    check("API: get_predictions(top_k=5)", _probs_valid,
          f"top class={predicted_class}, p={top_prob:.4f}, top5_sum={_probs_np.sum():.4f}")
except Exception as exc:
    check("API: get_predictions()", False, str(exc))
    predicted_class = 0

# ---------------------------------------------------------------------------
# 4. Gradient / CAM methods via scope.explain()
# ---------------------------------------------------------------------------

print("\n--- Python API: gradient / CAM methods ---\n")

cam_methods = ["gradcam", "gradcam++", "scorecam", "layercam"]
grad_methods = [
    "smoothgrad",
    "integrated_gradients",
    "vanilla_gradients",
    "guided_backprop",
]

_last_result = None
for mname in cam_methods + grad_methods:
    try:
        res = scope.explain(
            input_tensor,
            method=mname,
            target_class=predicted_class,
        )
        _ok, _detail = _attr_ok(res.attribution)
        check(f"API: explain({mname})", _ok, _detail)
        _last_result = res
    except Exception as exc:
        check(f"API: explain({mname})", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# 5. Perturbation methods
# ---------------------------------------------------------------------------

print("\n--- Python API: perturbation methods ---\n")

# Occlusion
try:
    res = scope.explain(
        input_tensor,
        method="occlusion",
        target_class=predicted_class,
    )
    _ok, _detail = _attr_ok(res.attribution)
    check("API: explain(occlusion)", _ok, _detail)
except Exception as exc:
    check("API: explain(occlusion)", False, traceback.format_exc().splitlines()[-1])

# RISE – use a small mask count to keep it fast
try:
    from scope_rx.methods.perturbation import RISE
    rise = RISE(model, num_masks=100, mask_size=7)
    res_rise = rise.explain(input_tensor, target_class=predicted_class)
    _ok, _detail = _attr_ok(res_rise.attribution)
    check("API: RISE.explain()", _ok, _detail)
except Exception as exc:
    check("API: RISE.explain()", False, traceback.format_exc().splitlines()[-1])

# MeaningfulPerturbation
try:
    res = scope.explain(
        input_tensor,
        method="meaningful_perturbation",
        target_class=predicted_class,
    )
    _ok, _detail = _attr_ok(res.attribution)
    check("API: explain(meaningful_perturbation)", _ok, _detail)
except Exception as exc:
    check("API: explain(meaningful_perturbation)", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# 6. Model-agnostic methods
# ---------------------------------------------------------------------------

print("\n--- Python API: model-agnostic methods ---\n")

# KernelSHAP – small sample count
try:
    from scope_rx.methods.model_agnostic import KernelSHAP
    shap_exp = KernelSHAP(model, num_samples=100, num_segments=20)
    res_shap = shap_exp.explain(input_tensor, target_class=predicted_class)
    _ok, _detail = _attr_ok(res_shap.attribution)
    check("API: KernelSHAP.explain()", _ok, _detail)
except Exception as exc:
    check("API: KernelSHAP.explain()", False, traceback.format_exc().splitlines()[-1])

# LIME – small sample count
try:
    from scope_rx.methods.model_agnostic import LIME
    lime_exp = LIME(model, num_samples=100, num_segments=20)
    res_lime = lime_exp.explain(input_tensor, target_class=predicted_class)
    _ok, _detail = _attr_ok(res_lime.attribution)
    check("API: LIME.explain()", _ok, _detail)
except Exception as exc:
    check("API: LIME.explain()", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# 7. compare_methods()  +  ComparisonResult helpers
# ---------------------------------------------------------------------------

print("\n--- Python API: compare_methods ---\n")

try:
    comparison = scope.compare_methods(
        input_tensor,
        methods=["gradcam", "smoothgrad", "integrated_gradients"],
        target_class=predicted_class,
    )
    _cmp_methods = set(comparison.methods)
    _cmp_expected = {"gradcam", "smoothgrad", "integrated_gradients"}
    _cmp_missing = _cmp_expected - _cmp_methods
    check("API: compare_methods(3 methods)",
          len(comparison.methods) == 3 and not _cmp_missing,
          str(comparison.methods) if not _cmp_missing else f"missing: {_cmp_missing}")
except Exception as exc:
    check("API: compare_methods()", False, traceback.format_exc().splitlines()[-1])
    comparison = None

if comparison is not None:
    try:
        fig = comparison.visualize_all(
            image=img_np,
            save_path="outputs/compare_all.png",
        )
        plt.close("all")
        _fsize = os.path.getsize("outputs/compare_all.png")
        check("API: comparison.visualize_all()",
              os.path.exists("outputs/compare_all.png") and _fsize > 5000,
              f"file size={_fsize} bytes")
    except Exception as exc:
        check("API: comparison.visualize_all()", False, traceback.format_exc().splitlines()[-1])

    try:
        import pandas as pd  # type: ignore[import-not-found]
        df = comparison.to_dataframe()
        check("API: comparison.to_dataframe()", len(df) > 0,
              df[["method", "target_class", "confidence"]].to_string(index=False))
    except ImportError:
        check("API: comparison.to_dataframe()", True,
              "pandas not installed – skipped (not a scope-rx dependency)")
    except Exception as exc:
        check("API: comparison.to_dataframe()", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# 8. ExplanationResult helpers
# ---------------------------------------------------------------------------

print("\n--- Python API: ExplanationResult ---\n")

# Grab a fresh GradCAM result for these tests
result = scope.explain(
    input_tensor,
    method="gradcam",
    target_class=predicted_class,
)

try:
    heatmap = result.to_heatmap(colormap="jet")
    _hm_h, _hm_w = result.attribution.shape[-2], result.attribution.shape[-1]
    _hm_shape_ok = heatmap.ndim == 3 and heatmap.shape[2] == 3
    _hm_dtype_ok = heatmap.dtype == np.uint8 or float(heatmap.max()) <= 1.0
    _hm_range_ok = float(heatmap.min()) >= 0
    _hm_vary_ok = np.std(heatmap) > 0          # not a solid colour
    check("result.to_heatmap()",
          _hm_shape_ok and _hm_dtype_ok and _hm_range_ok and _hm_vary_ok,
          f"shape={heatmap.shape} dtype={heatmap.dtype} std={np.std(heatmap):.2f}")
except Exception as exc:
    check("result.to_heatmap()", False, str(exc))

try:
    overlay = result.overlay(img_np, alpha=0.6)
    _ov_shape_ok = overlay is not None and overlay.shape == img_np.shape
    _ov_range_ok = float(overlay.min()) >= 0
    _ov_vary_ok = np.std(overlay) > 0
    check("result.overlay()",
          _ov_shape_ok and _ov_range_ok and _ov_vary_ok,
          f"shape={overlay.shape} (expected {img_np.shape})")
except Exception as exc:
    check("result.overlay()", False, str(exc))

try:
    result.visualize(image=img_np, show=False, save_path="outputs/result_visualize.png")
    plt.close("all")
    _fsize = os.path.getsize("outputs/result_visualize.png")
    check("result.visualize()",
          os.path.exists("outputs/result_visualize.png") and _fsize > 2000,
          f"{_fsize} bytes")
except Exception as exc:
    check("result.visualize()", False, str(exc))

try:
    result.save("outputs/result_save.png")
    _fsize = os.path.getsize("outputs/result_save.png")
    check("result.save() → png",
          os.path.exists("outputs/result_save.png") and _fsize > 500,
          f"{_fsize} bytes")
except Exception as exc:
    check("result.save() → png", False, str(exc))

try:
    result.save("outputs/result_save.npy")
    _arr = np.load("outputs/result_save.npy")
    check("result.save() → npy",
          os.path.exists("outputs/result_save.npy") and _arr.size > 0,
          f"loaded shape={_arr.shape}")
except Exception as exc:
    check("result.save() → npy", False, str(exc))

try:
    check("result.normalized_attribution",
          (
              float(result.normalized_attribution.min()) >= 0.0
              and float(result.normalized_attribution.max()) <= 1.0
              and float(result.normalized_attribution.max()) > 0.0      # not all-zero
              and float(np.std(result.normalized_attribution)) > 1e-6   # has variation
          ))
except Exception as exc:
    check("result.normalized_attribution", False, str(exc))

try:
    check("result.shape", isinstance(result.shape, tuple))
except Exception as exc:
    check("result.shape", False, str(exc))

try:
    check("result.__repr__", "ExplanationResult" in repr(result))
except Exception as exc:
    check("result.__repr__", False, str(exc))

# ---------------------------------------------------------------------------
# 9. Utility functions
# ---------------------------------------------------------------------------

print("\n--- Utils ---\n")

try:
    from scope_rx.utils import normalize_attribution, smooth_attribution, threshold_attribution
    attr = result.attribution

    norm = normalize_attribution(attr)
    _norm_min, _norm_max = float(norm.min()), float(norm.max())
    check("Utils: normalize_attribution() minmax",
          _norm_min >= 0.0 and _norm_max <= 1.0 and _norm_max > 0.5,   # actually rescaled
          f"[{_norm_min:.4f}, {_norm_max:.4f}]")

    norm_abs = normalize_attribution(attr, method="abs_max")
    check("Utils: normalize_attribution() abs_max",
          norm_abs is not None and np.all(np.isfinite(norm_abs)) and float(np.abs(norm_abs).max()) <= 1.0,
          f"max_abs={float(np.abs(norm_abs).max()):.4f}")

    norm_pct = normalize_attribution(attr, method="percentile")
    check("Utils: normalize_attribution() percentile",
          norm_pct is not None and np.all(np.isfinite(norm_pct)),
          f"range=[{float(norm_pct.min()):.4f}, {float(norm_pct.max()):.4f}]")

    smoothed = smooth_attribution(attr, sigma=2.0)
    check("Utils: smooth_attribution()",
          smoothed.shape == attr.shape and np.all(np.isfinite(smoothed)),
          str(smoothed.shape))

    thresholded = threshold_attribution(attr, threshold=0.5)
    # threshold_attribution zeros out values < 0.5 but keeps originals above.
    # So every non-zero entry must be >= 0.5 (when attr is normalized [0,1]).
    _thr_norm = normalize_attribution(attr)  # ensure [0, 1] before checking
    _thr_result = threshold_attribution(_thr_norm, threshold=0.5)
    _below_thresh_nonzero = float(np.any((_thr_result > 0) & (_thr_result < 0.5)))
    check("Utils: threshold_attribution()",
          thresholded is not None and not _below_thresh_nonzero,
          f"nonzero below threshold={_below_thresh_nonzero}")
except Exception as exc:
    check("Utils: postprocessing", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.utils import to_numpy, to_tensor, ensure_4d

    t = to_tensor(img_np)
    check("Utils: to_tensor(numpy→Tensor)",
          isinstance(t, torch.Tensor) and t.ndim >= 3 and bool(torch.all(torch.isfinite(t))),
          str(t.shape))

    n = to_numpy(t)
    check("Utils: to_numpy(Tensor→numpy)",
          isinstance(n, np.ndarray) and n.ndim >= 3 and np.all(np.isfinite(n)),
          str(n.shape))

    t4d = ensure_4d(input_tensor.squeeze(0))
    check("Utils: ensure_4d(3D→4D)", t4d.dim() == 4, str(t4d.shape))

    t4d_already = ensure_4d(input_tensor)
    check("Utils: ensure_4d(4D→4D, no-op)", t4d_already.dim() == 4, str(t4d_already.shape))
except Exception as exc:
    check("Utils: tensor utils", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.utils import normalize_image, denormalize_image

    norm_img = normalize_image(img_np.astype(np.float32) / 255.0)
    check("Utils: normalize_image()",
          norm_img is not None and np.all(np.isfinite(norm_img)),
          f"range=[{float(norm_img.min()):.4f}, {float(norm_img.max()):.4f}]")

    # denormalize expects (H, W, C) float array
    inp_hwc = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    denorm = denormalize_image(inp_hwc)
    check("Utils: denormalize_image()",
          denorm is not None and np.all(np.isfinite(denorm)) and float(denorm.max()) > 0,
          f"range=[{float(denorm.min()):.4f}, {float(denorm.max()):.4f}]")
except Exception as exc:
    check("Utils: normalize/denormalize", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# 10. Visualization functions
# ---------------------------------------------------------------------------

print("\n--- Visualization ---\n")

try:
    from scope_rx.visualization import plot_attribution
    fig = plot_attribution(
        result.attribution,
        image=img_np,
        title="GradCAM",
        colormap="hot",
        show=False,
        save_path="outputs/plot_attr.png",
    )
    plt.close("all")
    _fsize = os.path.getsize("outputs/plot_attr.png")
    check("Viz: plot_attribution() with image",
          os.path.exists("outputs/plot_attr.png") and _fsize > 5000,
          f"{_fsize} bytes")
except Exception as exc:
    check("Viz: plot_attribution() with image", False, traceback.format_exc().splitlines()[-1])

try:
    fig = plot_attribution(result.attribution, show=False,
                           save_path="outputs/plot_attr_noimg.png")
    plt.close("all")
    _fsize = os.path.getsize("outputs/plot_attr_noimg.png")
    check("Viz: plot_attribution() without image",
          os.path.exists("outputs/plot_attr_noimg.png") and _fsize > 2000,
          f"{_fsize} bytes")
except Exception as exc:
    check("Viz: plot_attribution() without image", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.visualization import plot_comparison
    ig_result = scope.explain(input_tensor, method="integrated_gradients",
                              target_class=predicted_class)
    attributions = {
        "GradCAM": result.attribution,
        "Integrated Gradients": ig_result.attribution,
    }
    fig = plot_comparison(
        attributions,
        image=img_np,
        save_path="outputs/plot_compare.png",
        show=False,
    )
    plt.close("all")
    _fsize = os.path.getsize("outputs/plot_compare.png")
    check("Viz: plot_comparison()",
          os.path.exists("outputs/plot_compare.png") and _fsize > 5000,
          f"{_fsize} bytes")
except Exception as exc:
    check("Viz: plot_comparison()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.visualization import overlay_attribution
    ov = overlay_attribution(result.attribution, img_np)
    _ov_shape_ok = ov is not None and ov.ndim == 3 and ov.shape == img_np.shape
    _ov_vary_ok = np.std(ov) > 0
    check("Viz: overlay_attribution()",
          _ov_shape_ok and _ov_vary_ok,
          f"shape={ov.shape} std={np.std(ov):.2f}")
except Exception as exc:
    check("Viz: overlay_attribution()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.visualization import export_visualization
    export_visualization(result.attribution, "outputs/export.png", image=img_np)
    _fsize_png = os.path.getsize("outputs/export.png")
    check("Viz: export_visualization() → png",
          os.path.exists("outputs/export.png") and _fsize_png > 2000,
          f"{_fsize_png} bytes")
    export_visualization(result.attribution, "outputs/export.npy")
    _npy = np.load("outputs/export.npy")
    check("Viz: export_visualization() → npy",
          os.path.exists("outputs/export.npy") and _npy.size > 0,
          f"loaded shape={_npy.shape}")
    export_visualization(result.attribution, "outputs/export.npz")
    _npz = np.load("outputs/export.npz")
    check("Viz: export_visualization() → npz",
          os.path.exists("outputs/export.npz") and len(_npz.files) > 0,
          f"keys={list(_npz.files)}")
except Exception as exc:
    check("Viz: export_visualization()", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# 11. Metrics
# ---------------------------------------------------------------------------

print("\n--- Metrics ---\n")

from scope_rx.core.wrapper import auto_detect_target_layer
from scope_rx.methods.gradient import GradCAM

_target_layer_name = auto_detect_target_layer(model)
if _target_layer_name is None:
    raise RuntimeError("Could not auto-detect a target layer on resnet50")
_target_layer = dict(model.named_modules())[_target_layer_name]
_gc_explainer = GradCAM(model, target_layer=_target_layer)

try:
    from scope_rx.metrics import insertion_deletion_auc
    ins_auc, del_auc = insertion_deletion_auc(
        model, input_tensor, result.attribution,
        target_class=predicted_class,
        num_steps=20,
    )
    # Both are probabilities: must be in [0, 1]. Insertion should exceed deletion
    # for a useful attribution (preserving salient pixels keeps confidence high).
    _auc_ok = (
        isinstance(ins_auc, float) and isinstance(del_auc, float)
        and 0.0 <= ins_auc <= 1.0 and 0.0 <= del_auc <= 1.0
        and ins_auc >= del_auc  # meaningful attribution: inserting beats deleting
    )
    check("Metric: insertion_deletion_auc()", _auc_ok,
          f"insertion={ins_auc:.4f}, deletion={del_auc:.4f}")
except Exception as exc:
    check("Metric: insertion_deletion_auc()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import faithfulness_score
    fs = faithfulness_score(
        model, input_tensor, result.attribution,
        target_class=predicted_class,
        num_steps=20,
    )
    # Faithfulness = (insertion_auc + (1 - deletion_auc)) / 2 → always in [0, 1]
    check("Metric: faithfulness_score()",
          isinstance(fs, float) and np.isfinite(fs) and 0.0 <= fs <= 1.0,
          f"score={fs:.4f}")
except Exception as exc:
    check("Metric: faithfulness_score()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import aopc_score
    ao = aopc_score(
        model, input_tensor, result.attribution,
        target_class=predicted_class,
        percentages=[10, 20, 30],
    )
    # AOPC is a difference of probabilities: finite and roughly in [-1, 1]
    check("Metric: aopc_score()",
          isinstance(ao, float) and np.isfinite(ao) and -1.05 <= ao <= 1.05,
          f"score={ao:.4f}")
except Exception as exc:
    check("Metric: aopc_score()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import sufficiency_score
    ss = sufficiency_score(
        model, input_tensor, result.attribution,
        target_class=predicted_class,
        percentage=20,
    )
    # Sufficiency is a probability of the target class: must be in [0, 1]
    check("Metric: sufficiency_score()",
          isinstance(ss, float) and np.isfinite(ss) and 0.0 <= ss <= 1.0,
          f"score={ss:.4f}")
except Exception as exc:
    check("Metric: sufficiency_score()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import sensitivity_score
    sens = sensitivity_score(
        _gc_explainer, input_tensor,
        target_class=predicted_class,
        num_samples=3,
        noise_level=0.05,
    )
    # Sensitivity is an L2 distance between attributions: must be non-negative
    check("Metric: sensitivity_score()",
          isinstance(sens, float) and np.isfinite(sens) and sens >= 0.0,
          f"score={sens:.4f}")
except Exception as exc:
    check("Metric: sensitivity_score()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import max_sensitivity
    ms = max_sensitivity(
        _gc_explainer, input_tensor,
        target_class=predicted_class,
        num_samples=3,
        noise_level=0.05,
    )
    check("Metric: max_sensitivity()",
          isinstance(ms, (float, np.floating)) and np.isfinite(ms) and float(ms) >= 0.0,
          f"score={float(ms):.4f}")
except Exception as exc:
    check("Metric: max_sensitivity()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import avg_sensitivity
    av = avg_sensitivity(
        _gc_explainer, input_tensor,
        target_class=predicted_class,
        num_samples=3,
        noise_level=0.05,
    )
    check("Metric: avg_sensitivity()",
          isinstance(av, float) and np.isfinite(av) and av >= 0.0,
          f"score={av:.4f}")
except Exception as exc:
    check("Metric: avg_sensitivity()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import stability_score
    similar = [input_tensor + torch.randn_like(input_tensor) * 0.01 for _ in range(3)]
    stab = stability_score(_gc_explainer, similar, target_class=predicted_class)
    # Stability is mean of (corr+1)/2 pairwise similarities: range [0, 1]
    check("Metric: stability_score()",
          isinstance(stab, float) and np.isfinite(stab) and 0.0 <= stab <= 1.0,
          f"score={stab:.4f}")
except Exception as exc:
    check("Metric: stability_score()", False, traceback.format_exc().splitlines()[-1])

try:
    from scope_rx.metrics import explanation_consistency
    cons = explanation_consistency(
        _gc_explainer, input_tensor,
        target_class=predicted_class,
        num_runs=3,
    )
    # Consistency is exp(-variance * 10): always in (0, 1]
    check("Metric: explanation_consistency()",
          isinstance(cons, float) and np.isfinite(cons) and 0.0 < cons <= 1.0,
          f"score={cons:.4f}")
except Exception as exc:
    check("Metric: explanation_consistency()", False, traceback.format_exc().splitlines()[-1])

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
passed = sum(1 for _, ok in _results if ok)
total = len(_results)
print(f"  Results: {passed}/{total} passed")
print("=" * 65)

failed = [(label, ok) for label, ok in _results if not ok]
if failed:
    print("\nFailed tests:")
    for label, _ in failed:
        print(f"  \033[91m✗\033[0m  {label}")
    print()
    sys.exit(1)
else:
    print("\n  \033[92mAll tests passed!\033[0m\n")
    sys.exit(0)
