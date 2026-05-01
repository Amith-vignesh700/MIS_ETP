"""
ml_engine/inference.py
=======================
End-to-end inference helper: .mat file → JSON-ready prediction dict.

Used by the Django API view (api/views.py) to:
    1. Load the trained CNN-LSTM weights from disk (once, at startup)
    2. Accept a raw .mat file upload
    3. Run Phase-1 signal processing (DFT → Hankelisation → DMD)
    4. Prepare a PyTorch tensor and run model inference
    5. Return a structured dict suitable for JSON serialisation,
       including a Base64-encoded strain heatmap PNG.

Public API
----------
    model_instance = load_model(weights_path, device)
    result         = run_inference(model_instance, mat_filepath, device)
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Django
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .architecture    import CNNLSTMElastography
from .signal_processing import process_rf_pipeline


# ── Default model configuration (must match training hyperparameters) ─────────
DEFAULT_CONFIG: dict[str, Any] = {
    "cnn_channels": [1, 16, 32, 64, 128],
    "lstm_hidden":  256,
    "lstm_layers":  2,
    "n_classes":    3,
    "patch_size":   64,
    "dropout":      0.3,
}

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "weights", "cnn_lstm_final.pth"
)

# ── Inference patch / temporal window settings ────────────────────────────────
PATCH_SIZE = DEFAULT_CONFIG["patch_size"]
N_FRAMES   = 8


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    weights_path: str  = DEFAULT_WEIGHTS,
    device:       str  = "cpu",
    config:       dict[str, Any] | None = None,
) -> CNNLSTMElastography:
    """
    Instantiate the CNN-LSTM model and load trained weights.

    Parameters
    ----------
    weights_path : str  — path to .pth checkpoint file
    device       : str  — 'cpu' or 'cuda'
    config       : dict — architecture config (defaults to DEFAULT_CONFIG)

    Returns
    -------
    model : CNNLSTMElastography  in eval() mode
    """
    cfg = config or DEFAULT_CONFIG

    model = CNNLSTMElastography(
        cnn_channels = cfg["cnn_channels"],
        lstm_hidden  = cfg["lstm_hidden"],
        lstm_layers  = cfg["lstm_layers"],
        n_classes    = cfg["n_classes"],
        patch_size   = cfg["patch_size"],
        dropout      = cfg.get("dropout", 0.3),
    )

    if os.path.isfile(weights_path):
        ckpt       = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[ml_engine] Loaded weights from: {weights_path}")
    else:
        warnings.warn(
            f"[ml_engine] Weight file not found at '{weights_path}'. "
            "Model running with random initialisation (demo mode).",
            RuntimeWarning,
            stacklevel=2,
        )

    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Tensor Preparation
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_tensor(
    rf_clean: np.ndarray,
    patch_size: int = PATCH_SIZE,
    n_frames:   int = N_FRAMES,
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Crop, window and convert a clean RF stack into a model-ready tensor.

    Crops a central spatial patch and the first ``n_frames`` temporal frames.
    Shape of returned tensor: (1, T, 1, H, W)

    Parameters
    ----------
    rf_clean   : (D, W, F) float64 DMD-reconstructed RF stack
    patch_size : spatial crop size (H = W = patch_size)
    n_frames   : number of temporal frames to use
    device     : torch device string

    Returns
    -------
    tensor : (1, T, 1, H, W) float32 PyTorch tensor on `device`
    """
    D, W, F = rf_clean.shape

    # Central spatial crop
    d0 = max(0, (D - patch_size) // 2)
    w0 = max(0, (W - patch_size) // 2)
    d0 = min(d0, max(0, D - patch_size))
    w0 = min(w0, max(0, W - patch_size))

    patch = rf_clean[d0: d0 + patch_size, w0: w0 + patch_size, :]   # (P, P, F)

    # Pad depth / width if smaller than patch_size
    pd = max(0, patch_size - patch.shape[0])
    pw = max(0, patch_size - patch.shape[1])
    if pd > 0 or pw > 0:
        patch = np.pad(patch, ((0, pd), (0, pw), (0, 0)), mode="edge")

    # Temporal window
    t_end   = min(n_frames, F)
    patch_t = patch[:, :, :t_end]                                    # (P, P, T)

    # Pad time axis if needed
    if patch_t.shape[2] < n_frames:
        pad_f   = n_frames - patch_t.shape[2]
        patch_t = np.pad(patch_t, ((0, 0), (0, 0), (0, pad_f)), mode="edge")

    # Normalise to [-1, 1]
    vmax = np.abs(patch_t).max()
    if vmax > 0:
        patch_t = patch_t / vmax

    # (P, P, T) → (T, P, P) → (1, T, 1, P, P)
    tensor = torch.from_numpy(patch_t).float()               # (P, P, T)
    tensor = tensor.permute(2, 0, 1).unsqueeze(1)            # (T, 1, P, P)
    tensor = tensor.unsqueeze(0)                              # (1, T, 1, P, P)
    return tensor.to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  Strain Heatmap Rendering
# ══════════════════════════════════════════════════════════════════════════════

def _render_strain_heatmap(displacement: np.ndarray) -> str:
    """
    Render the mean displacement field as a colourised PNG and return
    it as a Base64-encoded string suitable for JSON embedding.

    Parameters
    ----------
    displacement : (T, H, W) or (H, W) mean strain map (numpy)

    Returns
    -------
    b64_png : str — base64-encoded PNG image
    """
    if displacement.ndim == 3:
        strain_map = displacement.mean(axis=0)   # (H, W)
    else:
        strain_map = displacement

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(strain_map, cmap="jet", aspect="auto", origin="upper")
    ax.set_title("Strain Heatmap", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Displacement (a.u.)")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    b64_png = base64.b64encode(buf.read()).decode("utf-8")
    return b64_png


# ══════════════════════════════════════════════════════════════════════════════
#  Main Inference Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    model:       CNNLSTMElastography,
    mat_filepath: str,
    device:      str = "cpu",
    patch_size:  int = PATCH_SIZE,
    n_frames:    int = N_FRAMES,
    dft_low:     float = 0.05,
    dft_high:    float = 0.45,
    dmd_damping: float = 0.85,
) -> dict[str, Any]:
    """
    Full inference pipeline: .mat file → structured prediction dict.

    Workflow
    --------
    1. Phase-1 signal processing (DFT filter → Hankelisation → DMD reconstruction)
    2. Prepare (1, T, 1, H, W) tensor
    3. Forward pass through the CNN-LSTM
    4. Compute class probabilities via Softmax
    5. Render strain heatmap as Base64 PNG

    Parameters
    ----------
    model        : loaded CNNLSTMElastography in eval() mode
    mat_filepath : str — path to .mat file (pre- or post-compression)
    device       : str — 'cpu' or 'cuda'
    patch_size   : spatial crop size
    n_frames     : temporal window
    dft_low      : DFT bandpass lower bound
    dft_high     : DFT bandpass upper bound
    dmd_damping  : |λ| threshold for DMD mode selection

    Returns
    -------
    result : dict with keys:
        diagnosis         — str  ('Benign' | 'Precancerous' | 'Malignant')
        class_index       — int  (0 | 1 | 2)
        confidence_scores — dict  {'Benign': float, 'Precancerous': float, 'Malignant': float}
        strain_heatmap_b64— str  Base64-encoded PNG
        dmd_rank          — int  DMD truncation rank used
        rf_shape          — list [D, W, F]
        error             — str | None  (non-None if processing failed)
    """
    error_msg = None

    try:
        # ── Phase-1: Signal Processing ────────────────────────────────────────
        proc = process_rf_pipeline(
            filepath     = mat_filepath,
            low_cutoff   = dft_low,
            high_cutoff  = dft_high,
            dmd_damping  = dmd_damping,
            n_augmented  = 0,         # no augmentation needed at inference
        )
        rf_clean = proc["rf_clean"]   # (D, W, F)
        dmd_rank = proc["dmd_rank"]
        D, W, F  = proc["shape"]

        # ── Tensor Preparation ────────────────────────────────────────────────
        x_tensor = _prepare_tensor(
            rf_clean, patch_size=patch_size, n_frames=n_frames, device=device
        )

        # ── Model Forward Pass ────────────────────────────────────────────────
        with torch.no_grad():
            logits, displacement_t = model(x_tensor)
            probs = torch.softmax(logits, dim=-1)   # (1, n_classes)

        # ── Results ──────────────────────────────────────────────────────────
        class_names = CNNLSTMElastography.CLASS_NAMES
        pred_idx    = int(probs.argmax(dim=-1).item())
        prob_dict   = {
            name: round(float(probs[0, i].item()), 4)
            for i, name in enumerate(class_names)
        }

        # Displacement numpy for heatmap
        disp_np    = displacement_t[0].cpu().numpy()   # (T, H, W)
        heatmap_b64 = _render_strain_heatmap(disp_np)

    except Exception as exc:
        error_msg   = str(exc)
        pred_idx    = -1
        prob_dict   = {"Benign": 0.0, "Precancerous": 0.0, "Malignant": 0.0}
        heatmap_b64 = ""
        dmd_rank    = -1
        D = W = F   = 0

    return {
        "diagnosis":           class_names[pred_idx] if pred_idx >= 0 else "Error",
        "class_index":         pred_idx,
        "confidence_scores":   prob_dict,
        "strain_heatmap_b64":  heatmap_b64,
        "dmd_rank":            dmd_rank,
        "rf_shape":            [D, W, F],
        "error":               error_msg,
    }
