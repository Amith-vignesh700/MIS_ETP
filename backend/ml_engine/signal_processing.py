"""
ml_engine/signal_processing.py
===============================
Signal processing pipeline for ultrasound RF data.

Phase 1 pipeline (called by notebooks and Django inference view):
    1. load_mat_file           — Load .mat v5 / v7.3-HDF5 files
    2. extract_rf_frames       — Detect RF key and return (D, W, F) array
    3. apply_2d_dft_bandpass   — 2D DFT frequency filtering (per frame)
    4. filter_rf_stack         — Apply DFT filter to every frame in a stack
    5. build_global_hankel_matrix — Time-delay embedding of RF stack
    6. exact_dmd               — Exact Dynamic Mode Decomposition
    7. dmd_reconstruct         — Reconstruct clean signal from dominant DMD modes
    8. inject_rayleigh_noise   — Sim-to-real domain randomisation (Rayleigh phase jitter)
    9. process_rf_pipeline     — Full Phase-1 pipeline in one call

Mathematical Basis
------------------
    • 2D DFT: F(u,v) = Σ_x Σ_y f(x,y)·exp(-j2π(ux/M + vy/N))
    • Hankel embedding: maps 1-D pixel evolution → high-dimensional state space
    • Exact DMD: X₂ ≈ AX₁; eigenvalues encode dynamics; damped modes = artefacts
    • Rayleigh noise: magnitude drawn from Rayleigh(σ), phase uniform ∈ [0, 2π)
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.io as sio
import h5py
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
#  I/O Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_mat_file(filepath: str) -> dict[str, Any]:
    """
    Load a MATLAB .mat file (v5 or v7.3 / HDF5) and return a flat dict.

    Strategy
    --------
    1. Try ``scipy.io.loadmat`` (handles MATLAB ≤ v7.2 format).
    2. On failure, fall back to ``h5py`` (MATLAB v7.3 = HDF5).

    Parameters
    ----------
    filepath : str — absolute or relative path to the .mat file

    Returns
    -------
    data : dict[str, np.ndarray]
        Filtered to exclude MATLAB meta-keys (starting with '__').
    """
    try:
        data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        data: dict[str, Any] = {}
        with h5py.File(filepath, "r") as f:
            def _extract(name: str, obj: h5py.Dataset | h5py.Group) -> None:
                if isinstance(obj, h5py.Dataset):
                    data[name] = np.array(obj)
            f.visititems(_extract)
        return data


def extract_rf_frames(mat_data: dict[str, Any]) -> np.ndarray:
    """
    Extract the RF signal matrix from a loaded .mat data dict.

    Tries common Concordia dataset key names in priority order.
    Returns a float64 array of shape (depth, width, n_frames).

    Parameters
    ----------
    mat_data : dict returned by ``load_mat_file``

    Returns
    -------
    rf_stack : (D, W, F) np.ndarray  float64

    Raises
    ------
    KeyError : if no recognised RF key is found
    """
    candidate_keys = ["rf_pre", "rf_post", "RF", "rfData", "IQpre", "IQpost",
                      "rf", "RFdata", "data"]

    for key in candidate_keys:
        if key in mat_data:
            arr = np.array(mat_data[key], dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]          # (D, W) → (D, W, 1)
            elif arr.ndim == 3:
                pass                                  # already (D, W, F)
            else:
                continue
            return arr

    raise KeyError(
        f"No RF key found in .mat file. Available keys: {list(mat_data.keys())}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — 2D Discrete Fourier Transform Bandpass Filter
# ══════════════════════════════════════════════════════════════════════════════

def apply_2d_dft_bandpass(
    rf_frame:    np.ndarray,
    low_cutoff:  float = 0.05,
    high_cutoff: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a 2-D DFT bandpass filter to one RF frame.

    Mathematical basis
    ------------------
        F(u,v) = Σ_x Σ_y f(x,y) · exp(-j2π(ux/M + vy/N))
    Components outside [low_cutoff, high_cutoff] · Nyquist are zeroed
    before reconstruction via IDFT.

    Parameters
    ----------
    rf_frame    : (D, W) 2-D RF image
    low_cutoff  : normalised frequency lower bound (fraction of Nyquist, default 0.05)
    high_cutoff : normalised frequency upper bound (fraction of Nyquist, default 0.45)

    Returns
    -------
    filtered_frame : (D, W) real-valued filtered RF
    F_raw          : (D, W) |FFT| of original (for visualisation)
    F_filtered     : (D, W) |FFT| after masking (for visualisation)
    mask           : (D, W) boolean bandpass mask
    """
    D, W = rf_frame.shape

    # 2D FFT, zero-centred
    F_shifted = fftshift(fft2(rf_frame))

    # Build radial bandpass mask in frequency domain
    u = fftshift(np.fft.fftfreq(D))
    v = fftshift(np.fft.fftfreq(W))
    U, V = np.meshgrid(u, v, indexing="ij")
    freq_mag = np.sqrt(U**2 + V**2)
    mask = (freq_mag >= low_cutoff) & (freq_mag <= high_cutoff)

    F_filtered_shifted = F_shifted * mask
    filtered_frame = np.real(ifft2(ifftshift(F_filtered_shifted)))

    return (
        filtered_frame,
        np.abs(F_shifted),
        np.abs(F_filtered_shifted),
        mask,
    )


def filter_rf_stack(
    rf_stack:    np.ndarray,
    low_cutoff:  float = 0.05,
    high_cutoff: float = 0.45,
) -> np.ndarray:
    """
    Apply 2-D DFT bandpass filter to every frame in the stack.

    Parameters
    ----------
    rf_stack    : (D, W, F) RF stack
    low_cutoff  : see ``apply_2d_dft_bandpass``
    high_cutoff : see ``apply_2d_dft_bandpass``

    Returns
    -------
    filtered_stack : (D, W, F) float64
    """
    D, W, F = rf_stack.shape
    filtered = np.zeros_like(rf_stack)
    for i in range(F):
        filtered[:, :, i], _, _, _ = apply_2d_dft_bandpass(
            rf_stack[:, :, i], low_cutoff=low_cutoff, high_cutoff=high_cutoff
        )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Hankelization (Time-Delay Embedding)
# ══════════════════════════════════════════════════════════════════════════════

def build_global_hankel_matrix(
    rf_stack: np.ndarray,
    delay:    int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Construct a global Hankel (time-delay embedding) data matrix.

    For an RF stack (D, W, F), each pixel has a temporal series of length F.
    Embedding with delay m yields (F−m+1) time-shifted snapshots.
    All pixel embeddings are stacked as rows → global matrix X.

    Hankel matrix for a single pixel signal s₁, …, s_F:
        H(s) = [[s₁, s₂, …, s_{F−m+1}],
                 [s₂, s₃, …, s_{F−m+2}],
                 …
                 [s_m, s_{m+1}, …, s_F]]  ∈ ℝ^{m × (F−m+1)}

    Parameters
    ----------
    rf_stack : (D, W, F) RF stack (typically DFT-filtered)
    delay    : embedding delay m (defaults to F // 2, minimum 2)

    Returns
    -------
    X      : (D·W·m, F−m+1) global Hankel data matrix
    m      : delay used
    """
    D, W, F = rf_stack.shape
    m = max(2, F // 2) if delay is None else int(delay)
    n_cols = F - m + 1

    pixel_series = rf_stack.reshape(D * W, F)            # (N_pixels, F)

    # Build Hankel index grid
    row_idx = np.arange(m)[:, None]                      # (m, 1)
    col_idx = np.arange(n_cols)[None, :]                 # (1, n_cols)
    indices = row_idx + col_idx                           # (m, n_cols)

    hankel_block = pixel_series[:, indices]               # (N_pixels, m, n_cols)
    X = hankel_block.reshape(D * W * m, n_cols)           # (D·W·m, n_cols)

    return X, m


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Exact Dynamic Mode Decomposition (DMD)
# ══════════════════════════════════════════════════════════════════════════════

def exact_dmd(
    X1:               np.ndarray,
    X2:               np.ndarray,
    svd_rank:         int | None = None,
    energy_threshold: float      = 0.999,
) -> dict[str, Any]:
    """
    Compute Exact Dynamic Mode Decomposition.

    Algorithm
    ---------
    1. SVD:  X₁ = U Σ V*          (truncate to rank r)
    2. Reduced operator: Ã = U_r* X₂ V_r Σ_r⁻¹
    3. Eigendecompose:  Ã W = W Λ
    4. Exact DMD modes: Φ = X₂ V_r Σ_r⁻¹ W
    5. Amplitudes b: least-squares fit to X₁[:, 0]

    Parameters
    ----------
    X1               : (n, m)  snapshot matrix at time t
    X2               : (n, m)  snapshot matrix at time t+1
    svd_rank         : explicit truncation rank (None → auto by energy)
    energy_threshold : cumulative singular value energy to retain (default 0.999)

    Returns
    -------
    dict with keys:
        eigenvalues     — (r,)   complex, encode dynamics
        modes           — (n, r) complex DMD spatial modes
        amplitudes      — (r,)   complex mode amplitudes at t=0
        rank            — int    truncation rank used
        singular_values — (k,)   full singular value spectrum
    """
    U, sigma, Vh = np.linalg.svd(X1, full_matrices=False)

    if svd_rank is not None:
        r = min(int(svd_rank), len(sigma))
    else:
        cum_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
        r = int(np.searchsorted(cum_energy, energy_threshold)) + 1

    r = max(r, 1)

    U_r  = U[:, :r]
    s_r  = sigma[:r]
    Vh_r = Vh[:r, :]
    V_r  = Vh_r.conj().T

    A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1.0 / s_r)   # (r, r)

    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = X2 @ V_r @ np.diag(1.0 / s_r) @ W                   # (n, r)

    b = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]          # (r,)

    return {
        "eigenvalues":     eigenvalues,
        "modes":           Phi,
        "amplitudes":      b,
        "rank":            r,
        "singular_values": sigma,
    }


def dmd_reconstruct(
    dmd_result:        dict[str, Any],
    n_timesteps:       int,
    damping_threshold: float = 0.85,
) -> np.ndarray:
    """
    Reconstruct the clean RF signal from dominant DMD modes.

    Speckle / artefact modes are highly damped:  |λ_k| << 1.
    Tissue mechanics correspond to modes near the unit circle |λ_k| ≈ 1.

    X_clean[:, t] = Σ_{k: |λ_k|≥τ} b_k · λ_k^t · Φ_k

    Parameters
    ----------
    dmd_result        : output of ``exact_dmd``
    n_timesteps       : number of frames to reconstruct
    damping_threshold : keep modes with |λ_k| >= threshold (default 0.85)

    Returns
    -------
    X_clean : (n, n_timesteps) real-valued reconstructed data matrix
    """
    eigenvalues = dmd_result["eigenvalues"]
    Phi         = dmd_result["modes"]
    b           = dmd_result["amplitudes"]

    mag       = np.abs(eigenvalues)
    keep_mask = mag >= damping_threshold
    n_kept    = int(keep_mask.sum())

    if n_kept == 0:
        warnings.warn(
            f"No DMD modes passed threshold {damping_threshold}. "
            "Using top-1 mode instead.", RuntimeWarning, stacklevel=2
        )
        keep_mask = mag >= mag.max()

    eigs_k = eigenvalues[keep_mask]   # (k,)
    Phi_k  = Phi[:, keep_mask]        # (n, k)
    b_k    = b[keep_mask]             # (k,)

    # Time dynamics: Vand[t, k] = λ_k^t
    time_grid = np.arange(n_timesteps)                         # (T,)
    Vand      = np.power(
        eigs_k[np.newaxis, :],                                  # (1, k)
        time_grid[:, np.newaxis]                                 # (T, 1)
    )                                                           # (T, k)

    X_clean = np.real(Phi_k @ np.diag(b_k) @ Vand.T)          # (n, T)
    return X_clean


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Sim-to-Real Domain Randomisation (Rayleigh Noise Injection)
# ══════════════════════════════════════════════════════════════════════════════

def inject_rayleigh_noise(
    rf_stack:     np.ndarray,
    sigma:        float = 0.1,
    n_augmented:  int   = 3,
    random_state: int   = 42,
) -> np.ndarray:
    """
    Inject physics-based Rayleigh-distributed phase jitter into the simulated
    RF stack to mimic real-world transducer noise patterns.

    The augmented RF signal is:
        rf_aug(x,y,t) = rf(x,y,t) + A(x,y,t) · exp(j·φ(x,y,t))
    where:
        A   ~ Rayleigh(σ)         (magnitude envelope)
        φ   ~ Uniform(0, 2π)     (random phase)

    Parameters
    ----------
    rf_stack     : (D, W, F) clean simulated RF stack
    sigma        : Rayleigh scale parameter (controls noise amplitude)
    n_augmented  : number of independent noisy copies to generate
    random_state : NumPy random seed for reproducibility

    Returns
    -------
    augmented : (n_augmented, D, W, F) array of augmented stacks
    """
    rng = np.random.default_rng(seed=random_state)
    D, W, F = rf_stack.shape
    augmented = np.empty((n_augmented, D, W, F), dtype=np.float64)

    for i in range(n_augmented):
        magnitude = rng.rayleigh(scale=sigma, size=(D, W, F))
        phase     = rng.uniform(0, 2 * np.pi, size=(D, W, F))
        noise     = magnitude * np.cos(phase)    # real part of complex noise
        augmented[i] = rf_stack + noise

    return augmented


# ══════════════════════════════════════════════════════════════════════════════
#  Full Phase-1 Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_rf_pipeline(
    filepath:          str,
    low_cutoff:        float      = 0.05,
    high_cutoff:       float      = 0.45,
    dmd_rank:          int | None = None,
    dmd_energy:        float      = 0.999,
    dmd_damping:       float      = 0.85,
    rayleigh_sigma:    float      = 0.1,
    n_augmented:       int        = 3,
    random_state:      int        = 42,
    return_all:        bool       = False,
) -> dict[str, Any]:
    """
    Full Phase-1 signal processing pipeline on one .mat file.

    Steps
    -----
    1. Load .mat → extract RF stack (D, W, F)
    2. 2D DFT bandpass filter
    3. Build global Hankel matrix  →  X₁, X₂
    4. Exact DMD  →  decompose into modes + eigenvalues
    5. Reconstruct clean RF signal from dominant modes
    6. Inject Rayleigh noise  →  augmented copies for domain randomisation

    Parameters
    ----------
    filepath       : path to .mat file
    low_cutoff     : DFT bandpass lower bound
    high_cutoff    : DFT bandpass upper bound
    dmd_rank       : explicit SVD truncation rank (None = auto)
    dmd_energy     : cumulative energy threshold for automatic rank
    dmd_damping    : |λ| threshold for mode selection
    rayleigh_sigma : Rayleigh noise scale
    n_augmented    : number of augmented copies to generate
    random_state   : random seed
    return_all     : if True, include intermediate arrays in output

    Returns
    -------
    results : dict with keys:
        rf_clean       — (D, W, F) DMD-reconstructed clean RF stack
        rf_augmented   — (n_augmented, D, W, F) Rayleigh-augmented copies
        dmd_eigenvalues— (r,) complex eigenvalues
        dmd_rank       — int, rank used
        shape          — (D, W, F) tuple
        filepath       — original filepath
        [if return_all]:
            rf_raw, rf_filtered, hankel_X, dmd_result
    """
    # 1. Load
    mat_data = load_mat_file(filepath)
    rf_raw   = extract_rf_frames(mat_data)
    D, W, F  = rf_raw.shape

    # 2. DFT filter
    rf_filtered = filter_rf_stack(rf_raw, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    # 3. Hankelization
    X_hankel, m = build_global_hankel_matrix(rf_filtered)
    X1 = X_hankel[:, :-1]
    X2 = X_hankel[:, 1:]

    # 4. Exact DMD
    dmd_result = exact_dmd(X1, X2, svd_rank=dmd_rank, energy_threshold=dmd_energy)

    # 5. DMD reconstruction → back to (D, W, F)
    n_states   = D * W * m
    n_cols_out = X_hankel.shape[1]
    X_clean_dm = dmd_reconstruct(dmd_result, n_timesteps=n_cols_out,
                                  damping_threshold=dmd_damping)   # (n_states, n_cols_out)

    # Average overlapping Hankel windows → (D*W, F) approximation
    #  X_clean_dm rows correspond to (pixel, delay_row); average delay dimension
    X_clean_pix = X_clean_dm.reshape(D * W, m, n_cols_out)
    # Reconstruct full F frames via mean of available contributions at each t
    rf_clean_flat = np.zeros((D * W, F))
    counts        = np.zeros(F)
    for lag in range(m):
        for col in range(n_cols_out):
            t = lag + col
            rf_clean_flat[:, t] += X_clean_pix[:, lag, col]
            counts[t] += 1
    counts = np.maximum(counts, 1)
    rf_clean_flat /= counts
    rf_clean = rf_clean_flat.reshape(D, W, F)

    # 6. Domain randomisation
    rf_augmented = inject_rayleigh_noise(
        rf_clean, sigma=rayleigh_sigma,
        n_augmented=n_augmented, random_state=random_state
    )

    output: dict[str, Any] = {
        "rf_clean":        rf_clean,
        "rf_augmented":    rf_augmented,
        "dmd_eigenvalues": dmd_result["eigenvalues"],
        "dmd_rank":        dmd_result["rank"],
        "shape":           (D, W, F),
        "filepath":        filepath,
    }

    if return_all:
        output.update({
            "rf_raw":       rf_raw,
            "rf_filtered":  rf_filtered,
            "hankel_X":     X_hankel,
            "dmd_result":   dmd_result,
        })

    return output
