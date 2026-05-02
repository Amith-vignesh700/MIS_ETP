# 📘 Complete Beginner's Guide
# Ultrasound Elastography for Label-Free Tissue Tumor Grading

> **Who this guide is for:** Someone who has never seen this codebase before.
> By the end, you will understand every file, every math concept, every model
> parameter, every design decision, and exactly how to run, train, and deploy
> the entire system from scratch.

---

## Table of Contents

1. [What This Project Does (Plain English)](#1-what-this-project-does-plain-english)
2. [Clinical Background — Why It Matters](#2-clinical-background--why-it-matters)
3. [High-Level System Map](#3-high-level-system-map)
4. [Datasets — Where the Data Comes From](#4-datasets--where-the-data-comes-from)
5. [Complete File-by-File Reference](#5-complete-file-by-file-reference)
6. [Phase 1 — Signal Processing (The Math, Line by Line)](#6-phase-1--signal-processing-the-math-line-by-line)
7. [Phase 2 — The Deep Learning Model (Architecture Deep Dive)](#7-phase-2--the-deep-learning-model-architecture-deep-dive)
8. [Phase 2B — ADMM Training Loop (Optimization Theory)](#8-phase-2b--admm-training-loop-optimization-theory)
9. [Phase 3 — Statistical Validation](#9-phase-3--statistical-validation)
10. [Phase 4 — Django Backend API](#10-phase-4--django-backend-api)
11. [Phase 5 — Frontend Clinical Interface](#11-phase-5--frontend-clinical-interface)
12. [Phase 6 — Docker Deployment](#12-phase-6--docker-deployment)
13. [All Model Parameters Explained](#13-all-model-parameters-explained)
14. [All Hyperparameters and Why They Were Chosen](#14-all-hyperparameters-and-why-they-were-chosen)
15. [Running Everything Step by Step](#15-running-everything-step-by-step)
16. [API Reference (Every Endpoint)](#16-api-reference-every-endpoint)
17. [How Data Flows Through the Entire System](#17-how-data-flows-through-the-entire-system)
18. [Troubleshooting Common Errors](#18-troubleshooting-common-errors)
19. [Glossary of Every Technical Term](#19-glossary-of-every-technical-term)
20. [Syllabus Integration Map](#20-syllabus-integration-map)

---

## 1. What This Project Does (Plain English)

Imagine a doctor pressing a small probe against a patient's breast to take an ultrasound scan. The probe records electrical signals (called **RF signals** — Radio Frequency signals) as it compresses the tissue. Stiff tissue (like a tumour) behaves differently under compression than soft healthy tissue.

This project takes those raw RF signal files, runs them through a chain of mathematics and deep learning, and automatically answers one question:

> **Is this tissue Benign, Precancerous, or Malignant?**

It does this **label-free** — meaning it was never shown labelled tumour images by a doctor. It learned purely from physics. The output is a probability like:
- Benign: 12%
- Precancerous: 78%
- Malignant: 10%

...plus a colour heatmap image showing which parts of the tissue are stiffest.

The whole system runs as a web application: a clinician uploads a `.mat` file through a browser, and within ~2 seconds receives a diagnosis with a visual strain map.

---

## 2. Clinical Background — Why It Matters

### The Problem with Traditional Elastography

Traditional ultrasound elastography works like this:
1. A radiologist manually draws a box around a suspicious area (Region of Interest)
2. Software compares tissue movement between pre-compression and post-compression frames
3. The radiologist **subjectively** decides if the strain ratio looks suspicious
4. If suspicious → biopsy (needle inserted to extract tissue)

Problems: it is time-consuming, observer-dependent, invasive, and expensive.

### What This System Replaces

| Traditional Step | This System |
|---|---|
| Manual ROI annotation | Automatic — whole-field processing |
| Subjective strain assessment | Quantitative CNN probability score |
| Biopsy for confirmation | Statistical confidence intervals from validated model |
| Single scan analysis | Temporal sequence analysis (multiple frames) |
| Trained radiologist required | Any clinician uploads a file |

### The Three Output Classes

| Class | What It Means | Tissue Properties |
|---|---|---|
| 🟢 **Benign** (Class 0) | Non-cancerous growth | Uniform stiffness, smooth deformation under compression |
| 🟡 **Precancerous** (Class 1) | Early dysplasia | Heterogeneous stiffness, irregular strain patterns |
| 🔴 **Malignant** (Class 2) | Invasive cancer | Very stiff, high contrast ratio vs surrounding tissue, irregular boundaries |

---

## 3. High-Level System Map

Below is the complete data flow from raw `.mat` file to clinical dashboard:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     COMPLETE SYSTEM OVERVIEW                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  TRAINING TIME (Notebooks 01–04)                                         ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │                                                                │      ║
║  │  Dataset 1 (Simulation) ──► Notebook 01 ──► Notebook 02       │      ║
║  │       .mat files               Phase 1       Pre-train CNN-LSTM│      ║
║  │                               (DMD)          saves weights.pth │      ║
║  │                                                     │          │      ║
║  │  Dataset 2 (Real/Phantom) ──► Notebook 03 ◄─────────┘         │      ║
║  │       .mat files               ADMM Fine-tune (unsupervised)   │      ║
║  │                                saves cnn_lstm_final.pth        │      ║
║  │                                     │                          │      ║
║  │                               Notebook 04                      │      ║
║  │                               Statistical Validation           │      ║
║  │                               (Wilcoxon, MLE, CI)              │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
║                                                                          ║
║  DEPLOYMENT TIME (Docker + Django + Nginx)                               ║
║  ┌────────────────────────────────────────────────────────────────┐      ║
║  │                                                                │      ║
║  │  Browser ──► Nginx (port 80) ──► Django (port 8000)           │      ║
║  │  uploads        serves           loads cnn_lstm_final.pth      │      ║
║  │  .mat file      frontend         once at startup               │      ║
║  │                 static files                                    │      ║
║  │                                     │                          │      ║
║  │                              signal_processing.py              │      ║
║  │                              (DFT → Hankel → DMD)              │      ║
║  │                                     │                          │      ║
║  │                              inference.py                      │      ║
║  │                              (tensor prep → model → JSON)      │      ║
║  │                                     │                          │      ║
║  │  Browser ◄── JSON response ◄────────┘                         │      ║
║  │  shows heatmap + probabilities                                  │      ║
║  └────────────────────────────────────────────────────────────────┘      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Datasets — Where the Data Comes From

Both datasets are provided free by the **IMPACT Research Lab at Concordia University, Montreal**.

### Dataset 1 — Simulation Database

**URL:** `https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/`

**What it contains:**
- Mathematically simulated RF ultrasound frames generated by Finite Element Analysis (FEA)
- Each `.mat` file contains a 3D array of shape `(depth, width, n_frames)` — depth and width are spatial dimensions, n_frames is the number of compression frames
- Accompanying ground-truth displacement and strain fields (this is rare and valuable)
- **No hardware noise** — perfectly clean mathematical signals

**Why we use it first (Phase 2A pre-training):**
Because the signals are mathematically perfect, the model can learn the *fundamental physics* of how tissue deforms under compression without being confused by hardware noise. Think of it as learning to walk before running.

**Format:** MATLAB `.mat` files (v5 or v7.3/HDF5). Keys typically include `rf_pre`, `rf_post`, `displacement`, `strain`.

---

### Dataset 2 — In-Vivo & Physical Phantom Database

**URL:** `https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/`

**What it contains:**
- Real RF ultrasound acquisitions from **physical breast tissue phantoms** (laboratory tissue mimics) and real in-vivo subjects
- Acquired with clinical-grade ultrasound transducers (realistic hardware)
- Contains realistic **speckle noise**, transducer artifacts, boundary reflections, and signal decorrelation
- **No ground-truth labels** — this is unsupervised data only

**Why we use it second (Phase 2B fine-tuning + Phase 3 validation):**
After the model learns physics from simulation, we expose it to real-world noise. The ADMM-based unsupervised fine-tuning forces the model to adapt its internal representations to real acquisition artifacts without needing any labels.

**Split strategy:**
- 80% → Training (unsupervised ADMM fine-tuning in Notebook 03)
- 20% → Hold-out (never touched until Phase 3 statistical validation in Notebook 04)

---

### Dataset-to-Phase Routing (Explicit Mapping)

```
Phase 1  (Notebook 01) → Dataset 1 only    → DFT + Hankel + DMD + Rayleigh augmentation
Phase 2A (Notebook 02) → Dataset 1 only    → Supervised CNN-LSTM pre-training
Phase 2B (Notebook 03) → Dataset 2 (train) → Unsupervised ADMM fine-tuning
Phase 3  (Notebook 04) → Dataset 2 (hold-out) → Statistical validation ONLY (model frozen)
Phase 4-6 (Django API) → New patient .mat  → Live inference
```

---

### How to Download the Datasets

```bash
# Navigate to each URL in your browser and download the .mat archives.
# Recommended directory structure after extraction:

data/
├── simulation/          ← Dataset 1
│   ├── case_001.mat
│   ├── case_002.mat
│   └── ...
└── phantom/             ← Dataset 2
    ├── phantom_001.mat
    ├── phantom_002.mat
    └── ...
```

Set the `DATA_ROOT` variable at the top of each notebook to point to these directories.

---

## 5. Complete File-by-File Reference

Below is every file in the project with a plain-English explanation of its purpose.

```
ultrasound-elastography-grading/
│
├── README.md                          ← Project overview and pipeline summary
├── docker-compose.yml                 ← Orchestrates Django + Nginx containers
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   ← Phase 1: Load .mat, DFT, Hankelization, DMD
│   ├── 02_simulation_pretrain.ipynb  ← Phase 2A: Pre-train model on simulation data
│   ├── 03_unsupervised_tuning.ipynb  ← Phase 2B: ADMM fine-tune on phantom data
│   └── 04_statistical_testing.ipynb  ← Phase 3: Wilcoxon test + CI + MLE
│
├── out/
│   └── processed/                    ← Auto-generated outputs from notebooks
│       ├── tensors/                  ← Preprocessed .npz arrays and JSON manifests
│       ├── models/                   ← Saved PyTorch .pth weight files
│       ├── figures/                  ← Heatmaps, confusion matrices, loss curves (.png)
│       └── metrics/                  ← Statistical results and training histories (.json)
│
├── backend/
│   ├── Dockerfile                    ← Builds Python/PyTorch container image
│   ├── requirements.txt              ← All Python package dependencies
│   ├── manage.py                     ← Django CLI (migrate, runserver, etc.)
│   │
│   ├── core/
│   │   ├── __init__.py               ← Empty package marker
│   │   ├── settings.py               ← Django config (security, DB, ML_ENGINE, CORS)
│   │   ├── urls.py                   ← Root URL router → /api/ and /admin/
│   │   └── wsgi.py                   ← Gunicorn entry point
│   │
│   ├── api/
│   │   ├── __init__.py               ← Empty package marker
│   │   ├── apps.py                   ← Loads CNN-LSTM model ONCE at startup
│   │   ├── urls.py                   ← Routes /api/predict/ and /api/health/
│   │   └── views.py                  ← PredictView + HealthCheckView (DRF)
│   │
│   └── ml_engine/
│       ├── __init__.py               ← Public API: exposes CNNLSTMElastography, load_model
│       ├── architecture.py           ← PyTorch model classes (ConvBlock, UNetEncoder, CNN-LSTM)
│       ├── signal_processing.py      ← DFT, Hankelization, DMD, Rayleigh noise
│       ├── inference.py              ← End-to-end: .mat → tensor → model → JSON
│       └── weights/
│           └── cnn_lstm_final.pth    ← Trained model weights (produced by Notebook 03)
│
├── frontend/
│   ├── index.html                    ← Full clinical dashboard (1,700+ lines)
│   ├── css/
│   │   └── style.css                 ← Medical-grade responsive CSS
│   └── js/
│       └── app.js                    ← Fetch API: sends .mat file, renders results
│
└── nginx/
    └── nginx.conf                    ← Reverse proxy config (routing, upload limits, timeouts)
```

---

### Detailed File Descriptions

#### `out/processed/`
This directory acts as the central storage for auto-generated outputs from running the pipeline notebooks. It represents the transition from raw data to deployable models.
- **`tensors/`**: Contains `.npz` files that hold the multi-frame, filtered tensor data resulting from Phase 1 processing (DMD). Also contains metadata files like `manifest.pkl` (lists all successfully processed files) and `test_files.json` (defines the hold-out set used in Phase 3).
- **`models/`**: Holds the saved `.pth` weight files from PyTorch training. This includes checkpoint files and the final `cnn_lstm_final.pth`, which is subsequently copied to the backend for live clinical inference.
- **`figures/`**: Contains all visual `.png` outputs generated during training and validation. Includes pre-training curves, LNCC/TV loss trajectories, strain heatmaps, and statistical validation confusion matrices.
- **`metrics/`**: Contains raw `.json` files representing training histories and statistical test results (Wilcoxon p-values, CI). These provide a granular view of epoch-by-epoch learning and the final clinical validation scores.

#### `backend/core/settings.py`
The central configuration hub for the Django backend. Key sections:

- **`SECRET_KEY`** — Django's cryptographic signing key. Never commit a real value to git. In production, set via environment variable.
- **`DEBUG`** — Set `False` in production. Controls error detail level and CORS settings.
- **`ALLOWED_HOSTS`** — Comma-separated list of valid hostnames. Prevents Host header attacks.
- **`INSTALLED_APPS`** — Registers `rest_framework`, `corsheaders`, and the `api` app.
- **`FILE_UPLOAD_MAX_MEMORY_SIZE`** — Set to 200 MB to accommodate large `.mat` files.
- **`REST_FRAMEWORK`** — DRF config: JSON-only renderer, multipart parser (for file uploads), 30 requests/hour throttle.
- **`ML_ENGINE`** — Dictionary that `api/apps.py` reads to find the model weights path and device (`cpu` or `cuda`).
- **`LOGGING`** — Routes `api.*` and `ml_engine.*` logs to console at INFO level.

#### `backend/core/urls.py`
Maps incoming HTTP paths to Django applications:
- `/admin/` → Django admin panel
- `/api/` → everything in `api/urls.py`
- `/media/` → uploaded files (only in DEBUG mode; Nginx handles this in production)

#### `backend/core/wsgi.py`
The "bridge" between Gunicorn and Django. Gunicorn calls `application = get_wsgi_application()` to start handling requests. The `--timeout 120` flag in the Dockerfile's CMD is critical here — DMD + CNN-LSTM inference on a CPU can take 30–90 seconds.

#### `backend/api/apps.py`
Django calls `ready()` once after all apps are initialized. This is where the CNN-LSTM model is loaded into RAM:
1. Reads `ML_ENGINE` settings
2. Calls `ml_engine.load_model()` to instantiate `CNNLSTMElastography` and load `.pth` weights
3. Stores the model as `ApiConfig.model`

Every subsequent API request reads from `ApiConfig.model` (one shared instance in memory) — no per-request weight loading.

#### `backend/api/views.py`
Two DRF views:

**`PredictView` (POST /api/predict/):**
1. Checks `mat_file` key exists in `request.FILES`
2. Validates file extension (`.mat` only) and size (≤ 200 MB)
3. Checks `ApiConfig.model` is not None
4. Saves file to `MEDIA_ROOT/uploads/<uuid>_filename.mat`
5. Calls `run_inference()` from `ml_engine/inference.py`
6. Deletes the temp file (always, even on error)
7. Returns structured JSON

**`HealthCheckView` (GET /api/health/):**
Returns `{"status": "ok", "model_loaded": true}` when ready. Returns HTTP 503 if model failed to load.

#### `backend/ml_engine/architecture.py`
All PyTorch model classes. See Section 7 for deep dive.

#### `backend/ml_engine/signal_processing.py`
Pure NumPy/SciPy functions implementing Phase 1. See Section 6 for deep dive.

#### `backend/ml_engine/inference.py`
Ties everything together for a single inference call:
- `load_model()` — instantiates model + loads weights
- `_prepare_tensor()` — crops, windows, normalises RF stack into `(1, T, 1, H, W)` tensor
- `_render_strain_heatmap()` — creates a jet-colourmap PNG and Base64-encodes it
- `run_inference()` — calls all of the above in sequence, returns the JSON dict

#### `nginx/nginx.conf`
Nginx configuration covering:
- `client_max_body_size 200M` — allows large `.mat` uploads
- `proxy_read_timeout 150s` — longer than Gunicorn's 120s to prevent premature 504 errors
- `/api/` → proxied to `django:8000` with keep-alive
- `/static/` → served directly from the `static_volume` Docker volume
- `/media/` → served directly from the `media_volume` Docker volume
- `/` → serves `frontend/index.html` with SPA fallback

#### `docker-compose.yml`
Defines two services:
- **`django`** — builds from `backend/Dockerfile`, mounts source code + model weights volumes, exposes port 8000 (internal only)
- **`nginx`** — uses official `nginx:1.25-alpine` image, exposes port 80 to the outside world, depends on `django`

Three named Docker volumes ensure data persists across container rebuilds:
- `static_volume` — Django's collected static files
- `media_volume` — temporary `.mat` uploads
- `model_weights` — the `.pth` file

---

## 6. Phase 1 — Signal Processing (The Math, Line by Line)

This phase is implemented in `backend/ml_engine/signal_processing.py` and demonstrated in `notebooks/01_data_preprocessing.ipynb`.

The goal: take a noisy raw RF stack → produce a clean, artifact-free RF stack ready for the neural network.

### Step 1: Loading `.mat` Files

MATLAB stores data in two formats:
- **v5 (< 7.2)** — loaded with `scipy.io.loadmat()`
- **v7.3 (HDF5)** — must be loaded with `h5py` because it is a compressed HDF5 archive

The function `load_mat_file()` tries `scipy.io` first, falls back to `h5py`:

```python
try:
    data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
except Exception:
    with h5py.File(filepath, 'r') as f:
        f.visititems(lambda name, obj: data.update({name: np.array(obj)}))
```

The function `extract_rf_frames()` then searches the dict for common key names: `rf_pre`, `rf_post`, `RF`, `rfData`, `IQpre`, `IQpost`, `rf`, `RFdata`, `data`. It returns a `float64` array of shape `(D, W, F)` where:
- `D` = depth samples (axial direction, typically 512–2048)
- `W` = width samples (lateral direction, typically 64–256)
- `F` = number of compression frames (typically 4–20)

---

### Step 2: 2D Discrete Fourier Transform (2D-DFT) Bandpass Filter

**Why:** Raw RF frames contain two types of unwanted signals:
- **Low frequencies** (< 5% of Nyquist): slow probe motion artifacts, DC offset
- **High frequencies** (> 45% of Nyquist): electronic quantization noise, aliasing

We want to keep only the frequencies corresponding to actual tissue mechanics.

**The Math:**

The 2D-DFT converts a spatial-domain frame `f(x,y)` into its frequency-domain representation `F(u,v)`:

```
F(u,v) = Σ_{x=0}^{M-1} Σ_{y=0}^{N-1}  f(x,y) · e^(-j2π(ux/M + vy/N))
```

where `u` is the axial spatial frequency and `v` is the lateral spatial frequency.

**Implementation:**

```python
F_shifted = fftshift(fft2(rf_frame))       # 2D FFT, zero-centred

# Build radial frequency magnitude at each pixel
U, V = np.meshgrid(fftfreq(D), fftfreq(W), indexing='ij')
freq_mag = np.sqrt(U**2 + V**2)

# Bandpass mask: keep frequencies between 5% and 45% of Nyquist
mask = (freq_mag >= 0.05) & (freq_mag <= 0.45)

# Apply mask and invert back to spatial domain
filtered = np.real(ifft2(ifftshift(F_shifted * mask)))
```

**Parameters:**
- `low_cutoff = 0.05` — removes DC and very slow drift (normalised, fraction of Nyquist)
- `high_cutoff = 0.45` — removes high-frequency noise (slightly below Nyquist at 0.5)

This is applied to every frame in the stack via `filter_rf_stack()`.

---

### Step 3: Hankelization (Time-Delay Embedding)

**Why:** Dynamic Mode Decomposition requires a data matrix where columns represent different time snapshots. But we need to enrich the temporal representation. Hankelization (also called time-delay embedding) converts a 1D signal into a 2D matrix, revealing hidden dynamical structure.

**The Math:**

For a single pixel's temporal signal `s = [s₁, s₂, ..., s_F]`, the Hankel matrix is:

```
H(s) = | s₁   s₂   s₃  ...  s_{F-m+1} |
        | s₂   s₃   s₄  ...  s_{F-m+2} |
        | ...                           |
        | s_m  s_{m+1}  ... s_F         |
```

Shape: `(m, F-m+1)` where `m = F // 2` (the embedding delay).

For an entire RF stack with `D*W` pixels, we stack all pixel Hankel matrices to get a **global Hankel matrix**:

```
X ∈ ℝ^(D·W·m , F-m+1)
```

**Why this matters:** Instead of treating each frame as independent, this matrix encodes *how each pixel's signal evolves over time*. This is the mathematical equivalent of "looking at the movie, not just individual frames."

**Implementation:**

```python
pixel_series = rf_stack.reshape(D * W, F)     # (N_pixels, F)
# Vectorized indexing trick to build all Hankel matrices simultaneously
row_idx = np.arange(m)[:, None]               # (m, 1)
col_idx = np.arange(n_cols)[None, :]          # (1, n_cols)
indices = row_idx + col_idx                    # (m, n_cols)
hankel_block = pixel_series[:, indices]        # (N_pixels, m, n_cols)
X = hankel_block.reshape(D * W * m, n_cols)   # (D·W·m, n_cols)
```

---

### Step 4: Exact Dynamic Mode Decomposition (DMD)

**What DMD does:** It finds the *best-fit linear dynamical system* `x_{t+1} ≈ A · x_t` governing the data, then extracts its eigenvalues and modes. Modes with eigenvalues near the unit circle `|λ| ≈ 1` are persistent, physically meaningful tissue dynamics. Modes with `|λ| << 1` are highly damped — these are artefacts (speckle noise, electronic drift).

**The 5-Step Algorithm:**

Given snapshot pairs `X₁` (all columns except last) and `X₂` (all columns except first) from the Hankel matrix:

**Step A — Truncated SVD of X₁:**
```
X₁ = U · Σ · V*    (truncated to rank r)
```
The rank `r` is chosen automatically by cumulative energy: keep enough singular values to explain 99.9% of signal energy (`energy_threshold=0.999`).

**Step B — Reduced linear operator:**
```
Ã = U_r* · X₂ · V_r · Σ_r⁻¹      shape: (r, r)
```
This is the full operator `A` projected into the r-dimensional subspace defined by the top-r left singular vectors. Working in this reduced space is numerically stable and computationally efficient.

**Step C — Eigendecomposition:**
```
Ã · W = W · Λ
```
`Λ` = diagonal matrix of eigenvalues (complex numbers). Each eigenvalue `λ_k` encodes: `|λ_k|` = growth/decay rate, `angle(λ_k)` = oscillation frequency.

**Step D — Exact DMD modes:**
```
Φ = X₂ · V_r · Σ_r⁻¹ · W      shape: (n, r)
```
Each column of `Φ` is a spatial mode — a pattern that grows/decays/oscillates at rate `λ_k`.

**Step E — Mode amplitudes via least squares:**
```
b = Φ† · x₁     (least-squares fit to first snapshot)
```

**Reconstruction from dominant modes only:**
```python
keep_mask = np.abs(eigenvalues) >= 0.85   # discard speckle modes
X_clean[:, t] = Φ_kept @ diag(b_kept) @ (eigenvalues_kept ** t)
```

**Parameter `dmd_damping = 0.85`** — the threshold below which modes are considered artefacts. Tuned empirically: ultrasound speckle has `|λ| < 0.5`, tissue mechanics have `|λ| > 0.85`.

After reconstruction, the output is re-shaped back to `(D, W, F)` via a windowed averaging scheme that handles the overlapping Hankel windows.

---

### Step 5: Rayleigh Noise Injection (Sim-to-Real Augmentation)

**Why:** The model is pre-trained on simulation data (clean, perfect signals). Real ultrasound transducers produce Rayleigh-distributed amplitude noise because the signal is a sum of many random backscatterers. Injecting this noise pattern into simulation data makes it look more like real data — bridging the "sim-to-real gap."

**The Rayleigh Distribution:**
Rayleigh noise models the amplitude envelope of narrowband noise. If two independent Gaussian random variables `X ~ N(0,σ²)` and `Y ~ N(0,σ²)`, then `R = sqrt(X²+Y²)` follows a Rayleigh distribution. This matches how multiple tissue scatterers add up.

**Implementation:**
```python
magnitude = rng.rayleigh(scale=sigma, size=(D, W, F))   # Rayleigh amplitude
phase     = rng.uniform(0, 2*np.pi, size=(D, W, F))     # Random phase
noise     = magnitude * np.cos(phase)                    # Real part only
rf_augmented = rf_clean + noise
```

**Parameter `sigma = 0.1`** — Rayleigh scale parameter. Controls the amplitude of injected noise. Set to 10% of typical RF signal amplitude based on empirical comparison between Dataset 1 and Dataset 2 noise levels.

**Parameter `n_augmented = 3`** — generates 3 independent noisy copies of each simulation file, tripling the effective dataset size.

---

## 7. Phase 2 — The Deep Learning Model (Architecture Deep Dive)

All model code is in `backend/ml_engine/architecture.py`.

### Why a CNN-LSTM Hybrid?

- **CNN alone** would only see spatial patterns in individual frames — it cannot model deformation *over time*
- **LSTM alone** would only see temporal patterns in pixel intensity — it cannot model spatial tissue structure
- **CNN-LSTM** processes each frame spatially (CNN), then tracks how the spatial features *change over time* (LSTM). This perfectly matches ultrasound elastography: spatial tissue patterns deforming through a temporal compression sequence.

### Building Block: `ConvBlock`

The smallest unit of the spatial extractor. Each `ConvBlock` applies:

```
Input (B, C_in, H, W)
    │
    Conv2d(C_in → C_out, kernel=3×3, padding=1, bias=False)
    │
    BatchNorm2d(C_out)          ← Normalizes activations, stabilizes training
    │
    GELU()                      ← Smooth activation (better gradient flow than ReLU)
    │
    Conv2d(C_out → C_out, kernel=3×3, padding=1, bias=False)
    │
    BatchNorm2d(C_out)
    │
    GELU()
    │
    Dropout2d(p=0.1)            ← Spatial dropout: drops entire feature channels
    │
Output (B, C_out, H, W)
```

**Why `bias=False` with BatchNorm?** Because BatchNorm has its own learnable bias parameter `β`. Using both would be redundant.

**Why GELU instead of ReLU?** GELU (Gaussian Error Linear Unit) `= x·Φ(x)` is smooth everywhere and allows small negative values, which helps gradient flow in deep networks.

**Why `Dropout2d` (spatial dropout)?** Standard dropout drops individual pixels randomly. Spatial dropout drops entire feature *channels*, which is more effective for 2D feature maps because adjacent pixels are highly correlated.

---

### Spatial Extractor: `UNetEncoder`

Stacks `ConvBlock`s with `MaxPool2d` downsampling between them, following the U-Net encoder pattern:

```
Input: (B, 1, 64, 64)         ← 1 channel (grayscale RF frame)
    │
ConvBlock(1 → 16)             ← Learn simple edges and speckle gradients
    │
MaxPool2d(2,2)                ← 64×64 → 32×32
    │
ConvBlock(16 → 32)            ← Learn texture patterns
    │
MaxPool2d(2,2)                ← 32×32 → 16×16
    │
ConvBlock(32 → 64)            ← Learn mid-level speckle features
    │
MaxPool2d(2,2)                ← 16×16 → 8×8
    │
ConvBlock(64 → 128)           ← Learn high-level tissue structure patterns
    │
AdaptiveAvgPool2d(1,1)        ← 8×8 → 1×1 (global average, spatial invariance)
    │
Output: (B, 128)              ← 128-dimensional feature vector per frame
```

**Channel progression `[1, 16, 32, 64, 128]`:** Each level doubles the number of features while halving the spatial resolution, a standard U-Net pattern. 128 final channels empirically provide sufficient capacity for speckle pattern discrimination without over-parameterisation.

**`AdaptiveAvgPool2d(1,1)`:** Averages all spatial positions into a single value per channel. This gives spatial invariance — the model does not need to know *where* in the frame a feature is, only *that* it exists. This is important because tissue can be anywhere in the scan.

---

### Temporal Tracker: Bidirectional LSTM

After the CNN processes each frame independently, we have a sequence:
```
[feat_1, feat_2, ..., feat_T]    each feat_t ∈ ℝ^128
```

This sequence goes into a 2-layer Bidirectional LSTM:

```
[feat_1, ..., feat_T] → Bi-LSTM → lstm_out ∈ (B, T, 512)
                                    h_final  ∈ (B, 512)
```

**Why Bidirectional?** A standard LSTM only processes time forward: it knows about past frames when predicting the current one. A bidirectional LSTM runs a second LSTM *backwards* through the sequence. The outputs of both directions are concatenated. For elastography, this is valuable because we can see the full compression sequence — the model can use information from later frames when interpreting earlier ones.

**LSTM hidden size `lstm_hidden=256`:** Each LSTM direction has 256 units. Bidirectional concatenation gives `256 × 2 = 512`-dimensional output.

**LSTM layers `lstm_layers=2`:** Two stacked LSTM layers. The first layer learns simple temporal patterns; the second layer learns patterns of patterns. With 2 layers and `dropout=0.3` between them.

**`h_final` construction:**
```python
h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 512)
```
`h_n[-2]` is the final hidden state of the forward LSTM (last layer). `h_n[-1]` is the final hidden state of the backward LSTM (last layer). Concatenating gives the model's "summary" of the entire compression sequence.

---

### Classification Head

Takes `h_final ∈ (B, 512)` and produces class logits:

```
h_final (B, 512)
    │
LayerNorm(512)                ← Normalise before linear layers (stabilises training)
    │
Linear(512 → 128)
    │
GELU()
    │
Dropout(0.3)
    │
Linear(128 → 64)
    │
GELU()
    │
Dropout(0.15)
    │
Linear(64 → 3)
    │
Output: logits (B, 3)         ← Raw scores for [Benign, Precancerous, Malignant]
```

During inference, `softmax(logits)` converts these to probabilities summing to 1.0.

**Why `LayerNorm` before the classifier?** The LSTM outputs can have very different scales depending on the input. LayerNorm normalises each sample independently, making training stable regardless of input scale.

---

### Displacement / Strain Map Head

This second output head is what makes the model *physics-informed*. Instead of only outputting a class, the model also outputs a dense 2D displacement field for every frame in the sequence:

```
lstm_out (B, T, 512)
    │
Linear(512 → 256)             ← Applied per timestep
    │
GELU()
    │
Linear(256 → r²)              ← r = patch_size / (2^n_pools) = 64/8 = 8
    │
Reshape → (B·T, 1, 8, 8)
    │
Upsample(bilinear, 64×64)     ← Upsample back to patch resolution
    │
Output: displacement (B, T, 64, 64)   ← Dense strain map per timestep
```

**Why this matters for training:** The displacement head is supervised by the LNCC and TV losses (see Section 8). Without this second output, the model would be a pure black-box classifier. With it, the model must also *produce physically plausible displacement fields*, which acts as a powerful constraint that forces the model to learn genuine tissue mechanics rather than spurious correlations.

---

### Weight Initialisation

```python
# Convolutional layers: Kaiming Normal (He initialisation)
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Linear layers: Xavier Normal (Glorot initialisation)
nn.init.xavier_normal_(m.weight)
nn.init.zeros_(m.bias)

# Normalisation layers: identity initialisation
nn.init.ones_(m.weight)
nn.init.zeros_(m.bias)
```

**Why Kaiming for Conv, Xavier for Linear?** Kaiming is derived for ReLU-like activations and preserves variance through convolutional layers with fan-out connections. Xavier is better for linear layers where the activation function is approximately linear. Using the wrong initialisation can cause vanishing/exploding gradients during the first few training steps.

---

### Loss Functions

#### LNCC Loss (Local Normalised Cross-Correlation)

Measures acoustic tracking fidelity by comparing the model's predicted displacement between consecutive frames.

**Formula:**
```
LNCC(u,v) = [Σ_Ω (u-ū)(v-v̄)] / sqrt(Σ(u-ū)² · Σ(v-v̄)²)

L_LNCC = 1 - mean(LNCC)      ← Loss is 0 when perfect correlation, 1 when none
```

**Window size `win=9`:** Computes LNCC within 9×9 local windows. Local (not global) correlation is used because tissue properties vary spatially — a 9×9 window captures local texture without mixing different tissue types.

**Implementation:** Uses `F.conv2d` with a uniform kernel to efficiently compute local means and variances across all positions simultaneously.

#### Spatio-Temporal Total Variation Loss

Enforces physical continuity: real tissue cannot have discontinuous displacement fields (it doesn't tear randomly).

```
TV(d) = Σ_{x,y,t} |∂d/∂x| + |∂d/∂y| + |∂d/∂t|

     = mean(|d[:,  :, 1:, :]  - d[:,  :, :-1, :]|)   ← axial gradient
     + mean(|d[:,  :,  :, 1:] - d[:,  :,  :, :-1]|)  ← lateral gradient
     + mean(|d[:, 1:,  :,  :] - d[:, :-1, :,  :]|)   ← temporal gradient
```

The temporal gradient term is particularly important: it penalises large frame-to-frame changes in the displacement field, enforcing smooth tissue motion trajectories.

#### Focal Loss

Handles the severe class imbalance in the dataset (Precancerous samples are rare):

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- `γ = 2.0` (focusing parameter): down-weights easy examples. If the model is already 99% confident about a Benign sample, `(1-0.99)^2 = 0.0001` — nearly zero contribution to loss. Hard examples contribute much more.
- `α_t` (class weight): inverse-frequency weighting for each class. If Precancerous has 10% of the data, it gets a weight of 10x.

**Why not standard cross-entropy?** Standard cross-entropy treats all mistakes equally. With class imbalance, the model can achieve low loss by always predicting Benign, ignoring rare Precancerous cases. Focal Loss forces the model to focus on the hard, rare cases.

---

### Full Forward Pass

```python
def forward(self, x):         # x: (B, T, 1, H, W)
    B, T, C, H, W = x.shape

    # 1. CNN: process all frames simultaneously
    x_flat = x.view(B * T, C, H, W)        # (B·T, 1, H, W)
    feats  = self.cnn(x_flat)               # (B·T, 128)
    feats  = feats.view(B, T, -1)           # (B, T, 128)

    # 2. LSTM: temporal sequence processing
    lstm_out, (h_n, _) = self.lstm(feats)  # lstm_out: (B, T, 512)
    h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 512)

    # 3. Classification head
    logits = self.classifier(h_final)       # (B, 3)

    # 4. Displacement head
    disp_flat    = self.disp_head(lstm_out.reshape(B*T, -1))  # (B·T, 64)
    disp_spatial = disp_flat.view(B*T, 1, 8, 8)
    disp_up      = self.disp_upsample(disp_spatial)           # (B·T, 1, 64, 64)
    displacement = disp_up.view(B, T, H, W)                   # (B, T, 64, 64)

    return logits, displacement
```

---

## 8. Phase 2B — ADMM Training Loop (Optimization Theory)

This is the most mathematically sophisticated part of the project. Standard gradient descent (SGD/Adam) cannot optimise the Total Variation loss because TV is **non-smooth** (its gradient does not exist at points where the displacement field is zero).

### The Problem

The full training objective is:
```
min_θ  L_LNCC(θ) + α·L_TV(D(θ)) + β·L_focal(θ)
         smooth        non-smooth      smooth
```

We cannot simply call `loss.backward()` on `L_TV` because the subgradient at zero is undefined.

### The Solution: ADMM (Alternating Direction Method of Multipliers)

ADMM introduces an **auxiliary variable** `z` (a copy of the displacement field) and splits the problem:

```
min_{θ,z}  L_smooth(θ) + L_TV(z)

subject to:  D(θ) = z          ← constraint: the copy must equal the real displacement
```

The **Augmented Lagrangian** adds a penalty for violating the constraint:
```
L_ρ(θ, z, u) = L_smooth(θ) + L_TV(z) + (ρ/2)·‖D(θ) - z + u‖²
```

where `u` is the **dual variable** (Lagrange multiplier) and `ρ` is the **penalty parameter** (controls how strictly the constraint is enforced).

### The Three ADMM Steps

At each outer iteration `k`:

**Step A — θ-update (SGD step):**
```
θ^{k+1} = argmin_θ  L_smooth(θ) + (ρ/2)·‖D(θ) - z^k + u^k‖²
```
Solved by one (or a few) steps of Adam on the *smooth* part of the loss. The second term drives the model's displacement output `D(θ)` to match the current estimate `z^k - u^k`.

**Step B — z-update (Soft-Thresholding):**
```
z^{k+1} = prox_{L_TV/ρ}(D(θ^{k+1}) + u^k)
         = soft_threshold(D(θ^{k+1}) + u^k,  λ/ρ)
```

The proximal operator of the L1 norm (which TV decomposes into) has a closed-form solution: element-wise soft-thresholding:
```python
def soft_threshold(z, threshold):
    return torch.sign(z) * torch.clamp(torch.abs(z) - threshold, min=0.0)
```

This is exact (not approximate) and computationally cheap.

**Step C — u-update (dual variable):**
```
u^{k+1} = u^k + D(θ^{k+1}) - z^{k+1}
```

The dual variable accumulates the constraint violation over iterations. This ensures the constraint `D(θ) = z` is satisfied as `k → ∞`.

### Convergence

ADMM provably converges for convex problems. For the non-convex deep learning problem here, convergence is empirically stable because:
1. The TV term is convex in `z` (exact solution at each step)
2. The SGD step uses a small learning rate, making small, stable θ updates
3. The penalty `ρ` balances constraint enforcement vs objective minimisation

Typical hyperparameters: `ρ = 0.1`, `λ_TV = 0.01`, learning rate `1e-4`, 5–10 inner SGD steps per ADMM iteration.

---

## 9. Phase 3 — Statistical Validation

Implemented in `notebooks/04_statistical_testing.ipynb`. The model must be *proven statistically better* than a baseline before clinical deployment.

### What is CNR (Contrast-to-Noise Ratio)?

Elastographic CNR measures how well the model's strain map distinguishes tumour from background:

```
CNR = |μ_tumor - μ_background| / sqrt(σ²_tumor + σ²_background)
```

A higher CNR means the tumour region has clearly different strain than surrounding tissue. CNR > 3.0 is generally considered clinically acceptable.

### Maximum Likelihood Estimation (MLE)

We treat the CNR values across all hold-out test cases as i.i.d. samples from a Gaussian distribution `CNR_i ~ N(μ, σ²)`:

**MLE of mean:**
```
μ̂_MLE = (1/N) · Σᵢ CNR_i        ← the sample mean IS the MLE for Gaussian
```

**MLE of variance (biased):**
```
σ̂²_MLE = (1/N) · Σᵢ (CNR_i - μ̂)²    ← biased by factor (N-1)/N vs sample variance
```

**Why MLE? Why not just use `np.mean()`?** We use it because we want to give a *statistical justification* for our point estimate. MLE provides the maximum probability estimate given the observed data under the Gaussian model assumption.

### 95% Confidence Interval

```
CI = μ̂ ± 1.96 · (σ̂/√N)
```

The `1.96` comes from the 97.5th percentile of the standard Normal distribution (two-tailed 95% interval). This interval means: "If we repeated this experiment 100 times, 95 of those intervals would contain the true mean CNR."

For clinical deployment, we require:
1. `μ̂_MLE > 4.0` — mean CNR above clinical threshold
2. CI width `< 1.0` — tight enough to be clinically meaningful
3. p-value `< 0.05` — statistically better than baseline

### Wilcoxon Signed-Rank Test

**Why not a t-test?** The t-test assumes normally distributed data. CNR values from real ultrasound phantoms may not be normal (asymmetric, heavy-tailed). The Wilcoxon signed-rank test is non-parametric — it makes no distribution assumptions.

**Setup:**
- Pairs: `(cnr_model[i], cnr_baseline[i])` — same phantom scan, two different processing methods
- H₀ (null hypothesis): median(cnr_model) ≤ median(cnr_baseline) — model is no better
- H₁ (alternative): median(cnr_model) > median(cnr_baseline) — model is better (one-tailed)

```python
from scipy.stats import wilcoxon
stat, p_value = wilcoxon(cnr_model, cnr_baseline, alternative='greater')
if p_value < 0.05:
    print("Model validated for clinical deployment")
```

**How Wilcoxon works:**
1. Compute differences `d_i = cnr_model[i] - cnr_baseline[i]`
2. Rank the absolute differences `|d_i|` from smallest to largest
3. Sum the ranks of positive differences → W⁺; sum ranks of negatives → W⁻
4. Test statistic: `W = min(W⁺, W⁻)`. Under H₀, W follows a known distribution.
5. p-value: probability of observing W this extreme under H₀.

---

## 10. Phase 4 — Django Backend API

### Why Django? (Not Flask, FastAPI, etc.)

- **Django REST Framework (DRF)** provides production-grade serialisation, throttling, authentication, and file upload handling out of the box
- **`apps.py` ready()** provides a clean, one-time model loading hook with no third-party dependencies
- **Django admin** provides a free web UI for managing uploaded files and monitoring requests
- Django's maturity means security patches, CSRF protection, and SQL injection protection are already built in

### The Model Loading Strategy

The key insight in `api/apps.py`:

```python
class ApiConfig(AppConfig):
    model  = None    # class-level variable — shared across all requests
    device = "cpu"

    def ready(self):
        ApiConfig.model = load_model(weights_path, device, arch_cfg)
```

`model` is a class-level attribute (not instance-level). Django creates one `ApiConfig` instance per process. All HTTP requests handled by that process share the same model object. This means:
- Model weights are loaded from disk **once** at startup (takes 5–15 s)
- Every request is served from **in-memory weights** (< 1 ms overhead)
- Memory usage: ~50 MB for this model size on CPU

In `views.py`:
```python
api_cfg = apps.get_app_config("api")
model   = api_cfg.model    # reads the class-level attribute
```

### Request Lifecycle

```
Client sends: POST /api/predict/ + multipart .mat file
    │
    ▼
Nginx: checks content-length ≤ 200 MB, forwards to Django
    │
    ▼
PredictView.post():
    ├── 1. Check "mat_file" in request.FILES
    ├── 2. Validate extension (.mat) and size (≤ 200 MB)
    ├── 3. Check ApiConfig.model is not None
    ├── 4. Save to /app/media/uploads/<uuid>_filename.mat
    ├── 5. Call run_inference(model, filepath, device)
    │       ├── process_rf_pipeline()      ← signal_processing.py
    │       ├── _prepare_tensor()          ← inference.py
    │       ├── model(tensor)              ← CNN-LSTM forward pass
    │       └── _render_strain_heatmap()   ← matplotlib → base64 PNG
    ├── 6. Delete temp file (always)
    └── 7. Return JSON response
```

### Security Considerations

- **UUID-prefixed filenames:** `<uuid>_original_name.mat` prevents filename collisions and path traversal attacks
- **Extension whitelist:** Only `.mat` files accepted (checked by string comparison)
- **Size limit enforced at two levels:** Nginx `client_max_body_size 200M` and Django `FILE_UPLOAD_MAX_MEMORY_SIZE`
- **Temp file cleanup:** `finally:` block in `views.py` always deletes the uploaded file, even on error
- **Rate throttling:** `30/hour` for anonymous requests (prevents API abuse)
- **No `SECRET_KEY` in code:** Read from environment variable

---

## 11. Phase 5 — Frontend Clinical Interface

The frontend is a single `index.html` file (~1,700 lines) served statically by Nginx.

### Technology Choices

- **TailwindCSS** (CDN): utility-first CSS, no build step required
- **Vanilla JavaScript** with the Fetch API: no React/Vue overhead; the UI is simple enough
- **Google Fonts:** Playfair Display (serif, for headings) + Plus Jakarta Sans (sans-serif, for body)
- **No external JS dependencies:** reduces attack surface and load time

### Colour Palette (Medical Grade)

| Variable | Hex | Usage |
|---|---|---|
| `dr-cream` | `#F9F7F2` | Background (warm white, easy on eyes in clinical environments) |
| `dr-dark` | `#2C1810` | Primary text (dark brown, not pure black — reduces eye strain) |
| `dr-brown` | `#4A3B32` | Secondary text, muted elements |
| `dr-accent` | `#8B2635` | Deep maroon — highlights, CTAs, critical indicators |
| `dr-gold` | `#D4AF37` | Subtle highlights, warning states |

### Key UI Sections

1. **Navigation bar** — Sticky, becomes opaque with blur effect on scroll
2. **Hero / Overview section** — Project description, technology badges
3. **Diagnostic Tool section (`#diagnose`)** — File upload zone, scanning animation, result panel
4. **Care Protocols section (`#care`)** — 5-stage dietary/clinical recommendation cards (highlighted based on diagnosis)
5. **Metrics section (`#metrics`)** — Performance targets table

### File Upload Flow (JavaScript)

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("/predict", {
    method: "POST",
    body: formData,        // sends as multipart/form-data automatically
});

const data = await response.json();
// data.grade → 0-4 severity level
// data.confidence → percentage
// data.time → processing time
```

### Scanning Animation

While the API call is in-flight:
- A red scanning line (`scan-line`) animates down the upload zone via CSS `@keyframes scan`
- A loading overlay covers the result panel with a spinner
- The submit button text changes to "Processing Image..."

### Error Handling

`try/catch` wraps the entire fetch call. On network error or non-200 response:
- `showToast()` displays a red toast notification (bottom-right)
- The button is re-enabled
- The loading overlay is hidden

---

## 12. Phase 6 — Docker Deployment

### Container Architecture

```
Internet → Port 80
              │
         ┌────▼────────────────────────────────────────┐
         │  Nginx Container (nginx:1.25-alpine)         │
         │                                              │
         │  /         → frontend/index.html             │
         │  /static/  → static_volume (Django assets)  │
         │  /media/   → media_volume (uploads)          │
         │  /api/     → proxy → django:8000             │
         │  /admin/   → proxy → django:8000             │
         └─────────────────┬────────────────────────────┘
                           │ Internal Docker network
                           ▼
         ┌─────────────────────────────────────────────────┐
         │  Django Container (python:3.10-slim)             │
         │                                                   │
         │  Gunicorn: 2 workers × 2 threads                  │
         │  Timeout: 120 seconds                             │
         │  Port: 8000 (not exposed externally)             │
         │                                                   │
         │  On startup: loads cnn_lstm_final.pth into RAM   │
         │  On request: DMD → CNN-LSTM → JSON response      │
         └───────────────────────────────────────────────────┘

Shared Docker Volumes:
  static_volume  ← Django collectstatic output (CSS, JS, admin assets)
  media_volume   ← Temp .mat uploads (cleaned after inference)
  model_weights  ← Persistent: cnn_lstm_final.pth survives rebuilds
```

### Dockerfile Breakdown

```dockerfile
FROM python:3.10-slim              ← Minimal base image (~150MB vs ~1GB full)

RUN apt-get install -y \
    libgl1-mesa-glx \             ← Required by matplotlib (rendering backend)
    libglib2.0-0 \                ← Required by OpenCV (if used)
    libhdf5-dev \                 ← Required by h5py for .mat v7.3 files
    curl                          ← Required by Docker healthcheck

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt   ← Cached layer: only re-runs if requirements.txt changes

COPY . .

RUN python manage.py collectstatic --noinput   ← Gather static files into /app/static/

RUN adduser --disabled-password appuser
USER appuser                       ← Run as non-root for security

CMD ["gunicorn", "core.wsgi:application",
     "--bind", "0.0.0.0:8000",
     "--workers", "2",             ← 2 workers: each loads model into RAM (~50MB each)
     "--threads", "2",             ← 2 threads per worker: handles concurrent I/O
     "--timeout", "120",           ← 2 min timeout: allows DMD + CNN-LSTM on CPU
     "--log-level", "info"]
```

**Why 2 workers?** Each Gunicorn worker is a separate OS process that loads its own copy of the PyTorch model. More workers = more RAM. On a typical 4GB RAM server, 2 workers is the sweet spot. Increase to 4 on an 8GB+ server.

**Why 120s timeout?** On a modern CPU (e.g., AWS t3.medium), DMD on a large `.mat` file takes ~15s, and CNN-LSTM forward pass takes ~5s. 120s provides a 5× safety margin.

### Docker Compose Commands

```bash
# Build and start everything in background
docker-compose up --build -d

# View logs
docker-compose logs -f django
docker-compose logs -f nginx

# Stop everything (keep volumes)
docker-compose down

# Stop everything and delete volumes (clean slate)
docker-compose down -v

# Restart just the Django container (e.g. after updating model weights)
docker-compose restart django

# Check container health
docker-compose ps

# Open Django shell inside container
docker-compose exec django python manage.py shell

# Run Django migrations inside container
docker-compose exec django python manage.py migrate
```

### Environment Variables (`.env` file)

Create a `.env` file in the project root before first run:

```bash
# .env — DO NOT COMMIT THIS FILE

SECRET_KEY=your-generated-secret-key-here
DJANGO_DEBUG=False
ALLOWED_HOSTS=localhost,your-server-ip,your-domain.com
MODEL_WEIGHTS_PATH=/app/ml_engine/weights/cnn_lstm_final.pth
INFERENCE_DEVICE=cpu
MEDIA_ROOT_PATH=/app/media
```

Generate a secret key:
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

---

## 13. All Model Parameters Explained

### `CNNLSTMElastography` Constructor Parameters

| Parameter | Default | What It Controls | Effect of Changing |
|---|---|---|---|
| `cnn_channels` | `[1,16,32,64,128]` | Channel progression in U-Net encoder | More channels = more capacity, more RAM, slower training |
| `lstm_hidden` | `256` | LSTM hidden units per direction | Larger = more temporal memory, more parameters |
| `lstm_layers` | `2` | Number of stacked LSTM layers | More layers = deeper temporal reasoning |
| `n_classes` | `3` | Output classes | Must be 3 for Benign/Precancerous/Malignant |
| `patch_size` | `64` | Spatial crop size (H×W of input) | Larger = more context, more RAM, must divide evenly |
| `dropout` | `0.3` | Dropout probability throughout model | Higher = more regularisation, slower convergence |

### `process_rf_pipeline()` Parameters

| Parameter | Default | What It Controls |
|---|---|---|
| `low_cutoff` | `0.05` | DFT bandpass lower bound (fraction of Nyquist) |
| `high_cutoff` | `0.45` | DFT bandpass upper bound |
| `dmd_rank` | `None` | SVD truncation rank (None = auto by energy) |
| `dmd_energy` | `0.999` | Cumulative energy threshold for auto rank selection |
| `dmd_damping` | `0.85` | |λ| threshold: modes below this are discarded as artefacts |
| `rayleigh_sigma` | `0.1` | Rayleigh noise amplitude (fraction of signal amplitude) |
| `n_augmented` | `3` | Number of noisy copies to generate per simulation file |
| `random_state` | `42` | NumPy random seed for reproducibility |

### `run_inference()` Parameters

| Parameter | Default | What It Controls |
|---|---|---|
| `patch_size` | `64` | Central spatial crop size fed to model |
| `n_frames` | `8` | Number of temporal frames used (from start of sequence) |
| `dft_low` | `0.05` | Passed to signal processing |
| `dft_high` | `0.45` | Passed to signal processing |
| `dmd_damping` | `0.85` | Passed to DMD reconstruction |

### Settings `ML_ENGINE` Dictionary (settings.py)

| Key | Default | Meaning |
|---|---|---|
| `WEIGHTS_PATH` | `backend/ml_engine/weights/cnn_lstm_final.pth` | Where to load the .pth file from |
| `DEVICE` | `cpu` | `cpu` or `cuda` (GPU) |
| `PATCH_SIZE` | `64` | Must match training |
| `N_FRAMES` | `8` | Must match training |
| `CNN_CHANNELS` | `[1,16,32,64,128]` | Must match training |
| `LSTM_HIDDEN` | `256` | Must match training |
| `LSTM_LAYERS` | `2` | Must match training |
| `N_CLASSES` | `3` | Must match training |

**Critical:** Every architecture parameter in `ML_ENGINE` must exactly match the values used when `cnn_lstm_final.pth` was trained. Mismatches cause a `RuntimeError: size mismatch` when loading weights.

---

## 14. All Hyperparameters and Why They Were Chosen

### Training Hyperparameters (Notebook 02 — Pre-training)

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate; better than SGD for sparse gradients in early training |
| Learning rate | `1e-4` | Standard for Adam with CNN-LSTM; lower than usual due to LSTM stability |
| Batch size | `8` | Limited by CPU RAM during development; increase to 32+ on GPU |
| Epochs | `50` | Pre-training converges in ~30 epochs; 50 provides safety margin |
| LR scheduler | CosineAnnealing | Gradually reduces LR without needing manual tuning |
| Weight decay | `1e-5` | L2 regularisation to prevent overfitting on clean simulation data |

### ADMM Fine-tuning Hyperparameters (Notebook 03)

| Hyperparameter | Value | Justification |
|---|---|---|
| ADMM penalty ρ | `0.1` | Balances constraint enforcement vs objective; too high → oscillation |
| TV weight λ | `0.01` | Weak TV regularisation; strong would over-smooth the displacement field |
| LNCC weight α | `1.0` | Primary unsupervised loss |
| Focal weight β | `0.5` | Secondary; reduced because no labels in Phase 2B |
| Inner SGD steps | `5` | Steps per ADMM iteration; too many → constraint violated |
| ADMM iterations | `100` | Per training sample; convergence typically within 50 |
| Freeze layers | Encoder only | Early CNN layers frozen; only LSTM + heads fine-tuned in Phase 2B |

### DFT Filter Design

| Parameter | Value | Justification |
|---|---|---|
| `low_cutoff` | `0.05` | Removes DC component and ultra-low frequency probe drift |
| `high_cutoff` | `0.45` | Stays below Nyquist (0.5) to avoid aliasing; removes high-freq noise |

This range (5–45% of Nyquist) was chosen by analysing the power spectral density of Dataset 1 simulation files — true tissue motion energy is concentrated in this band.

### DMD Parameters

| Parameter | Value | Justification |
|---|---|---|
| `dmd_energy` | `0.999` | Retain 99.9% of signal energy; excludes noise-dominated singular values |
| `dmd_damping` | `0.85` | Speckle modes have |λ| < 0.5; tissue modes have |λ| ≈ 0.95–1.0; 0.85 is a safe threshold |

### Rayleigh Noise

| Parameter | Value | Justification |
|---|---|---|
| `sigma` | `0.1` | Measured by comparing SNR of Dataset 1 vs Dataset 2; ~10% noise-to-signal ratio in real data |
| `n_augmented` | `3` | Triples simulation dataset size; more copies only marginally improve domain adaptation |

---

## 15. Running Everything Step by Step

### Prerequisites

```bash
# Check Python version (must be 3.10+)
python3 --version

# Check Docker
docker --version
docker-compose --version

# Check CUDA (optional, for GPU training)
nvidia-smi
```

### Step 1: Clone and Set Up

```bash
git clone https://github.com/your-username/ultrasound-elastography-grading.git
cd ultrasound-elastography-grading

# Create Python virtual environment for notebooks
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# OR:
venv\Scripts\activate       # Windows

pip install -r backend/requirements.txt
```

### Step 2: Download Datasets

```bash
mkdir -p data/simulation data/phantom
# Download Dataset 1 to data/simulation/
# Download Dataset 2 to data/phantom/
```

### Step 3: Run Notebooks in Order

```bash
cd notebooks/
jupyter lab

# In Jupyter:
# 1. Open 01_data_preprocessing.ipynb
#    - Set DATA_ROOT = "../data/simulation"
#    - Run all cells
#    - Outputs: processed .npy arrays in data/processed/

# 2. Open 02_simulation_pretrain.ipynb
#    - Set PROCESSED_ROOT = "../data/processed"
#    - Run all cells (takes 30–120 min on CPU, 5–15 min on GPU)
#    - Outputs: backend/ml_engine/weights/pretrain_checkpoint.pth

# 3. Open 03_unsupervised_tuning.ipynb
#    - Set PHANTOM_ROOT = "../data/phantom"
#    - Set PRETRAIN_WEIGHTS = "../backend/ml_engine/weights/pretrain_checkpoint.pth"
#    - Run all cells (takes 1–4 hours on CPU, 20–60 min on GPU)
#    - Outputs: backend/ml_engine/weights/cnn_lstm_final.pth

# 4. Open 04_statistical_testing.ipynb
#    - Set HOLDOUT_ROOT = "../data/phantom/holdout"
#    - Set WEIGHTS = "../backend/ml_engine/weights/cnn_lstm_final.pth"
#    - Run all cells
#    - Outputs: p-value, CI, CNR plots
```

### Step 4: Deploy with Docker

```bash
# Create environment file
cp .env.example .env
# Edit .env with your values (SECRET_KEY, ALLOWED_HOSTS, etc.)

# Ensure weights are in place
ls backend/ml_engine/weights/cnn_lstm_final.pth   # Must exist

# Build and start
docker-compose up --build -d

# Monitor startup
docker-compose logs -f django
# Look for: "[ml_engine] Loaded weights from: ..."
# Look for: "[api.apps] Model loaded successfully — ready for inference."

# Test health endpoint
curl http://localhost/api/health/
# Expected: {"status": "ok", "model_loaded": true, "device": "cpu"}

### Step 5: The Development Way (Without Docker)

This is the fastest way to test changes during development without waiting for Docker to build.

**1. Run the Django Backend:**
Open your first terminal window:
```bash
cd backend/
# (Optional) python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

**2. Run the Frontend:**
Open a second terminal window:
```bash
cd frontend/
# Use Python's built-in server to serve static files on port 80
python -m http.server 80
```

**3. Access the Dashboard:**
Visit `http://localhost` in your browser. 

> [!NOTE]
> Since `CORS_ALLOW_ALL_ORIGINS = DEBUG` is set to `True` by default when `DJANGO_DEBUG=True` (or omitted) in your `.env`, the frontend on port 80 can talk to the backend on port 8000 without any security blocks.


# Open browser
open http://localhost
```

### Step 5: Test the API

```bash
# Upload a .mat file
curl -X POST http://localhost/api/predict/ \
     -F "mat_file=@/path/to/test_scan.mat"

# Expected response:
{
  "status": "success",
  "diagnosis": "Benign",
  "class_index": 0,
  "confidence_scores": {
    "Benign": 0.8823,
    "Precancerous": 0.0912,
    "Malignant": 0.0265
  },
  "strain_heatmap_b64": "iVBORw0KGgoAAAA...",
  "dmd_rank": 14,
  "rf_shape": [512, 128, 10],
  "filename": "test_scan.mat",
  "warning": null
}
```

### Step 6: Development Server (No Docker)

```bash
cd backend/
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py runserver 0.0.0.0:8000

# In another terminal:
# Test directly at http://localhost:8000/api/health/
```

---

## 16. API Reference (Every Endpoint)

### POST `/api/predict/`

**Purpose:** Accepts a `.mat` file, runs Phase 1 signal processing + CNN-LSTM inference, returns diagnosis.

**Request:**
```
Method: POST
Content-Type: multipart/form-data
Body field: mat_file (required)
Max size: 200 MB
```

**Success Response (HTTP 200):**
```json
{
  "status":             "success",
  "diagnosis":          "Precancerous",
  "class_index":        1,
  "confidence_scores":  {
    "Benign":           0.1200,
    "Precancerous":     0.7800,
    "Malignant":        0.1000
  },
  "strain_heatmap_b64": "<base64-encoded PNG string>",
  "dmd_rank":           14,
  "rf_shape":           [512, 128, 10],
  "filename":           "patient_scan.mat",
  "warning":            null
}
```

**Error Responses:**

| HTTP Code | When | Body |
|---|---|---|
| 400 | No file, wrong extension | `{"status":"error","message":"..."}` |
| 422 | Signal processing failed | `{"status":"error","message":"..."}` |
| 503 | Model not loaded | `{"status":"error","message":"..."}` |
| 500 | Server crash | `{"status":"error","message":"..."}` |

**Field Descriptions:**

| Field | Type | Description |
|---|---|---|
| `diagnosis` | string | `"Benign"`, `"Precancerous"`, or `"Malignant"` |
| `class_index` | int | 0, 1, or 2 |
| `confidence_scores` | object | Softmax probabilities, all three sum to 1.0 |
| `strain_heatmap_b64` | string | Base64-encoded jet-colormap PNG, 400×400 px |
| `dmd_rank` | int | Number of DMD modes retained (quality indicator) |
| `rf_shape` | array | `[depth, width, n_frames]` of the loaded RF stack |
| `filename` | string | Original uploaded filename |
| `warning` | string/null | Non-null if processing succeeded with caveats |

---

### GET `/api/health/`

**Purpose:** Liveness and readiness probe.

**Success Response (HTTP 200):**
```json
{
  "status":       "ok",
  "model_loaded": true,
  "device":       "cpu"
}
```

**Failure Response (HTTP 503):**
```json
{
  "status":       "degraded",
  "model_loaded": false,
  "detail":       "Model weights not loaded. Check server logs."
}
```

---

## 17. How Data Flows Through the Entire System

Here is the exact function call chain from `.mat` file to JSON response:

```
PredictView.post()                          [api/views.py]
  │
  ├── _validate_mat_file(uploaded_file)      validates extension + size
  ├── _save_upload(uploaded_file)            saves to media/uploads/<uuid>.mat
  │
  └── run_inference(model, filepath, device) [ml_engine/inference.py]
        │
        ├── process_rf_pipeline(filepath)   [ml_engine/signal_processing.py]
        │     │
        │     ├── load_mat_file(filepath)
        │     │     ├── sio.loadmat()        (try v5 first)
        │     │     └── h5py.File()          (fallback to v7.3)
        │     │
        │     ├── extract_rf_frames(mat_data)
        │     │     └── returns (D, W, F) float64 array
        │     │
        │     ├── filter_rf_stack(rf_raw)
        │     │     └── apply_2d_dft_bandpass() per frame
        │     │           └── fftshift(fft2(frame)) → mask → ifft2()
        │     │
        │     ├── build_global_hankel_matrix(rf_filtered)
        │     │     └── returns X ∈ ℝ^(D·W·m, F-m+1)
        │     │
        │     ├── exact_dmd(X1, X2)
        │     │     ├── np.linalg.svd(X1)    → U, Σ, V*
        │     │     ├── Ã = U*·X2·V·Σ⁻¹
        │     │     ├── np.linalg.eig(Ã)     → eigenvalues, W
        │     │     └── Φ = X2·V·Σ⁻¹·W      (DMD modes)
        │     │
        │     └── dmd_reconstruct(dmd_result)
        │           └── keep modes with |λ| ≥ 0.85
        │           └── reshape back to (D, W, F)
        │
        ├── _prepare_tensor(rf_clean)
        │     ├── central spatial crop: (D, W, F) → (64, 64, F)
        │     ├── temporal window: keep first 8 frames
        │     ├── normalise to [-1, 1] by max absolute value
        │     └── permute + reshape → (1, 8, 1, 64, 64) float32 tensor
        │
        ├── model(x_tensor)                  [ml_engine/architecture.py]
        │     ├── CNN: UNetEncoder per frame  → (8, 128) features
        │     ├── LSTM: sequence tracking     → (1, 8, 512) lstm_out
        │     │                                  (1, 512) h_final
        │     ├── classifier(h_final)         → (1, 3) logits
        │     └── disp_head(lstm_out)         → (1, 8, 64, 64) displacement
        │
        ├── torch.softmax(logits)             → (1, 3) probabilities
        │
        └── _render_strain_heatmap(disp_np)
              ├── mean across T frames        → (64, 64) strain map
              ├── matplotlib imshow(jet cmap) → PNG in BytesIO buffer
              └── base64.b64encode()          → string
```

---

## 18. Troubleshooting Common Errors

### "No RF key found in .mat file"
The `extract_rf_frames()` function could not find a recognised key. Solution: open your `.mat` file in MATLAB or Python and check available keys:
```python
import scipy.io as sio
data = sio.loadmat('your_file.mat')
print([k for k in data.keys() if not k.startswith('__')])
```
Then add your key to the `candidate_keys` list in `signal_processing.py`.

### "No DMD modes passed threshold 0.85"
All eigenvalues were below the damping threshold — the signal is too noisy or too short. Try:
1. Lowering `dmd_damping` to `0.5`
2. Checking that the `.mat` file has `F ≥ 4` frames
3. Increasing `dft_high_cutoff` slightly (more signal passes the filter)

### "RuntimeError: size mismatch for classifier.4.weight"
Architecture parameters in `settings.py` do not match the trained weights. Check that `CNN_CHANNELS`, `LSTM_HIDDEN`, `LSTM_LAYERS`, and `N_CLASSES` exactly match the values used during training.

### "Model running with random initialisation (demo mode)"
The `.pth` file was not found at `WEIGHTS_PATH`. Solution:
```bash
ls backend/ml_engine/weights/cnn_lstm_final.pth
# If missing, copy the file:
cp /path/to/cnn_lstm_final.pth backend/ml_engine/weights/
# In Docker:
docker-compose exec django ls /app/ml_engine/weights/
```

### "504 Gateway Timeout"
Nginx timed out waiting for Django. The inference took longer than 150s. Solutions:
1. Increase `proxy_read_timeout` in `nginx.conf`
2. Increase `--timeout` in Dockerfile Gunicorn CMD
3. Use a GPU (`INFERENCE_DEVICE=cuda` in `.env`)
4. Check that your `.mat` file isn't unusually large

### "413 Request Entity Too Large"
The uploaded file exceeded Nginx's limit. Solution: increase `client_max_body_size` in `nginx.conf` and `FILE_UPLOAD_MAX_MEMORY_SIZE` in `settings.py`.

### ImportError on `from ml_engine import ...`
Django cannot find the `ml_engine` package. Ensure `backend/` is the working directory and `ml_engine/` exists inside it with an `__init__.py` file.

### Gunicorn "CRITICAL WORKER TIMEOUT"
Worker was killed because inference exceeded `--timeout`. Increase timeout (see above) or reduce model complexity.

---

## 19. Glossary of Every Technical Term

| Term | Definition |
|---|---|
| **RF Signal** | Radio Frequency signal — the raw electrical signal output by an ultrasound transducer before any image processing |
| **Elastography** | Imaging technique that measures tissue stiffness by tracking how tissue deforms under mechanical stress |
| **Strain** | A measure of tissue deformation — how much a tissue region compressed relative to its original size |
| **Strain Map** | A 2D image where each pixel colour represents local tissue stiffness (e.g., jet colormap: blue=soft, red=stiff) |
| **CNR** | Contrast-to-Noise Ratio — measures how distinctly a tumour region stands out from background tissue in the strain map |
| **Label-Free** | No manual annotations/labels were needed to train the model (fully unsupervised fine-tuning) |
| **.mat file** | MATLAB data file format. Contains 3D arrays of ultrasound RF data. v5 = old format, v7.3 = HDF5-based |
| **2D-DFT** | 2D Discrete Fourier Transform — converts a spatial image into its frequency components |
| **Bandpass filter** | Keeps only a frequency range of interest; removes both very low and very high frequencies |
| **Hankelization** | Converting a 1D time signal into a 2D matrix using time-delay embedding; reveals hidden dynamical structure |
| **DMD** | Dynamic Mode Decomposition — decomposes a time-varying signal into spatial modes and their temporal dynamics |
| **SVD** | Singular Value Decomposition — factorises a matrix as U·Σ·V*; used inside DMD for dimensionality reduction |
| **Eigenvalue** | In DMD context: complex number λ encoding a mode's growth rate (|λ|) and oscillation frequency (angle(λ)) |
| **Rayleigh distribution** | Probability distribution describing ultrasound speckle noise amplitude; arises from sums of many small random scatterers |
| **Domain Gap** | Difference between training data distribution (simulation) and test data distribution (real hardware) |
| **Sim-to-Real** | Techniques to transfer knowledge learned from simulation to real-world environments |
| **CNN** | Convolutional Neural Network — uses sliding convolutional filters to detect spatial patterns in images |
| **U-Net** | Encoder-decoder CNN architecture originally designed for medical image segmentation; used here as encoder only |
| **LSTM** | Long Short-Term Memory — recurrent neural network cell that can learn long-range temporal dependencies |
| **Bidirectional LSTM** | LSTM that processes the sequence in both forward and backward directions simultaneously |
| **BatchNorm** | Batch Normalisation — normalises layer activations across the batch; stabilises and accelerates training |
| **GELU** | Gaussian Error Linear Unit — smooth activation function `x·Φ(x)`; outperforms ReLU in many settings |
| **Dropout** | Randomly zeroing activations during training to prevent overfitting |
| **Softmax** | Converts raw logits to probabilities summing to 1.0: `p_i = exp(z_i) / Σ exp(z_j)` |
| **Logits** | Raw unnormalised scores output by the classification head before softmax |
| **LNCC** | Local Normalised Cross-Correlation — measures similarity between two displacement fields locally |
| **Total Variation** | Penalises the total magnitude of spatial/temporal gradients in the displacement field |
| **Focal Loss** | Modified cross-entropy that down-weights easy examples; addresses class imbalance |
| **ADMM** | Alternating Direction Method of Multipliers — splits a non-smooth optimisation into solvable sub-problems |
| **Proximal operator** | Generalised gradient step for non-smooth functions; prox of L1 = soft-thresholding |
| **Soft-thresholding** | `sign(v)·max(|v|−τ, 0)` — shrinks values toward zero, used as proximal operator for TV |
| **Augmented Lagrangian** | Adds a penalty term for constraint violation to the Lagrangian of a constrained optimisation problem |
| **Dual variable** | Lagrange multiplier in ADMM; accumulates constraint violations to ensure eventual satisfaction |
| **MLE** | Maximum Likelihood Estimation — finds parameter values that maximise the probability of observed data |
| **Confidence Interval** | Range that contains the true parameter with stated probability (e.g., 95% CI) |
| **Wilcoxon Test** | Non-parametric statistical test comparing two paired samples without assuming normality |
| **p-value** | Probability of observing results at least as extreme as the data if the null hypothesis were true |
| **DRF** | Django REST Framework — Python library for building REST APIs with Django |
| **Gunicorn** | Green Unicorn — WSGI HTTP server for Python web applications; production replacement for Django's `runserver` |
| **WSGI** | Web Server Gateway Interface — standard interface between Python web apps and HTTP servers |
| **Nginx** | High-performance HTTP server and reverse proxy |
| **Reverse proxy** | Server that forwards client requests to backend servers, handling load balancing, SSL, and static files |
| **Multipart form data** | HTTP content type for file uploads, sends files as binary parts of the request body |
| **Base64** | Encoding scheme that converts binary data (PNG bytes) to ASCII text safe for JSON transmission |
| **Docker** | Platform for running applications in isolated containers |
| **Docker Compose** | Tool for defining and running multi-container Docker applications |
| **Named volume** | Docker storage that persists beyond container lifecycle (survives `docker restart`) |
| **Health check** | Automated test that verifies a service is running correctly; used by Docker and load balancers |

---

## 20. Syllabus Integration Map

This project directly implements every topic from the three course units:

### Unit 1 — Linear Algebra and Signal Processing

| Topic | Implementation | File | Code Function |
|---|---|---|---|
| **2D Discrete Fourier Transform** | Bandpass filter for RF noise removal | `signal_processing.py` | `apply_2d_dft_bandpass()` |
| **Inverse DFT** | Reconstruction after frequency-domain filtering | `signal_processing.py` | `ifft2(ifftshift(...))` |
| **Hankel Matrix** | Time-delay embedding of pixel temporal signals | `signal_processing.py` | `build_global_hankel_matrix()` |
| **Singular Value Decomposition** | Core of Exact DMD; rank selection by energy | `signal_processing.py` | `np.linalg.svd(X1)` |
| **Matrix Operations** | Reduced operator Ã = U*·X₂·V·Σ⁻¹ | `signal_processing.py` | `exact_dmd()` |
| **Eigendecomposition** | Finding DMD modes and eigenvalues | `signal_processing.py` | `np.linalg.eig(A_tilde)` |
| **Least Squares** | DMD mode amplitude estimation | `signal_processing.py` | `np.linalg.lstsq()` |
| **Dynamic Mode Decomposition** | Artefact rejection and signal reconstruction | `signal_processing.py` | `exact_dmd()`, `dmd_reconstruct()` |

### Unit 2 — Optimisation and Deep Learning

| Topic | Implementation | File | Code Location |
|---|---|---|---|
| **Stochastic Gradient Descent** | θ-update in ADMM inner loop | `03_unsupervised_tuning.ipynb` | `optimizer.step()` |
| **Loss Functions (LNCC)** | Unsupervised acoustic tracking loss | `architecture.py` | `LNCCLoss.forward()` |
| **Loss Functions (Focal)** | Imbalanced classification loss | `architecture.py` | `FocalLoss.forward()` |
| **Total Variation Regularization** | Displacement field smoothness | `architecture.py` | `SpatioTemporalTVLoss.forward()` |
| **Matrix Splitting (ADMM)** | Split smooth + non-smooth objectives | `03_unsupervised_tuning.ipynb` | ADMM iteration loop |
| **Proximal Algorithms** | z-update via soft-thresholding | `architecture.py` | `soft_threshold()` |
| **Augmented Lagrangian** | Full ADMM objective formulation | `03_unsupervised_tuning.ipynb` | L_ρ computation |
| **Dual Variable Update** | u^{k+1} = u^k + D(θ) - z | `03_unsupervised_tuning.ipynb` | `u = u + d_theta - z` |
| **Kaiming / Xavier Init** | Weight initialisation for stable training | `architecture.py` | `_init_weights()` |
| **Batch Normalisation** | Activations normalised per mini-batch | `architecture.py` | `nn.BatchNorm2d()` |

### Unit 3 — Statistical Estimation and Hypothesis Testing

| Topic | Implementation | File | Code Location |
|---|---|---|---|
| **Maximum Likelihood Estimation** | MLE of mean CNR under Gaussian model | `04_statistical_testing.ipynb` | `mu_mle = cnr_values.mean()` |
| **MLE Variance Estimation** | Biased MLE of σ² | `04_statistical_testing.ipynb` | `sigma2_mle = ((cnr - mu)**2).mean()` |
| **95% Confidence Intervals** | CI = μ̂ ± 1.96·σ̂/√N | `04_statistical_testing.ipynb` | CI computation |
| **Null Hypothesis Formulation** | H₀: model CNR ≤ baseline CNR | `04_statistical_testing.ipynb` | Written in cell markdown |
| **Wilcoxon Signed-Rank Test** | Non-parametric paired comparison | `04_statistical_testing.ipynb` | `wilcoxon(cnr_model, cnr_baseline)` |
| **p-value Interpretation** | Reject H₀ if p < 0.05 | `04_statistical_testing.ipynb` | `if p_value < 0.05:` |
| **Statistical Sampling** | Hold-out split as random sample | `04_statistical_testing.ipynb` | 20% hold-out from Dataset 2 |

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════════╗
║           ULTRASOUND ELASTOGRAPHY — QUICK REFERENCE                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  DATASETS:                                                           ║
║  • Dataset 1 (Simulation) → Phase 1 + Phase 2A only                 ║
║  • Dataset 2 (Phantom)    → Phase 2B (train 80%) + Phase 3 (test 20%)║
║                                                                      ║
║  PIPELINE ORDER:                                                     ║
║  .mat → DFT filter → Hankel → DMD → tensor (1,8,1,64,64)            ║
║      → CNN (per frame) → Bi-LSTM → classifier → softmax → JSON      ║
║                                                                      ║
║  KEY PARAMETERS:                                                     ║
║  patch_size=64, n_frames=8, lstm_hidden=256, cnn=[1,16,32,64,128]   ║
║  dmd_damping=0.85, dft_low=0.05, dft_high=0.45                      ║
║                                                                      ║
║  KEY FILES:                                                          ║
║  signal_processing.py  → Phase 1 (DFT, Hankel, DMD)                 ║
║  architecture.py       → Model classes + Loss functions              ║
║  inference.py          → End-to-end: .mat → JSON                     ║
║  views.py              → HTTP: .mat upload → run_inference() → JSON  ║
║  apps.py               → Load model ONCE at startup                  ║
║  nginx.conf            → 200MB upload limit, 150s timeout            ║
║                                                                      ║
║  DEPLOY:  docker-compose up --build -d                               ║
║  TEST:    curl -X POST localhost/api/predict/ -F mat_file=@scan.mat  ║
║  HEALTH:  curl localhost/api/health/                                 ║
║  TRAIN:   Run notebooks 01 → 02 → 03 → 04 in order                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

*Developed for Concordia University · ENCS · Advanced Topics in Machine Learning*
*Physics-Informed Deep Learning | ADMM Optimization | Statistical Validation | Clinical Deployment*
