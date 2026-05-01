# 🔬 Ultrasound Elastography for Label-Free Tissue Tumor Grading

> **A physics-informed deep learning pipeline for autonomous, label-free tumor grading using ultrasound elastography — integrating Dynamic Mode Decomposition, CNN-LSTM hybrid networks, ADMM optimization, and statistical validation into a containerized clinical web application.**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Clinical Motivation](#-clinical-motivation)
3. [Technology Stack](#-technology-stack)
4. [Datasets & Routing Strategy](#️-datasets--routing-strategy)
5. [Dataset-to-Phase Mapping](#-dataset-to-phase-mapping)
6. [Architecture Pipeline](#️-architecture-pipeline)
   - [Phase 1: Dynamic Data Processing & Sim-to-Real Augmentation](#phase-1-dynamic-data-processing--sim-to-real-augmentation)
   - [Phase 2: Physics-Informed CNN-LSTM Training](#phase-2-physics-informed-cnn-lstm-architecture)
   - [Phase 3: Clinical Robustness & Statistical Validation](#phase-3-clinical-robustness--statistical-validation)
   - [Phase 4: Django Backend API](#phase-4-django-backend-api-development)
   - [Phase 5: Frontend Clinical Interface](#phase-5-frontend-clinical-interface-htmlcssjs)
   - [Phase 6: Productionization & Containerization](#phase-6-productionization--containerization)
7. [Mathematical Foundations](#-mathematical-foundations)
8. [Project File Structure](#-project-file-structure)
9. [Setup & Installation](#-setup--installation)
10. [API Reference](#-api-reference)
11. [Running the Notebooks](#-running-the-notebooks)
12. [Results & Validation Metrics](#-results--validation-metrics)
13. [Syllabus Unit Integration Map](#-syllabus-unit-integration-map)
14. [Future Work](#-future-work)
15. [License & Citation](#-license--citation)

---

## 📖 Project Overview

This project presents an advanced, **physics-informed deep learning pipeline** for label-free tissue tumor grading using ultrasound elastography. By modeling tissue compression as a **dynamic spatio-temporal system** rather than a static image comparison, the architecture achieves high-fidelity strain estimation without requiring manual annotation or biopsy labels.

The system categorizes tumors into three distinct diagnostic classes:

| Class | Clinical Description |
|---|---|
| 🟢 **Benign** | Non-cancerous, uniform stiffness profile |
| 🟡 **Precancerous** | Heterogeneous stiffness, early-stage dysplasia |
| 🔴 **Malignant** | Stiff, irregular lesion with high contrast ratio |

Clinicians receive **multi-class confidence scores** and **statistical 95% confidence intervals** on every prediction, served through a secure, containerized web application.

---

## 🏥 Clinical Motivation

Traditional ultrasound elastography requires:
- Manual palpation and subjective strain assessment by a trained radiologist
- Biopsy confirmation (invasive, costly, delayed results)
- Region-of-interest (ROI) annotation that is time-consuming and observer-dependent

This pipeline **eliminates all of the above** by:
1. Learning tissue deformation mechanics from physics (DMD, ADMM, TV regularization)
2. Producing a quantitative, reproducible strain map from raw RF ultrasound frames
3. Classifying tumors **label-free** — no annotated training data required for fine-tuning
4. Providing **hypothesis-tested statistical confidence** before any clinical deployment

---

## 🧰 Technology Stack

| Layer | Tool | Purpose |
|---|---|---|
| **Deep Learning** | PyTorch 2.x | CNN-LSTM model definition, training, inference |
| **Signal Processing** | SciPy, NumPy | 2D-DFT, Hankel matrices, DMD |
| **Data I/O** | `scipy.io`, `h5py` | Loading MATLAB `.mat` files (v5 and v7.3) |
| **Statistical Analysis** | SciPy Stats | MLE, Wilcoxon test, confidence intervals |
| **Backend API** | Django 4.x, DRF | REST endpoint for `.mat` upload and inference |
| **Frontend UI** | HTML5, CSS3, Vanilla JS | Clinical dashboard with Fetch API |
| **WSGI Server** | Gunicorn | Production Python application server |
| **Reverse Proxy** | Nginx | HTTP routing, static file serving, large uploads |
| **Containerization** | Docker, Docker Compose | Reproducible deployment to any Linux cloud |
| **Notebooks** | Jupyter Lab | Experimentation, training, validation |

---

## 🗄️ Datasets & Routing Strategy

To bridge the **sim-to-real domain gap** and achieve maximum clinical accuracy, this project utilizes a **Hybrid Transfer Learning Strategy** across two primary datasets provided by Concordia University IMPACT Lab:

### Dataset 1 — Simulation Database
**URL:** https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/

- Mathematically generated RF ultrasound frames from finite element simulation
- Contains ground-truth displacement and strain fields
- No transducer noise, no speckle decorrelation artifacts
- **Role:** Pre-training — teaches the network the fundamental physics of wave propagation and tissue mechanics in a clean mathematical environment

### Dataset 2 — In-Vivo & Phantom Database
**URL:** https://users.encs.concordia.ca/~impact/ultrasound-elastography-dataset-for-unsupervised-training/

- Real ultrasound acquisitions from physical breast tissue phantoms and in-vivo subjects
- Contains realistic transducer artifacts, speckle noise, and boundary effects
- Acquired with clinical-grade ultrasound hardware
- **Role:** Domain adaptation (unsupervised fine-tuning) + Hold-out testing

---

## 🗺️ Dataset-to-Phase Mapping

This is the **explicit, production-ready routing** of which dataset feeds each phase of the pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATASET ROUTING DIAGRAM                          │
│                                                                     │
│  Dataset 1 (Simulation) ──────────────────────────────────────────┐│
│                          │                                         ││
│                    ┌─────▼──────┐   ┌──────────────────────┐      ││
│                    │  Phase 1   │──▶│  Phase 2A            │      ││
│                    │ Processing │   │  CNN-LSTM Pre-Train  │      ││
│                    └────────────┘   └──────────────────────┘      ││
│                                                                    ││
│  Dataset 2 (In-Vivo) ────────────────────────────────────────────┐││
│                       │                                           │││
│               ┌───────┴───────┐                                   │││
│               │               │                                   │││
│          ┌────▼─────┐   ┌─────▼────┐   ┌──────────────────────┐  │││
│          │Train     │   │Hold-out  │──▶│  Phase 3             │  │││
│          │Split     │   │Split     │   │  Statistical Validation│ │││
│          └────┬─────┘   └──────────┘   └──────────────────────┘  │││
│               │                                                   │││
│        ┌──────▼──────┐                                            │││
│        │  Phase 2B   │                                            │││
│        │  Fine-Tune  │                                            │││
│        └─────────────┘                                            │││
└─────────────────────────────────────────────────────────────────────┘
```

| Phase | Dataset Used | Split | Action |
|---|---|---|---|
| **Phase 1** | Dataset 1 (Simulation) | Full | 2D-DFT, Hankelization, DMD, Rayleigh noise injection |
| **Phase 2A** | Dataset 1 (processed) | Full | Supervised CNN-LSTM pre-training |
| **Phase 2B** | Dataset 2 (In-Vivo) | Train (80%) | Unsupervised ADMM fine-tuning, label-free |
| **Phase 3** | Dataset 2 (Phantom) | Hold-out (20%) | Locked `model.eval()`, statistical validation only |
| **Phase 4-6** | New patient `.mat` files | N/A | Live clinical inference via Django API |

---

## 🏗️ Architecture Pipeline

### Phase 1: Dynamic Data Processing & Sim-to-Real Augmentation
> *Integrates Syllabus Unit 1: Fourier Transform, Hankel Matrices, Dynamic Mode Decomposition*

Raw ultrasound `.mat` files are noisy and subject to signal decorrelation. Phase 1 treats the RF signals as a continuous **spatio-temporal sequence** governed by wave mechanics.

#### Step 1.1 — Data Ingestion
```python
import scipy.io as sio
import h5py

# For MATLAB v5 files:
data = sio.loadmat('rf_pre.mat')

# For MATLAB v7.3 files (HDF5):
with h5py.File('rf_pre.mat', 'r') as f:
    rf_frames = np.array(f['rf_data'])
```

#### Step 1.2 — 2D Discrete Fourier Transform (2D-DFT)
The RF data matrix `X ∈ ℝ^(M×N)` is transformed into the frequency domain to analyze axial (temporal) and lateral (spatial) frequency components simultaneously:

```
X̂[k₁, k₂] = Σ_m Σ_n X[m, n] · e^(-j2π(mk₁/M + nk₂/N))
```

A **band-pass mask** in the frequency domain removes low-frequency motion artifacts and high-frequency quantization noise before reconstruction via inverse 2D-DFT.

#### Step 1.3 — Hankelization (Time-Delay Embedding)
The pixel temporal evolution `x(t) = [x₁, x₂, ..., x_T]` is mapped into the high-dimensional Hankel matrix `H ∈ ℝ^(p×q)`:

```
H = | x₁  x₂  x₃  ... x_q  |
    | x₂  x₃  x₄  ... x_q+1|
    | ...                   |
    | x_p x_p+1 ...  x_T   |
```

where `p + q - 1 = T`. This **time-delay embedding** converts the 1D signal into a matrix form suitable for Dynamic Mode Decomposition.

#### Step 1.4 — Exact Dynamic Mode Decomposition (DMD)
DMD assumes the linear relationship `H_{k+1} ≈ A · H_k` and computes the best-fit linear operator `A` via SVD:

```
1. SVD: H ≈ UΣV*
2. Ã = U*AU = U*H'V Σ⁻¹   (reduced operator)
3. Eigendecompose: ÃW = WΛ
4. Exact DMD modes: Φ = H'VΣ⁻¹W
5. Reconstruct: X_clean = Σ_r (φ_r · b_r · λ_r^t)
```

**Artifact rejection:** Modes with `|λ_r| << 1` (highly damped — speckle noise) are dropped. Only dominant, physically meaningful modes are kept for reconstruction.

#### Step 1.5 — Domain Randomization (Rayleigh Noise)
To bridge the sim-to-real gap, Rayleigh-distributed phase jitter is injected into simulation data:

```python
sigma = 0.1  # Rayleigh scale parameter
noise = sigma * np.sqrt(-2 * np.log(np.random.uniform(0, 1, rf.shape)))
rf_augmented = rf + noise * np.exp(1j * 2 * np.pi * np.random.uniform(size=rf.shape))
```

---

### Phase 2: Physics-Informed CNN-LSTM Architecture
> *Integrates Syllabus Unit 2: SGD, Loss Functions, Matrix Splitting, Proximal Algorithms, ADMM*

#### Step 2.1 — Network Architecture: CNN-LSTM Hybrid

```
Input: RF Frame Sequence [B, T, 1, H, W]
         │
         ▼
┌─────────────────────┐
│  U-Net CNN Encoder  │  ← Extracts spatial speckle features per frame
│  (Shared Weights)   │    Conv → BN → ReLU × 4 levels
│  Output: [B,T,C,h,w]│
└─────────────────────┘
         │
         ▼ Flatten spatial → [B, T, C*h*w]
┌─────────────────────┐
│   Bi-LSTM Layer     │  ← Tracks tissue deformation across T frames
│   hidden_dim=256    │    Learns compression dynamics
│   Output: [B,T,256] │
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────────────┐
│Strain │ │ Classification   │
│Map    │ │ Head             │
│Output │ │ FC(256→128→3)    │
│[B,H,W]│ │ Softmax → 3 cls  │
└───────┘ └──────────────────┘
```

#### Step 2.2 — Physics-Informed Loss Function

The total training loss combines three physically motivated terms:

```
L_total = α·L_LNCC + β·L_TV + γ·L_focal
```

**Local Normalized Cross-Correlation (LNCC)** — Data fidelity for acoustic tracking:
```
LNCC(I, J) = [Σ(I-Ī)(J-J̄)]² / [Σ(I-Ī)² · Σ(J-J̄)²]
L_LNCC = 1 - mean(LNCC)
```

**Spatio-Temporal Total Variation (TV)** — Physical continuity regularizer:
```
L_TV = Σ_{x,y,t} √[(u_{x+1,y,t} - u_{x,y,t})² + (u_{x,y+1,t} - u_{x,y,t})² + λ_t(u_{x,y,t+1} - u_{x,y,t})²]
```

**Focal Loss** — Handles class imbalance for rare Precancerous class:
```
L_focal = -Σ α_c · (1 - p_c)^γ · log(p_c)
```
where `γ=2` (focusing parameter) and `α_c` is the inverse-frequency class weight.

#### Step 2.3 — ADMM Optimization Loop

Standard SGD cannot minimize non-smooth Total Variation. The problem is split via ADMM:

**Augmented Lagrangian:**
```
L_ρ(θ, z, u) = L_smooth(θ) + L_TV(z) + (ρ/2)||D(θ) - z + u||²
```

**ADMM Iteration:**
```
Step A (θ-update): θ^{k+1} = argmin_θ [L_smooth(θ) + (ρ/2)||D(θ) - z^k + u^k||²]
                   ↳ Solved via SGD/Adam on smooth gradients

Step B (z-update):  z^{k+1} = prox_{L_TV/ρ}(D(θ^{k+1}) + u^k)
                   ↳ Solved via Soft-Thresholding (Proximal Operator)
                   ↳ shrink(v, λ) = sign(v) · max(|v| - λ, 0)

Step C (u-update):  u^{k+1} = u^k + D(θ^{k+1}) - z^{k+1}
                   ↳ Dual variable update (Lagrange multiplier)
```

---

### Phase 3: Clinical Robustness & Statistical Validation
> *Integrates Syllabus Unit 3: Statistical Estimation Theory, Hypothesis Testing*

#### Step 3.1 — Elastographic CNR Computation
Elastographic Contrast-to-Noise Ratio (CNR) quantifies the ability to distinguish tumor from background:

```
CNR = |μ_tumor - μ_background| / √(σ²_tumor + σ²_background)
```

#### Step 3.2 — Maximum Likelihood Estimation (MLE)
Treating CNR values as i.i.d. samples from a Gaussian distribution `CNR_i ~ N(μ, σ²)`:

```
μ̂_MLE = (1/N) Σᵢ CNR_i          (sample mean)
σ̂²_MLE = (1/N) Σᵢ (CNR_i - μ̂)²  (biased MLE variance)

95% CI: μ̂ ± 1.96 · (σ̂/√N)
```

#### Step 3.3 — Wilcoxon Signed-Rank Hypothesis Test

**Null Hypothesis H₀:** The deep learning model's median CNR ≤ baseline standard algorithm CNR

**Alternative H₁:** The model's median CNR > baseline (one-tailed test)

```python
from scipy.stats import wilcoxon
stat, p_value = wilcoxon(cnr_model, cnr_baseline, alternative='greater')
# Reject H₀ if p_value < 0.05
```

A `p-value < 0.05` is required to validate the model for clinical deployment.

---

### Phase 4: Django Backend API Development

#### Project Structure
```
backend/
├── core/settings.py     ← ALLOWED_HOSTS, INSTALLED_APPS, FILE_UPLOAD_MAX_MEMORY_SIZE
├── core/urls.py         ← include('api.urls')
├── api/apps.py          ← Model loaded into memory on Django startup
├── api/views.py         ← PredictView: .mat upload → DMD → inference → JSON
├── api/urls.py          ← path('predict/', PredictView.as_view())
└── ml_engine/
    ├── architecture.py  ← CNNLSTMElastography class
    └── signal_processing.py ← hankelization(), exact_dmd(), apply_dft_filter()
```

#### API Endpoint
```
POST /api/predict/
Content-Type: multipart/form-data

Body:
  rf_pre  : <.mat file>  # Pre-compression RF frame stack
  rf_post : <.mat file>  # Post-compression RF frame stack

Response 200 OK:
{
  "diagnosis": "Precancerous",
  "confidence": {
    "Benign": 0.12,
    "Precancerous": 0.78,
    "Malignant": 0.10
  },
  "cnr_estimate": 4.32,
  "confidence_interval_95": [3.98, 4.66],
  "strain_heatmap_b64": "<base64-encoded PNG>",
  "processing_time_ms": 1840
}
```

---

### Phase 5: Frontend Clinical Interface (HTML/CSS/JS)

The frontend is a **medical-grade, responsive clinical dashboard** served as static files through Nginx:

- **File Upload Zone:** Drag-and-drop `.mat` file pair (pre/post compression)
- **Strain Heatmap Viewer:** Base64-decoded image rendered dynamically via JavaScript
- **Confidence Bar Chart:** Animated progress bars showing 3-class probabilities
- **CNR Gauge:** Visual indicator with 95% CI bounds
- **Status Indicators:** Real-time processing spinner and error handling

---

### Phase 6: Productionization & Containerization

#### Docker Architecture
```
Internet
   │
   ▼
┌──────────────────────────────────────────┐
│  Nginx Container  (Port 80)              │
│  ├── Serves: /frontend → static HTML     │
│  ├── Forwards: /api/ → Django:8000       │
│  └── client_max_body_size: 500M          │
└───────────────────┬──────────────────────┘
                    │ proxy_pass
                    ▼
┌──────────────────────────────────────────┐
│  Django + Gunicorn Container (Port 8000) │
│  ├── 4 Gunicorn workers                  │
│  ├── timeout: 120s (for .mat processing) │
│  └── CNN-LSTM model loaded in memory     │
└──────────────────────────────────────────┘
```

**Deploy with a single command:**
```bash
docker-compose up --build -d
```

---

## 📐 Mathematical Foundations

| Concept | Formula | Used In |
|---|---|---|
| **2D-DFT** | `X̂[k₁,k₂] = ΣΣ X[m,n]·e^(-j2π(mk₁/M+nk₂/N))` | Phase 1 |
| **Hankel Matrix** | `H[i,j] = x[i+j-1]` | Phase 1 |
| **DMD** | `A ≈ H'VΣ⁻¹U*` | Phase 1 |
| **LNCC Loss** | `L = 1 - E[(I-Ī)(J-J̄)] / (σ_I·σ_J)` | Phase 2 |
| **TV Regularization** | `TV(u) = Σ|∇u|` | Phase 2 |
| **Focal Loss** | `FL = -α(1-p)^γ log(p)` | Phase 2 |
| **ADMM** | `θ,z,u` alternating updates | Phase 2 |
| **Soft-Threshold** | `S_λ(v) = sign(v)·max(|v|-λ,0)` | Phase 2 |
| **MLE** | `μ̂ = (1/N)Σx_i` | Phase 3 |
| **95% CI** | `μ̂ ± 1.96·σ̂/√N` | Phase 3 |
| **Wilcoxon** | Non-parametric paired test | Phase 3 |

---

## 📁 Project File Structure

```
ultrasound-elastography-grading/
│
├── .gitignore                          # Ignores .mat files, venv, __pycache__, *.pth
├── README.md                           # This exhaustive pipeline documentation
├── docker-compose.yml                  # Orchestrates Django and Nginx containers
│
├── notebooks/                          # Research, Training, and Validation
│   ├── 01_data_preprocessing.ipynb     # Phase 1: .mat loading, 2D-DFT, Hankelization, DMD
│   ├── 02_simulation_pretrain.ipynb    # Phase 2A: CNN-LSTM pre-training on Dataset 1
│   ├── 03_unsupervised_tuning.ipynb    # Phase 2B: ADMM fine-tuning on Dataset 2
│   └── 04_statistical_testing.ipynb   # Phase 3: Wilcoxon test, MLE, Confidence Intervals
│
├── backend/                            # Django REST API & PyTorch Inference
│   ├── Dockerfile                      # Python 3.10-slim + PyTorch CPU + SciPy
│   ├── requirements.txt                # All Python dependencies
│   ├── manage.py                       # Django CLI
│   │
│   ├── core/                           # Django project settings
│   │   ├── __init__.py
│   │   ├── settings.py                 # Hosts, apps, file upload config
│   │   ├── urls.py                     # Root URL routing
│   │   └── wsgi.py                     # Gunicorn entry point
│   │
│   ├── api/                            # ML inference API app
│   │   ├── __init__.py
│   │   ├── apps.py                     # Loads .pth model into memory at startup
│   │   ├── urls.py                     # /predict/ route
│   │   └── views.py                    # .mat upload → DMD → inference → JSON
│   │
│   ├── ml_engine/                      # Pure-Python ML logic (decoupled from Django)
│   │   ├── __init__.py
│   │   ├── architecture.py             # CNNLSTMElastography PyTorch class
│   │   ├── signal_processing.py        # hankelization(), exact_dmd(), dft_filter()
│   │   └── weights/
│   │       └── cnn_lstm_final.pth      # Trained model weights (NOT committed to Git)
│   │
│   └── static/                         # Django collectstatic output
│
├── frontend/                           # Clinical UI (Phase 5)
│   ├── index.html                      # Dashboard: upload, heatmap, confidence bars
│   ├── css/
│   │   └── style.css                   # Medical-grade responsive styling
│   └── js/
│       └── app.js                      # Async Fetch API → DOM updates
│
└── nginx/                              # Reverse Proxy (Phase 6)
    └── nginx.conf                      # Routes traffic, serves static, handles big uploads
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Docker Desktop ≥ 24.x and Docker Compose ≥ 2.x
- OR: Python 3.10+, pip, Jupyter Lab (for notebook-only usage)

### Option A: Full Docker Deployment (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/your-username/ultrasound-elastography-grading.git
cd ultrasound-elastography-grading

# 2. Place your trained model weights
cp /path/to/cnn_lstm_final.pth backend/ml_engine/weights/

# 3. Build and launch all containers
docker-compose up --build -d

# 4. Access the clinical dashboard
open http://localhost:80
```

### Option B: Notebook-Only (Development & Training)
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install torch torchvision scipy numpy h5py matplotlib seaborn jupyter

# 3. Launch Jupyter Lab
jupyter lab --notebook-dir=notebooks/

# 4. Run notebooks in order:
#    01 → 02 → 03 → 04
```

### Option C: Django Development Server
```bash
cd backend/
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

---

## 📡 API Reference

### `POST /api/predict/`
Accepts a pre/post compression `.mat` file pair and returns a full diagnostic JSON payload.

**Request:**
```http
POST /api/predict/ HTTP/1.1
Content-Type: multipart/form-data

rf_pre=<binary .mat file>
rf_post=<binary .mat file>
```

**Response `200 OK`:**
```json
{
    "diagnosis": "Precancerous",
    "confidence": {
        "Benign": 0.12,
        "Precancerous": 0.78,
        "Malignant": 0.10
    },
    "cnr_estimate": 4.32,
    "confidence_interval_95": [3.98, 4.66],
    "strain_heatmap_b64": "iVBORw0KGgoAAAA...",
    "processing_time_ms": 1840,
    "dmd_modes_kept": 12,
    "model_version": "cnn_lstm_v1.0"
}
```

**Error `400 Bad Request`:**
```json
{
    "error": "Both rf_pre and rf_post .mat files are required."
}
```

---

## 📓 Running the Notebooks

The four notebooks are designed to be **run sequentially**:

| Notebook | Phase | Dataset | Output |
|---|---|---|---|
| `01_data_preprocessing.ipynb` | Phase 1 | Dataset 1 (Simulation) | DMD-filtered `.npy` arrays |
| `02_simulation_pretrain.ipynb` | Phase 2A | Dataset 1 processed | Pre-trained `.pth` weights |
| `03_unsupervised_tuning.ipynb` | Phase 2B | Dataset 2 train split | Final `cnn_lstm_final.pth` |
| `04_statistical_testing.ipynb` | Phase 3 | Dataset 2 hold-out | p-value, CI, CNR plots |

Update the `DATA_ROOT` path variable at the top of each notebook to point to your local dataset directory.

---

## 📊 Results & Validation Metrics

*(To be populated after training. Target values shown below.)*

| Metric | Target | Achieved |
|---|---|---|
| Mean CNR (DL model) | > 4.0 | — |
| Wilcoxon p-value | < 0.05 | — |
| 95% Confidence Interval Width | < 1.0 | — |
| Benign Classification Accuracy | > 90% | — |
| Precancerous Recall (Sensitivity) | > 80% | — |
| Malignant Precision | > 85% | — |
| Mean API Processing Time | < 3000 ms | — |

---

## 🗂️ Syllabus Unit Integration Map

| Syllabus Unit | Mathematical Concept | Pipeline Location |
|---|---|---|
| **Unit 1** | 2D Discrete Fourier Transform | Phase 1 / Notebook 01 |
| **Unit 1** | Hankel Matrix Construction | Phase 1 / Notebook 01 |
| **Unit 1** | Dynamic Mode Decomposition (DMD) | Phase 1 / Notebook 01 |
| **Unit 2** | Stochastic Gradient Descent (SGD) | Phase 2A / Notebook 02 |
| **Unit 2** | Focal Loss / Cross-Entropy | Phase 2 / Notebook 02–03 |
| **Unit 2** | Matrix Splitting (ADMM) | Phase 2B / Notebook 03 |
| **Unit 2** | Proximal Algorithm (Soft-Thresholding) | Phase 2B / Notebook 03 |
| **Unit 2** | Total Variation Regularization | Phase 2B / Notebook 03 |
| **Unit 3** | Maximum Likelihood Estimation (MLE) | Phase 3 / Notebook 04 |
| **Unit 3** | 95% Confidence Intervals | Phase 3 / Notebook 04 |
| **Unit 3** | Wilcoxon Signed-Rank Test | Phase 3 / Notebook 04 |

---

## 🔭 Future Work

- [ ] **GPU Deployment:** Add CUDA-enabled PyTorch Docker image for sub-500ms inference
- [ ] **Multi-Frame DMD:** Extend DMD to full 3D tensor (axial × lateral × temporal)
- [ ] **Transformer Backbone:** Replace LSTM with temporal self-attention for longer sequences
- [ ] **Active Learning:** Integrate clinician feedback loop to iteratively refine Precancerous boundary
- [ ] **DICOM Integration:** Accept standard DICOM ultrasound format in addition to `.mat`
- [ ] **ONNX Export:** Export PyTorch model to ONNX for cross-platform edge deployment
- [ ] **Federated Learning:** Multi-hospital training without sharing patient data

---

## 📄 License & Citation

This project is developed for academic and research purposes at **Concordia University, Montreal**.

Datasets are the intellectual property of the **IMPACT Research Group, Concordia University**:
- https://users.encs.concordia.ca/~impact/

If you use this pipeline or its methodology in your research, please cite:

```bibtex
@misc{ultrasound_elastography_grading_2024,
  title   = {Ultrasound Elastography for Label-Free Tissue Tumor Grading},
  author  = {[Author Names]},
  year    = {2024},
  school  = {Concordia University, Department of Electrical and Computer Engineering},
  note    = {Physics-informed CNN-LSTM with ADMM optimization and Wilcoxon validation}
}
```

---

*Developed as a comprehensive integration of linear algebra, continuous optimization, statistical estimation, and deep learning for clinical diagnostic applications.*  
*Concordia University · ENCS · Advanced Topics in Machine Learning*
