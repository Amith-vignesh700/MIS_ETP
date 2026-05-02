"""
ml_engine/architecture.py
=========================
PyTorch CNN-LSTM hybrid architecture for ultrasound elastography
label-free tissue tumour grading (Benign / Precancerous / Malignant).

Classes
-------
    ConvBlock             — Double-conv BN-GELU encoder unit
    UNetEncoder           — Modified U-Net spatial feature extractor
    CNNLSTMElastography   — Full model: CNN → Bi-LSTM → Classifier + Displacement Head

Usage
-----
    from ml_engine.architecture import CNNLSTMElastography
    model = CNNLSTMElastography()
    logits, displacement = model(x)   # x: (B, T, 1, H, W)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    Double convolution block with Batch Normalisation and GELU activation.

    Architecture per block:
        Conv2d(in→out, 3×3, pad=1) → BN → GELU →
        Conv2d(out→out, 3×3, pad=1) → BN → GELU →
        Dropout2d(p)

    Parameters
    ----------
    in_ch   : int   — input channels
    out_ch  : int   — output channels
    dropout : float — spatial dropout probability (default 0.1)
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Spatial Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════

class UNetEncoder(nn.Module):
    """
    Modified U-Net encoder: extracts deep speckle pattern features from
    individual RF frames.  Returns a global feature vector per frame via
    adaptive average pooling.

    Parameters
    ----------
    channels : list[int]
        Channel progression, e.g. [1, 16, 32, 64, 128].
        len(channels)-1 encoder blocks are created.
        MaxPool2d(2,2) is inserted between all but the last pair.

    Output
    ------
    feats : (B, feat_dim) where feat_dim = channels[-1]
    """

    def __init__(self, channels: list[int] | None = None) -> None:
        super().__init__()
        if channels is None:
            channels = [1, 16, 32, 64, 128]

        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.encoders.append(ConvBlock(channels[i], channels[i + 1]))
            if i < len(channels) - 2:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.feat_dim    = channels[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_ch, H, W)

        Returns
        -------
        feats : (B, feat_dim)
        """
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        feats = self.global_pool(x).squeeze(-1).squeeze(-1)   # (B, feat_dim)
        return feats


# ══════════════════════════════════════════════════════════════════════════════
#  Loss Functions (self-contained for use in inference / fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

class LNCCLoss(nn.Module):
    """
    Local Normalised Cross-Correlation Loss.

    Compares consecutive predicted displacement frames as an unsupervised
    acoustic-tracking fidelity measure.

    LNCC(u, v) = 1 - Σ_Ω (u-ū)(v-v̄) / sqrt(Σ(u-ū)² · Σ(v-v̄)²)

    Parameters
    ----------
    win : int — local window size (default 9)
    """

    def __init__(self, win: int = 9) -> None:
        super().__init__()
        self.win = win

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, T, H, W) predicted displacement field
        target : (B, T, H, W) reference field (optional; uses consecutive frames if None)

        Returns
        -------
        loss : scalar tensor  (1 − mean LNCC)
        """
        if target is None:
            if pred.shape[1] < 2:
                return torch.tensor(0.0, device=pred.device)
            u = pred[:, :-1]
            v = pred[:, 1:]
        else:
            u, v = pred, target

        B, T1, H, W = u.shape
        w   = self.win
        pad = w // 2

        u_flat = u.reshape(B * T1, 1, H, W)
        v_flat = v.reshape(B * T1, 1, H, W)

        kernel  = torch.ones(1, 1, w, w, device=u.device) / (w * w)
        u_mean  = F.conv2d(u_flat, kernel, padding=pad)
        v_mean  = F.conv2d(v_flat, kernel, padding=pad)

        u_c = u_flat - u_mean
        v_c = v_flat - v_mean

        cross = F.conv2d(u_c * v_c, kernel, padding=pad)
        u_sq  = F.conv2d(u_c * u_c, kernel, padding=pad)
        v_sq  = F.conv2d(v_c * v_c, kernel, padding=pad)

        lncc = cross / (torch.sqrt(u_sq * v_sq) + 1e-8)
        return 1.0 - lncc.mean()


class SpatioTemporalTVLoss(nn.Module):
    """
    Spatio-Temporal Total Variation regularisation.

    TV(d) = Σ_{x,y,t} |∂d/∂x| + |∂d/∂y| + |∂d/∂t|

    Enforces physically smooth, continuous displacement fields
    (tissue does not tear randomly under compression).
    """

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        disp : (B, T, H, W)

        Returns
        -------
        loss : scalar tensor (mean TV)
        """
        dx = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
        dy = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
        dt = torch.abs(disp[:, 1:, :, :] - disp[:, :-1, :, :])
        return dx.mean() + dy.mean() + dt.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class imbalanced classification.

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    Addresses severe class imbalance (Precancerous samples are rare).

    Parameters
    ----------
    gamma        : float     — focusing parameter (default 2.0)
    class_weights: list[float] | None — per-class α weights
    reduction    : str       — 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma:         float             = 2.0,
        class_weights: list[float] | None = None,
        reduction:     str               = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer(
                "alpha", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.alpha = None   # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) raw class scores
        targets : (B,)   integer class indices

        Returns
        -------
        loss : scalar tensor
        """
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════════════════
#  ADMM Proximal Operator
# ══════════════════════════════════════════════════════════════════════════════

def soft_threshold(z: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Element-wise soft-thresholding (proximal operator for ℓ₁ / TV).

    prox_{τ‖·‖₁}(z) = sign(z) · max(|z| − τ, 0)

    Parameters
    ----------
    z         : arbitrary tensor
    threshold : τ = λ / ρ

    Returns
    -------
    Thresholded tensor of same shape.
    """
    return torch.sign(z) * torch.clamp(torch.abs(z) - threshold, min=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Full CNN-LSTM Model
# ══════════════════════════════════════════════════════════════════════════════

class CNNLSTMElastography(nn.Module):
    """
    Physics-Informed CNN-LSTM for Ultrasound Elastography Tumour Grading.

    Architecture
    ------------
    For each frame t:
        RF_frame_t  →  UNetEncoder  →  feat_t          (spatial features, dim=feat_dim)

    Across the temporal sequence:
        [feat_1, ..., feat_T]  →  Bi-LSTM  →  lstm_out  (B, T, lstm_hidden×2)
                                              h_final   (B, lstm_hidden×2)

    Two heads on h_final / lstm_out:
        Classification Head  : h_final   → FC → Softmax → (B, n_classes)
        Displacement Head    : lstm_out  → FC → Upsample → (B, T, H, W)

    The displacement output is the estimated dense 2-D strain / displacement
    field, used directly by the unsupervised LNCC + TV losses.

    Parameters
    ----------
    cnn_channels : list[int]  — U-Net channel progression (default [1,16,32,64,128])
    lstm_hidden  : int        — LSTM units per direction (default 256)
    lstm_layers  : int        — stacked LSTM layers (default 2)
    n_classes    : int        — number of diagnostic classes (default 3)
    patch_size   : int        — spatial dimension of input patches (default 64)
    dropout      : float      — dropout probability used throughout (default 0.3)

    Inputs / Outputs
    ----------------
    Input  : x            — (B, T, 1, H, W)  batch of RF frame sequences
    Output : logits       — (B, n_classes)    raw class scores
             displacement — (B, T, H, W)      dense displacement field
    """

    CLASS_NAMES: list[str] = ["Benign", "Precancerous", "Malignant"]

    def __init__(
        self,
        cnn_channels: list[int] | None = None,
        lstm_hidden:  int              = 256,
        lstm_layers:  int              = 2,
        n_classes:    int              = 3,
        patch_size:   int              = 64,
        dropout:      float            = 0.3,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [1, 16, 32, 64, 128]

        self.patch_size  = patch_size
        self.lstm_hidden = lstm_hidden
        self.n_classes   = n_classes

        # ── Spatial Extractor ─────────────────────────────────────────────────
        self.cnn     = UNetEncoder(channels=cnn_channels)
        feat_dim     = cnn_channels[-1]          # = 128 by default

        # ── Temporal Tracker ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = feat_dim,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2           # bidirectional concat

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

        # ── Displacement / Strain Map Head ────────────────────────────────────
        # Number of spatial divisions from pooling: (len-2) MaxPool2d layers
        n_pools       = len(cnn_channels) - 2
        reduced       = max(patch_size // (2 ** n_pools), 4)
        self.disp_head = nn.Sequential(
            nn.Linear(lstm_out_dim, reduced * reduced),
        )
        self.disp_upsample = nn.Upsample(
            size=(patch_size, patch_size), mode="bilinear", align_corners=False
        )

        self._init_weights()

    # ── Weight Initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, 1, H, W)

        Returns
        -------
        logits       : (B, n_classes)
        displacement : (B, T, H, W)
        """
        B, T, C, H, W = x.shape

        # CNN: each frame processed independently
        x_flat = x.view(B * T, C, H, W)            # (B·T, 1, H, W)
        feats  = self.cnn(x_flat)                   # (B·T, feat_dim)
        feats  = feats.view(B, T, -1)               # (B, T, feat_dim)

        # LSTM: temporal deformation tracking
        lstm_out, (h_n, _) = self.lstm(feats)       # (B, T, lstm_out_dim)

        # Concatenate final hidden states from both directions
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, lstm_out_dim)

        # Classification
        logits = self.classifier(h_final)           # (B, n_classes)

        # Displacement field per timestep
        disp_flat    = self.disp_head(
            lstm_out.reshape(B * T, -1)
        )                                           # (B·T, r²)
        r            = int(math.isqrt(disp_flat.shape[-1]))
        disp_spatial = disp_flat.view(B * T, 1, r, r)
        disp_up      = self.disp_upsample(disp_spatial)    # (B·T, 1, H, W)
        displacement = disp_up.view(B, T, H, W)    # (B, T, H, W)

        return logits, displacement

    # ── Convenience: probabilities + class name ───────────────────────────────

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> dict[str, object]:
        """
        Run inference and return a human-readable prediction dict.

        Returns
        -------
        {
            "class_idx":    int,
            "class_name":   str,
            "probabilities": {"Benign": float, "Precancerous": float, "Malignant": float},
            "displacement":  Tensor (B, T, H, W),
        }
        """
        self.eval()
        logits, displacement = self.forward(x)
        probs    = torch.softmax(logits, dim=-1)    # (B, n_classes)
        pred_idx = int(probs.argmax(dim=-1)[0].item())

        return {
            "class_idx":  pred_idx,
            "class_name": self.CLASS_NAMES[pred_idx],
            "probabilities": {
                name: round(float(probs[0, i].item()), 4)
                for i, name in enumerate(self.CLASS_NAMES)
            },
            "displacement": displacement,
        }

    # ── Parameter summary ────────────────────────────────────────────────────

    def parameter_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"CNNLSTMElastography | "
            f"Total params: {total:,} | "
            f"Trainable: {trainable:,} ({100 * trainable / total:.1f}%)"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  High-Level Training Orchestrators
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for supervised pre-training.
    L = L_LNCC + λ_tv * L_TV + L_focal
    """

    def __init__(
        self,
        lncc_win:      int         = 9,
        tv_lambda:     float       = 0.05,
        focal_gamma:   float       = 2.0,
        class_weights: list[float] | None = None,
        n_classes:     int         = 3,
    ) -> None:
        super().__init__()
        self.lncc_loss  = LNCCLoss(win=lncc_win)
        self.tv_loss    = SpatioTemporalTVLoss()
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, class_weights=class_weights, reduction="mean"
        )
        self.tv_lambda = tv_lambda

    def forward(
        self,
        logits:       torch.Tensor,
        displacement: torch.Tensor,
        targets:      torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_lncc  = self.lncc_loss(displacement)
        l_tv    = self.tv_loss(displacement)
        l_focal = self.focal_loss(logits, targets)

        total = l_lncc + self.tv_lambda * l_tv + l_focal
        return total, {
            "lncc":  float(l_lncc.item()),
            "tv":    float(l_tv.item()),
            "focal": float(l_focal.item()),
        }


class ADMMOptimizer:
    """
    ADMM (Alternating Direction Method of Multipliers) Orchestrator.
    Used for unsupervised domain adaptation (Phase 2B).

    Splits the problem into:
        - θ-step: SGD on LNCC + ADMM penalty
        - z-step: Proximal update for TV (soft-thresholding)
        - u-step: Dual variable update
    """

    def __init__(
        self,
        model:        nn.Module,
        lr:           float = 1e-4,
        momentum:     float = 0.9,
        weight_decay: float = 1e-4,
        rho:          float = 0.1,
        tv_lambda:    float = 0.05,
        inner_iters:  int   = 3,
    ) -> None:
        self.model       = model
        self.rho         = rho
        self.tv_lambda   = tv_lambda
        self.inner_iters = inner_iters

        # Only optimize parameters with requires_grad=True
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.sgd = torch.optim.SGD(
            trainable, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        self.z: torch.Tensor | None = None
        self.u: torch.Tensor | None = None

    def _init_dual(self, d: torch.Tensor) -> None:
        if self.z is None or self.z.shape != d.shape:
            self.z = torch.zeros_like(d)
            self.u = torch.zeros_like(d)

    def step(
        self,
        x:         torch.Tensor,
        targets:   torch.Tensor,
        criterion: PhysicsInformedLoss,
    ) -> dict[str, float]:
        """Performs one full ADMM iteration."""
        total_loss_val = 0.0
        metrics_agg: dict[str, float] = {"lncc": 0, "tv": 0, "focal": 0}

        for _ in range(self.inner_iters):
            self.sgd.zero_grad()
            logits, displacement = self.model(x)
            self._init_dual(displacement.detach())

            # Smooth parts (LNCC + Focal)
            loss_pi, parts = criterion(logits, displacement, targets)

            # ADMM penalty: (ρ/2)‖d - z + u‖²
            penalty = (self.rho / 2.0) * torch.mean((displacement - self.z + self.u) ** 2)
            loss = loss_pi + penalty

            loss.backward()
            self.sgd.step()

            total_loss_val += float(loss.item())
            for k in metrics_agg:
                metrics_agg[k] += parts.get(k, 0.0)

        # z-step: Proximal update
        with torch.no_grad():
            _, d_final = self.model(x)
            self.z = soft_threshold(d_final + self.u, self.tv_lambda / self.rho)
            # u-step: Dual ascent
            self.u = self.u + d_final - self.z

        n = self.inner_iters
        return {
            "total_loss": total_loss_val / n,
            **{k: v / n for k, v in metrics_agg.items()},
        }
