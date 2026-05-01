"""
api/apps.py
===========
Django AppConfig for the `api` application.

Responsibility
--------------
Load the trained CNN-LSTM model weights **once** at server startup and store the
live model instance on this AppConfig object.  Every subsequent API request reads
from `ApiConfig.model` rather than re-loading the .pth file, avoiding per-request
I/O and PyTorch overhead.

Access pattern (in views.py)
-----------------------------
    from django.apps import apps
    model = apps.get_app_config("api").model
    device = apps.get_app_config("api").device

Startup flow
------------
    1. Django calls ready() automatically after all apps are initialized.
    2. ready() reads ML_ENGINE settings from settings.py.
    3. ml_engine.load_model() instantiates CNNLSTMElastography and loads weights.
    4. The model is set to eval() mode and kept in CPU/GPU memory for the
       entire lifetime of the Gunicorn worker process.

Notes
-----
    - If the .pth file is missing, load_model() emits a RuntimeWarning and
      continues with random weights (demo / development mode).
    - Heavy model loading in ready() adds ~5–15 s to cold-start time; this is
      acceptable because Gunicorn workers are long-lived.
"""

import logging

from django.apps import AppConfig

logger = logging.getLogger("api")


class ApiConfig(AppConfig):
    """Configuration for the api Django application."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "Elastography Inference API"

    # ── Shared state loaded once at startup ───────────────────────────────────
    model  = None   # CNNLSTMElastography instance
    device = "cpu"

    # ─────────────────────────────────────────────────────────────────────────
    def ready(self) -> None:
        """
        Called by Django once all apps are loaded.
        Loads the CNN-LSTM model into memory using settings from ML_ENGINE.
        """
        # Avoid running inside management commands that don't need the model
        # (e.g. migrate, collectstatic) by importing lazily.
        try:
            from django.conf import settings
            from ml_engine import load_model

            ml_cfg = getattr(settings, "ML_ENGINE", {})

            weights_path = ml_cfg.get("WEIGHTS_PATH", "")
            device       = ml_cfg.get("DEVICE", "cpu")

            # Build architecture config from settings (mirrors DEFAULT_CONFIG)
            arch_cfg = {
                "cnn_channels": ml_cfg.get("CNN_CHANNELS", [1, 16, 32, 64, 128]),
                "lstm_hidden":  ml_cfg.get("LSTM_HIDDEN",  256),
                "lstm_layers":  ml_cfg.get("LSTM_LAYERS",  2),
                "n_classes":    ml_cfg.get("N_CLASSES",    3),
                "patch_size":   ml_cfg.get("PATCH_SIZE",   64),
                "dropout":      ml_cfg.get("DROPOUT",      0.3),
            }

            logger.info(
                "[api.apps] Loading CNN-LSTM model on device='%s' from '%s'",
                device, weights_path,
            )

            ApiConfig.model  = load_model(
                weights_path = weights_path,
                device       = device,
                config       = arch_cfg,
            )
            ApiConfig.device = device

            logger.info("[api.apps] Model loaded successfully — ready for inference.")

        except Exception as exc:  # pragma: no cover
            # Never crash the server on model-load failure; views will handle None.
            logger.error(
                "[api.apps] Failed to load model at startup: %s", exc, exc_info=True
            )
            ApiConfig.model  = None
            ApiConfig.device = "cpu"
