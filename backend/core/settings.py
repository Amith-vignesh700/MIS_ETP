"""
core/settings.py
================
Django settings for the Ultrasound Elastography inference backend.

Environment variables (set via .env / docker-compose env_file):
    SECRET_KEY      — Django secret key (required in production)
    DJANGO_DEBUG    — 'True' | 'False'  (default False)
    ALLOWED_HOSTS   — comma-separated hostnames, e.g. 'localhost,myserver.com'
    MEDIA_ROOT_PATH — override default media directory (optional)

Security note
-------------
Never commit a real SECRET_KEY to version control.
Generate one with:  python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
"""

import os
from pathlib import Path

# ── Base directory ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-change-me-before-production-!!!"
)

DEBUG = os.environ.get("DJANGO_DEBUG", "False").strip().lower() == "true"

# Hosts: read comma-separated string from env, fall back to localhost
_raw_hosts = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1")
ALLOWED_HOSTS: list[str] = [h.strip() for h in _raw_hosts.split(",") if h.strip()]

# ── Application definition ────────────────────────────────────────────────────
INSTALLED_APPS = [
    # Django built-ins
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third-party
    "rest_framework",           # Django REST Framework
    "corsheaders",              # Cross-Origin Resource Sharing

    # Project apps
    "api",                      # ML inference API (Phase 4)
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",      # must be first
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # serve static files without Nginx in dev
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "core.wsgi.application"

# ── Database ──────────────────────────────────────────────────────────────────
# SQLite for development / simple deployment.
# Swap for PostgreSQL in production by setting DATABASE_URL env var.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ── Password validation ───────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ── Internationalisation ──────────────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE     = "UTC"
USE_I18N      = True
USE_TZ        = True

# ── Static files (CSS, JavaScript, collected by collectstatic) ────────────────
STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "static"      # target for `collectstatic` → shared volume

# WhiteNoise: compressed static file serving
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# ── Media files (.mat uploads from clinicians) ────────────────────────────────
MEDIA_URL  = "/media/"
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT_PATH", str(BASE_DIR / "media")))

# ── File upload limits ────────────────────────────────────────────────────────
# .mat files can be large; allow up to 200 MB per upload
DATA_UPLOAD_MAX_MEMORY_SIZE = 209_715_200   # 200 MB (in-memory)
FILE_UPLOAD_MAX_MEMORY_SIZE = 209_715_200   # threshold for temp file vs memory

# ── Django REST Framework ─────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.MultiPartParser",   # .mat file uploads
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.JSONParser",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",      # tighten in production
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "30/hour",                          # prevent abuse on public API
    },
}

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the frontend (served by Nginx on port 80) to call the API.
# Override with CORS_ALLOWED_ORIGINS env var in production.
CORS_ALLOW_ALL_ORIGINS = DEBUG   # only wide-open in DEBUG mode

CORS_ALLOWED_ORIGINS: list[str] = [
    "http://localhost",
    "http://localhost:80",
    "http://127.0.0.1",
]

CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]

# ── Default primary key type ──────────────────────────────────────────────────
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── ML Engine configuration ───────────────────────────────────────────────────
# These are read by api/apps.py to locate the trained model weights.
ML_ENGINE = {
    "WEIGHTS_PATH": os.environ.get(
        "MODEL_WEIGHTS_PATH",
        str(BASE_DIR / "ml_engine" / "weights" / "cnn_lstm_final.pth"),
    ),
    "DEVICE": os.environ.get("INFERENCE_DEVICE", "cpu"),
    "PATCH_SIZE": 64,
    "N_FRAMES":   8,
    "CNN_CHANNELS": [1, 16, 32, 64, 128],
    "LSTM_HIDDEN":  256,
    "LSTM_LAYERS":  2,
    "N_CLASSES":    3,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level":    "INFO",
    },
    "loggers": {
        "django":     {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "api":        {"handlers": ["console"], "level": "INFO",    "propagate": False},
        "ml_engine":  {"handlers": ["console"], "level": "INFO",    "propagate": False},
    },
}
