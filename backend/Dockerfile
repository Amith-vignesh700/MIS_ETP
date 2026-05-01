# ============================================================
# backend/Dockerfile
# Python 3.10 slim — Django + Gunicorn + PyTorch CPU + SciPy
# ============================================================

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1-mesa-glx / libglib2.0-0 : OpenCV / matplotlib rendering
# libhdf5-dev                     : h5py for MATLAB v7.3 .mat files
# curl                            : Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libhdf5-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Collect static files ──────────────────────────────────────────────────────
RUN python manage.py collectstatic --noinput

# ── Non-root user for security ────────────────────────────────────────────────
RUN adduser --disabled-password --gecos '' appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Expose Gunicorn port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Gunicorn: extended timeout for heavy .mat processing ─────────────────────
# --workers 2 : keep RAM usage low on CPU-only inference servers
# --timeout 120 : allow up to 2 min for DMD + CNN-LSTM forward pass
CMD ["gunicorn", \
     "core.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
