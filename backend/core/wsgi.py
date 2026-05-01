"""
core/wsgi.py
============
WSGI entry point for Gunicorn in production.

Gunicorn invocation (from Dockerfile CMD / docker-compose):
    gunicorn core.wsgi:application \
        --bind 0.0.0.0:8000 \
        --workers 2 \
        --threads 4 \
        --timeout 120 \
        --log-level info

Notes
-----
    --timeout 120   : Required; .mat file processing (DMD + CNN-LSTM) can take
                      30–90 s on CPU.  Gunicorn default (30 s) will kill the worker.
    --workers 2     : Keep low when each worker loads a full PyTorch model into RAM.
                      Increase only if you have sufficient VRAM/RAM headroom.
    --threads 4     : I/O-bound waiting (file read, JSON serialisation) benefits
                      from threads within each worker process.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

application = get_wsgi_application()
