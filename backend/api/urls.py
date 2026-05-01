"""
api/urls.py
===========
URL routing for the Elastography ML inference API.

All routes are prefixed with /api/ by core/urls.py.

Endpoints
---------
    POST /api/predict/  — Accept a .mat file upload, run the full
                          DMD + CNN-LSTM pipeline, return a JSON payload
                          containing diagnosis, confidence scores, and a
                          Base64-encoded strain heatmap.

    GET  /api/health/   — Lightweight liveness probe used by Docker Compose
                          healthcheck and load-balancer health checks.
                          Returns HTTP 200 {"status": "ok"} when the server
                          and the loaded model are both ready.
"""

from django.urls import path

from .views import PredictView, HealthCheckView

app_name = "api"

urlpatterns = [
    # ── Primary inference endpoint ────────────────────────────────────────────
    path("predict/", PredictView.as_view(), name="predict"),

    # ── Liveness / readiness probe ────────────────────────────────────────────
    path("health/",  HealthCheckView.as_view(), name="health"),
]
