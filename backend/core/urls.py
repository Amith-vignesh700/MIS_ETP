"""
core/urls.py
============
Root URL configuration for the Ultrasound Elastography inference backend.

Routes
------
    /admin/          → Django admin panel
    /api/            → ML inference API (api/urls.py)
    /media/<path>    → Uploaded .mat file serving (dev only; Nginx handles this in prod)

Notes
-----
    In production (Docker + Nginx), the Nginx reverse proxy intercepts /media/ and
    /static/ requests directly without hitting Gunicorn.  This urlpatterns block
    only adds those routes when DEBUG=True so `runserver` works for local dev.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django admin
    path("admin/", admin.site.urls),

    # ML inference API — all routes defined in api/urls.py
    path("api/", include("api.urls")),
]

# Serve uploaded media files in development (Nginx takes over in production)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
