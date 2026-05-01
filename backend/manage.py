#!/usr/bin/env python
"""
Django's command-line utility for administrative tasks.

Usage examples
--------------
    python manage.py runserver           # development server
    python manage.py migrate             # apply database migrations
    python manage.py collectstatic       # gather static files for Nginx
    python manage.py createsuperuser     # create admin user
    python manage.py shell               # interactive Django shell
"""

import os
import sys


def main() -> None:
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it is installed and "
            "available on your PYTHONPATH environment variable. "
            "Did you forget to activate a virtual environment?"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
