"""ASGI entrypoint for container and serverless hosts.

The application object lives in app.py, because Vercel resolves a Python
entrypoint by filename and reaches app.py first. This module re-exports it so
`uvicorn asgi:app` works too, which is the conventional name elsewhere.

Running `python3 integrated_system.py` is still the way to serve this locally.
"""
from app import app

__all__ = ["app"]
