"""Smart Turn v2 FastAPI entrypoint.

Thin shim that exposes the FastAPI application for uvicorn.
See `src/app/factory.py` for the application factory.
"""

from fastapi import FastAPI

from .app.factory import create_app


# Expose module-level app for uvicorn: `uvicorn src.server:app`
app: FastAPI = create_app()