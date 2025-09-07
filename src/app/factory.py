"""FastAPI application factory and lifecycle hooks."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI

from ..serving.batcher import batcher
from ..api.routes import health, status, raw
from ..runtime import runtime as rt


async def on_start() -> None:
    """Startup hook: spawn batcher, kick off compile, and write readiness file."""
    rt.logger.info(f"Smart Turn v2 server | device={rt.DEVICE} dtype={rt.DTYPE} buckets={rt.BATCH_BUCKETS}")
    asyncio.create_task(batcher())

    # Start compile in a daemon thread so the event loop isn't blocked
    threading.Thread(target=rt.compile_sync, daemon=True).start()

    # Mark ready file
    try:
        root = Path(__file__).resolve().parents[2]
        run_dir = root / ".run"; run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ready").write_text("ok", encoding="utf-8")
    except Exception as e:
        rt.logger.warning(f"Failed to write readiness file: {e}")
    rt.logger.info("Smart Turn v2 server ready (startup tasks scheduled).")


def create_app() -> FastAPI:
    """Application factory to allow easy testing and extension."""
    application = FastAPI()
    application.add_event_handler("startup", on_start)
    application.add_api_route("/health", health, methods=["GET"])
    application.add_api_route("/status", status, methods=["GET"])
    application.add_api_route("/raw", raw, methods=["POST"])
    return application


__all__ = ["create_app"]


