"""Simple header-based authorization helper."""

from __future__ import annotations

from fastapi import Request

from ..constants import AUTH_KEY


def auth_ok(request: Request) -> bool:
    """Return True when authorization passes based on AUTH_KEY requirement."""
    if not AUTH_KEY:
        return True
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    return isinstance(header, str) and header.strip() == f"Key {AUTH_KEY}"


__all__ = ["auth_ok"]


