"""HTTP server for the NRE tool-calling kernel (FastAPI)."""

from __future__ import annotations

from nre.server.app import app, create_app

__all__ = ["app", "create_app"]
