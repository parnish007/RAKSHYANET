"""Vercel serverless entrypoint for the RakshyaNet API.

Vercel's Python runtime looks for an ASGI ``app`` in a small set of default
locations; ``api/index.py`` is one of them. It imports this module once per cold
start and then hands it every request, so the only job here is to make the
``backend`` package importable and re-export the real application.

The path insert is not ceremony. A serverless function is imported with the
function's own directory on ``sys.path``, not the repository root, so
``import backend`` fails without it -- and it fails at cold start, which
surfaces as a 500 on the first request rather than as a build error.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.main import app  # noqa: E402  (import must follow the path fix)

__all__ = ["app"]
