"""
tests/conftest.py — shared fixtures for the Recipe RAG test suite.
"""

import pytest
import httpx

API_URL = "http://host.docker.internal:8041"   # adjust if different


def api_available() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


requires_api = pytest.mark.skipif(
    not api_available(),
    reason="Backend API not running — start with docker compose up",
)


def ask(query: str, timeout: int = 60) -> dict:
    """POST /recommend and return parsed JSON."""
    r = httpx.post(
        f"{API_URL}/recommend",
        data={"query": query},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()
