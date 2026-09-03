"""The UI's one way to reach the daemon.

Every request needs the loopback token now (see ``daemon/auth.py``), and a
token attached at twenty-odd call sites is a token forgotten at one of them.
So the base URL and the header live here, and the UI asks for a path.

The token is read from the same settings file the daemon writes it to. The UI
runs as the same user — that is the whole basis of the scheme — so reading it
is not a privilege it did not already have.
"""

from __future__ import annotations

from typing import Any

import httpx

DAEMON = "http://127.0.0.1:9100"

_cached_token: str | None = None


def auth_headers() -> dict[str, str]:
    """The token header, read once and remembered.

    A miss is not fatal: the daemon answers with a 403 naming the reason, which
    is a better failure than the UI refusing to start because it could not find
    a file the daemon may not have written yet.
    """
    global _cached_token
    if _cached_token is None:
        try:
            from daemon.auth import load_token

            _cached_token = load_token()
        except Exception:  # noqa: BLE001
            _cached_token = ""
    if not _cached_token:
        return {}
    from daemon.auth import TOKEN_HEADER

    return {TOKEN_HEADER: _cached_token}


def forget_token() -> None:
    """Drop the cached token, so the next call re-reads it."""
    global _cached_token
    _cached_token = None


def _merged(kwargs: dict[str, Any]) -> dict[str, Any]:
    headers = {**auth_headers(), **(kwargs.pop("headers", None) or {})}
    return {**kwargs, "headers": headers}


def get(path: str, **kwargs: Any) -> httpx.Response:
    return httpx.get(f"{DAEMON}{path}", **_merged(kwargs))


def post(path: str, **kwargs: Any) -> httpx.Response:
    return httpx.post(f"{DAEMON}{path}", **_merged(kwargs))


def patch(path: str, **kwargs: Any) -> httpx.Response:
    return httpx.patch(f"{DAEMON}{path}", **_merged(kwargs))


def delete(path: str, **kwargs: Any) -> httpx.Response:
    return httpx.delete(f"{DAEMON}{path}", **_merged(kwargs))


def stream(method: str, path: str, **kwargs: Any) -> Any:
    """A streaming request. Used as a context manager, like ``httpx.stream``."""
    return httpx.stream(method, f"{DAEMON}{path}", **_merged(kwargs))
