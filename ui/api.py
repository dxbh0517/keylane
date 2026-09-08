"""The UI's one way to reach the daemon.

Every request needs the loopback token now (see ``daemon/auth.py``), and a
token attached at twenty-odd call sites is a token forgotten at one of them.
So the base URL and the header live here, and the UI asks for a path.

The token is read from the same settings file the daemon writes it to. The UI
runs as the same user — that is the whole basis of the scheme — so reading it
is not a privilege it did not already have.
"""

from __future__ import annotations

import errno
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DAEMON = "http://127.0.0.1:9100"

_cached_token: str | None = None
_warned = False


def auth_headers() -> dict[str, str]:
    """The token header, read once and remembered.

    A miss is not fatal: the daemon answers with a 403 naming the reason, which
    is a better failure than the UI refusing to start because it could not find
    a file the daemon may not have written yet.
    """
    global _cached_token
    if _cached_token is None:
        try:
            # read_token, not load_token: the UI must never create a token the
            # daemon has not seen. See daemon/auth.py.
            from daemon.auth import read_token

            _cached_token = read_token()
        except Exception:  # noqa: BLE001
            logger.warning("could not read the API token", exc_info=True)
            _cached_token = ""
    if not _cached_token:
        # Said once, loudly, naming the file. Without this the only symptom is
        # a 403 from every authenticated route — the models list comes back
        # empty and nothing anywhere says why.
        global _warned
        if not _warned:
            _warned = True
            from daemon.paths import SETTINGS_PATH

            logger.error(
                "no API token in %s — every request to the daemon will be refused "
                "with 403. The daemon writes the token to its own data directory; "
                "if this path is inside a release, the two are looking at "
                "different files.",
                SETTINGS_PATH,
            )
        return {}
    from daemon.auth import TOKEN_HEADER

    return {TOKEN_HEADER: _cached_token}


def forget_token() -> None:
    """Drop the cached token, so the next call re-reads it."""
    global _cached_token, _warned
    _cached_token = None
    _warned = False


def describe_request_error(exc: BaseException) -> str:
    """What to show when a request to the daemon did not complete.

    httpx surfaces a refused connection as ``[Errno 111] Connection refused``,
    which is true and useless: the number is the kernel's name for "nothing
    is listening", and the process that should have been listening is the
    Keylane daemon. Name that, not the errno.
    """
    if _daemon_unreachable(exc):
        return (
            "the Keylane daemon is not running — nothing is listening on "
            "127.0.0.1:9100"
        )
    return str(exc)


def _daemon_unreachable(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno == errno.ECONNREFUSED:
            return True
        text = str(current).lower()
        if "connection refused" in text or "[errno 111]" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


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
