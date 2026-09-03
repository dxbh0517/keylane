"""Who may talk to the daemon.

The daemon is the authority on this machine: it can read every memory, replay
every session, and run the agent with its tools. It listens on loopback, and
loopback used to be treated as sufficient — it is not. Two things reach
127.0.0.1 that are not the user:

**A web page.** Any site the user visits can ``fetch("http://127.0.0.1:9100/
memories")``. The daemon previously answered with a CORS header saying every
origin was welcome, so the page could read the reply. The elaborate SSRF policy
in ``research/urlpolicy.py`` keeps the *model* off loopback; this keeps the
*browser* off it, which is the shorter path.

**Any other user or process on the machine.** Loopback is not per-user.

So there are two gates, and a request has to pass both:

1. **No ``Origin``.** A browser attaches ``Origin`` to every cross-origin
   request and a script cannot remove it. Keylane's own clients are not
   browsers and never send one, so refusing the header outright stops a
   malicious page even if it somehow learned the token.
2. **A token**, generated on first run into ``data/settings.json`` at mode
   0600 and sent as ``X-Keylane-Token``. Reading the token means already having
   read access to the user's own data directory.

``/health`` is exempt: it is how a launcher script asks whether the daemon is
up, it says nothing private, and the exemption keeps a shell one-liner working.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from typing import Any, Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

TOKEN_HEADER = "x-keylane-token"

# Paths that answer without a token. Keep this list short and boring: nothing
# here may reveal anything about the user or change any state.
PUBLIC_PATHS = frozenset({"/health"})

_ENV_TOKEN = "KEYLANE_TOKEN"


def _generate() -> str:
    return secrets.token_urlsafe(32)


def load_token() -> str:
    """The daemon's token, created and persisted on first call.

    An explicit ``KEYLANE_TOKEN`` in the environment wins, so a sandbox or a
    test can pin one without touching the user's settings file.
    """
    override = os.environ.get(_ENV_TOKEN, "").strip()
    if override:
        return override

    from daemon.config import get_section, save_settings
    from daemon.paths import SETTINGS_PATH

    existing = str(get_section("security").get("api_token", "") or "").strip()
    if existing:
        return existing

    token = _generate()
    save_settings("security", {"api_token": token})
    # The file holds a credential now, so it stops being world-readable. This
    # is best-effort: on a filesystem with no POSIX modes there is nothing to
    # tighten, and failing to start over it would be worse than the exposure.
    try:
        SETTINGS_PATH.chmod(0o600)
    except OSError:  # noqa: BLE001
        logger.warning("could not restrict permissions on %s", SETTINGS_PATH)
    logger.info("generated a new Keylane API token in %s", SETTINGS_PATH)
    return token


def token_matches(supplied: str) -> bool:
    """Constant-time comparison against the stored token."""
    expected = load_token()
    if not expected or not supplied:
        return False
    return hmac.compare_digest(supplied, expected)


def _refuse(reason: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": reason, "detail": detail}, status_code=403)


async def auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
) -> Any:
    """Admit one request, or refuse it with a reason a person can act on."""
    if request.method == "OPTIONS":
        # There is no CORS here on purpose, so a preflight has no useful answer.
        return _refuse(
            "cors_not_supported",
            "the Keylane daemon does not serve browsers and has no CORS policy",
        )

    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    if request.headers.get("origin"):
        return _refuse(
            "origin_refused",
            "requests carrying an Origin header are refused: this daemon is "
            "not a web API and a page in your browser must not reach it",
        )

    if not token_matches(request.headers.get(TOKEN_HEADER, "")):
        return _refuse(
            "token_required",
            f"send the token from data/settings.json as the {TOKEN_HEADER} header",
        )

    return await call_next(request)
