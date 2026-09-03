"""Asking GitHub what the latest Keylane is.

Two channels, because they answer different questions. ``stable`` reads the
latest published release — a tag someone decided was ready. ``main`` reads the
head of the default branch, for anyone who wants the edge.

Unauthenticated GitHub allows sixty requests an hour per IP, shared with
everything else on the machine that talks to it. A desktop assistant checking
for updates must not be what exhausts that, so the answer is cached in
``data/update.json`` and the network is not touched again within the cache
window. Nothing here is required for Keylane to work: every failure returns
"no idea", never an exception into a caller that only wanted to draw a badge.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from daemon.paths import DATA
from updater.version import VERSION, is_newer

logger = logging.getLogger(__name__)

REPO = "dxbh0517/keylane"
API = "https://api.github.com"

CHANNELS = ("stable", "main")
DEFAULT_CHANNEL = "stable"

# Sixty requests an hour, shared with the rest of the machine. Once an hour is
# plenty for something a person acts on maybe monthly.
CACHE_SECONDS = 3600
STATE_PATH = DATA / "update.json"

TIMEOUT = 10.0


@dataclass
class Release:
    """A version available upstream."""

    channel: str
    tag: str
    version: str
    published_at: str = ""
    notes: str = ""
    html_url: str = ""
    tarball_url: str = ""
    # GitHub does not publish a digest for the auto-generated tarball, so a
    # release that wants verified downloads attaches its own asset. When this
    # is empty, apply.py says so rather than pretending the download was
    # checked.
    sha256: str = ""
    asset_url: str = ""

    @property
    def is_newer(self) -> bool:
        return is_newer(self.version)


def _headers() -> dict[str, str]:
    return {
        "accept": "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
        "user-agent": f"keylane/{VERSION}",
    }


def _read_state() -> dict[str, Any]:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except OSError:
        logger.debug("could not cache the update check", exc_info=True)


def _find_digest(assets: list[dict[str, Any]]) -> tuple[str, str]:
    """(sha256, download url) from a release's own attached tarball, if any."""
    for asset in assets:
        name = str(asset.get("name", ""))
        if not name.endswith((".tar.gz", ".tgz")):
            continue
        # GitHub exposes a digest on assets uploaded through the API as
        # "sha256:<hex>"; older releases have none.
        digest = str(asset.get("digest", "") or "")
        sha = digest.split(":", 1)[1] if digest.startswith("sha256:") else ""
        return sha, str(asset.get("browser_download_url", ""))
    return "", ""


def _fetch_stable() -> Release | None:
    resp = httpx.get(f"{API}/repos/{REPO}/releases/latest", headers=_headers(), timeout=TIMEOUT)
    if resp.status_code == 404:
        # A repo with no published release yet. Not an error.
        return None
    resp.raise_for_status()
    data = resp.json()
    tag = str(data.get("tag_name", ""))
    sha, asset_url = _find_digest(data.get("assets") or [])
    return Release(
        channel="stable",
        tag=tag,
        version=tag.lstrip("vV"),
        published_at=str(data.get("published_at", "")),
        notes=str(data.get("body", "") or "")[:4000],
        html_url=str(data.get("html_url", "")),
        tarball_url=str(data.get("tarball_url", "")),
        sha256=sha,
        asset_url=asset_url,
    )


def _fetch_main() -> Release | None:
    resp = httpx.get(f"{API}/repos/{REPO}/commits/main", headers=_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    sha = str(data.get("sha", ""))[:7]
    commit = data.get("commit") or {}
    return Release(
        channel="main",
        tag="main",
        # The edge channel has no version of its own, so it always reads as
        # newer than the running one when the commit differs.
        version=f"{VERSION}+{sha}",
        published_at=str((commit.get("committer") or {}).get("date", "")),
        notes=str(commit.get("message", ""))[:4000],
        html_url=str(data.get("html_url", "")),
        tarball_url=f"{API}/repos/{REPO}/tarball/main",
    )


def latest(channel: str = DEFAULT_CHANNEL, *, force: bool = False) -> Release | None:
    """The newest release on *channel*, from cache unless it is stale.

    Returns None when there is nothing published, or when GitHub could not be
    reached — the caller is drawing a badge, not making a decision.
    """
    channel = channel if channel in CHANNELS else DEFAULT_CHANNEL
    state = _read_state()
    cached = (state.get("channels") or {}).get(channel) or {}
    age = time.time() - float(cached.get("checked_at") or 0)
    if not force and cached.get("release") and age < CACHE_SECONDS:
        return Release(**cached["release"])

    try:
        release = _fetch_stable() if channel == "stable" else _fetch_main()
    except httpx.HTTPError as exc:
        logger.info("update check failed: %s", exc)
        # Keep whatever was cached rather than forgetting it over one timeout.
        return Release(**cached["release"]) if cached.get("release") else None

    channels = dict(state.get("channels") or {})
    channels[channel] = {
        "checked_at": time.time(),
        "release": asdict(release) if release else None,
    }
    _write_state({**state, "channels": channels})
    return release


@dataclass
class UpdateStatus:
    """What Settings and the footer badge render."""

    current: str
    channel: str
    available: bool = False
    latest_version: str = ""
    tag: str = ""
    notes: str = ""
    html_url: str = ""
    checked_at: float = 0.0
    install: str = ""
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def check_for_update(channel: str = DEFAULT_CHANNEL, *, force: bool = False) -> UpdateStatus:
    from updater.apply import detect_shape
    from updater.version import git_revision

    release = latest(channel, force=force)
    state = _read_state()
    checked_at = float(((state.get("channels") or {}).get(channel) or {}).get("checked_at") or 0)
    shape = detect_shape()

    if release is None:
        return UpdateStatus(
            current=VERSION,
            channel=channel,
            checked_at=checked_at,
            install=shape.name,
            detail="no published release found, or GitHub could not be reached",
        )

    available = release.is_newer
    if channel == "main":
        # The edge channel compares commits, not versions: a checkout already
        # at that commit is not behind.
        head = git_revision()
        available = bool(head) and not release.version.endswith(head)

    return UpdateStatus(
        current=VERSION,
        channel=channel,
        available=available,
        latest_version=release.version,
        tag=release.tag,
        notes=release.notes,
        html_url=release.html_url,
        checked_at=checked_at,
        install=shape.name,
        extra={"verified_download": bool(release.sha256)},
    )
