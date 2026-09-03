"""Installing an update, whichever shape this install is.

Three shapes, and the difference matters enough to be an enum rather than an
``if``:

``CHECKOUT`` — a git working tree. Updating is a fetch and a fast-forward, and
it refuses on a dirty tree, because silently merging over someone's work in
progress is worse than not updating.

``RELEASE`` — the layout ``install.sh`` produces once it has been through the
migration below: ``~/.local/share/keylane/releases/<tag>`` with a ``current``
symlink and ``data/`` outside the release tree. Updating downloads the tarball,
verifies it, unpacks it beside the running copy, installs requirements, and
moves the symlink. Nothing is overwritten in place, so a rollback is a symlink
and a restart.

``UNKNOWN`` — an rsync install from before that layout, a container, a package.
Guessing here would mean writing over files nobody asked us to touch, so it
reports what it found and links the release page.

The two rules the whole module is built around: ``data/`` is never inside the
thing being replaced, and the running release is never modified.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import httpx

from daemon.paths import ROOT
from updater.github import Release, latest

logger = logging.getLogger(__name__)

# Keep the previous release so a rollback is instant, and one before it in case
# the previous one is what broke.
KEEP_RELEASES = 3

DOWNLOAD_TIMEOUT = 300.0

# Only these hosts, ever. A release URL that points anywhere else is a signal
# that something has gone wrong, not a redirect to follow.
ALLOWED_HOSTS = frozenset({"github.com", "api.github.com", "codeload.github.com",
                           "objects.githubusercontent.com"})

Progress = Callable[[str], None]


class UpdateError(RuntimeError):
    """An update could not be applied. The message is shown to the user."""


class InstallShape(Enum):
    CHECKOUT = "checkout"
    RELEASE = "release"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Layout:
    """Where a release install keeps its parts."""

    base: Path

    @property
    def releases(self) -> Path:
        return self.base / "releases"

    @property
    def current(self) -> Path:
        return self.base / "current"

    @property
    def data(self) -> Path:
        return self.base / "data"


def default_base() -> Path:
    return Path(os.environ.get("KEYLANE_HOME", Path.home() / ".local/share/keylane"))


def detect_shape(root: Path | None = None) -> InstallShape:
    """Which of the three this running copy is."""
    root = root or ROOT
    if (root / ".git").exists():
        return InstallShape.CHECKOUT
    # A release install runs out of releases/<tag>, reached through `current`.
    if root.parent.name == "releases" and (root.parent.parent / "current").exists():
        return InstallShape.RELEASE
    return InstallShape.UNKNOWN


def layout_for(root: Path | None = None) -> Layout | None:
    root = root or ROOT
    if detect_shape(root) is not InstallShape.RELEASE:
        return None
    return Layout(base=root.parent.parent)


# ── the checkout path ────────────────────────────────────────────────────


def _git(root: Path, *args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _update_checkout(root: Path, progress: Progress) -> str:
    dirty = _git(root, "status", "--porcelain")
    if dirty.returncode != 0:
        raise UpdateError(f"git could not read the tree: {dirty.stderr.strip()}")
    if dirty.stdout.strip():
        raise UpdateError(
            "this is a git checkout with uncommitted changes. Commit or stash "
            "them first — an update must not merge over work in progress."
        )

    progress("fetching…")
    fetched = _git(root, "fetch", "--tags", "origin")
    if fetched.returncode != 0:
        raise UpdateError(f"could not fetch: {fetched.stderr.strip()}")

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"
    progress(f"fast-forwarding {branch}…")
    merged = _git(root, "merge", "--ff-only", f"origin/{branch}")
    if merged.returncode != 0:
        raise UpdateError(
            f"could not fast-forward {branch}: {merged.stderr.strip() or merged.stdout.strip()}. "
            "The branch has diverged; resolve it by hand."
        )
    return merged.stdout.strip() or "already up to date"


# ── the release path ─────────────────────────────────────────────────────


def _check_host(url: str) -> None:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UpdateError(f"refusing a non-HTTPS download: {url}")
    if parts.hostname not in ALLOWED_HOSTS:
        raise UpdateError(f"refusing a download from an unexpected host: {parts.hostname}")


def _download(url: str, dest: Path, progress: Progress) -> str:
    """Fetch *url* to *dest*, returning its sha256."""
    _check_host(url)
    digest = hashlib.sha256()
    got = 0
    with httpx.stream(
        "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True,
        headers={"user-agent": "keylane-updater"},
    ) as resp:
        # A redirect is followed above, but only within the allowlist.
        _check_host(str(resp.url))
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                fh.write(chunk)
                digest.update(chunk)
                got += len(chunk)
                if total:
                    progress(f"downloading… {100 * got // total}%")
    return digest.hexdigest()


def _safe_extract(archive: Path, into: Path) -> Path:
    """Unpack a release tarball, refusing any member that escapes *into*.

    GitHub's tarballs wrap everything in one directory named for the commit,
    so the payload is that directory rather than the archive root.
    """
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        root_names = set()
        for member in tar.getmembers():
            target = (into / member.name).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise UpdateError(f"refusing an archive member outside the target: {member.name}")
            if member.issym() or member.islnk():
                raise UpdateError(f"refusing a link inside the archive: {member.name}")
            root_names.add(member.name.split("/", 1)[0])
        # Python 3.12 added extraction filters and 3.14 made "data" the
        # default; ask for it explicitly so the behaviour does not depend on
        # which interpreter this runs under. The checks above stand on their
        # own — this is a second pair of hands, not the policy.
        try:
            tar.extractall(into, filter="data")  # noqa: S202 — members checked above
        except TypeError:
            tar.extractall(into)  # noqa: S202 — Python < 3.12 has no filters

    if len(root_names) == 1:
        return into / next(iter(root_names))
    return into


def _install_requirements(release_dir: Path, base: Layout, progress: Progress) -> None:
    """Bring the venv in line with the new tree's requirements.txt.

    The venv lives beside the releases rather than inside one, so it survives
    a rollback — and so a release that only changed Python files needs no pip
    run at all.
    """
    venv_pip = base.base / ".venv/bin/pip"
    if not venv_pip.is_file():
        logger.info("no venv at %s; skipping dependency install", venv_pip)
        return
    requirements = release_dir / "requirements.txt"
    if not requirements.is_file():
        return
    progress("installing dependencies…")
    result = subprocess.run(
        [str(venv_pip), "install", "-q", "-r", str(requirements)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise UpdateError(
            f"dependency install failed, so the update was not applied: "
            f"{result.stderr.strip()[:400]}"
        )


def _prune_releases(base: Layout, keep: int = KEEP_RELEASES) -> None:
    live = base.current.resolve() if base.current.exists() else None
    entries = sorted(
        (p for p in base.releases.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in entries[keep:]:
        if live is not None and stale.resolve() == live:
            continue
        shutil.rmtree(stale, ignore_errors=True)


def _point_current_at(base: Layout, release_dir: Path) -> None:
    """Move the symlink atomically, so there is no moment with no current."""
    staging = base.base / ".current.new"
    if staging.exists() or staging.is_symlink():
        staging.unlink()
    staging.symlink_to(release_dir, target_is_directory=True)
    os.replace(staging, base.current)


def _restart_units(progress: Progress) -> None:
    progress("restarting…")
    for unit in ("keylane-daemon.service", "keylane-ui.service"):
        subprocess.run(
            ["systemctl", "--user", "try-restart", unit],
            capture_output=True,
            timeout=60,
            check=False,
        )


def _update_release(base: Layout, release: Release, progress: Progress) -> str:
    url = release.asset_url or release.tarball_url
    if not url:
        raise UpdateError(f"release {release.tag} has nothing to download")

    target = base.releases / release.tag
    if target.exists():
        raise UpdateError(
            f"{release.tag} is already unpacked at {target}. "
            "Roll forward to it with `keylane-update --use {release.tag}` "
            "or remove that directory to re-download."
        )

    with tempfile.TemporaryDirectory(prefix="keylane-update-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "release.tar.gz"
        got_sha = _download(url, archive, progress)

        if release.sha256:
            if got_sha != release.sha256:
                raise UpdateError(
                    "the download does not match the digest published with the "
                    f"release (expected {release.sha256[:12]}…, got {got_sha[:12]}…). "
                    "Nothing was installed."
                )
            progress("digest verified")
        else:
            # GitHub publishes no digest for the tarball it generates itself.
            # Say so rather than implying the download was checked.
            logger.warning(
                "release %s publishes no sha256; the download could not be verified "
                "beyond HTTPS to github.com",
                release.tag,
            )

        progress("unpacking…")
        payload = _safe_extract(archive, tmp_path / "unpacked")
        if not (payload / "daemon" / "main.py").is_file():
            raise UpdateError("the archive does not look like Keylane; nothing was installed")

        base.releases.mkdir(parents=True, exist_ok=True)
        shutil.copytree(payload, target)

    try:
        _install_requirements(target, base, progress)
    except UpdateError:
        # A release that cannot have its dependencies installed must not become
        # `current`, and must not be left behind to confuse the next attempt.
        shutil.rmtree(target, ignore_errors=True)
        raise

    previous = base.current.resolve() if base.current.exists() else None
    _point_current_at(base, target)
    _prune_releases(base)
    _restart_units(progress)

    was = previous.name if previous else "?"
    return f"updated from {was} to {release.tag}"


# ── the entry points ─────────────────────────────────────────────────────


def apply_update(
    channel: str = "stable",
    *,
    progress: Progress | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Install the newest release on *channel*. Raises UpdateError with a reason."""
    say: Progress = progress or (lambda _m: None)
    root = root or ROOT
    shape = detect_shape(root)

    if shape is InstallShape.CHECKOUT:
        say("updating the checkout…")
        return {"shape": shape.value, "detail": _update_checkout(root, say)}

    if shape is InstallShape.RELEASE:
        base = layout_for(root)
        assert base is not None
        release = latest(channel, force=True)
        if release is None:
            raise UpdateError("no release is published on that channel")
        if not release.is_newer and channel == "stable":
            return {"shape": shape.value, "detail": "already up to date"}
        say(f"installing {release.tag}…")
        return {"shape": shape.value, "detail": _update_release(base, release, say)}

    raise UpdateError(
        "this copy of Keylane was not installed in a way it can update itself — "
        "it is neither a git checkout nor a release install. Re-run "
        "scripts/install.sh to move to the release layout, or update by hand."
    )


def rollback(root: Path | None = None, *, progress: Progress | None = None) -> dict[str, str]:
    """Point `current` back at the previous release and restart."""
    say: Progress = progress or (lambda _m: None)
    base = layout_for(root or ROOT)
    if base is None:
        raise UpdateError("only a release install can roll back")

    live = base.current.resolve() if base.current.exists() else None
    entries = sorted(
        (p for p in base.releases.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    others = [p for p in entries if live is None or p.resolve() != live]
    if not others:
        raise UpdateError("there is no previous release to go back to")

    _point_current_at(base, others[0])
    _restart_units(say)
    return {"shape": "release", "detail": f"rolled back to {others[0].name}"}
