"""Updating from GitHub, and refusing to when it would be wrong.

The release path is the one that cannot be exercised by hand without
publishing something, so it is built here out of a real tarball, a real
symlink and a real swap.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from updater.apply import (
    InstallShape,
    UpdateError,
    detect_shape,
    layout_for,
    rollback,
)
from updater.github import Release
from updater.version import VERSION, is_newer, parse


# ── versions ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("candidate", "current", "newer"),
    [
        ("0.6.0", "0.5.0", True),
        ("v0.6.0", "0.5.0", True),
        ("0.5.1", "0.5.0", True),
        ("0.5.0", "0.5.0", False),
        ("0.4.9", "0.5.0", False),
        ("1.0.0", "0.9.9", True),
        # A tag with a pre-release suffix still compares on its numbers.
        ("0.6.0-rc1", "0.5.0", True),
    ],
)
def test_version_comparison(candidate: str, current: str, newer: bool) -> None:
    assert is_newer(candidate, current) is newer


def test_a_nonsense_version_does_not_explode() -> None:
    assert parse("not-a-version") == (0, 0, 0)


def test_the_shipped_version_is_parseable() -> None:
    assert parse(VERSION) > (0, 0, 0)


# ── install shapes ───────────────────────────────────────────────────────


def test_a_git_tree_is_a_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert detect_shape(tmp_path) is InstallShape.CHECKOUT


def test_a_bare_directory_is_unknown(tmp_path: Path) -> None:
    assert detect_shape(tmp_path) is InstallShape.UNKNOWN


def test_an_unknown_install_refuses_rather_than_guesses(tmp_path: Path) -> None:
    """Writing over files nobody asked us to touch is worse than not updating."""
    from updater.apply import apply_update

    with pytest.raises(UpdateError, match="not installed in a way"):
        apply_update("stable", root=tmp_path)


# ── the release layout ───────────────────────────────────────────────────


@pytest.fixture()
def release_install(tmp_path: Path) -> Path:
    """A base directory shaped the way install.sh leaves one."""
    base = tmp_path / "keylane"
    first = base / "releases" / "v0.1.0"
    (first / "daemon").mkdir(parents=True)
    (first / "daemon" / "main.py").write_text("# old", encoding="utf-8")
    (base / "data").mkdir()
    (base / "data" / "keylane.db").write_text("precious", encoding="utf-8")
    (base / "current").symlink_to(first, target_is_directory=True)
    return base


def test_a_release_install_is_recognised_through_current(release_install: Path) -> None:
    running = release_install / "current" / ""
    assert detect_shape(running.resolve()) is InstallShape.RELEASE
    layout = layout_for(running.resolve())
    assert layout is not None
    assert layout.base == release_install


def _tarball(files: dict[str, str], root: str = "keylane-abc123") -> bytes:
    """A GitHub-shaped tarball: everything under one top-level directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture()
def served(monkeypatch):
    """Serve a tarball to the updater without a network."""

    def _install(payload: bytes) -> str:
        def _download(url, dest, progress):  # noqa: ANN001
            from updater.apply import _check_host

            _check_host(url)
            Path(dest).write_bytes(payload)
            progress("downloading… 100%")
            return hashlib.sha256(payload).hexdigest()

        monkeypatch.setattr("updater.apply._download", _download)
        monkeypatch.setattr("updater.apply._install_requirements", lambda *a, **k: None)
        monkeypatch.setattr("updater.apply._restart_units", lambda *a, **k: None)
        return hashlib.sha256(payload).hexdigest()

    return _install


def _release(sha: str = "", tag: str = "v0.2.0") -> Release:
    return Release(
        channel="stable",
        tag=tag,
        version=tag.lstrip("v"),
        tarball_url=f"https://github.com/x/y/tarball/{tag}",
        sha256=sha,
    )


def test_an_update_swaps_the_symlink_and_keeps_data(release_install, served) -> None:
    from updater.apply import _update_release

    payload = _tarball({"daemon/main.py": "# new", "requirements.txt": "httpx\n"})
    sha = served(payload)

    layout = layout_for((release_install / "current").resolve())
    detail = _update_release(layout, _release(sha), lambda _m: None)

    assert "v0.2.0" in detail
    assert (release_install / "current").resolve().name == "v0.2.0"
    assert (release_install / "current" / "daemon" / "main.py").read_text() == "# new"
    # The old release is still there to go back to, and data was never touched.
    assert (release_install / "releases" / "v0.1.0").is_dir()
    assert (release_install / "data" / "keylane.db").read_text() == "precious"


def test_a_download_that_does_not_match_its_digest_installs_nothing(
    release_install, served
) -> None:
    from updater.apply import _update_release

    served(_tarball({"daemon/main.py": "# tampered"}))
    layout = layout_for((release_install / "current").resolve())

    with pytest.raises(UpdateError, match="does not match the digest"):
        _update_release(layout, _release("0" * 64), lambda _m: None)

    assert (release_install / "current").resolve().name == "v0.1.0"
    assert not (release_install / "releases" / "v0.2.0").exists()


def test_an_archive_that_is_not_keylane_installs_nothing(release_install, served) -> None:
    from updater.apply import _update_release

    sha = served(_tarball({"README.md": "wrong project"}))
    layout = layout_for((release_install / "current").resolve())

    with pytest.raises(UpdateError, match="does not look like Keylane"):
        _update_release(layout, _release(sha), lambda _m: None)
    assert (release_install / "current").resolve().name == "v0.1.0"


def test_a_failed_dependency_install_does_not_become_current(
    release_install, served, monkeypatch
) -> None:
    """A release whose requirements will not install must not be switched to."""
    from updater.apply import _update_release

    sha = served(_tarball({"daemon/main.py": "# new", "requirements.txt": "nope\n"}))

    def _boom(*_a, **_k):
        raise UpdateError("dependency install failed")

    monkeypatch.setattr("updater.apply._install_requirements", _boom)
    layout = layout_for((release_install / "current").resolve())

    with pytest.raises(UpdateError, match="dependency install failed"):
        _update_release(layout, _release(sha), lambda _m: None)

    assert (release_install / "current").resolve().name == "v0.1.0"
    # And it is cleaned up, so the next attempt does not trip over it.
    assert not (release_install / "releases" / "v0.2.0").exists()


def test_rollback_returns_to_the_previous_release(release_install, served, monkeypatch) -> None:
    from updater.apply import _update_release

    sha = served(_tarball({"daemon/main.py": "# new"}))
    layout = layout_for((release_install / "current").resolve())
    _update_release(layout, _release(sha), lambda _m: None)
    assert (release_install / "current").resolve().name == "v0.2.0"

    monkeypatch.setattr("updater.apply._restart_units", lambda *a, **k: None)
    result = rollback((release_install / "current").resolve())
    assert "v0.1.0" in result["detail"]
    assert (release_install / "current").resolve().name == "v0.1.0"


# ── what may be downloaded ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/x/y/tarball/v1",       # not HTTPS
        "https://evil.example/keylane.tar.gz",    # not GitHub
        "https://raw.gitbub.com/x.tar.gz",        # a typo is not an allowlist entry
    ],
)
def test_only_github_over_https_is_downloaded(url: str) -> None:
    from updater.apply import _check_host

    with pytest.raises(UpdateError):
        _check_host(url)


def test_github_hosts_are_allowed() -> None:
    from updater.apply import _check_host

    for url in (
        "https://api.github.com/repos/x/y/tarball/main",
        "https://codeload.github.com/x/y/tar.gz/refs/tags/v1",
        "https://objects.githubusercontent.com/blob",
    ):
        _check_host(url)


def test_an_archive_cannot_escape_its_directory(tmp_path: Path) -> None:
    """A member named ../../ would otherwise write outside the release tree."""
    from updater.apply import _safe_extract

    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo("../../escaped.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with pytest.raises(UpdateError, match="outside the target"):
        _safe_extract(archive, tmp_path / "into")
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_a_symlink_in_an_archive_is_refused(tmp_path: Path) -> None:
    from updater.apply import _safe_extract

    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("keylane/passwd")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(UpdateError, match="link inside the archive"):
        _safe_extract(archive, tmp_path / "into")


# ── the checkout path ────────────────────────────────────────────────────


def test_a_dirty_checkout_is_not_merged_over(tmp_path: Path, monkeypatch) -> None:
    from updater.apply import _update_checkout

    class _Result:
        returncode = 0
        stdout = " M agent/loop.py\n"
        stderr = ""

    monkeypatch.setattr("updater.apply._git", lambda *a, **k: _Result())
    with pytest.raises(UpdateError, match="uncommitted changes"):
        _update_checkout(tmp_path, lambda _m: None)
