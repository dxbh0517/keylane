"""Where an install keeps its data, given where its code lives.

A release lays the two out deliberately apart:

    ~/.local/share/keylane/
      releases/<tag>/     the code — never contains data/
      current -> …        what the systemd units follow
      data/               memories, models, settings

so that replacing the code cannot touch the data. `root / "data"` is therefore
right for a checkout and, inside a release, names a directory that by design
does not exist. `settings.json` is missing, the API token with it, and every
authenticated route answers 403 — the models list comes back empty with
nothing anywhere saying why.

It stayed hidden because the systemd units set KEYLANE_DATA explicitly. The UI
started cold by `--toggle`, or any script run by hand, did not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daemon.paths import default_data_dir

BASE = Path("/home/someone/.local/share/keylane")


def test_a_release_keeps_its_data_beside_the_releases_not_inside_one() -> None:
    assert default_data_dir(BASE / "releases" / "v0.7.1") == BASE / "data"


def test_every_release_of_the_same_install_shares_one_data_directory() -> None:
    """This is what makes an update and a rollback safe."""
    old = default_data_dir(BASE / "releases" / "v0.6.0")
    new = default_data_dir(BASE / "releases" / "v0.7.1")
    assert old == new == BASE / "data"


def test_an_unresolved_current_symlink_resolves_too() -> None:
    """Path.resolve() normally follows it; a caller passing it must still work."""
    assert default_data_dir(BASE / "current") == BASE / "data"


def test_a_checkout_keeps_data_inside_itself() -> None:
    """The development layout, which is where `root / "data"` was right."""
    checkout = Path("/home/someone/Documents/Code/keylane")
    assert default_data_dir(checkout) == checkout / "data"


def test_a_directory_that_merely_ends_in_a_version_is_not_a_release() -> None:
    """Only the `releases/` parent identifies the layout, not the name."""
    odd = Path("/opt/keylane/v0.7.1")
    assert default_data_dir(odd) == odd / "data"


@pytest.mark.parametrize("name", ["releases", "current"])
def test_the_data_directory_is_never_inside_the_replaceable_part(name: str) -> None:
    """The property the whole layout exists to guarantee."""
    root = BASE / "releases" / "v1.0.0" if name == "releases" else BASE / "current"
    data = default_data_dir(root)
    assert not str(data).startswith(str(BASE / "releases"))


def test_an_explicit_env_override_still_wins(monkeypatch, tmp_path: Path) -> None:
    """The systemd units set it; nothing here may override what they say."""
    monkeypatch.setenv("KEYLANE_DATA", str(tmp_path / "elsewhere"))
    import importlib

    import daemon.paths as paths

    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATA == tmp_path / "elsewhere"
    finally:
        monkeypatch.delenv("KEYLANE_DATA", raising=False)
        importlib.reload(paths)
