"""URL admission and shell argument policy.

Both exist because an allowlist of *names* is not a policy: `web_fetch` will
fetch whatever URL it is handed, and `cat` will read whatever path it is handed.
"""

from __future__ import annotations

import pytest

from daemon.shellpolicy import CommandNotAllowed, check_command, read_roots
from research.urlpolicy import UrlNotAllowed, check_url, is_public_address

import ipaddress


# ── outbound URL policy ──────────────────────────────────────────────────


@pytest.fixture()
def no_dns(monkeypatch):
    """Resolve every hostname to one address the test names."""

    def _install(address: str) -> None:
        monkeypatch.setattr("research.urlpolicy._resolve", lambda host, port: [address])

    return _install


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9100/memories",
        "http://[::1]/",
        "http://10.0.0.5/",
        "http://192.168.1.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[64:ff9b::7f00:1]/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_non_public_literals_are_refused(url: str) -> None:
    """Keylane's own API is on 127.0.0.1:9100 with no auth."""
    with pytest.raises(UrlNotAllowed):
        check_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/"])
def test_only_http_schemes_are_fetched(url: str) -> None:
    with pytest.raises(UrlNotAllowed, match="http"):
        check_url(url)


def test_credentials_in_the_url_are_refused() -> None:
    with pytest.raises(UrlNotAllowed, match="credentials"):
        check_url("http://user:secret@example.com/")


def test_a_name_resolving_to_loopback_is_refused(no_dns) -> None:
    """The DNS answer decides, not the spelling of the host."""
    no_dns("127.0.0.1")
    with pytest.raises(UrlNotAllowed, match="non-public"):
        check_url("https://totally-public.example/")


def test_a_public_name_is_admitted(no_dns) -> None:
    no_dns("93.184.216.34")
    assert check_url("https://example.com/path") == ["93.184.216.34"]


def test_translation_prefixes_are_refused_whichever_target_they_carry() -> None:
    """NAT64 and IPv4-mapped forms are ways to spell a v4 address in v6.

    Both are refused wholesale — a desktop assistant has no reason to reach
    anything through one, and admitting the public-target case would mean
    trusting the unwrap to be exhaustive.
    """
    assert is_public_address(ipaddress.ip_address("64:ff9b::a00:5")) is False
    assert is_public_address(ipaddress.ip_address("64:ff9b::808:808")) is False
    assert is_public_address(ipaddress.ip_address("::ffff:10.0.0.5")) is False
    assert is_public_address(ipaddress.ip_address("::ffff:8.8.8.8")) is False
    # An ordinary global v6 address is still fine.
    assert is_public_address(ipaddress.ip_address("2606:4700::1111")) is True


# ── shell argument policy ────────────────────────────────────────────────

ALLOWLIST = ["ls", "pwd", "date", "whoami", "cat", "head", "tail", "grep", "wc"]


@pytest.fixture()
def roots(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    return [tmp_path.resolve()]


def _check(command, args, roots):
    check_command(command, args, allowlist=ALLOWLIST, roots=roots)


def test_a_path_inside_a_root_is_allowed(roots) -> None:
    _check("cat", [str(roots[0] / "notes.txt")], roots)


@pytest.mark.parametrize("path", ["/etc/passwd", "~/.ssh/id_rsa", "../../../etc/shadow"])
def test_a_path_outside_every_root_is_refused(roots, path: str) -> None:
    with pytest.raises(CommandNotAllowed, match="outside"):
        _check("cat", [path], roots)


def test_a_symlink_out_of_a_root_is_refused(roots, tmp_path) -> None:
    """resolve() follows the link, so the target is what is checked."""
    escape = tmp_path / "escape"
    escape.symlink_to("/etc/passwd")
    with pytest.raises(CommandNotAllowed, match="outside"):
        _check("cat", [str(escape)], roots)


def test_an_unlisted_command_is_refused(roots) -> None:
    with pytest.raises(CommandNotAllowed, match="not allowlisted"):
        _check("curl", ["https://example.com"], roots)


def test_flags_that_read_arbitrary_files_are_refused(roots) -> None:
    """grep -f reads its patterns from a file the path check never sees."""
    with pytest.raises(CommandNotAllowed, match="-f"):
        _check("grep", ["-f", "/etc/passwd", "x"], roots)


def test_bundled_short_flags_are_expanded(roots) -> None:
    _check("grep", ["-rn", "hello", str(roots[0])], roots)
    with pytest.raises(CommandNotAllowed, match="-f"):
        _check("grep", ["-rnf", "/etc/passwd", "x"], roots)


def test_a_flag_value_is_not_treated_as_a_path(roots) -> None:
    _check("grep", ["-A", "3", "hello", str(roots[0] / "notes.txt")], roots)
    _check("grep", ["-A3", "hello", str(roots[0] / "notes.txt")], roots)
    _check("head", ["-n", "20", str(roots[0] / "notes.txt")], roots)


def test_numeric_shorthand_still_works(roots) -> None:
    _check("head", ["-20", str(roots[0] / "notes.txt")], roots)


def test_greps_first_positional_is_a_pattern_not_a_path(roots) -> None:
    _check("grep", ["/etc/passwd", str(roots[0] / "notes.txt")], roots)


def test_commands_without_paths_need_no_roots() -> None:
    _check("whoami", [], [])
    _check("pwd", [], [])


def test_roots_default_to_the_keylane_checkout() -> None:
    from daemon.paths import ROOT

    assert read_roots(None) == [ROOT.resolve()]
    assert read_roots([]) == [ROOT.resolve()]
