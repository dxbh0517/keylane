"""Argument-aware admission for the `shell` tool.

An allowlist of command *names* that includes ``cat``, ``head``, ``tail`` and
``grep`` is an allowlist of the entire filesystem: the names are harmless and
the arguments are the whole risk. So each allowlisted command declares which
flags it accepts and which of its positional arguments are paths, and every
path argument must resolve inside a configured root.

``shell_read_roots`` defaults to the Keylane checkout, which keeps "read the
log", "grep the config" working while leaving ``~/.ssh`` out of reach. Widening
it is a deliberate edit in Settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from daemon.paths import ROOT


class CommandNotAllowed(ValueError):
    """A command that may not run. The message is shown to the model."""


@dataclass(frozen=True)
class CommandSpec:
    """What one allowlisted command may be handed."""

    # Flags accepted verbatim, e.g. "-n". Anything else starting with "-" is refused.
    flags: frozenset[str] = field(default_factory=frozenset)
    # Flags that consume the following argument as a value rather than a path.
    valued: frozenset[str] = field(default_factory=frozenset)
    # How many leading positionals are not paths (grep's pattern is not a file).
    skip_positionals: int = 0
    # Whether remaining positionals are paths at all.
    takes_paths: bool = True


_NUMERIC_SHORTHAND = frozenset({"head", "tail"})

SPECS: dict[str, CommandSpec] = {
    "pwd": CommandSpec(takes_paths=False),
    "whoami": CommandSpec(takes_paths=False),
    "date": CommandSpec(flags=frozenset({"-u", "-R", "-I"}), takes_paths=False),
    "ls": CommandSpec(
        flags=frozenset({"-l", "-a", "-A", "-h", "-t", "-r", "-S", "-1", "-la", "-lh", "-lah"})
    ),
    "cat": CommandSpec(flags=frozenset({"-n", "-b", "-s", "-E"})),
    "head": CommandSpec(flags=frozenset({"-n", "-c", "-q"}), valued=frozenset({"-n", "-c"})),
    "tail": CommandSpec(flags=frozenset({"-n", "-c", "-q"}), valued=frozenset({"-n", "-c"})),
    "grep": CommandSpec(
        flags=frozenset(
            {"-i", "-n", "-r", "-R", "-l", "-L", "-c", "-v", "-w", "-x",
             "-E", "-F", "-H", "-h", "-o", "-s", "-A", "-B", "-C", "-m"}
        ),
        valued=frozenset({"-A", "-B", "-C", "-m"}),
        skip_positionals=1,
    ),
    "wc": CommandSpec(flags=frozenset({"-l", "-w", "-c", "-m"})),
}


def read_roots(configured: list[str] | None) -> list[Path]:
    """Resolve the configured roots, falling back to the Keylane checkout."""
    raw = configured if configured else [str(ROOT)]
    roots: list[Path] = []
    for entry in raw:
        try:
            roots.append(Path(entry).expanduser().resolve())
        except OSError:
            continue
    return roots


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _check_path(raw: str, roots: list[Path]) -> None:
    if not roots:
        raise CommandNotAllowed(
            "file arguments are not permitted: no directories are listed in "
            "security.shell_read_roots"
        )
    # resolve() follows symlinks, so a link pointing out of a root is caught.
    try:
        resolved = Path(raw).expanduser().resolve()
    except OSError as exc:
        raise CommandNotAllowed(f"could not resolve path {raw!r}: {exc}") from exc
    if not _within(resolved, roots):
        allowed = ", ".join(str(r) for r in roots)
        raise CommandNotAllowed(
            f"{raw!r} is outside the permitted directories ({allowed})"
        )


def _expand_bundle(command: str, arg: str, spec: CommandSpec) -> bool:
    """Validate a short-flag bundle. Returns True if it still needs a value.

    ``-rn`` is two flags; ``-A3`` is one flag carrying its value; ``-A`` alone
    takes the next argument.
    """
    letters = arg[1:]
    for position, letter in enumerate(letters):
        flag = f"-{letter}"
        if flag not in spec.flags:
            raise CommandNotAllowed(
                f"flag {flag!r} (in {arg!r}) is not permitted for {command}"
            )
        if flag in spec.valued:
            # Anything after it in the bundle is the value, not more flags.
            return position == len(letters) - 1
    return False


def check_command(command: str, args: list[str], *, allowlist: list[str], roots: list[Path]) -> None:
    """Admit one command invocation, or raise :class:`CommandNotAllowed`.

    The message names the reason so the model can correct the call rather than
    retry it unchanged.
    """
    if command not in allowlist:
        raise CommandNotAllowed(f"command not allowlisted: {command}")
    spec = SPECS.get(command)
    if spec is None:
        raise CommandNotAllowed(
            f"{command} is allowlisted but has no argument policy; "
            "add one in daemon/shellpolicy.py before using it"
        )

    positionals = 0
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--":
            continue
        if arg.startswith("-") and arg != "-":
            # `head -20` and `tail -5` are the classic shorthand for -n.
            if command in _NUMERIC_SHORTHAND and arg[1:].isdigit():
                continue
            if arg.startswith("--"):
                if arg not in spec.flags:
                    raise CommandNotAllowed(f"flag {arg!r} is not permitted for {command}")
                continue
            # `-rn` means `-r -n`, and `-A3` carries its value in the bundle.
            if _expand_bundle(command, arg, spec):
                if index >= len(args):
                    raise CommandNotAllowed(f"flag {arg!r} needs a value")
                index += 1  # consume the value; it is not a path
            continue

        positionals += 1
        if positionals <= spec.skip_positionals:
            continue
        if not spec.takes_paths:
            raise CommandNotAllowed(f"{command} does not take file arguments")
        _check_path(arg, roots)
