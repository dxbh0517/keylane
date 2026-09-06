"""The control port that carries the daemon's drawing requests to the UI.

The protocol grew from a 64-byte `recv` that only ever had to carry the word
"toggle". A walkthrough step is orders of magnitude bigger than that, and a
truncated one is invalid JSON rather than a short command, so the framing is
worth testing on its own.
"""

from __future__ import annotations

import json
import os
import socket
import threading

# See tests/test_spotlight.py: importing ui.main re-execs without this.
os.environ.setdefault("KEYLANE_LAYER_SHELL_PRIMED", "1")

from ui.main import CONTROL_MAX_BYTES, _read_command  # noqa: E402


def _send_and_read(payload: bytes) -> str:
    """Write *payload* down a real socket and read one command back off it.

    The write goes on its own thread. A payload larger than the socket buffer
    blocks in `sendall` until someone drains it, and doing both in one thread
    deadlocks — which is exactly the case the size cap exists for.
    """
    client, server = socket.socketpair()

    def _write() -> None:
        try:
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
        except OSError:
            # The reader hit its cap and stopped; that is the point of the test.
            pass

    writer = threading.Thread(target=_write, daemon=True)
    with client, server:
        writer.start()
        line = _read_command(server)
    writer.join(timeout=5)
    return line


def test_a_bare_verb_is_read() -> None:
    assert _send_and_read(b"toggle\n") == "toggle"


def test_a_verb_with_a_payload_survives_intact() -> None:
    body = json.dumps({"shapes": [{"kind": "ring", "target_text": "Export"}]})
    line = _send_and_read(f"annotate {body}\n".encode())
    verb, _, rest = line.partition(" ")
    assert verb == "annotate"
    assert json.loads(rest)["shapes"][0]["target_text"] == "Export"


def test_a_payload_larger_than_one_read_is_reassembled() -> None:
    """The old 64-byte read truncated anything real into invalid JSON."""
    shapes = [{"kind": "ring", "x": i, "y": i, "text": f"step {i}"} for i in range(200)]
    body = json.dumps({"shapes": shapes})
    assert len(body) > 8192
    line = _send_and_read(f"annotate {body}\n".encode())
    _, _, rest = line.partition(" ")
    assert len(json.loads(rest)["shapes"]) == 200


def test_only_the_first_line_is_taken() -> None:
    assert _send_and_read(b"toggle\nmic\n") == "toggle"


def test_a_command_with_no_newline_still_arrives() -> None:
    """A client that closes instead of terminating the line is still understood."""
    assert _send_and_read(b"mic") == "mic"


def test_an_unterminated_flood_stops_at_the_cap() -> None:
    """A client that never sends a newline must not be able to exhaust memory."""
    line = _send_and_read(b"x" * (CONTROL_MAX_BYTES * 2))
    assert len(line) <= CONTROL_MAX_BYTES


def test_an_empty_request_reads_as_nothing() -> None:
    # The caller turns this into the default command rather than guessing here.
    assert _send_and_read(b"") == ""
