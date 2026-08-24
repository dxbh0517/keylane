"""Import the orb's pure helpers without pulling in GTK.

``launcher.result_orb`` needs a display to import. These two functions are
plain data transforms, so they are loaded from source instead — the tests
exercise the real code, not a copy.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "launcher" / "result_orb.py"
_WANTED = {"_canvas_text", "_panel_width_for"}

_module = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_namespace: dict[str, object] = {
    "Any": object,
    "PANEL_WIDTH": 440,
    "PANEL_WIDTH_COMPACT": 300,
    "COMPACT_CHARS": 90,
}
for node in _module.body:
    if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
        exec(compile(ast.Module([node], []), str(_SOURCE), "exec"), _namespace)

canvas_text = _namespace["_canvas_text"]
panel_width = _namespace["_panel_width_for"]


def loader_states() -> dict[str, tuple[float, float, float]]:
    """The loader's state palette, read without importing GTK."""
    source = (Path(__file__).resolve().parent.parent / "launcher" / "loader.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "STATE_COLOURS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("STATE_COLOURS not found in launcher/loader.py")
