"""Validate OpenVINO IR weight files on disk."""

from __future__ import annotations

import re
from pathlib import Path

_MIN_BIN_BYTES = 8192
_WEIGHT_ATTR = re.compile(r'<data[^>]+value="([^"]+\.bin)"', re.IGNORECASE)


def weight_files_for_xml(xml_path: Path) -> list[Path]:
    """Return .bin files this IR expects (companion name + embedded refs)."""
    expected: list[Path] = []
    companion = xml_path.with_suffix(".bin")
    expected.append(companion)
    try:
        text = xml_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return expected
    for match in _WEIGHT_ATTR.finditer(text):
        name = match.group(1).strip()
        if name:
            expected.append(xml_path.parent / name)
    # de-dupe preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for path in expected:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def missing_weights(model_dir: Path) -> list[str]:
    """Names of missing or empty weight files required by IR xml in *model_dir*."""
    if not model_dir.is_dir():
        return ["model directory missing"]

    xml_files = sorted(model_dir.glob("openvino*.xml"))
    if not xml_files:
        return ["no openvino*.xml files"]

    missing: list[str] = []
    checked: set[Path] = set()
    for xml_path in xml_files:
        for bin_path in weight_files_for_xml(xml_path):
            if bin_path in checked:
                continue
            checked.add(bin_path)
            if not bin_path.is_file():
                missing.append(bin_path.name)
            elif bin_path.stat().st_size < _MIN_BIN_BYTES:
                missing.append(f"{bin_path.name} (empty or truncated)")

    return missing


def is_model_complete(model_dir: Path) -> bool:
    return not missing_weights(model_dir)


def purge_incomplete(model_dir: Path) -> None:
    """Remove partial IR trees so snapshot_download can fetch cleanly."""
    if not model_dir.is_dir():
        return
    for pattern in ("openvino*.xml", "openvino*.bin", "*.safetensors"):
        for path in model_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
