from pathlib import Path

from npu.weights import is_model_complete, missing_weights


def test_missing_weights_detects_partial_download(tmp_path: Path):
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "openvino_model.xml").write_text(
        '<net><layer><data value="openvino_model.bin"/></layer></net>',
        encoding="utf-8",
    )
    assert not is_model_complete(model)
    missing = missing_weights(model)
    assert "openvino_model.bin" in missing


def test_complete_when_bins_present(tmp_path: Path):
    model = tmp_path / "qwen"
    model.mkdir()
    (model / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (model / "openvino_model.bin").write_bytes(b"\0" * 16384)
    assert is_model_complete(model)
