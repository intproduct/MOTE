from __future__ import annotations

import json
from pathlib import Path

from fitmotn.export import classify_model_path, is_exported_fitmotn_dir, is_raw_fitmotn_checkpoint
from fitmotn.export.format import EXPORT_CONFIG_FILENAME, EXPORT_MANIFEST_FILENAME, FITMOTN_AUTO_MAP
from fitmotn.export.validate import validate_export_layout


def _write_export(path: Path) -> None:
    path.mkdir()
    (path / EXPORT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "format_name": "fitmotn_hf_export",
                "format_version": 1,
                "export_stage": "metadata_only",
                "hf_roundtrip_ready": False,
                "vllm_ready": False,
            }
        ),
        encoding="utf-8",
    )
    (path / EXPORT_CONFIG_FILENAME).write_text(
        json.dumps({"base_model_name_or_path": "fake-base"}),
        encoding="utf-8",
    )
    (path / "README.md").write_text("export", encoding="utf-8")


def test_classify_model_path_cases(tmp_path):
    assert classify_model_path(tmp_path / "missing") == "missing"

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fitmotn_state.pt").write_bytes(b"marker")
    assert is_raw_fitmotn_checkpoint(raw)
    assert classify_model_path(raw) == "raw_fitmotn_checkpoint"

    raw_json = tmp_path / "raw_json"
    raw_json.mkdir()
    (raw_json / "fitmotn_state.json").write_text("{}", encoding="utf-8")
    assert classify_model_path(raw_json) == "raw_fitmotn_checkpoint"

    exported = tmp_path / "exported"
    _write_export(exported)
    assert is_exported_fitmotn_dir(exported)
    assert classify_model_path(exported) == "exported_fitmotn_dir"

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    assert classify_model_path(hf_dir) == "hf_model_dir"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert classify_model_path(empty) == "unknown"


def test_exported_markers_take_precedence_only_when_recognizable(tmp_path):
    both = tmp_path / "both"
    _write_export(both)
    (both / "fitmotn_state.pt").write_bytes(b"marker")
    assert classify_model_path(both) == "exported_fitmotn_dir"

    invalid_export_with_raw = tmp_path / "invalid_export_with_raw"
    invalid_export_with_raw.mkdir()
    (invalid_export_with_raw / EXPORT_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (invalid_export_with_raw / EXPORT_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    (invalid_export_with_raw / "fitmotn_state.json").write_text("{}", encoding="utf-8")
    assert classify_model_path(invalid_export_with_raw) == "raw_fitmotn_checkpoint"


def test_validate_export_layout(tmp_path):
    exported = tmp_path / "exported"
    _write_export(exported)
    result = validate_export_layout(exported)
    assert result.ok
    assert result.errors == []

    missing_manifest = tmp_path / "missing_manifest"
    _write_export(missing_manifest)
    (missing_manifest / EXPORT_MANIFEST_FILENAME).unlink()
    result = validate_export_layout(missing_manifest)
    assert not result.ok
    assert any(err["code"] == "missing_file" for err in result.errors)

    with_config = tmp_path / "with_config"
    _write_export(with_config)
    (with_config / "config.json").write_text("{}", encoding="utf-8")
    result = validate_export_layout(with_config)
    assert result.ok
    assert any(warning["code"] == "placeholder_hf_config" for warning in result.warnings)


def test_validate_hf_roundtrip_layout(tmp_path):
    exported = tmp_path / "hf_roundtrip"
    exported.mkdir()
    (exported / EXPORT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "format_name": "fitmotn_hf_export",
                "format_version": 1,
                "export_stage": "hf_roundtrip",
                "hf_roundtrip_ready": True,
                "vllm_ready": False,
                "exported_code_ready": True,
                "auto_map_ready": True,
            }
        ),
        encoding="utf-8",
    )
    (exported / EXPORT_CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "base_model_name_or_path": "fake-base",
                "hf_roundtrip_ready": True,
                "vllm_ready": False,
                "exported_code_ready": True,
                "auto_map_ready": True,
            }
        ),
        encoding="utf-8",
    )
    (exported / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fitmotn",
                "architectures": ["FitMoTNForCausalLM"],
                "auto_map": FITMOTN_AUTO_MAP,
                "fitmotn_patch_config": {"patch_backend": "motn", "E": 4},
            }
        ),
        encoding="utf-8",
    )
    (exported / "model.safetensors").write_bytes(b"weights")
    (exported / "configuration_fitmotn.py").write_text("# config\n", encoding="utf-8")
    (exported / "modeling_fitmotn.py").write_text("# model\n", encoding="utf-8")
    (exported / "README.md").write_text("export", encoding="utf-8")

    result = validate_export_layout(exported)

    assert result.ok
    assert result.errors == []
