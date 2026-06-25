from __future__ import annotations

import json
import sys

import pytest

from fitmotn.eval.vllm_runner import evaluate_with_vllm, inspect_vllm_compatibility
from fitmotn.export.format import EXPORT_CONFIG_FILENAME, EXPORT_MANIFEST_FILENAME


def test_vllm_raw_checkpoint_guard_mentions_export_hf(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fitmotn_state.pt").write_bytes(b"marker")
    (raw / "fitmotn_state.json").write_text(
        json.dumps({"checkpoint_format": "patch_state_only_v2", "layers_to_patch": [0]}),
        encoding="utf-8",
    )

    report = inspect_vllm_compatibility(str(raw))
    assert report["supports_vllm_eval"] is False
    assert report["reason"] == "raw_fitmotn_checkpoint_requires_export_hf"
    with pytest.raises(NotImplementedError, match="export_hf"):
        evaluate_with_vllm(str(raw), fit_cfg=None)


def test_vllm_metadata_only_export_guard_mentions_stage_4(tmp_path):
    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / EXPORT_MANIFEST_FILENAME).write_text(
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
    (exported / EXPORT_CONFIG_FILENAME).write_text(
        json.dumps({"base_model_name_or_path": "fake-base"}),
        encoding="utf-8",
    )

    report = inspect_vllm_compatibility(str(exported))
    assert report["supports_vllm_eval"] is False
    assert report["reason"] == "metadata_only_export_not_vllm_ready"
    with pytest.raises(NotImplementedError, match="metadata-only.*Stage 4B/4C"):
        evaluate_with_vllm(str(exported), fit_cfg=None)


def test_ordinary_hf_dir_does_not_trigger_fitmotn_guard(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "vllm", None)

    report = inspect_vllm_compatibility(str(hf_dir))
    assert report["supports_vllm_eval"] is True
    assert report["path_kind"] == "hf_model_dir"
    with pytest.raises(ImportError, match="vLLM is required"):
        evaluate_with_vllm(str(hf_dir), fit_cfg=None)
