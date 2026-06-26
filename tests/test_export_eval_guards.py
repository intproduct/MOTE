from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import fitmotn.cli.eval_auto as eval_auto
import fitmotn.cli.eval_hf as eval_hf
from fitmotn.export.format import EXPORT_CONFIG_FILENAME, EXPORT_MANIFEST_FILENAME


def _cfg():
    return SimpleNamespace(model=SimpleNamespace(trust_remote_code=False, torch_dtype="float32"))


def _write_exported(path, *, stage: str) -> None:
    path.mkdir()
    (path / EXPORT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "format_name": "fitmotn_hf_export",
                "format_version": 1,
                "export_stage": stage,
                "hf_roundtrip_ready": stage == "hf_roundtrip",
                "vllm_ready": False,
            }
        ),
        encoding="utf-8",
    )
    (path / EXPORT_CONFIG_FILENAME).write_text(
        json.dumps({"base_model_name_or_path": "fake-base"}),
        encoding="utf-8",
    )


def test_eval_hf_forces_trust_remote_code_for_hf_roundtrip_export(tmp_path, monkeypatch):
    exported = tmp_path / "exported"
    _write_exported(exported, stage="hf_roundtrip")
    calls = {}

    def fake_loader(*args, **kwargs):
        calls["kwargs"] = kwargs
        return object(), object(), "float32"

    monkeypatch.setattr(eval_hf, "load_causal_lm_and_tokenizer", fake_loader)

    eval_hf._load_model_and_tokenizer(exported, _cfg(), "cpu")

    assert calls["kwargs"]["trust_remote_code"] is True


def test_eval_auto_forces_trust_remote_code_for_hf_roundtrip_export(tmp_path, monkeypatch):
    exported = tmp_path / "exported"
    _write_exported(exported, stage="hf_roundtrip")
    calls = {}

    def fake_loader(*args, **kwargs):
        calls["kwargs"] = kwargs
        return object(), object(), "float32"

    monkeypatch.setattr(eval_auto, "load_causal_lm_and_tokenizer", fake_loader)

    eval_auto._load_model_and_tokenizer(exported, _cfg(), "cpu")

    assert calls["kwargs"]["trust_remote_code"] is True


def test_eval_guards_reject_metadata_only_export(tmp_path):
    exported = tmp_path / "metadata_only"
    _write_exported(exported, stage="metadata_only")

    with pytest.raises(ValueError, match="Metadata-only FitMoTN exports are not loadable"):
        eval_hf._load_model_and_tokenizer(exported, _cfg(), "cpu")
    with pytest.raises(ValueError, match="Metadata-only FitMoTN exports are not loadable"):
        eval_auto._load_model_and_tokenizer(exported, _cfg(), "cpu")
