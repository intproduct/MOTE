from __future__ import annotations

import json

import pytest

from MOTE.cli import build_sft_release as build_cli
from MOTE.cli import preflight_sft_release as preflight_cli


def test_build_cli_source_manifest_uses_production_pipeline(monkeypatch, tmp_path, capsys):
    source_manifest = tmp_path / "sources.json"
    source_manifest.write_text(json.dumps({"format": "fitmotn_sft_source_manifest_v1"}), encoding="utf-8")
    tokenizer = object()
    monkeypatch.setattr(build_cli.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    observed = {}

    def fake_build(manifest, output_dir, **kwargs):
        observed.update({"manifest": manifest, "output_dir": output_dir, **kwargs})
        return {"accepted_count": 1}

    monkeypatch.setattr(build_cli, "build_release_from_source_manifest", fake_build)
    assert (
        build_cli.main(
            [
                "--source_manifest",
                str(source_manifest),
                "--output_dir",
                str(tmp_path / "release"),
                "--tokenizer",
                "qwen-local",
                "--tokenizer_revision",
                "qwen-commit-abc",
                "--max_length",
                "1280",
                "--pipeline_version",
                "stage3",
            ]
        )
        == 0
    )
    assert observed["tokenizer"] is tokenizer
    assert observed["tokenizer_revision"] == "qwen-commit-abc"
    assert observed["max_length"] == 1280
    assert json.loads(capsys.readouterr().out)["accepted_count"] == 1


def test_build_cli_requires_tokenizer_revision_for_source_manifest(monkeypatch, tmp_path):
    source_manifest = tmp_path / "sources.json"
    source_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build_cli.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="tokenizer_revision"):
        build_cli.main(
            [
                "--source_manifest",
                str(source_manifest),
                "--output_dir",
                str(tmp_path / "release"),
                "--tokenizer",
                "qwen-local",
                "--max_length",
                "1280",
                "--pipeline_version",
                "stage3",
            ]
        )


def test_preflight_cli_reports_without_training(monkeypatch, capsys):
    monkeypatch.setattr(
        preflight_cli,
        "preflight_frozen_sft_release",
        lambda release_dir, min_records: {"ok": True, "consumed_records": min_records, "release_dir": release_dir},
    )
    assert preflight_cli.main(["--release_dir", "release", "--min_records", "1000"]) == 0
    assert json.loads(capsys.readouterr().out)["consumed_records"] == 1000
