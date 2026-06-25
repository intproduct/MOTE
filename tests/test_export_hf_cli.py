from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fitmotn.cli.export_hf import main
from fitmotn.export.format import EXPORT_CONFIG_FILENAME, EXPORT_MANIFEST_FILENAME


def _write_raw_checkpoint(path: Path, *, base_model: str = "fake-base") -> None:
    path.mkdir()
    (path / "fitmotn_state.pt").write_bytes(b"marker")
    (path / "fitmotn_state.json").write_text(
        json.dumps(
            {
                "base_model_path": base_model,
                "tokenizer_path": "fake-tokenizer",
                "checkpoint_format": "patch_state_only_v2",
                "checkpoint_name": "final_model",
                "layers_to_patch": [0, 1],
                "patch_backend": "motn",
                "patch_cfg": {"patch_backend": "motn", "E": 4, "topk": 2},
                "global_step": 3,
                "updates_done": 3,
                "resolved_model_dtype": "torch.bfloat16",
            }
        ),
        encoding="utf-8",
    )


def test_dry_run_writes_no_files(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw_checkpoint(raw)

    assert main(["--checkpoint_dir", str(raw), "--output_dir", str(out), "--dry_run"]) == 0

    assert not out.exists()


def test_metadata_only_export_writes_manifest_config_and_readme(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw_checkpoint(raw)

    assert main(["--checkpoint_dir", str(raw), "--output_dir", str(out)]) == 0

    assert (out / EXPORT_MANIFEST_FILENAME).exists()
    assert (out / EXPORT_CONFIG_FILENAME).exists()
    assert (out / "README.md").exists()
    assert not (out / "config.json").exists()
    manifest = json.loads((out / EXPORT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["format_name"] == "fitmotn_hf_export"
    assert manifest["format_version"] == 1
    assert manifest["export_stage"] == "metadata_only"
    assert manifest["hf_roundtrip_ready"] is False
    assert manifest["vllm_ready"] is False


def test_non_empty_output_requires_overwrite(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw_checkpoint(raw)
    out.mkdir()
    (out / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        main(["--checkpoint_dir", str(raw), "--output_dir", str(out)])

    assert main(["--checkpoint_dir", str(raw), "--output_dir", str(out), "--overwrite"]) == 0
    assert (out / "old.txt").exists()
    assert (out / EXPORT_MANIFEST_FILENAME).exists()


def test_copy_tokenizer_false_does_not_import_transformers(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw_checkpoint(raw)
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert main(["--checkpoint_dir", str(raw), "--output_dir", str(out)]) == 0


def test_rejects_no_metadata_only_and_hf_input(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw_checkpoint(raw)
    with pytest.raises(ValueError, match="metadata_only"):
        main(["--checkpoint_dir", str(raw), "--output_dir", str(out), "--no-metadata_only"])

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="ordinary HF"):
        main(["--checkpoint_dir", str(hf_dir), "--output_dir", str(out)])
