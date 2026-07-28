from __future__ import annotations

import json

import pytest

from MOTE.config.loader import load_config_from_json
from MOTE.data.release import build_frozen_sft_release, iter_frozen_sft_records, validate_frozen_sft_release
from MOTE.tasks.reasoning_specs import build_task_mixture_tasks


class TinyTokenizer:
    name_or_path = "tiny"
    vocab_size = 512
    eos_token = "<eos>"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 10 for char in str(text)]


def _rows():
    return [
        {
            "source_name": "demo",
            "source_revision": "r1",
            "source_split": "train",
            "source_row_id": "0",
            "prompt": "Q",
            "target": "A",
        },
        {
            "source_name": "demo",
            "source_revision": "r1",
            "source_split": "train",
            "source_row_id": "1",
            "prompt": "question-too-long",
            "target": "answer-too-long",
        },
        {"source_name": "demo", "prompt": "missing identity", "target": "A"},
    ]


def test_release_build_is_atomic_validated_and_quarantines_rejected_rows(tmp_path):
    output = tmp_path / "release"
    manifest = build_frozen_sft_release(
        _rows(), output, tokenizer=TinyTokenizer(), max_length=8, pipeline_version="pipeline-v1"
    )
    assert manifest["accepted_count"] == 1
    assert manifest["quarantine_count"] == 2
    assert manifest["reason_counts"] == {"missing_source_identity": 1, "overlong": 1}
    assert validate_frozen_sft_release(output)["ok"] is True
    records = list(iter_frozen_sft_records(output))
    assert len(records) == 1
    assert records[0]["source_row_id"] == "0"
    assert records[0]["data_diagnostics"]["supervised_tokens"] == 2
    quarantine = [json.loads(line) for line in (output / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["reason_code"] for row in quarantine} == {"missing_source_identity", "overlong"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_frozen_sft_release(_rows(), output, tokenizer=TinyTokenizer(), max_length=8, pipeline_version="pipeline-v1")


def test_release_manifest_fingerprint_is_reproducible(tmp_path):
    first = build_frozen_sft_release(
        _rows(), tmp_path / "first", tokenizer=TinyTokenizer(), max_length=8, pipeline_version="pipeline-v1"
    )
    second = build_frozen_sft_release(
        _rows(), tmp_path / "second", tokenizer=TinyTokenizer(), max_length=8, pipeline_version="pipeline-v1"
    )
    assert first["manifest_fingerprint"] == second["manifest_fingerprint"]
    assert first["files"] == second["files"]


def test_release_validation_detects_modified_accepted_data(tmp_path):
    output = tmp_path / "release"
    build_frozen_sft_release(_rows(), output, tokenizer=TinyTokenizer(), max_length=8, pipeline_version="pipeline-v1")
    with (output / "accepted.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    report = validate_frozen_sft_release(output)
    assert report["ok"] is False
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_config_can_select_a_valid_frozen_release_exclusively(tmp_path):
    release = tmp_path / "release"
    build_frozen_sft_release(_rows(), release, tokenizer=TinyTokenizer(), max_length=8, pipeline_version="pipeline-v1")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "seq_len_run": 8,
                    "use_frozen_sft_release": True,
                    "frozen_sft_release_dir": str(release),
                    "frozen_sft_exclusive": True,
                    "source_sampling_mode": "deterministic_strict",
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config_from_json(config_path)
    tasks = build_task_mixture_tasks(cfg.data)
    assert [task.name for task in tasks] == ["frozen_sft_release"]
    assert cfg.data.fail_on_dynamic_skip is True


def test_config_rejects_unknown_sampling_mode(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data": {"source_sampling_mode": "random_magic"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source_sampling_mode"):
        load_config_from_json(path)


def test_release_quarantines_duplicate_source_identity_and_reports_sources(tmp_path):
    rows = [_rows()[0], {**_rows()[0], "target": "changed"}]
    output = tmp_path / "release"
    manifest = build_frozen_sft_release(
        rows, output, tokenizer=TinyTokenizer(), max_length=64, pipeline_version="pipeline-v1"
    )
    assert manifest["accepted_count"] == 1
    assert manifest["reason_counts"] == {"duplicate_source_identity": 1}
    assert manifest["source_counts"] == {"demo": {"accepted": 1, "quarantine": 1}}


def test_release_aborts_on_unexpected_tokenizer_failure(tmp_path):
    class BrokenTokenizer(TinyTokenizer):
        def encode(self, text, add_special_tokens=False):
            raise AssertionError("implementation failure")

    output = tmp_path / "release"
    with pytest.raises(RuntimeError, match="refusing to quarantine an unknown error"):
        build_frozen_sft_release(
            [_rows()[0]], output, tokenizer=BrokenTokenizer(), max_length=64, pipeline_version="pipeline-v1"
        )
    assert not output.exists()
