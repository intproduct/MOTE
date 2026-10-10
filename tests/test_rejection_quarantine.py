from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fitmotn.data.contracts import content_hash
from fitmotn.data.release import tokenizer_fingerprint
from fitmotn.data.rejection_quarantine import (
    FORMAT,
    REJECTIONS_FILENAME,
    RejectionQuarantine,
    sha256_file,
    task_policy_fingerprint,
    task_policy_payload,
    validate_rejection_quarantine,
)


def _task():
    return SimpleNamespace(
        name="reasoning",
        kind="jsonl",
        path="/frozen/source.jsonl",
        split="train",
        hf_name=None,
        hf_config=None,
        dataset_name="unit",
        max_samples=None,
        length_policy={"max_chars": 8},
        supervision_mode="full_trace",
        trace_selection_policy="short",
        supports_skip=True,
    )


def _tokenizer():
    return SimpleNamespace(
        name_or_path="/frozen/tokenizer",
        vocab_size=32,
        special_tokens_map={"eos_token": "</s>", "pad_token": "</s>"},
        eos_token_id=1,
        bos_token_id=0,
        pad_token_id=1,
        chat_template=None,
    )


def _build(tmp_path):
    task = _task()
    raw = {"question": "q", "solution": "too long"}
    record = {
        "task": task.name,
        "reason": "overlong",
        "trace_strategy": "short",
        "raw_content_hash": content_hash(raw),
        "first_source_row_index": 3,
    }
    records = tmp_path / REJECTIONS_FILENAME
    records.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format": FORMAT,
        "task_policy": task_policy_payload([task]),
        "task_policy_fingerprint": task_policy_fingerprint([task]),
        "max_length": 1280,
        "tokenizer_fingerprint": tokenizer_fingerprint(_tokenizer()),
        "sources": {
            task.name: {
                "rows": 4,
                "dataset_fingerprint": "source-fp",
            }
        },
        "files": {
            REJECTIONS_FILENAME: {
                "sha256": sha256_file(records),
                "rows": 1,
            }
        },
    }
    manifest["manifest_fingerprint"] = content_hash(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return task, raw


def test_rejection_quarantine_allows_only_exact_frozen_record(tmp_path):
    task, raw = _build(tmp_path)
    quarantine = RejectionQuarantine(tmp_path, [task])
    assert quarantine.allows(task.name, "overlong", raw)
    assert not quarantine.allows(task.name, "missing_fields", raw)
    assert not quarantine.allows(task.name, "overlong", {**raw, "solution": "changed"})


def test_rejection_quarantine_rejects_policy_drift(tmp_path):
    task, _ = _build(tmp_path)
    task.length_policy = {"max_chars": 9}
    with pytest.raises(RuntimeError, match="task-policy fingerprint mismatch"):
        RejectionQuarantine(tmp_path, [task])


def test_rejection_quarantine_detects_file_tamper(tmp_path):
    _build(tmp_path)
    with (tmp_path / REJECTIONS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    report = validate_rejection_quarantine(tmp_path)
    assert not report["ok"]
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_rejection_quarantine_rejects_max_length_drift(tmp_path):
    task, _ = _build(tmp_path)
    with pytest.raises(RuntimeError, match="max-length mismatch"):
        RejectionQuarantine(tmp_path, [task], max_length=1024)


def test_rejection_quarantine_rejects_tokenizer_drift(tmp_path):
    task, _ = _build(tmp_path)
    changed = _tokenizer()
    changed.vocab_size = 33
    with pytest.raises(RuntimeError, match="tokenizer fingerprint mismatch"):
        RejectionQuarantine(tmp_path, [task], tokenizer=changed, max_length=1280)


def test_rejection_quarantine_accepts_bound_tokenizer_and_length(tmp_path):
    task, raw = _build(tmp_path)
    quarantine = RejectionQuarantine(tmp_path, [task], tokenizer=_tokenizer(), max_length=1280)
    assert quarantine.allows(task.name, "overlong", raw)
