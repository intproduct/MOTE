from __future__ import annotations

import json

import pytest

from MOTE.data.clean_release import (
    build_clean_release_from_registry,
    iter_clean_records,
    validate_clean_release,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _source(name, path, plane, adapter, fields):
    return {
        "source_name": name,
        "kind": "jsonl",
        "path": str(path),
        "split": "train",
        "revision": f"{name}-commit-123",
        "data_plane": plane,
        "adapter": adapter,
        "fields": fields,
        "license": "apache-2.0",
        "allowed_uses": ["research", "training"],
        "acquired_at": "2026-07-29T00:00:00Z",
    }


def test_clean_release_supports_five_data_planes_without_tokenizer_binding(tmp_path):
    rows = {
        "pretraining": ([{"text": "A factual document."}], "text", {"text": "text"}),
        "general_sft": ([{"prompt": "Say hi", "response": "Hello"}], "prompt_response", {"prompt": "prompt", "response": "response"}),
        "reasoning_sft": ([{"question": "1+1?", "solution": "Compute.\n\n#### 2"}], "prompt_response", {"prompt": "question", "response": "solution"}),
        "code_pretraining": ([{"content": "def add(a, b):\n    return a + b"}], "text", {"text": "content"}),
        "code_sft": ([{"instruction": "Fix it", "answer": "Use a bounds check."}], "prompt_response", {"prompt": "instruction", "response": "answer"}),
    }
    sources = []
    for plane, (plane_rows, adapter, fields) in rows.items():
        path = tmp_path / f"{plane}.jsonl"
        _write_jsonl(path, plane_rows)
        sources.append(_source(plane, path, plane, adapter, fields))
    output = tmp_path / "clean"
    manifest = build_clean_release_from_registry(
        {"format": "fitmotn_data_registry_v1", "pipeline_version": "clean-v1", "sources": sources}, output
    )
    assert manifest["accepted_count"] == 5
    assert manifest["data_plane_counts"] == {plane: 1 for plane in sorted(rows)}
    assert "tokenizer_fingerprint" not in manifest
    records = list(iter_clean_records(output))
    assert {record["data_plane"] for record in records} == set(rows)
    assert all(record["license"] == "apache-2.0" for record in records)
    assert all(record["allowed_uses"] == ["research", "training"] for record in records)
    assert validate_clean_release(output)["ok"] is True


@pytest.mark.parametrize("missing", ["revision", "license", "allowed_uses", "acquired_at"])
def test_clean_registry_fails_closed_on_missing_governance_fields(tmp_path, missing):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "hello"}])
    source = _source("demo", path, "pretraining", "text", {"text": "text"})
    source.pop(missing)
    with pytest.raises(ValueError, match=missing):
        build_clean_release_from_registry(
            {"format": "fitmotn_data_registry_v1", "pipeline_version": "clean-v1", "sources": [source]},
            tmp_path / "clean",
        )


def test_clean_registry_rejects_invalid_acquisition_timestamp(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "hello"}])
    source = _source("demo", path, "pretraining", "text", {"text": "text"})
    source["acquired_at"] = "sometime"
    with pytest.raises(ValueError, match="ISO-8601"):
        build_clean_release_from_registry(
            {"format": "fitmotn_data_registry_v1", "pipeline_version": "clean-v1", "sources": [source]},
            tmp_path / "clean",
        )


def test_exact_dedup_is_scoped_by_data_plane(tmp_path):
    first = tmp_path / "pretrain.jsonl"
    second = tmp_path / "code.jsonl"
    _write_jsonl(first, [{"text": "shared source text"}])
    _write_jsonl(second, [{"text": "shared source text"}])
    registry = {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "sources": [
            _source("documents", first, "pretraining", "text", {"text": "text"}),
            _source("code", second, "code_pretraining", "text", {"text": "text"}),
        ],
    }
    manifest = build_clean_release_from_registry(registry, tmp_path / "clean")
    assert manifest["accepted_count"] == 2


def test_clean_release_quarantines_raw_and_normalized_duplicates_across_sources(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"text": "same"}, {"text": "normalized value  \r\n"}])
    _write_jsonl(second, [{"text": "same"}, {"text": "normalized value"}])
    registry = {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "sources": [
            _source("first", first, "pretraining", "text", {"text": "text"}),
            _source("second", second, "pretraining", "text", {"text": "text"}),
        ],
    }
    manifest = build_clean_release_from_registry(registry, tmp_path / "clean")
    assert manifest["accepted_count"] == 2
    assert manifest["reason_counts"] == {"duplicate_normalized_content": 1, "duplicate_raw_content": 1}


def test_clean_release_detects_near_duplicates_and_benchmark_contamination(tmp_path):
    source_path = tmp_path / "source.jsonl"
    benchmark_path = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        source_path,
        [
            {"text": "alpha beta gamma delta epsilon zeta eta theta"},
            {"text": "alpha beta gamma delta epsilon zeta eta iota"},
            {"text": "benchmark private question"},
            {"text": "a clean unrelated document"},
        ],
    )
    _write_jsonl(benchmark_path, [{"question": "benchmark private question"}])
    registry = {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "near_duplicate": {"enabled": True, "threshold": 0.65, "num_perm": 32, "bands": 8, "shingle_size": 2},
        "benchmarks": [
            {
                "name": "private_eval",
                "kind": "jsonl",
                "path": str(benchmark_path),
                "split": "test",
                "revision": "benchmark-commit-1",
                "text_fields": ["question"],
            }
        ],
        "sources": [_source("documents", source_path, "pretraining", "text", {"text": "text"})],
    }
    manifest = build_clean_release_from_registry(registry, tmp_path / "clean")
    assert manifest["accepted_count"] == 2
    assert manifest["reason_counts"]["near_duplicate"] == 1
    assert manifest["reason_counts"]["benchmark_exact_contamination"] == 1
    quarantine = [json.loads(line) for line in (tmp_path / "clean" / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["reason_code"] for row in quarantine} == {"near_duplicate", "benchmark_exact_contamination"}


def test_clean_release_validation_detects_tampering_and_refuses_overwrite(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "valid content"}])
    registry = {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "sources": [_source("demo", path, "pretraining", "text", {"text": "text"})],
    }
    output = tmp_path / "clean"
    build_clean_release_from_registry(registry, output)
    with pytest.raises(FileExistsError):
        build_clean_release_from_registry(registry, output)
    with (output / "accepted.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    assert validate_clean_release(output)["ok"] is False


def test_clean_release_parallel_preparation_preserves_registry_order(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": f"document {index}"} for index in range(12)])
    registry = {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "execution": {"workers": 2, "chunk_size": 3},
        "sources": [_source("demo", path, "pretraining", "text", {"text": "text"})],
    }
    manifest = build_clean_release_from_registry(registry, tmp_path / "clean")
    records = list(iter_clean_records(tmp_path / "clean"))
    assert manifest["execution_policy"] == {"workers": 2, "chunk_size": 3, "ordered_output": True}
    assert [record["source_row_id"] for record in records] == [str(index) for index in range(12)]


def test_sensitive_and_external_quality_signals_are_auditable(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(
        path,
        [
            {"text": "Contact person@example.com for details", "judge_score": 0.75},
            {"text": "-----BEGIN PRIVATE KEY----- secret material", "judge_score": 0.1},
        ],
    )
    source = _source("demo", path, "pretraining", "text", {"text": "text"})
    source["quality_score_fields"] = {"external_judge": "judge_score"}
    registry = {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "quality_policy": {"reject_sensitive_flags": ["private_key"]},
        "sources": [source],
    }
    manifest = build_clean_release_from_registry(registry, tmp_path / "clean")
    assert manifest["reason_counts"] == {"sensitive_content": 1}
    record = next(iter_clean_records(tmp_path / "clean"))
    assert record["sensitive_flags"] == ["email"]
    assert record["quality_scores"]["external_judge"] == 0.75
