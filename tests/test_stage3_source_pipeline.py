from __future__ import annotations

import json

import pytest

from MOTE.data.release import build_frozen_sft_release, preflight_frozen_sft_release, validate_frozen_record
from MOTE.data.source_pipeline import (
    SourceContext,
    SourceAdapterError,
    adapt_source_row,
    build_release_from_source_manifest,
    deduplicate_canonical_rows,
)


class CharacterTokenizer:
    name_or_path = "fixture-qwen"
    vocab_size = 4096
    eos_token = "<eos>"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 10 for char in str(text)]


def _context(adapter: str, row_id: str = "0") -> SourceContext:
    return SourceContext(
        source_name=adapter,
        source_revision="fixture-sha256:abc123",
        source_split="train",
        source_row_id=row_id,
        adapter=adapter,
        adapter_version="v1",
    )


@pytest.mark.parametrize(
    "adapter,row",
    [
        ("gsm8k_main", {"question": "2+2?", "answer": "Compute 2+2. #### 4"}),
        ("gsm8k_socratic", {"question": "3+3?", "answer": "Add them. #### 6"}),
        ("svamp", {"Body": "Sam has 2.", "Question": "He gets 3 more. Total?", "Equation": "2+3=5", "Answer": "5"}),
        (
            "synthetic_arithmetic",
            {"question": "4+5?", "solution": "4+5=9", "answer": "9", "program_answer": "9", "generation_seed": 7, "template_type": "add", "difficulty": {"digits": 1}},
        ),
        ("hendrycks_math", {"problem": "Evaluate 1+2.", "solution": r"It is \boxed{3}."}),
        ("metamath", {"query": "Variation of q", "original_question": "Base q", "response": "Reasoning. #### 7"}),
        (
            "openr1",
            {"problem": "OpenR1 q", "generations": ["bad #### 1", "verified #### 2"], "correctness_math_verify": [False, True]},
        ),
        ("numinamath", {"problem": "Numina q", "solution": r"Reasoning. \boxed{8}"}),
        (
            "openthoughts",
            {"problem": "Thought q", "conversations": [{"from": "assistant", "value": "Reasoning. #### 9"}], "correct": True},
        ),
    ],
)
def test_supported_source_adapters_emit_structured_canonical(adapter, row):
    canonical = adapt_source_row(adapter, row, _context(adapter), tokenizer=CharacterTokenizer())
    for key in (
        "source_name",
        "source_revision",
        "source_split",
        "source_row_id",
        "source_sample_id",
        "derived_sample_id",
        "problem",
        "reasoning",
        "final_answer",
        "answer_type",
        "problem_group_id",
        "normalized_problem_hash",
        "problem_answer_hash",
        "verification_method",
        "quality_tier",
        "pipeline_version",
        "format_version",
        "prompt",
        "target",
    ):
        assert canonical.get(key) not in (None, "")
    assert canonical["target"].endswith(canonical["final_answer"])


def test_openr1_verified_trace_must_agree_with_explicit_reference_answer():
    row = {
        "problem": "q",
        "answer": "3",
        "generations": ["verified reasoning #### 2"],
        "correctness_math_verify": [True],
    }
    with pytest.raises(SourceAdapterError) as exc:
        adapt_source_row("openr1", row, _context("openr1"), tokenizer=CharacterTokenizer())
    assert getattr(exc.value, "reason_code", None) == "verified_answer_mismatch"


def test_proof_problem_is_typed_as_proof_and_may_use_proof_as_final_target():
    canonical = adapt_source_row(
        "hendrycks_math",
        {"problem": "Prove that the sum of two even integers is even.", "solution": "Let the integers be 2a and 2b; their sum is 2(a+b)."},
        _context("hendrycks_math"),
        tokenizer=CharacterTokenizer(),
    )
    assert canonical["task_type"] == "proof"
    assert canonical["answer_type"] == "proof"
    assert canonical["final_answer"] == canonical["reasoning"]


def test_answer_prefix_is_normalized_before_canonical_publication():
    canonical = adapt_source_row(
        "generic_reasoning",
        {"problem": "q", "reasoning": "work", "final_answer": "The answer is: 4", "quality_tier": "B"},
        _context("generic_reasoning"),
        tokenizer=CharacterTokenizer(),
    )
    assert canonical["final_answer"] == "4"
    assert "The answer is" not in canonical["target"]


def _canonical(row_id: str, problem: str, reasoning: str, answer: str, *, verified: bool, tier: str, source: str):
    return adapt_source_row(
        "generic_reasoning",
        {
            "problem": problem,
            "reasoning": reasoning,
            "final_answer": answer,
            "verification_result": verified,
            "quality_tier": tier,
        },
        SourceContext(source, "fixture-sha256:abc", "train", row_id, "generic_reasoning", "v1"),
        tokenizer=CharacterTokenizer(),
    )


def test_exact_and_problem_group_dedup_are_auditable_and_ranked():
    rows = [
        _canonical("0", "same problem", "same reasoning", "1", verified=False, tier="B", source="weak"),
        _canonical("1", "same problem", "same reasoning", "1", verified=True, tier="A", source="strong"),
        _canonical("2", "same problem", "short verified", "1", verified=True, tier="A", source="strong2"),
        _canonical("3", "same problem", "another verified solution", "1", verified=True, tier="A", source="strong3"),
    ]
    prepared = deduplicate_canonical_rows(rows, tokenizer=CharacterTokenizer(), max_per_problem_group=2)
    winners = [row for row in prepared if not row.get("quarantine_reason_code")]
    rejected = [row["quarantine_reason_code"] for row in prepared if row.get("quarantine_reason_code")]
    assert len(winners) == 2
    assert "duplicate_canonical_content" in rejected
    assert "problem_group_exposure_limit" in rejected
    assert all(row["verification_result"] is True for row in winners)


def test_source_manifest_builds_release_and_extended_quality_report(tmp_path):
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "".join(
            json.dumps({"question": f"q{idx}", "answer": f"work #### {idx}"}) + "\n"
            for idx in range(3)
        ),
        encoding="utf-8",
    )
    source_manifest = {
        "format": "fitmotn_sft_source_manifest_v1",
        "format_version": "math_raw_v2",
        "max_problem_group_solutions": 2,
        "planned_consumed_sequences": 3,
        "pretrain_data_plane": {"the_stack": {"status": "disabled", "reason": "no_code experiment"}},
        "sources": [
            {
                "source_name": "gsm8k_main",
                "adapter": "gsm8k_main",
                "kind": "jsonl",
                "path": str(source_path),
                "split": "train",
                "revision": "fixture-release-v1",
            }
        ],
    }
    output = tmp_path / "release"
    manifest = build_release_from_source_manifest(
        source_manifest,
        output,
        tokenizer=CharacterTokenizer(),
        tokenizer_revision="fixture-tokenizer-rev",
        max_length=1280,
        pipeline_version="sft-v2-stage3",
    )
    assert manifest["accepted_count"] == 3
    assert manifest["format_version"] == "math_raw_v2"
    assert manifest["source_counts"]["gsm8k_main"] == {"scanned": 3, "accepted": 3, "quarantine": 0}
    assert manifest["quality_tier_counts"] == {"A": 3}
    assert manifest["verification_counts"] == {"verified": 3}
    assert manifest["token_statistics"]["total_tokens"]["max"] <= 1280
    assert manifest["source_token_contributions"]["gsm8k_main"]["sequence_fraction"] == 1.0
    assert manifest["pretrain_data_plane"]["the_stack"]["status"] == "disabled"
    assert manifest["exposure_plan"]["estimated_source_epochs"] == 1


def test_source_manifest_quarantines_duplicate_source_row_identity_before_content_dedup(tmp_path):
    source_path = tmp_path / "rows.jsonl"
    source_path.write_text(
        json.dumps({"id": "same", "question": "q1", "answer": "work #### 1"})
        + "\n"
        + json.dumps({"id": "same", "question": "q2", "answer": "work #### 2"})
        + "\n",
        encoding="utf-8",
    )
    manifest = build_release_from_source_manifest(
        {
            "format": "fitmotn_sft_source_manifest_v1",
            "planned_consumed_sequences": 1,
            "pretrain_data_plane": {"the_stack": {"status": "disabled", "reason": "fixture no_code"}},
            "sources": [
                {
                    "source_name": "gsm8k_main",
                    "adapter": "gsm8k_main",
                    "kind": "jsonl",
                    "path": str(source_path),
                    "split": "train",
                    "revision": "fixture-v1",
                    "row_id_field": "id",
                }
            ],
        },
        tmp_path / "release",
        tokenizer=CharacterTokenizer(),
        tokenizer_revision="fixture-tokenizer-v1",
        max_length=1280,
        pipeline_version="stage3",
    )
    assert manifest["accepted_count"] == 1
    assert manifest["reason_counts"]["duplicate_source_identity"] == 1


def test_source_manifest_requires_explicit_the_stack_data_plane_status(tmp_path):
    with pytest.raises(ValueError, match="the_stack.status"):
        build_release_from_source_manifest(
            {
                "format": "fitmotn_sft_source_manifest_v1",
                "planned_consumed_sequences": 1,
                "sources": [{"source_name": "x", "adapter": "gsm8k_main"}],
            },
            tmp_path / "release",
            tokenizer=CharacterTokenizer(),
            tokenizer_revision="fixture-tokenizer-v1",
            max_length=1280,
            pipeline_version="stage3",
        )


def test_release_exact_content_duplicate_is_quarantined_even_with_distinct_source_identity(tmp_path):
    first = _canonical("0", "same", "work", "1", verified=True, tier="A", source="s1")
    second = _canonical("1", "same", "work", "1", verified=True, tier="A", source="s2")
    manifest = build_frozen_sft_release(
        [first, second],
        tmp_path / "release",
        tokenizer=CharacterTokenizer(),
        max_length=1280,
        pipeline_version="stage3",
        format_version="math_raw_v2",
    )
    assert manifest["accepted_count"] == 1
    assert manifest["reason_counts"]["duplicate_canonical_content"] == 1


def test_seq1280_overlong_sample_is_quarantined_and_indexed_for_long_context(tmp_path):
    row = _canonical("0", "q", "long reasoning " * 200, "1", verified=True, tier="A", source="fixture")
    output = tmp_path / "release"
    manifest = build_frozen_sft_release(
        [row],
        output,
        tokenizer=CharacterTokenizer(),
        max_length=1280,
        pipeline_version="stage3",
        format_version="math_raw_v2",
    )
    assert manifest["accepted_count"] == 0
    assert manifest["quarantine_count"] == 1
    assert manifest["long_context_count"] == 1
    assert manifest["reason_counts"]["overlong"] == 1
    long_rows = [json.loads(line) for line in (output / "long_context.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(long_rows) == 1
    assert long_rows[0]["reason_code"] == "overlong"
    assert long_rows[0]["data_diagnostics"]["overflow_tokens"] > 0


def test_frozen_loader_preflight_consumes_1000_records_without_dynamic_processing(tmp_path):
    rows = [
        {
            "source_name": "fixture",
            "source_revision": "fixture-v1",
            "source_split": "train",
            "source_row_id": str(index),
            "prompt": f"Q{index}",
            "target": f"A{index}",
        }
        for index in range(1000)
    ]
    output = tmp_path / "release"
    build_frozen_sft_release(
        rows,
        output,
        tokenizer=CharacterTokenizer(),
        max_length=1280,
        pipeline_version="stage3-preflight",
    )
    report = preflight_frozen_sft_release(output, min_records=1000)
    assert report["ok"] is True
    assert report["consumed_records"] == 1000
    assert report["dynamic_tokenization"] is False
    assert report["quarantine_visible_to_loader"] is False


def test_preflight_refuses_to_claim_1000_record_coverage_for_tiny_release(tmp_path):
    output = tmp_path / "release"
    build_frozen_sft_release(
        [
            {
                "source_name": "fixture",
                "source_revision": "fixture-v1",
                "source_split": "train",
                "source_row_id": "0",
                "prompt": "Q",
                "target": "A",
            }
        ],
        output,
        tokenizer=CharacterTokenizer(),
        max_length=1280,
        pipeline_version="stage3-preflight",
    )
    with pytest.raises(RuntimeError, match="requires at least 1000"):
        preflight_frozen_sft_release(output, min_records=1000)


def test_frozen_record_validator_rejects_target_mask_and_eos_anomalies(tmp_path):
    row = _canonical("0", "q", "work", "1", verified=True, tier="A", source="fixture")
    output = tmp_path / "release"
    manifest = build_frozen_sft_release(
        [row],
        output,
        tokenizer=CharacterTokenizer(),
        max_length=1280,
        pipeline_version="stage3",
        format_version="math_raw_v2",
    )
    accepted = json.loads((output / "accepted.jsonl").read_text(encoding="utf-8"))
    bad_mask = dict(accepted)
    bad_mask["labels"] = list(accepted["labels"])
    bad_mask["labels"][-2] = -100
    with pytest.raises(RuntimeError, match="target mask"):
        validate_frozen_record(bad_mask, manifest)
    bad_eos = dict(accepted)
    bad_eos["input_ids"] = list(accepted["input_ids"])
    bad_eos["input_ids"][-2] = CharacterTokenizer.eos_token_id
    with pytest.raises(RuntimeError, match="EOS invariant"):
        validate_frozen_record(bad_eos, manifest)
