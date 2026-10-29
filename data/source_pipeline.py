from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tasks.answer_extraction import (
    AnswerValidationError,
    clean_text,
    extract_final_answer,
    normalize_math_answer,
    split_reasoning_and_final_answer,
    validate_final_answer,
)
from ..tasks.reasoning_normalization import ReasoningNormalizationError, select_openr1_trace
from ..tasks.synthetic_arithmetic import build_synthetic_arithmetic_dataset
from .contracts import CanonicalSample, content_hash, normalized_problem_hash
from .release import build_frozen_sft_release
from .sft_format import FORMAT_VERSION, parse_math_raw_v2_target, render_math_raw_v2


SOURCE_MANIFEST_FORMAT = "fitmotn_sft_source_manifest_v1"
ADAPTER_VERSION = "stage3-v1"
VAGUE_REVISIONS = {"", "latest", "main", "master", "head", "unknown", "default"}
QUALITY_ORDER = {"Q": 0, "C": 1, "B": 2, "A": 3}
SOURCE_AUTHORITY = {
    "synthetic_arithmetic": 100,
    "gsm8k_main": 95,
    "gsm8k_socratic": 94,
    "openr1": 90,
    "hendrycks_math": 85,
    "svamp": 80,
    "metamath": 65,
    "numinamath": 60,
    "openthoughts": 55,
    "generic_reasoning": 50,
}


class SourceAdapterError(ValueError):
    def __init__(self, reason_code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


@dataclass(frozen=True)
class SourceContext:
    source_name: str
    source_revision: str
    source_split: str
    source_row_id: str
    adapter: str
    adapter_version: str = ADAPTER_VERSION
    pipeline_version: str = "sft-v2-stage3"
    format_version: str = FORMAT_VERSION


def _revision_is_vague(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in VAGUE_REVISIONS or "replace" in normalized or "todo" in normalized


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _conversation_text(value: Any, roles: set[str]) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or item.get("from") or "").strip().lower()
        if role in roles:
            text = clean_text(item.get("content") or item.get("value") or item.get("text"))
            if text:
                parts.append(text)
    return clean_text(parts[-1] if parts else "")


def _task_type(row: Mapping[str, Any]) -> str:
    value = clean_text(_first(row, "task_type", "type", "problem_type")).lower()
    return "proof" if "proof" in value else "numeric"


def _split_solution(solution: str, explicit_answer: Any = None) -> tuple[str, str]:
    reasoning, parsed = split_reasoning_and_final_answer(solution)
    answer = clean_text(explicit_answer) or parsed or extract_final_answer(solution)
    return clean_text(reasoning), clean_text(answer)


def _adapter_fields(adapter: str, row: Mapping[str, Any], tokenizer) -> dict[str, Any]:
    adapter = str(adapter).strip().lower()
    metadata: dict[str, Any] = {}
    task_type = _task_type(row)
    if adapter in {"gsm8k_main", "gsm8k_socratic"}:
        problem = clean_text(row.get("question"))
        solution_raw = clean_text(row.get("answer"))
        reasoning, answer = _split_solution(solution_raw)
        verification_method, verification_result, quality_tier = "authoritative_reference", True, "A"
    elif adapter == "svamp":
        problem = clean_text([row.get("Body"), row.get("Question")])
        reasoning = clean_text(_first(row, "Equation", "equation", "solution"))
        answer = clean_text(_first(row, "Answer", "answer"))
        verification_method, verification_result, quality_tier = "authoritative_reference", True, "A"
    elif adapter == "synthetic_arithmetic":
        problem = clean_text(row.get("question"))
        reasoning = clean_text(row.get("solution"))
        answer = clean_text(row.get("answer"))
        program_answer = clean_text(row.get("program_answer"))
        if not program_answer:
            raise SourceAdapterError("missing_program_verification", "synthetic sample has no program_answer")
        if answer != program_answer:
            raise SourceAdapterError(
                "synthetic_answer_mismatch",
                "synthetic answer does not match program recomputation",
                details={"answer": answer, "program_answer": program_answer},
            )
        metadata.update(
            {
                "generation_seed": row.get("generation_seed"),
                "template_type": row.get("template_type"),
                "difficulty": row.get("difficulty"),
            }
        )
        verification_method, verification_result, quality_tier = "program_recompute", True, "A"
    elif adapter == "hendrycks_math":
        problem = clean_text(row.get("problem"))
        solution_raw = clean_text(row.get("solution"))
        reasoning, answer = _split_solution(solution_raw, _first(row, "final_answer", "answer"))
        verification_method, verification_result, quality_tier = "authoritative_solution_parse", True, "A"
    elif adapter == "metamath":
        problem = clean_text(_first(row, "original_question", "query", "question"))
        solution_raw = clean_text(_first(row, "response", "solution"))
        reasoning, answer = _split_solution(solution_raw, _first(row, "final_answer", "answer"))
        metadata["query"] = clean_text(row.get("query"))
        metadata["original_question"] = clean_text(row.get("original_question"))
        verification_method, verification_result, quality_tier = "upstream_reference_parse", None, "B"
    elif adapter == "openr1":
        try:
            selected = select_openr1_trace(row, tokenizer=tokenizer)
        except ReasoningNormalizationError as exc:
            raise SourceAdapterError(exc.reason_code, str(exc), details=exc.details) from exc
        problem = clean_text(_first(row, "problem", "question", "prompt"))
        reasoning, parsed_answer = _split_solution(selected["solution"])
        explicit_answer = clean_text(_first(row, "final_answer", "answer", "ground_truth", "expected_answer"))
        if explicit_answer and parsed_answer and normalize_math_answer(explicit_answer) != normalize_math_answer(parsed_answer):
            raise SourceAdapterError(
                "verified_answer_mismatch",
                "selected verified OpenR1 trace disagrees with the explicit reference answer",
                details={"explicit_answer": explicit_answer, "selected_trace_answer": parsed_answer},
            )
        answer = explicit_answer or parsed_answer
        metadata.update({key: value for key, value in selected.items() if key != "solution"})
        verification_method, verification_result, quality_tier = "correctness_math_verify", True, "A"
    elif adapter == "numinamath":
        problem = clean_text(_first(row, "problem", "question")) or _conversation_text(row.get("messages"), {"user", "human"})
        solution_raw = clean_text(_first(row, "solution", "reasoning")) or _conversation_text(
            row.get("messages"), {"assistant", "gpt", "model"}
        )
        reasoning, answer = _split_solution(solution_raw, _first(row, "final_answer", "answer"))
        verification_method, verification_result, quality_tier = "upstream_reference_parse", None, "B"
    elif adapter == "openthoughts":
        correct = row.get("correct")
        if correct is False or (isinstance(correct, str) and correct.strip().lower() == "false"):
            raise SourceAdapterError("incorrect_trace", "OpenThoughts row is marked incorrect")
        problem = clean_text(_first(row, "problem", "question")) or _conversation_text(
            row.get("conversations"), {"user", "human"}
        )
        problem = problem or _conversation_text(row.get("messages"), {"user", "human"})
        solution_raw = clean_text(_first(row, "solution", "reasoning")) or _conversation_text(
            row.get("conversations"), {"assistant", "gpt", "model"}
        )
        solution_raw = solution_raw or _conversation_text(row.get("messages"), {"assistant", "gpt", "model"})
        reasoning, answer = _split_solution(solution_raw, _first(row, "final_answer", "answer"))
        verification_result = True if correct is True or str(correct).strip().lower() == "true" else None
        verification_method = "upstream_correct_flag" if verification_result else "parser_only"
        quality_tier = "B" if verification_result else "C"
    elif adapter == "generic_reasoning":
        problem = clean_text(_first(row, "problem", "question", "prompt"))
        reasoning = clean_text(_first(row, "reasoning", "solution", "response"))
        answer = clean_text(_first(row, "final_answer", "answer")) or extract_final_answer(reasoning)
        verification_result = row.get("verification_result")
        if verification_result is not None and not isinstance(verification_result, bool):
            raise SourceAdapterError("invalid_verification_value", "verification_result must be boolean or null")
        verification_method = clean_text(row.get("verification_method")) or "provided_by_source_manifest"
        quality_tier = clean_text(row.get("quality_tier")).upper() or ("A" if verification_result else "B")
    else:
        raise SourceAdapterError("unsupported_source_adapter", f"unsupported adapter={adapter!r}")
    return {
        "problem": problem,
        "reasoning": reasoning,
        "final_answer": answer,
        "task_type": task_type,
        "verification_method": verification_method,
        "verification_result": verification_result,
        "quality_tier": quality_tier,
        "metadata": metadata,
    }


def adapt_source_row(adapter: str, row: Mapping[str, Any], context: SourceContext, *, tokenizer) -> dict[str, Any]:
    if _revision_is_vague(context.source_revision):
        raise SourceAdapterError("invalid_source_revision", f"source revision is vague: {context.source_revision!r}")
    fields = _adapter_fields(adapter, row, tokenizer)
    problem = clean_text(fields["problem"])
    reasoning = clean_text(fields["reasoning"])
    if fields["task_type"] != "proof" and re.search(
        r"\b(?:prove|show\s+that|demonstrate)\b", problem, flags=re.IGNORECASE
    ):
        fields["task_type"] = "proof"
        if not clean_text(fields["final_answer"]):
            fields["final_answer"] = reasoning
    if not problem or not reasoning:
        raise SourceAdapterError("missing_fields", "source adapter produced empty problem or reasoning")
    try:
        answer_info = validate_final_answer(reasoning, fields["final_answer"], task_type=fields["task_type"])
    except AnswerValidationError as exc:
        raise SourceAdapterError(exc.reason_code, str(exc)) from exc
    rendered = render_math_raw_v2(problem, reasoning, answer_info["final_answer"])
    parsed = parse_math_raw_v2_target(rendered["target"])
    if parsed["reasoning"] != reasoning or parsed["final_answer"] != answer_info["final_answer"]:
        raise SourceAdapterError("format_roundtrip_mismatch", "math_raw_v2 render/parse round-trip changed content")
    group_id = normalized_problem_hash(problem)
    metadata = {
        **dict(fields.get("metadata") or {}),
        "adapter": context.adapter,
        "adapter_version": context.adapter_version,
        "raw_source_content_hash": content_hash(row),
    }
    canonical_row = {
        "source_name": context.source_name,
        "source_revision": context.source_revision,
        "source_split": context.source_split,
        "source_row_id": str(context.source_row_id),
        "data_plane": "sft",
        "task_type": fields["task_type"],
        "problem": problem,
        "reasoning": reasoning,
        "final_answer": answer_info["final_answer"],
        "answer_type": answer_info["answer_type"],
        "problem_group_id": group_id,
        "verification_method": fields["verification_method"],
        "verification_result": fields["verification_result"],
        "quality_tier": fields["quality_tier"],
        "pipeline_version": context.pipeline_version,
        "format_version": context.format_version,
        "prompt": rendered["prompt"],
        "target": rendered["target"],
        "metadata": metadata,
    }
    return CanonicalSample.from_mapping(canonical_row, pipeline_version=context.pipeline_version).to_dict()


def _tokenized_length(row: Mapping[str, Any], tokenizer) -> int:
    return (
        len(tokenizer.encode(str(row.get("prompt", "")), add_special_tokens=False))
        + len(tokenizer.encode(str(row.get("target", "")), add_special_tokens=False))
        + (1 if getattr(tokenizer, "eos_token_id", None) is not None else 0)
    )


def _canonical_exact_hash(row: Mapping[str, Any]) -> str:
    return content_hash(
        {
            "problem": clean_text(row.get("problem")),
            "reasoning": clean_text(row.get("reasoning")),
            "final_answer": clean_text(row.get("final_answer")),
            "format_version": row.get("format_version"),
        }
    )


def _rank_key(row: Mapping[str, Any], tokenizer) -> tuple[Any, ...]:
    metadata = dict(row.get("metadata") or {})
    adapter = str(metadata.get("adapter") or row.get("source_name") or "")
    return (
        -int(row.get("verification_result") is True),
        -QUALITY_ORDER.get(str(row.get("quality_tier") or "Q").upper(), 0),
        -SOURCE_AUTHORITY.get(adapter, 0),
        _tokenized_length(row, tokenizer),
        str(row.get("source_name") or ""),
        str(row.get("source_row_id") or ""),
    )


def _mark_quarantine(row: Mapping[str, Any], reason: str, **details: Any) -> dict[str, Any]:
    value = dict(row)
    value["quarantine_reason_code"] = str(reason)
    value["quarantine_details"] = dict(details)
    return value


def deduplicate_canonical_rows(
    rows: Sequence[Mapping[str, Any]], *, tokenizer, max_per_problem_group: int = 2
) -> list[dict[str, Any]]:
    if int(max_per_problem_group) <= 0:
        raise ValueError("max_per_problem_group must be > 0")
    indexed = [(index, dict(row)) for index, row in enumerate(rows)]
    exact_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    pre_rejected: list[tuple[int, dict[str, Any]]] = []
    for index, row in indexed:
        if row.get("quarantine_reason_code"):
            pre_rejected.append((index, row))
            continue
        exact_hash = _canonical_exact_hash(row)
        exact_groups[exact_hash].append((index, row))
    exact_winners: list[tuple[int, dict[str, Any]]] = []
    rejected = list(pre_rejected)
    for exact_hash, candidates in exact_groups.items():
        ordered = sorted(candidates, key=lambda item: _rank_key(item[1], tokenizer))
        exact_winners.append(ordered[0])
        for index, row in ordered[1:]:
            rejected.append((index, _mark_quarantine(row, "duplicate_canonical_content", canonical_content_hash=exact_hash)))
    problem_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in exact_winners:
        problem_groups[str(item[1].get("problem_group_id") or normalized_problem_hash(item[1].get("problem")))].append(item)
    winners: list[tuple[int, dict[str, Any]]] = []
    for group_id, candidates in problem_groups.items():
        ordered = sorted(candidates, key=lambda item: _rank_key(item[1], tokenizer))
        winners.extend(ordered[: int(max_per_problem_group)])
        for index, row in ordered[int(max_per_problem_group) :]:
            rejected.append((index, _mark_quarantine(row, "problem_group_exposure_limit", problem_group_id=group_id)))
    return [row for _, row in sorted(winners, key=lambda item: item[0])] + [
        row for _, row in sorted(rejected, key=lambda item: item[0])
    ]


class _SqliteCanonicalStager:
    """Disk-backed deterministic dedup staging for large offline source builds."""

    def __init__(self, directory: Path, *, tokenizer, max_per_problem_group: int) -> None:
        if int(max_per_problem_group) <= 0:
            raise ValueError("max_per_problem_group must be > 0")
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=".sft-stage3-", suffix=".sqlite3", dir=directory, delete=False)
        handle.close()
        self.path = Path(handle.name).resolve()
        self.tokenizer = tokenizer
        self.max_per_problem_group = int(max_per_problem_group)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute(
            """
            CREATE TABLE candidates (
                row_index INTEGER PRIMARY KEY,
                row_json TEXT NOT NULL,
                pre_rejected INTEGER NOT NULL,
                exact_hash TEXT NOT NULL,
                problem_group_id TEXT NOT NULL,
                verified_rank INTEGER NOT NULL,
                quality_rank INTEGER NOT NULL,
                authority_rank INTEGER NOT NULL,
                token_length INTEGER NOT NULL,
                source_name TEXT NOT NULL,
                source_row_id TEXT NOT NULL,
                source_identity TEXT
            )
            """
        )
        self.connection.execute(
            "CREATE UNIQUE INDEX unique_source_identity ON candidates(source_identity) WHERE source_identity IS NOT NULL"
        )
        self.count = 0

    def add(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        rejected = int(bool(value.get("quarantine_reason_code")))
        metadata = dict(value.get("metadata") or {})
        adapter = str(metadata.get("adapter") or value.get("source_name") or "")
        source_identity = None
        if not rejected:
            source_identity = content_hash(
                {
                    "source_name": value.get("source_name"),
                    "source_revision": value.get("source_revision"),
                    "source_split": value.get("source_split"),
                    "source_row_id": value.get("source_row_id"),
                }
            )
        row_values = (
            self.count,
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
            rejected,
            "" if rejected else _canonical_exact_hash(value),
            "" if rejected else str(value.get("problem_group_id") or normalized_problem_hash(value.get("problem"))),
            -int(value.get("verification_result") is True),
            -QUALITY_ORDER.get(str(value.get("quality_tier") or "Q").upper(), 0),
            -SOURCE_AUTHORITY.get(adapter, 0),
            0 if rejected else _tokenized_length(value, self.tokenizer),
            str(value.get("source_name") or ""),
            str(value.get("source_row_id") or ""),
            source_identity,
        )
        try:
            self.connection.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row_values)
        except sqlite3.IntegrityError as exc:
            if source_identity is None or "unique" not in str(exc).lower():
                raise
            value = _mark_quarantine(value, "duplicate_source_identity", source_sample_id=source_identity)
            self.connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.count,
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
                    1,
                    "",
                    "",
                    0,
                    0,
                    0,
                    0,
                    str(value.get("source_name") or ""),
                    str(value.get("source_row_id") or ""),
                    None,
                ),
            )
        self.count += 1
        if self.count % 1000 == 0:
            self.connection.commit()

    @staticmethod
    def _rank_order() -> str:
        return "verified_rank, quality_rank, authority_rank, token_length, source_name, source_row_id, row_index"

    def iter_prepared(self) -> Iterator[dict[str, Any]]:
        self.connection.commit()
        rank_order = self._rank_order()
        common = f"""
            WITH exact_ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY exact_hash ORDER BY {rank_order}) AS exact_position
                FROM candidates WHERE pre_rejected = 0
            ), group_ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY problem_group_id ORDER BY {rank_order}) AS group_position
                FROM exact_ranked WHERE exact_position = 1
            )
        """
        winner_query = common + " SELECT row_json FROM group_ranked WHERE group_position <= ? ORDER BY row_index"
        for (row_json,) in self.connection.execute(winner_query, (self.max_per_problem_group,)):
            yield json.loads(row_json)
        duplicate_query = common + " SELECT row_json, exact_hash FROM exact_ranked WHERE exact_position > 1 ORDER BY row_index"
        for row_json, exact_hash in self.connection.execute(duplicate_query):
            yield _mark_quarantine(json.loads(row_json), "duplicate_canonical_content", canonical_content_hash=exact_hash)
        group_query = common + " SELECT row_json, problem_group_id FROM group_ranked WHERE group_position > ? ORDER BY row_index"
        for row_json, group_id in self.connection.execute(group_query, (self.max_per_problem_group,)):
            yield _mark_quarantine(json.loads(row_json), "problem_group_exposure_limit", problem_group_id=group_id)
        for (row_json,) in self.connection.execute(
            "SELECT row_json FROM candidates WHERE pre_rejected = 1 ORDER BY row_index"
        ):
            yield json.loads(row_json)

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            if self.path.is_file():
                self.path.unlink()


def _iter_json_lines(path: Path, *, compressed: bool = False) -> Iterator[dict[str, Any]]:
    opener = gzip.open if compressed else Path.open
    if compressed:
        handle_context = opener(path, "rt", encoding="utf-8")
    else:
        handle_context = opener(path, "r", encoding="utf-8")
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"source JSONL row {line_number} is not an object")
            yield value


def _load_source(spec: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    kind = str(spec.get("kind") or "load_from_disk").strip().lower()
    if kind == "synthetic_arithmetic":
        return build_synthetic_arithmetic_dataset(int(spec["num_samples"]), int(spec["seed"]))
    path = Path(os.path.expandvars(str(spec.get("path") or ""))).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"source cache not found: {path}")
    if kind == "jsonl":
        return _iter_json_lines(path)
    if kind == "jsonl_gz":
        return _iter_json_lines(path, compressed=True)
    if kind == "load_from_disk":
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError("datasets is required for load_from_disk sources") from exc
        loaded = load_from_disk(str(path))
        split = str(spec.get("split") or "train")
        if isinstance(loaded, Mapping):
            if split not in loaded:
                raise KeyError(f"source cache has no split={split!r}; available={sorted(loaded.keys())}")
            loaded = loaded[split]
        return loaded
    raise ValueError(f"unsupported local source kind={kind!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_revision(spec: Mapping[str, Any], loaded: Any) -> str:
    revision = str(spec.get("revision") or "").strip()
    if not _revision_is_vague(revision):
        return revision
    fingerprint = getattr(loaded, "_fingerprint", None)
    if fingerprint:
        return f"cache-fingerprint:{fingerprint}"
    path_text = os.path.expandvars(str(spec.get("path") or ""))
    path = Path(path_text).expanduser().resolve() if path_text else None
    if path is not None and path.is_file():
        return f"sha256:{_file_sha256(path)}"
    raise SourceAdapterError("invalid_source_revision", "source revision is vague and no immutable cache fingerprint is available")


def build_release_from_source_manifest(
    source_manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    tokenizer,
    tokenizer_revision: str,
    max_length: int,
    pipeline_version: str,
) -> dict[str, Any]:
    if source_manifest.get("format") != SOURCE_MANIFEST_FORMAT:
        raise ValueError(f"unsupported source manifest format={source_manifest.get('format')!r}")
    if str(source_manifest.get("format_version") or FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError(f"only format_version={FORMAT_VERSION!r} is supported")
    if _revision_is_vague(tokenizer_revision):
        raise ValueError("tokenizer_revision must be an immutable commit/release fingerprint")
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen release: {output_path}")
    source_specs = list(source_manifest.get("sources") or [])
    if not source_specs:
        raise ValueError("source manifest requires at least one source")
    pretrain_data_plane = dict(source_manifest.get("pretrain_data_plane") or {})
    the_stack = dict(pretrain_data_plane.get("the_stack") or {})
    if str(the_stack.get("status") or "").strip().lower() not in {"loaded", "disabled"}:
        raise ValueError(
            "source manifest must explicitly record pretrain_data_plane.the_stack.status as loaded or disabled"
        )
    planned_consumed_sequences = int(source_manifest.get("planned_consumed_sequences", 0) or 0)
    if planned_consumed_sequences <= 0:
        raise ValueError("source manifest requires planned_consumed_sequences > 0 for exposure auditing")
    stager = _SqliteCanonicalStager(
        output_path.parent,
        tokenizer=tokenizer,
        max_per_problem_group=int(source_manifest.get("max_problem_group_solutions", 2)),
    )
    adapter_versions: dict[str, str] = {}
    seen_source_names: set[str] = set()
    try:
        for source_spec in source_specs:
            source_name = str(source_spec.get("source_name") or "").strip()
            adapter = str(source_spec.get("adapter") or source_name).strip().lower()
            if not source_name:
                raise ValueError("each source requires source_name")
            if source_name in seen_source_names:
                raise ValueError(f"duplicate source_name in source manifest: {source_name}")
            seen_source_names.add(source_name)
            loaded = _load_source(source_spec)
            revision = _resolve_revision(source_spec, loaded)
            split = str(source_spec.get("split") or "train")
            adapter_version = str(source_spec.get("adapter_version") or ADAPTER_VERSION)
            adapter_versions[source_name] = adapter_version
            min_tier = str(source_spec.get("minimum_quality_tier") or "B").upper()
            if min_tier not in QUALITY_ORDER:
                raise ValueError(f"invalid minimum_quality_tier={min_tier!r}")
            row_id_field = source_spec.get("row_id_field")
            for index, row in enumerate(loaded):
                raw_row = dict(row)
                row_id = raw_row.get(str(row_id_field)) if row_id_field else index
                context = SourceContext(
                    source_name=source_name,
                    source_revision=revision,
                    source_split=split,
                    source_row_id=str(row_id),
                    adapter=adapter,
                    adapter_version=adapter_version,
                    pipeline_version=str(pipeline_version),
                    format_version=FORMAT_VERSION,
                )
                try:
                    canonical = adapt_source_row(adapter, raw_row, context, tokenizer=tokenizer)
                    if QUALITY_ORDER[canonical["quality_tier"]] < QUALITY_ORDER[min_tier]:
                        canonical = _mark_quarantine(
                            canonical,
                            "quality_tier_below_minimum",
                            minimum_quality_tier=min_tier,
                            observed_quality_tier=canonical["quality_tier"],
                        )
                except SourceAdapterError as exc:
                    canonical = {
                        "source_name": source_name,
                        "source_revision": revision,
                        "source_split": split,
                        "source_row_id": str(row_id),
                        "pipeline_version": str(pipeline_version),
                        "format_version": FORMAT_VERSION,
                        "quarantine_reason_code": exc.reason_code,
                        "quarantine_details": exc.details,
                    }
                stager.add(canonical)
        return build_frozen_sft_release(
            stager.iter_prepared(),
            output_dir,
            tokenizer=tokenizer,
            max_length=int(max_length),
            pipeline_version=str(pipeline_version),
            format_version=FORMAT_VERSION,
            tokenizer_revision=str(tokenizer_revision),
            adapter_versions=adapter_versions,
            release_metadata={
                "source_manifest_format": SOURCE_MANIFEST_FORMAT,
                "dedup_staging": "sqlite_disk_backed_v1",
                "planned_consumed_sequences": planned_consumed_sequences,
                "pretrain_data_plane": pretrain_data_plane,
                "problem_group_policy": {
                    "max_solutions": int(source_manifest.get("max_problem_group_solutions", 2)),
                    "ranking": "verified,quality,source_authority,tokenized_length,stable_identity",
                },
            },
        )
    finally:
        stager.close()
