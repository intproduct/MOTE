from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ..chat_formatting import make_standard_chat_supervised_example
from ..tasks.answer_extraction import AnswerValidationError, validate_final_answer
from .sft_format import FORMAT_VERSION, parse_math_raw_v2_target
from .contracts import CanonicalSample, DataContractError, ReasonCode, content_hash
from .supervision import SupervisionError
from .tokenization import make_supervised_example


RELEASE_FORMAT = "fitmotn_frozen_sft_release_v2"
SUPPORTED_RELEASE_FORMATS = {"fitmotn_frozen_sft_release_v1", RELEASE_FORMAT}
MANIFEST_FILENAME = "manifest.json"
ACCEPTED_FILENAME = "accepted.jsonl"
QUARANTINE_FILENAME = "quarantine.jsonl"
LONG_CONTEXT_FILENAME = "long_context.jsonl"


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    return content_hash(payload)


def _materialize_sample(sample: CanonicalSample, tokenizer, *, max_length: int) -> dict[str, Any]:
    if sample.problem or sample.reasoning or sample.final_answer:
        try:
            validated = validate_final_answer(sample.reasoning, sample.final_answer, task_type=sample.task_type)
        except AnswerValidationError as exc:
            raise DataContractError(exc.reason_code, str(exc)) from exc
        if validated["final_answer"] != sample.final_answer or validated["answer_type"] != sample.answer_type:
            raise DataContractError(
                ReasonCode.INVALID_SCHEMA,
                "structured canonical answer normalization/type does not match declared fields",
            )
        if sample.format_version == FORMAT_VERSION:
            try:
                parsed = parse_math_raw_v2_target(sample.target)
            except ValueError as exc:
                raise DataContractError(ReasonCode.FORMAT_ROUNDTRIP_MISMATCH, str(exc)) from exc
            if parsed != {"reasoning": sample.reasoning, "final_answer": sample.final_answer}:
                raise DataContractError(
                    ReasonCode.FORMAT_ROUNDTRIP_MISMATCH,
                    "math_raw_v2 target does not round-trip to canonical reasoning/final_answer",
                )
    if sample.messages:
        tokenized = make_standard_chat_supervised_example(
            tokenizer,
            sample.messages,
            max_len=int(max_length),
            add_eos=True,
        )
        format_type = "chat"
    else:
        tokenized = make_supervised_example(
            tokenizer,
            sample.prompt,
            sample.target,
            max_len=int(max_length),
            add_eos=True,
        )
        format_type = "prompt_target"
    result = sample.to_dict()
    result.update(
        {
            "format_type": format_type,
            "input_ids": tokenized["input_ids"].tolist(),
            "labels": tokenized["labels"].tolist(),
            "data_diagnostics": dict(tokenized["data_diagnostics"]),
        }
    )
    if "loss_weights" in tokenized:
        result["loss_weights"] = tokenized["loss_weights"].tolist()
    return result


def _quarantine_record(row: Mapping[str, Any], reason_code: str, message: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "source_name": row.get("source_name"),
        "source_revision": row.get("source_revision"),
        "source_split": row.get("source_split"),
        "source_row_id": row.get("source_row_id"),
        "reason_code": str(reason_code),
        "message": str(message),
        "details": dict(details or {}),
    }


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(int(value) for value in values)

    def percentile(percent: int) -> int:
        index = max(0, min(len(ordered) - 1, (len(ordered) * percent + 99) // 100 - 1))
        return int(ordered[index])

    return {
        "count": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": int(ordered[-1]),
    }


def build_frozen_sft_release(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    tokenizer,
    max_length: int,
    pipeline_version: str,
    format_version: str | None = None,
    tokenizer_revision: str | None = None,
    adapter_versions: Mapping[str, str] | None = None,
    release_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an immutable, deterministic SFT release with atomic publication."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen release: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))).resolve()
    accepted_path = temporary / ACCEPTED_FILENAME
    quarantine_path = temporary / QUARANTINE_FILENAME
    long_context_path = temporary / LONG_CONTEXT_FILENAME
    accepted_count = 0
    quarantine_count = 0
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    seen_source_ids: dict[str, str] = {}
    seen_canonical_hashes: dict[str, str] = {}
    accepted_problem_groups: set[str] = set()
    quality_tier_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    token_values: dict[str, list[int]] = {
        "prompt_tokens": [],
        "target_tokens": [],
        "total_tokens": [],
        "supervised_tokens": [],
    }
    source_supervised_tokens: Counter[str] = Counter()
    long_context_count = 0
    supervised_tokens = 0
    try:
        with accepted_path.open("w", encoding="utf-8", newline="\n") as accepted_handle, quarantine_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as quarantine_handle, long_context_path.open("w", encoding="utf-8", newline="\n") as long_context_handle:
            for raw_row in rows:
                row = dict(raw_row)
                source_name = str(row.get("source_name") or "<missing>")
                source_counts.setdefault(source_name, Counter())["scanned"] += 1
                sample = None
                try:
                    if row.get("quarantine_reason_code"):
                        raise DataContractError(
                            str(row["quarantine_reason_code"]),
                            "row was rejected by the offline source pipeline",
                            details=dict(row.get("quarantine_details") or {}),
                        )
                    sample = CanonicalSample.from_mapping(row, pipeline_version=str(pipeline_version))
                    if format_version and str(format_version) != "legacy" and sample.format_version != str(format_version):
                        raise DataContractError(
                            ReasonCode.INVALID_SCHEMA,
                            f"sample format_version={sample.format_version!r} does not match release format_version={format_version!r}",
                        )
                    source_id = sample.identity.source_sample_id
                    if source_id in seen_source_ids:
                        raise DataContractError(
                            ReasonCode.DUPLICATE_SOURCE_IDENTITY,
                            "duplicate source identity in release input",
                            details={
                                "source_sample_id": source_id,
                                "first_raw_content_hash": seen_source_ids[source_id],
                                "duplicate_raw_content_hash": sample.raw_content_hash,
                            },
                        )
                    seen_source_ids[source_id] = sample.raw_content_hash
                    canonical_hash = sample.canonical_content_hash
                    if canonical_hash in seen_canonical_hashes:
                        raise DataContractError(
                            ReasonCode.DUPLICATE_CANONICAL_CONTENT,
                            "duplicate canonical content in release input",
                            details={
                                "canonical_content_hash": canonical_hash,
                                "first_source_sample_id": seen_canonical_hashes[canonical_hash],
                                "duplicate_source_sample_id": source_id,
                            },
                        )
                    seen_canonical_hashes[canonical_hash] = source_id
                    materialized = _materialize_sample(sample, tokenizer, max_length=int(max_length))
                    accepted_handle.write(_json_line(materialized))
                    accepted_count += 1
                    source_name = sample.identity.source_name
                    source_counts.setdefault(source_name, Counter())["accepted"] += 1
                    diagnostics = dict(materialized["data_diagnostics"])
                    supervised = int(diagnostics["supervised_tokens"])
                    supervised_tokens += supervised
                    source_supervised_tokens[source_name] += supervised
                    for key in token_values:
                        token_values[key].append(int(diagnostics[key]))
                    if sample.quality_tier:
                        quality_tier_counts[sample.quality_tier] += 1
                    verification_counts["verified" if sample.verification_result is True else "unverified"] += 1
                    if sample.problem_group_id:
                        accepted_problem_groups.add(sample.problem_group_id)
                except DataContractError as exc:
                    reason_counts[exc.reason_code] += 1
                    quarantine_count += 1
                    source_name = str(row.get("source_name") or "<missing>")
                    source_counts.setdefault(source_name, Counter())["quarantine"] += 1
                    quarantine_handle.write(_json_line(_quarantine_record(row, exc.reason_code, str(exc), exc.details)))
                except SupervisionError as exc:
                    reason_counts[exc.reason_code] += 1
                    quarantine_count += 1
                    source_name = str(row.get("source_name") or "<missing>")
                    source_counts.setdefault(source_name, Counter())["quarantine"] += 1
                    if exc.reason_code == ReasonCode.OVERLONG.value and sample is not None:
                        long_context_count += 1
                        long_context_handle.write(
                            _json_line(
                                {
                                    **sample.to_dict(),
                                    "data_diagnostics": exc.diagnostics.to_dict(),
                                    "reason_code": exc.reason_code,
                                }
                            )
                        )
                    quarantine_handle.write(
                        _json_line(_quarantine_record(row, exc.reason_code, str(exc), exc.diagnostics.to_dict()))
                    )
                except Exception as exc:
                    source = {
                        "source_name": row.get("source_name"),
                        "source_revision": row.get("source_revision"),
                        "source_split": row.get("source_split"),
                        "source_row_id": row.get("source_row_id"),
                    }
                    raise RuntimeError(
                        f"unexpected release-build failure for source identity {source}; refusing to quarantine an unknown error"
                    ) from exc

        files = {
            ACCEPTED_FILENAME: {"sha256": _sha256_file(accepted_path), "rows": accepted_count},
            QUARANTINE_FILENAME: {"sha256": _sha256_file(quarantine_path), "rows": quarantine_count},
            LONG_CONTEXT_FILENAME: {"sha256": _sha256_file(long_context_path), "rows": long_context_count},
        }
        source_contributions = {}
        for source_name, counts in sorted(source_counts.items()):
            accepted_for_source = int(counts.get("accepted", 0))
            source_contributions[source_name] = {
                "accepted_sequences": accepted_for_source,
                "supervised_tokens": int(source_supervised_tokens.get(source_name, 0)),
                "sequence_fraction": float(accepted_for_source / accepted_count) if accepted_count else 0.0,
                "supervised_token_fraction": (
                    float(source_supervised_tokens.get(source_name, 0) / supervised_tokens) if supervised_tokens else 0.0
                ),
            }
        release_metadata = dict(release_metadata or {})
        planned_consumed_sequences = int(release_metadata.get("planned_consumed_sequences", 0) or 0)
        estimated_source_epochs = (
            (planned_consumed_sequences + accepted_count - 1) // accepted_count
            if planned_consumed_sequences > 0 and accepted_count > 0
            else None
        )
        manifest = {
            "format": RELEASE_FORMAT,
            "pipeline_version": str(pipeline_version),
            "format_version": str(format_version or "legacy"),
            "adapter_versions": dict(sorted(dict(adapter_versions or {}).items())),
            "max_length": int(max_length),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "tokenizer_revision": str(tokenizer_revision or getattr(tokenizer, "name_or_path", "")),
            "tokenizer_spec": {
                "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            },
            "accepted_count": accepted_count,
            "quarantine_count": quarantine_count,
            "long_context_count": long_context_count,
            "supervised_tokens": supervised_tokens,
            "reason_counts": dict(sorted(reason_counts.items())),
            "source_counts": {
                source_name: {
                    "scanned": int(counts.get("scanned", 0)),
                    "accepted": int(counts.get("accepted", 0)),
                    "quarantine": int(counts.get("quarantine", 0)),
                }
                for source_name, counts in sorted(source_counts.items())
            },
            "loaded_sources": sorted(source_name for source_name, counts in source_counts.items() if counts.get("scanned", 0)),
            "verification_counts": dict(sorted(verification_counts.items())),
            "quality_tier_counts": dict(sorted(quality_tier_counts.items())),
            "problem_group_count": len(accepted_problem_groups),
            "problem_group_rejected_count": int(
                reason_counts.get(ReasonCode.PROBLEM_GROUP_EXPOSURE_LIMIT.value, 0)
            ),
            "exact_duplicate_count": int(reason_counts.get(ReasonCode.DUPLICATE_CANONICAL_CONTENT.value, 0)),
            "token_statistics": {key: _distribution(values) for key, values in token_values.items()},
            "source_token_contributions": source_contributions,
            "anomaly_counts": {
                "answer_equals_solution": int(reason_counts.get(ReasonCode.ANSWER_EQUALS_SOLUTION.value, 0)),
                "overlong": int(reason_counts.get(ReasonCode.OVERLONG.value, 0)),
                "empty_supervision": int(reason_counts.get(ReasonCode.EMPTY_SUPERVISION.value, 0)),
                "eos_anomaly": 0,
                "mask_anomaly": 0,
            },
            "pretrain_data_plane": dict(release_metadata.get("pretrain_data_plane") or {}),
            "release_metadata": release_metadata,
            "exposure_plan": {
                "planned_consumed_sequences": planned_consumed_sequences or None,
                "estimated_source_epochs": estimated_source_epochs,
                "estimated_max_sample_exposure": estimated_source_epochs,
            },
            "files": files,
        }
        manifest["manifest_fingerprint"] = content_hash(manifest)
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        validation = validate_frozen_sft_release(temporary)
        if not validation["ok"]:
            raise RuntimeError(f"frozen release validation failed before publication: {validation['errors']}")
        os.replace(temporary, target)
        return manifest
    except Exception:
        if temporary.exists() and temporary.parent == target.parent and temporary.name.startswith(f".{target.name}.tmp-"):
            shutil.rmtree(temporary)
        raise


def validate_frozen_sft_release(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir).expanduser().resolve()
    errors: list[str] = []
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {MANIFEST_FILENAME}"], "manifest": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid manifest: {exc}"], "manifest": None}
    if manifest.get("format") not in SUPPORTED_RELEASE_FORMATS:
        errors.append(f"unsupported release format: {manifest.get('format')!r}")
    expected_fingerprint = manifest.get("manifest_fingerprint")
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("manifest_fingerprint", None)
    if expected_fingerprint != content_hash(fingerprint_payload):
        errors.append("manifest fingerprint mismatch")
    required_files = [ACCEPTED_FILENAME, QUARANTINE_FILENAME]
    if LONG_CONTEXT_FILENAME in dict(manifest.get("files") or {}):
        required_files.append(LONG_CONTEXT_FILENAME)
    for filename in required_files:
        file_path = (root / filename).resolve()
        if file_path.parent != root or not file_path.is_file():
            errors.append(f"missing release file: {filename}")
            continue
        expected = dict(manifest.get("files", {}).get(filename) or {})
        if expected.get("sha256") != _sha256_file(file_path):
            errors.append(f"sha256 mismatch: {filename}")
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
        except OSError as exc:
            errors.append(f"cannot read {filename}: {exc}")
            continue
        if int(expected.get("rows", -1)) != row_count:
            errors.append(f"row count mismatch: {filename}")
    return {"ok": not errors, "errors": errors, "manifest": manifest}


def iter_frozen_sft_records(release_dir: str | Path) -> Iterator[dict[str, Any]]:
    report = validate_frozen_sft_release(release_dir)
    if not report["ok"]:
        raise RuntimeError(f"invalid frozen SFT release: {report['errors']}")
    path = Path(release_dir).expanduser().resolve() / ACCEPTED_FILENAME
    manifest = dict(report.get("manifest") or {})
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            yield validate_frozen_record(value, manifest, location=f"line {line_number}")


def validate_frozen_record(
    value: Mapping[str, Any], manifest: Mapping[str, Any], *, location: str = "record"
) -> dict[str, Any]:
    record = dict(value)
    input_ids = record.get("input_ids")
    labels = record.get("labels")
    if not isinstance(input_ids, list) or not isinstance(labels, list):
        raise RuntimeError(f"invalid frozen {location}: input_ids/labels must be lists")
    if len(input_ids) != len(labels):
        raise RuntimeError(f"frozen token/label length mismatch at {location}")
    max_length = int(manifest.get("max_length", 0) or 0)
    if not input_ids or (max_length > 0 and len(input_ids) > max_length):
        raise RuntimeError(f"frozen length invariant failed at {location}")
    supervised_positions = [index for index, label in enumerate(labels) if int(label) != -100]
    if not supervised_positions:
        raise RuntimeError(f"frozen record has empty supervision at {location}")
    first_supervised = supervised_positions[0]
    strict_math_raw = str(manifest.get("format_version") or "") == FORMAT_VERSION
    if strict_math_raw and any(int(label) == -100 for label in labels[first_supervised:]):
        raise RuntimeError(f"frozen target mask invariant failed at {location}")
    diagnostics = dict(record.get("data_diagnostics") or {})
    if diagnostics:
        if int(diagnostics.get("total_tokens", -1)) != len(input_ids):
            raise RuntimeError(f"frozen diagnostics total_tokens mismatch at {location}")
        if strict_math_raw and int(diagnostics.get("prompt_tokens", -1)) != first_supervised:
            raise RuntimeError(f"frozen diagnostics prompt_tokens/mask mismatch at {location}")
        if strict_math_raw and int(diagnostics.get("target_tokens", -1)) != len(input_ids) - first_supervised:
            raise RuntimeError(f"frozen diagnostics target_tokens/mask mismatch at {location}")
        if int(diagnostics.get("supervised_tokens", -1)) != len(supervised_positions):
            raise RuntimeError(f"frozen diagnostics supervised_tokens mismatch at {location}")
        if int(diagnostics.get("overflow_tokens", -1)) != 0:
            raise RuntimeError(f"frozen record reports truncation/overflow at {location}")
    eos_id = dict(manifest.get("tokenizer_spec") or {}).get("eos_token_id")
    if strict_math_raw and eos_id is not None:
        eos_count = sum(int(token) == int(eos_id) for token in input_ids)
        if eos_count != 1 or int(input_ids[-1]) != int(eos_id):
            raise RuntimeError(f"frozen EOS invariant failed at {location}")
    return record


def preflight_frozen_sft_release(release_dir: str | Path, *, min_records: int = 1000) -> dict[str, Any]:
    report = validate_frozen_sft_release(release_dir)
    if not report["ok"]:
        raise RuntimeError(f"invalid frozen SFT release: {report['errors']}")
    manifest = dict(report["manifest"] or {})
    available = int(manifest.get("accepted_count", 0))
    required = int(min_records)
    if required <= 0:
        raise ValueError("min_records must be > 0")
    if available < required:
        raise RuntimeError(f"preflight requires at least {required} accepted records; release has {available}")
    consumed = 0
    openr1_unverified = 0
    answer_prefix_residue = 0
    seen_source_ids: set[str] = set()
    seen_canonical_hashes: set[str] = set()
    for record in iter_frozen_sft_records(release_dir):
        source_id = str(record.get("source_sample_id") or "")
        canonical_hash = str(record.get("canonical_content_hash") or "")
        if source_id and source_id in seen_source_ids:
            raise RuntimeError("preflight found duplicate source_sample_id")
        if canonical_hash and canonical_hash in seen_canonical_hashes:
            raise RuntimeError("preflight found duplicate canonical_content_hash")
        seen_source_ids.add(source_id)
        seen_canonical_hashes.add(canonical_hash)
        metadata = dict(record.get("metadata") or {})
        if str(metadata.get("adapter") or "") == "openr1" and record.get("verification_result") is not True:
            openr1_unverified += 1
        if str(record.get("final_answer") or "").strip().lower().startswith("the answer is"):
            answer_prefix_residue += 1
        consumed += 1
        if consumed >= required:
            break
    if consumed != required:
        raise RuntimeError(f"preflight expected {required} records but consumed {consumed}")
    if openr1_unverified or answer_prefix_residue:
        raise RuntimeError(
            f"preflight quality invariant failed: openr1_unverified={openr1_unverified} "
            f"answer_prefix_residue={answer_prefix_residue}"
        )
    return {
        "ok": True,
        "consumed_records": consumed,
        "accepted_count": available,
        "tokenizer_fingerprint": manifest.get("tokenizer_fingerprint"),
        "openr1_unverified": openr1_unverified,
        "answer_prefix_residue": answer_prefix_residue,
        "dynamic_tokenization": False,
        "quarantine_visible_to_loader": False,
    }
