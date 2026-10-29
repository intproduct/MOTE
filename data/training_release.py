from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..chat_formatting import make_standard_chat_supervised_example
from .clean_release import DATA_PLANES, SFT_PLANES, TEXT_PLANES, iter_clean_records, validate_clean_release
from .contracts import content_hash
from .release import tokenizer_fingerprint
from .supervision import SupervisionError
from .tokenization import make_supervised_example


BOUND_RELEASE_FORMAT = "fitmotn_bound_training_release_v1"
BOUND_RECORD_FORMAT = "fitmotn_bound_training_record_v1"
ACCEPTED_FILENAME = "accepted.jsonl"
QUARANTINE_FILENAME = "quarantine.jsonl"
MANIFEST_FILENAME = "manifest.json"
VAGUE_REVISIONS = {"", "latest", "main", "master", "unknown", "head", "todo"}


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket(length: int, boundaries: Sequence[int]) -> str:
    for boundary in boundaries:
        if length <= int(boundary):
            return f"le_{int(boundary)}"
    return f"gt_{int(boundaries[-1])}" if boundaries else "unbucketed"


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(int(value) for value in values)

    def percentile(percent: int) -> int:
        index = max(0, min(len(ordered) - 1, (len(ordered) * percent + 99) // 100 - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean": float(statistics.fmean(ordered)),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
    }


def _tokenize_clean_record(record: Mapping[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    plane = str(record.get("data_plane") or "")
    if plane in TEXT_PLANES:
        ids = [int(value) for value in tokenizer.encode(str(record.get("text") or ""), add_special_tokens=False)]
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and (not ids or ids[-1] != int(eos_id)):
            ids.append(int(eos_id))
        if not ids:
            raise ValueError("empty_token_sequence")
        if len(ids) > int(max_length):
            raise ValueError("overlong")
        diagnostics = {
            "prompt_tokens": 0,
            "target_tokens": len(ids),
            "total_tokens": len(ids),
            "supervised_tokens": len(ids),
            "max_length": int(max_length),
            "overflow_tokens": 0,
            "truncation_action": "none",
        }
        input_ids, labels = ids, list(ids)
    elif plane in SFT_PLANES:
        if record.get("messages"):
            tokenized = make_standard_chat_supervised_example(
                tokenizer, list(record["messages"]), max_len=int(max_length), add_eos=True
            )
        else:
            tokenized = make_supervised_example(
                tokenizer,
                str(record.get("prompt") or ""),
                str(record.get("response") or ""),
                max_len=int(max_length),
                add_eos=True,
            )
        input_ids = [int(value) for value in tokenized["input_ids"].tolist()]
        labels = [int(value) for value in tokenized["labels"].tolist()]
        diagnostics = dict(tokenized["data_diagnostics"])
    else:
        raise ValueError(f"unsupported_data_plane:{plane}")
    return {
        "format": BOUND_RECORD_FORMAT,
        "bound_record_id": content_hash(
            {"clean_record_id": record.get("record_id"), "input_ids": input_ids, "labels": labels}
        ),
        "clean_record_id": str(record.get("record_id") or ""),
        "source_name": record.get("source_name"),
        "source_revision": record.get("source_revision"),
        "source_split": record.get("source_split"),
        "source_row_id": record.get("source_row_id"),
        "data_plane": plane,
        "input_ids": input_ids,
        "labels": labels,
        "data_diagnostics": diagnostics,
        "license": record.get("license"),
        "allowed_uses": list(record.get("allowed_uses") or []),
    }


def _packed_record(members: list[dict[str, Any]], max_length: int, boundaries: Sequence[int]) -> dict[str, Any]:
    input_ids = [token for member in members for token in member["input_ids"]]
    labels = [token for member in members for token in member["labels"]]
    member_ids = [member["clean_record_id"] for member in members]
    plane = members[0]["data_plane"]
    licenses = sorted({str(member.get("license") or "") for member in members})
    result = {
        "format": BOUND_RECORD_FORMAT,
        "bound_record_id": content_hash({"members": member_ids, "input_ids": input_ids}),
        "clean_record_id": member_ids[0] if len(member_ids) == 1 else content_hash(member_ids),
        "source_name": "packed",
        "source_revision": "derived-from-clean-release",
        "source_split": "train",
        "source_row_id": content_hash(member_ids),
        "data_plane": plane,
        "input_ids": input_ids,
        "labels": labels,
        "data_diagnostics": {
            "prompt_tokens": 0,
            "target_tokens": len(input_ids),
            "total_tokens": len(input_ids),
            "supervised_tokens": len(input_ids),
            "max_length": int(max_length),
            "overflow_tokens": 0,
            "truncation_action": "none",
        },
        "license": licenses[0] if len(licenses) == 1 else "mixed",
        "licenses": licenses,
        "allowed_uses": sorted(set.intersection(*(set(member.get("allowed_uses") or []) for member in members))),
        "packing": {"mode": "pretrain_greedy", "member_count": len(members), "member_record_ids": member_ids},
    }
    result["length_bucket"] = _bucket(len(input_ids), boundaries)
    return result


def _iter_tokenized_records(
    records: Iterator[dict[str, Any]], tokenizer, max_length: int, *, workers: int, prefetch_factor: int
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None, Exception | None]]:
    if workers == 1:
        for record in records:
            try:
                yield record, _tokenize_clean_record(record, tokenizer, max_length), None
            except (SupervisionError, ValueError) as exc:
                yield record, None, exc
        return
    pending: deque[tuple[dict[str, Any], Future]] = deque()
    limit = workers * prefetch_factor
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fitmotn-tokenizer") as executor:
        for record in records:
            pending.append((record, executor.submit(_tokenize_clean_record, record, tokenizer, max_length)))
            if len(pending) < limit:
                continue
            current, future = pending.popleft()
            try:
                yield current, future.result(), None
            except (SupervisionError, ValueError) as exc:
                yield current, None, exc
        while pending:
            current, future = pending.popleft()
            try:
                yield current, future.result(), None
            except (SupervisionError, ValueError) as exc:
                yield current, None, exc


def bind_clean_release(
    clean_release_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer,
    tokenizer_revision: str,
    max_length: int,
    selected_planes: Sequence[str],
    length_buckets: Sequence[int] | None = None,
    packing_mode: str = "none",
    tokenization_workers: int = 1,
    prefetch_factor: int = 4,
) -> dict[str, Any]:
    clean_report = validate_clean_release(clean_release_dir)
    if not clean_report["ok"]:
        raise RuntimeError(f"invalid clean release: {clean_report['errors']}")
    normalized_revision = str(tokenizer_revision or "").strip()
    if normalized_revision.lower() in VAGUE_REVISIONS or "replace_with" in normalized_revision.lower():
        raise ValueError("tokenizer_revision must be immutable")
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    planes = sorted({str(value).strip().lower() for value in selected_planes})
    if not planes or any(plane not in DATA_PLANES for plane in planes):
        raise ValueError("selected_planes must contain supported data planes")
    objectives = {"causal_lm" if plane in TEXT_PLANES else "sft" for plane in planes}
    if len(objectives) != 1:
        raise ValueError("a bound release cannot mix causal_lm and sft data planes")
    objective = next(iter(objectives))
    packing_mode = str(packing_mode or "none").strip().lower()
    if packing_mode not in {"none", "pretrain_greedy"}:
        raise ValueError("packing_mode must be none or pretrain_greedy")
    if packing_mode != "none" and objective != "causal_lm":
        raise ValueError("packing is only supported for causal pretraining data; SFT cross-sample packing is disabled")
    tokenization_workers = int(tokenization_workers)
    prefetch_factor = int(prefetch_factor)
    if tokenization_workers <= 0 or tokenization_workers > 256:
        raise ValueError("tokenization_workers must be in [1, 256]")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")
    boundaries = sorted({int(value) for value in list(length_buckets or [])})
    if any(value <= 0 or value > max_length for value in boundaries):
        raise ValueError("length bucket boundaries must be in (0, max_length]")
    if boundaries and boundaries[-1] != max_length:
        boundaries.append(max_length)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing bound release: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))).resolve()
    accepted_count = 0
    accepted_source_records = 0
    quarantine_count = 0
    skipped_plane_count = 0
    reason_counts: Counter[str] = Counter()
    plane_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    pack_buffer: list[dict[str, Any]] = []
    pack_plane: str | None = None
    try:
        with (temporary / ACCEPTED_FILENAME).open("w", encoding="utf-8", newline="\n") as accepted_handle, (
            temporary / QUARANTINE_FILENAME
        ).open("w", encoding="utf-8", newline="\n") as quarantine_handle:

            def write_record(value: dict[str, Any]) -> None:
                nonlocal accepted_count
                value.setdefault("length_bucket", _bucket(len(value["input_ids"]), boundaries))
                accepted_handle.write(_json_line(value))
                accepted_count += 1
                plane_counts[value["data_plane"]] += 1
                bucket_counts[value["length_bucket"]] += 1
                token_lengths.append(len(value["input_ids"]))

            def flush_pack() -> None:
                nonlocal pack_buffer, pack_plane
                if pack_buffer:
                    write_record(_packed_record(pack_buffer, max_length, boundaries))
                    pack_buffer = []
                    pack_plane = None

            def selected_records() -> Iterator[dict[str, Any]]:
                nonlocal skipped_plane_count
                for clean_record in iter_clean_records(clean_release_dir):
                    if str(clean_record.get("data_plane") or "") not in planes:
                        skipped_plane_count += 1
                        continue
                    yield clean_record

            for clean_record, bound, tokenization_error in _iter_tokenized_records(
                selected_records(),
                tokenizer,
                max_length,
                workers=tokenization_workers,
                prefetch_factor=prefetch_factor,
            ):
                plane = str(clean_record.get("data_plane") or "")
                if tokenization_error is not None:
                    exc = tokenization_error
                    reason = str(getattr(exc, "reason_code", str(exc)))
                    if ":" in reason or " " in reason:
                        reason = "tokenization_error"
                    reason_counts[reason] += 1
                    quarantine_count += 1
                    quarantine_handle.write(
                        _json_line(
                            {
                                "clean_record_id": clean_record.get("record_id"),
                                "source_name": clean_record.get("source_name"),
                                "data_plane": plane,
                                "reason_code": reason,
                                "details": {"message": str(exc)},
                            }
                        )
                    )
                    continue
                if bound is None:
                    raise RuntimeError("tokenization worker returned neither a record nor an error")
                accepted_source_records += 1
                if packing_mode == "none":
                    write_record(bound)
                    continue
                if pack_plane is not None and (pack_plane != plane or sum(len(item["input_ids"]) for item in pack_buffer) + len(bound["input_ids"]) > max_length):
                    flush_pack()
                pack_plane = plane
                pack_buffer.append(bound)
            flush_pack()
        files = {
            filename: {"sha256": _sha256_file(temporary / filename), "rows": rows}
            for filename, rows in ((ACCEPTED_FILENAME, accepted_count), (QUARANTINE_FILENAME, quarantine_count))
        }
        clean_manifest = dict(clean_report["manifest"] or {})
        manifest = {
            "format": BOUND_RELEASE_FORMAT,
            "record_format": BOUND_RECORD_FORMAT,
            "clean_release_fingerprint": clean_manifest.get("manifest_fingerprint"),
            "clean_pipeline_version": clean_manifest.get("pipeline_version"),
            "tokenizer_revision": normalized_revision,
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "chat_template_fingerprint": content_hash(str(getattr(tokenizer, "chat_template", "") or "")),
            "tokenizer_spec": {
                "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            },
            "max_length": max_length,
            "selected_planes": planes,
            "training_objective": objective,
            "packing_mode": packing_mode,
            "tokenization_execution": {
                "workers": tokenization_workers,
                "prefetch_factor": prefetch_factor,
                "ordered_output": True,
            },
            "length_bucket_boundaries": boundaries,
            "accepted_count": accepted_count,
            "accepted_source_records": accepted_source_records,
            "quarantine_count": quarantine_count,
            "skipped_plane_count": skipped_plane_count,
            "reason_counts": dict(sorted(reason_counts.items())),
            "data_plane_counts": dict(sorted(plane_counts.items())),
            "length_bucket_counts": dict(sorted(bucket_counts.items())),
            "token_statistics": _distribution(token_lengths),
            "files": files,
        }
        manifest["manifest_fingerprint"] = content_hash(manifest)
        (temporary / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report = validate_bound_training_release(temporary)
        if not report["ok"]:
            raise RuntimeError(f"bound release validation failed: {report['errors']}")
        os.replace(temporary, target)
        return manifest
    except Exception:
        if temporary.exists() and temporary.parent == target.parent and temporary.name.startswith(f".{target.name}.tmp-"):
            shutil.rmtree(temporary)
        raise


def validate_bound_training_release(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {MANIFEST_FILENAME}"], "manifest": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid manifest: {exc}"], "manifest": None}
    errors: list[str] = []
    if manifest.get("format") != BOUND_RELEASE_FORMAT:
        errors.append(f"unsupported bound release format={manifest.get('format')!r}")
    expected_fingerprint = manifest.get("manifest_fingerprint")
    payload = dict(manifest)
    payload.pop("manifest_fingerprint", None)
    if expected_fingerprint != content_hash(payload):
        errors.append("manifest fingerprint mismatch")
    for filename in (ACCEPTED_FILENAME, QUARANTINE_FILENAME):
        path = root / filename
        expected = dict(manifest.get("files", {}).get(filename) or {})
        if not path.is_file():
            errors.append(f"missing release file: {filename}")
            continue
        if expected.get("sha256") != _sha256_file(path):
            errors.append(f"sha256 mismatch: {filename}")
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        if int(expected.get("rows", -1)) != rows:
            errors.append(f"row count mismatch: {filename}")
    return {"ok": not errors, "errors": errors, "manifest": manifest}


def validate_bound_record(record: Mapping[str, Any], manifest: Mapping[str, Any], *, location: str = "record") -> dict[str, Any]:
    value = dict(record)
    if value.get("format") != BOUND_RECORD_FORMAT:
        raise RuntimeError(f"invalid bound record format at {location}")
    input_ids = value.get("input_ids")
    labels = value.get("labels")
    if not isinstance(input_ids, list) or not isinstance(labels, list) or not input_ids or len(input_ids) != len(labels):
        raise RuntimeError(f"invalid bound token/label lengths at {location}")
    max_length = int(manifest.get("max_length", 0) or 0)
    if max_length <= 0 or len(input_ids) > max_length:
        raise RuntimeError(f"bound length invariant failed at {location}")
    objective = str(manifest.get("training_objective") or "")
    if objective == "causal_lm" and any(int(token) != int(label) for token, label in zip(input_ids, labels)):
        raise RuntimeError(f"bound causal labels mismatch at {location}")
    if objective == "sft":
        if not any(int(label) == -100 for label in labels) or not any(int(label) != -100 for label in labels):
            raise RuntimeError(f"bound SFT mask invariant failed at {location}")
    eos_id = dict(manifest.get("tokenizer_spec") or {}).get("eos_token_id")
    if eos_id is not None and int(input_ids[-1]) != int(eos_id):
        raise RuntimeError(f"bound EOS invariant failed at {location}")
    diagnostics = dict(value.get("data_diagnostics") or {})
    if int(diagnostics.get("total_tokens", -1)) != len(input_ids) or int(diagnostics.get("overflow_tokens", -1)) != 0:
        raise RuntimeError(f"bound diagnostics invariant failed at {location}")
    return value


def iter_bound_training_records(release_dir: str | Path) -> Iterator[dict[str, Any]]:
    report = validate_bound_training_release(release_dir)
    if not report["ok"]:
        raise RuntimeError(f"invalid bound release: {report['errors']}")
    manifest = dict(report["manifest"] or {})
    with (Path(release_dir).expanduser().resolve() / ACCEPTED_FILENAME).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield validate_bound_record(json.loads(line), manifest, location=f"line {line_number}")


def preflight_bound_training_release(release_dir: str | Path, *, min_records: int = 1000) -> dict[str, Any]:
    report = validate_bound_training_release(release_dir)
    if not report["ok"]:
        raise RuntimeError(f"invalid bound release: {report['errors']}")
    manifest = dict(report["manifest"] or {})
    required = int(min_records)
    available = int(manifest.get("accepted_count", 0))
    if required <= 0:
        raise ValueError("min_records must be > 0")
    if available < required:
        raise RuntimeError(f"preflight requires at least {required} records; release has {available}")
    seen: set[str] = set()
    consumed = 0
    for record in iter_bound_training_records(release_dir):
        record_id = str(record.get("bound_record_id") or "")
        if not record_id or record_id in seen:
            raise RuntimeError("preflight found missing or duplicate bound_record_id")
        seen.add(record_id)
        consumed += 1
        if consumed >= required:
            break
    return {
        "ok": True,
        "consumed_records": consumed,
        "accepted_count": available,
        "training_objective": manifest.get("training_objective"),
        "tokenizer_fingerprint": manifest.get("tokenizer_fingerprint"),
        "dynamic_tokenization": False,
        "quarantine_visible_to_loader": False,
    }
