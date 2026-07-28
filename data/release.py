from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ..chat_formatting import make_standard_chat_supervised_example
from .contracts import CanonicalSample, DataContractError, ReasonCode, content_hash
from .supervision import SupervisionError
from .tokenization import make_supervised_example


RELEASE_FORMAT = "fitmotn_frozen_sft_release_v1"
MANIFEST_FILENAME = "manifest.json"
ACCEPTED_FILENAME = "accepted.jsonl"
QUARANTINE_FILENAME = "quarantine.jsonl"


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


def build_frozen_sft_release(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    tokenizer,
    max_length: int,
    pipeline_version: str,
) -> dict[str, Any]:
    """Build an immutable, deterministic SFT release with atomic publication."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen release: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))).resolve()
    accepted_path = temporary / ACCEPTED_FILENAME
    quarantine_path = temporary / QUARANTINE_FILENAME
    accepted_count = 0
    quarantine_count = 0
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    seen_source_ids: dict[str, str] = {}
    supervised_tokens = 0
    try:
        with accepted_path.open("w", encoding="utf-8", newline="\n") as accepted_handle, quarantine_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as quarantine_handle:
            for raw_row in rows:
                row = dict(raw_row)
                try:
                    sample = CanonicalSample.from_mapping(row, pipeline_version=str(pipeline_version))
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
                    materialized = _materialize_sample(sample, tokenizer, max_length=int(max_length))
                    accepted_handle.write(_json_line(materialized))
                    accepted_count += 1
                    source_name = sample.identity.source_name
                    source_counts.setdefault(source_name, Counter())["accepted"] += 1
                    supervised_tokens += int(materialized["data_diagnostics"]["supervised_tokens"])
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
        }
        manifest = {
            "format": RELEASE_FORMAT,
            "pipeline_version": str(pipeline_version),
            "max_length": int(max_length),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "accepted_count": accepted_count,
            "quarantine_count": quarantine_count,
            "supervised_tokens": supervised_tokens,
            "reason_counts": dict(sorted(reason_counts.items())),
            "source_counts": {
                source_name: {
                    "accepted": int(counts.get("accepted", 0)),
                    "quarantine": int(counts.get("quarantine", 0)),
                }
                for source_name, counts in sorted(source_counts.items())
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
    if manifest.get("format") != RELEASE_FORMAT:
        errors.append(f"unsupported release format: {manifest.get('format')!r}")
    expected_fingerprint = manifest.get("manifest_fingerprint")
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("manifest_fingerprint", None)
    if expected_fingerprint != content_hash(fingerprint_payload):
        errors.append("manifest fingerprint mismatch")
    for filename in (ACCEPTED_FILENAME, QUARANTINE_FILENAME):
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
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value.get("input_ids"), list) or not isinstance(value.get("labels"), list):
                raise RuntimeError(f"invalid frozen record at line {line_number}")
            if len(value["input_ids"]) != len(value["labels"]):
                raise RuntimeError(f"frozen record token/label length mismatch at line {line_number}")
            yield value
