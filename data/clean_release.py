from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .contracts import content_hash, stable_json_bytes
from .text_normalization import normalize_text


REGISTRY_FORMAT = "fitmotn_data_registry_v1"
CLEAN_RECORD_FORMAT = "fitmotn_clean_record_v1"
CLEAN_RELEASE_FORMAT = "fitmotn_clean_release_v1"
ACCEPTED_FILENAME = "accepted.jsonl"
QUARANTINE_FILENAME = "quarantine.jsonl"
MANIFEST_FILENAME = "manifest.json"

DATA_PLANES = {
    "pretraining",
    "general_sft",
    "reasoning_sft",
    "code_pretraining",
    "code_sft",
}
SFT_PLANES = {"general_sft", "reasoning_sft", "code_sft"}
TEXT_PLANES = {"pretraining", "code_pretraining"}
VAGUE_REVISIONS = {"", "latest", "main", "master", "unknown", "head", "todo"}
SENSITIVE_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
    ),
}


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_vague_revision(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in VAGUE_REVISIONS or "replace_with" in normalized


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and tag.lower() in {"p", "div", "br", "li", "section", "article", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and tag.lower() in {"p", "div", "li", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def _strip_html(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return "".join(parser.parts)


def normalize_governed_text(value: Any, *, strip_html: bool = False) -> str:
    text = str(value or "")
    if strip_html:
        text = _strip_html(text)
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    return normalize_text(text)


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key and key in row and row[key] is not None:
            return row[key]
    return None


def _normalize_messages(value: Any, role_map: Mapping[str, str], *, strip_html: bool) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    defaults = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant", "system": "system", "tool": "tool"}
    defaults.update({str(key).strip().lower(): str(role).strip().lower() for key, role in role_map.items()})
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = defaults.get(str(_first(item, "role", "from") or "").strip().lower(), str(_first(item, "role", "from") or "").strip().lower())
        content = normalize_governed_text(_first(item, "content", "value", "text"), strip_html=strip_html)
        if role not in {"system", "user", "assistant", "tool"} or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _canonicalize_row(row: Mapping[str, Any], source: Mapping[str, Any], row_index: int, pipeline_version: str) -> dict[str, Any]:
    plane = str(source["data_plane"])
    adapter = str(source.get("adapter") or ("text" if plane in TEXT_PLANES else "prompt_response")).strip().lower()
    fields = dict(source.get("fields") or {})
    strip_html = bool(source.get("strip_html", False))
    raw_payload: Any
    text = ""
    prompt = ""
    response = ""
    messages: list[dict[str, str]] = []
    if adapter == "text":
        raw_text = _first(row, str(fields.get("text") or ""), "text", "content", "body", "document", "code")
        raw_payload = {"text": raw_text}
        text = normalize_governed_text(raw_text, strip_html=strip_html)
        if not text:
            raise ValueError("missing_text")
    elif adapter == "prompt_response":
        raw_prompt = _first(row, str(fields.get("prompt") or ""), "prompt", "instruction", "question", "problem")
        raw_response = _first(row, str(fields.get("response") or ""), "response", "output", "answer", "solution", "reasoning")
        raw_payload = {"prompt": raw_prompt, "response": raw_response}
        prompt = normalize_governed_text(raw_prompt, strip_html=strip_html)
        response = normalize_governed_text(raw_response, strip_html=strip_html)
        if not prompt or not response:
            raise ValueError("missing_prompt_or_response")
    elif adapter == "messages":
        raw_messages = _first(row, str(fields.get("messages") or ""), "messages", "conversations", "conversation")
        raw_payload = {"messages": raw_messages}
        messages = _normalize_messages(raw_messages, dict(source.get("role_map") or {}), strip_html=strip_html)
        if not messages or not any(message["role"] == "assistant" for message in messages):
            raise ValueError("missing_valid_messages")
    else:
        raise ValueError(f"unsupported_adapter:{adapter}")

    if messages:
        normalized_content = "\n".join(f"<{item['role']}>\n{item['content']}" for item in messages)
        contamination_text = "\n".join(item["content"] for item in messages if item["role"] in {"user", "system"})
    elif plane in SFT_PLANES:
        normalized_content = f"<user>\n{prompt}\n<assistant>\n{response}"
        contamination_text = prompt
    else:
        normalized_content = text
        contamination_text = text
    source_row_id = str(row.get(str(source.get("row_id_field")))) if source.get("row_id_field") else str(row_index)
    source_identity = {
        "source_name": source["source_name"],
        "source_revision": source["revision"],
        "source_split": source["split"],
        "source_row_id": source_row_id,
    }
    raw_content_hash = content_hash(raw_payload)
    normalized_content_hash = content_hash({"data_plane": plane, "content": normalized_content})
    record_id = content_hash({**source_identity, "pipeline_version": pipeline_version, "normalized_content_hash": normalized_content_hash})
    preserved = {key: row.get(key) for key in list(source.get("preserve_fields") or []) if key in row}
    external_scores: dict[str, float] = {}
    for score_name, field_name in dict(source.get("quality_score_fields") or {}).items():
        raw_score = row.get(str(field_name))
        if raw_score is None:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_external_quality_score") from exc
        if not math.isfinite(score):
            raise ValueError("invalid_external_quality_score")
        external_scores[str(score_name)] = score
    return {
        "format": CLEAN_RECORD_FORMAT,
        "record_id": record_id,
        **source_identity,
        "data_plane": plane,
        "adapter": adapter,
        "text": text,
        "prompt": prompt,
        "response": response,
        "messages": messages,
        "normalized_content": normalized_content,
        "contamination_text": contamination_text,
        "raw_content_hash": raw_content_hash,
        "normalized_content_hash": normalized_content_hash,
        "license": source["license"],
        "allowed_uses": list(source["allowed_uses"]),
        "acquired_at": source["acquired_at"],
        "language": str(source.get("language") or ""),
        "domain": str(source.get("domain") or ""),
        "quality_flags": [],
        "quality_scores": external_scores,
        "sensitive_flags": [],
        "provenance": {
            "registry_format": REGISTRY_FORMAT,
            "pipeline_version": pipeline_version,
            "upstream_metadata": dict(source.get("upstream_metadata") or {}),
            "preserved_fields": preserved,
        },
    }


def _quality_diagnostics(text: str) -> dict[str, float | int]:
    length = len(text)
    controls = sum(unicodedata.category(char).startswith("C") and char not in "\n\t" for char in text)
    replacements = text.count("\ufffd")
    alnum = sum(char.isalnum() for char in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = len(lines) - len(set(lines))
    return {
        "char_count": length,
        "control_ratio": controls / max(1, length),
        "replacement_ratio": replacements / max(1, length),
        "alnum_ratio": alnum / max(1, length),
        "repeated_line_ratio": repeated / max(1, len(lines)),
    }


def _quality_gate(record: dict[str, Any], policy: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    text = str(record["normalized_content"])
    diagnostics = _quality_diagnostics(text)
    flags: list[str] = []
    checks = [
        (diagnostics["char_count"] < int(policy.get("min_chars", 1)), "too_short"),
        (diagnostics["char_count"] > int(policy.get("max_chars", 10_000_000)), "too_long_chars"),
        (diagnostics["control_ratio"] > float(policy.get("max_control_ratio", 0.01)), "control_character_ratio"),
        (diagnostics["replacement_ratio"] > float(policy.get("max_replacement_ratio", 0.001)), "replacement_character_ratio"),
        (diagnostics["alnum_ratio"] < float(policy.get("min_alnum_ratio", 0.0)), "low_alnum_ratio"),
        (diagnostics["repeated_line_ratio"] > float(policy.get("max_repeated_line_ratio", 0.8)), "repeated_lines"),
    ]
    for failed, reason in checks:
        if failed:
            flags.append(reason)
    record["quality_flags"] = flags
    record["quality_scores"] = {**dict(record.get("quality_scores") or {}), **diagnostics}
    sensitive_flags = sorted(name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text))
    record["sensitive_flags"] = sensitive_flags
    rejected_sensitive = sorted(set(sensitive_flags) & {str(value) for value in list(policy.get("reject_sensitive_flags") or [])})
    if flags:
        return flags[0], diagnostics
    if rejected_sensitive:
        return "sensitive_content", {**diagnostics, "sensitive_flags": rejected_sensitive}
    return None, diagnostics


def _prepare_source_row(job: tuple[dict[str, Any], dict[str, Any], int, str, dict[str, Any]]) -> tuple[int, str, dict[str, Any] | None, str | None, dict[str, Any]]:
    row, source, row_index, pipeline_version, source_policy = job
    source_row_id = str(row.get(str(source.get("row_id_field")))) if source.get("row_id_field") else str(row_index)
    try:
        record = _canonicalize_row(row, source, row_index, pipeline_version)
        reason, details = _quality_gate(record, source_policy)
        return row_index, source_row_id, record, reason, details
    except ValueError as exc:
        message = str(exc)
        reason = message if re.fullmatch(r"[a-z0-9_]+", message) else "invalid_source_record"
        return row_index, source_row_id, None, reason, {"message": message}


def _iter_jsonl(path: Path, compressed: bool = False) -> Iterator[dict[str, Any]]:
    opener = gzip.open if compressed else Path.open
    args = (path, "rt") if compressed else (path, "r")
    with opener(*args, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object")
            yield value


def _load_local_source(spec: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    kind = str(spec.get("kind") or "load_from_disk").strip().lower()
    path = Path(os.path.expandvars(str(spec.get("path") or ""))).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"source cache not found: {path}")
    if kind == "jsonl":
        return _iter_jsonl(path)
    if kind == "jsonl_gz":
        return _iter_jsonl(path, compressed=True)
    if kind in {"load_from_disk", "parquet"}:
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:
            raise RuntimeError("datasets is required for load_from_disk/parquet sources") from exc
        dataset_path = path / "dataset" if kind == "load_from_disk" and (path / "source_receipt.json").is_file() else path
        loaded = load_from_disk(str(dataset_path)) if kind == "load_from_disk" else load_dataset("parquet", data_files=str(path), split="train")
        split = str(spec.get("split") or "train")
        if isinstance(loaded, Mapping):
            if split not in loaded:
                raise KeyError(f"source cache has no split={split!r}")
            loaded = loaded[split]
        return loaded
    raise ValueError(f"unsupported local source kind={kind!r}")


@dataclass(frozen=True)
class _NearPolicy:
    enabled: bool
    threshold: float
    num_perm: int
    bands: int
    shingle_size: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "_NearPolicy":
        raw = dict(value or {})
        enabled = bool(raw.get("enabled", False))
        threshold = float(raw.get("threshold", 0.85))
        num_perm = int(raw.get("num_perm", 64))
        bands = int(raw.get("bands", 8))
        shingle_size = int(raw.get("shingle_size", 5))
        if not 0 < threshold <= 1:
            raise ValueError("near_duplicate.threshold must be in (0, 1]")
        if num_perm <= 0 or bands <= 0 or num_perm % bands:
            raise ValueError("near_duplicate.num_perm must be positive and divisible by bands")
        if shingle_size <= 0:
            raise ValueError("near_duplicate.shingle_size must be > 0")
        return cls(enabled, threshold, num_perm, bands, shingle_size)


class _GovernanceIndex:
    _PRIME = (1 << 61) - 1

    def __init__(self, path: Path, near_policy: _NearPolicy) -> None:
        self.near_policy = near_policy
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("CREATE TABLE exact_keys (kind TEXT, value TEXT, record_id TEXT, PRIMARY KEY(kind, value))")
        self.connection.execute("CREATE TABLE signatures (namespace TEXT, record_id TEXT, signature BLOB, PRIMARY KEY(namespace, record_id))")
        self.connection.execute("CREATE TABLE bands (namespace TEXT, band_index INTEGER, band_hash TEXT, record_id TEXT)")
        self.connection.execute("CREATE INDEX band_lookup ON bands(namespace, band_index, band_hash)")
        self._permutations = [self._permutation(index) for index in range(near_policy.num_perm)]

    @staticmethod
    def _permutation(index: int) -> tuple[int, int]:
        digest = hashlib.sha256(f"fitmotn-minhash-v1:{index}".encode("ascii")).digest()
        a = int.from_bytes(digest[:8], "big") | 1
        b = int.from_bytes(digest[8:16], "big")
        return a, b

    def signature(self, text: str) -> tuple[int, ...]:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
        size = self.near_policy.shingle_size
        shingles = ["\x1f".join(tokens[index : index + size]) for index in range(max(1, len(tokens) - size + 1))]
        if not tokens:
            shingles = [""]
        bases = [int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "big") % self._PRIME for item in shingles]
        return tuple(min((a * value + b) % self._PRIME for value in bases) for a, b in self._permutations)

    @staticmethod
    def _pack(signature: Sequence[int]) -> bytes:
        return struct.pack(f">{len(signature)}Q", *signature)

    def _unpack(self, value: bytes) -> tuple[int, ...]:
        return struct.unpack(f">{self.near_policy.num_perm}Q", value)

    def has_exact(self, kind: str, value: str) -> str | None:
        row = self.connection.execute("SELECT record_id FROM exact_keys WHERE kind=? AND value=?", (kind, value)).fetchone()
        return None if row is None else str(row[0])

    def add_exact(self, kind: str, value: str, record_id: str) -> None:
        self.connection.execute("INSERT INTO exact_keys VALUES (?, ?, ?)", (kind, value, record_id))

    def find_near(self, namespace: str, signature: Sequence[int]) -> tuple[str, float] | None:
        if not self.near_policy.enabled:
            return None
        rows_per_band = self.near_policy.num_perm // self.near_policy.bands
        candidate_ids: set[str] = set()
        for band_index in range(self.near_policy.bands):
            start = band_index * rows_per_band
            band_hash = hashlib.sha256(self._pack(signature[start : start + rows_per_band])).hexdigest()
            for (record_id,) in self.connection.execute(
                "SELECT record_id FROM bands WHERE namespace=? AND band_index=? AND band_hash=?",
                (namespace, band_index, band_hash),
            ):
                candidate_ids.add(str(record_id))
        for record_id in sorted(candidate_ids):
            row = self.connection.execute(
                "SELECT signature FROM signatures WHERE namespace=? AND record_id=?", (namespace, record_id)
            ).fetchone()
            if row is None:
                continue
            other = self._unpack(row[0])
            similarity = sum(left == right for left, right in zip(signature, other)) / len(signature)
            if similarity >= self.near_policy.threshold:
                return record_id, similarity
        return None

    def add_near(self, namespace: str, record_id: str, signature: Sequence[int]) -> None:
        if not self.near_policy.enabled:
            return
        packed = self._pack(signature)
        self.connection.execute("INSERT INTO signatures VALUES (?, ?, ?)", (namespace, record_id, packed))
        rows_per_band = self.near_policy.num_perm // self.near_policy.bands
        for band_index in range(self.near_policy.bands):
            start = band_index * rows_per_band
            band_hash = hashlib.sha256(self._pack(signature[start : start + rows_per_band])).hexdigest()
            self.connection.execute("INSERT INTO bands VALUES (?, ?, ?, ?)", (namespace, band_index, band_hash, record_id))

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _validate_source(source: Mapping[str, Any], intended_use: str) -> dict[str, Any]:
    result = dict(source)
    required = ["source_name", "kind", "path", "split", "revision", "data_plane", "license", "allowed_uses", "acquired_at"]
    for field in required:
        value = result.get(field)
        if value is None or value == "" or (field == "allowed_uses" and not value):
            raise ValueError(f"source {result.get('source_name')!r} requires non-empty {field}")
    if _is_vague_revision(result["revision"]):
        raise ValueError(f"source {result['source_name']!r} revision must be immutable")
    plane = str(result["data_plane"]).strip().lower()
    if plane not in DATA_PLANES:
        raise ValueError(f"unsupported data_plane={plane!r}")
    result["data_plane"] = plane
    allowed = sorted({str(value).strip().lower() for value in result["allowed_uses"] if str(value).strip()})
    if not allowed:
        raise ValueError(f"source {result['source_name']!r} requires non-empty allowed_uses")
    if intended_use and intended_use not in allowed:
        raise ValueError(f"source {result['source_name']!r} does not allow intended_use={intended_use!r}")
    result["allowed_uses"] = allowed
    result["source_name"] = str(result["source_name"]).strip()
    result["revision"] = str(result["revision"]).strip()
    result["split"] = str(result["split"]).strip()
    result["license"] = str(result["license"]).strip().lower()
    result["acquired_at"] = str(result["acquired_at"]).strip()
    try:
        datetime.fromisoformat(result["acquired_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"source {result['source_name']!r} acquired_at must be ISO-8601") from exc
    return result


def _load_benchmarks(registry: Mapping[str, Any], index: _GovernanceIndex) -> int:
    count = 0
    for benchmark in list(registry.get("benchmarks") or []):
        if not bool(benchmark.get("enabled", True)):
            continue
        name = str(benchmark.get("name") or "").strip()
        if not name or _is_vague_revision(benchmark.get("revision")):
            raise ValueError("each benchmark requires name and immutable revision")
        fields = [str(field) for field in list(benchmark.get("text_fields") or []) if str(field)]
        if not fields:
            raise ValueError(f"benchmark {name!r} requires text_fields")
        for row_index, row in enumerate(_load_local_source(benchmark)):
            text = normalize_governed_text("\n".join(str(row.get(field) or "") for field in fields))
            if not text:
                continue
            record_id = f"{name}:{row_index}"
            exact = content_hash(text)
            if not index.has_exact("benchmark_content", exact):
                index.add_exact("benchmark_content", exact, record_id)
            if index.near_policy.enabled:
                index.add_near("benchmark", record_id, index.signature(text))
            count += 1
    return count


def build_clean_release_from_registry(registry: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    if registry.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"unsupported registry format={registry.get('format')!r}")
    pipeline_version = str(registry.get("pipeline_version") or "").strip()
    if not pipeline_version:
        raise ValueError("registry requires pipeline_version")
    intended_use = str(registry.get("intended_use") or "training").strip().lower()
    raw_sources = [source for source in list(registry.get("sources") or []) if bool(source.get("enabled", True))]
    if not raw_sources:
        raise ValueError("registry requires at least one source")
    sources = [_validate_source(source, intended_use) for source in raw_sources]
    names = [source["source_name"] for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("registry source_name values must be unique")
    near_policy = _NearPolicy.from_mapping(registry.get("near_duplicate"))
    quality_policy = dict(registry.get("quality_policy") or {})
    execution = dict(registry.get("execution") or {})
    workers = int(execution.get("workers", 1))
    worker_chunk_size = int(execution.get("chunk_size", 64))
    if workers <= 0 or workers > 256:
        raise ValueError("execution.workers must be in [1, 256]")
    if worker_chunk_size <= 0:
        raise ValueError("execution.chunk_size must be > 0")
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing clean release: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))).resolve()
    index = _GovernanceIndex(temporary / ".governance.sqlite3", near_policy)
    active_executor: ProcessPoolExecutor | None = None
    accepted_count = 0
    quarantine_count = 0
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    plane_counts: Counter[str] = Counter()
    try:
        benchmark_count = _load_benchmarks(registry, index)
        with (temporary / ACCEPTED_FILENAME).open("w", encoding="utf-8", newline="\n") as accepted_handle, (
            temporary / QUARANTINE_FILENAME
        ).open("w", encoding="utf-8", newline="\n") as quarantine_handle:
            for source in sources:
                source_name = source["source_name"]
                source_counts.setdefault(source_name, Counter())
                source_policy = {**quality_policy, **dict(source.get("quality_policy") or {})}
                jobs = (
                    (dict(row), source, row_index, pipeline_version, source_policy)
                    for row_index, row in enumerate(_load_local_source(source))
                )
                if workers > 1:
                    active_executor = ProcessPoolExecutor(max_workers=workers)
                    prepared_rows = active_executor.map(_prepare_source_row, jobs, chunksize=worker_chunk_size)
                else:
                    prepared_rows = map(_prepare_source_row, jobs)
                for row_index, source_row_id, record, reason, details in prepared_rows:
                    source_counts[source_name]["scanned"] += 1
                    if record is not None:
                        source_identity_hash = content_hash(
                            {key: record[key] for key in ("source_name", "source_revision", "source_split", "source_row_id")}
                        )
                        if reason is None and index.has_exact("source_identity", source_identity_hash):
                            reason, details = "duplicate_source_identity", {"source_identity_hash": source_identity_hash}
                        raw_kind = f"raw_content:{record['data_plane']}"
                        normalized_kind = f"normalized_content:{record['data_plane']}"
                        if reason is None and index.has_exact(raw_kind, record["raw_content_hash"]):
                            reason, details = "duplicate_raw_content", {"raw_content_hash": record["raw_content_hash"]}
                        if reason is None and index.has_exact(normalized_kind, record["normalized_content_hash"]):
                            reason, details = "duplicate_normalized_content", {"normalized_content_hash": record["normalized_content_hash"]}
                        contamination_hash = content_hash(record["contamination_text"])
                        benchmark_match = index.has_exact("benchmark_content", contamination_hash)
                        if reason is None and benchmark_match:
                            reason, details = "benchmark_exact_contamination", {"benchmark_record_id": benchmark_match}
                        signature = index.signature(record["normalized_content"]) if reason is None and near_policy.enabled else ()
                        contamination_signature = index.signature(record["contamination_text"]) if reason is None and near_policy.enabled else ()
                        if reason is None and near_policy.enabled:
                            near_match = index.find_near(f"accepted:{record['data_plane']}", signature)
                            if near_match:
                                reason, details = "near_duplicate", {"matched_record_id": near_match[0], "estimated_similarity": near_match[1]}
                        if reason is None and near_policy.enabled:
                            benchmark_near = index.find_near("benchmark", contamination_signature)
                            if benchmark_near:
                                reason, details = "benchmark_near_contamination", {
                                    "benchmark_record_id": benchmark_near[0],
                                    "estimated_similarity": benchmark_near[1],
                                }
                        if reason is None:
                            index.add_exact("source_identity", source_identity_hash, record["record_id"])
                            index.add_exact(raw_kind, record["raw_content_hash"], record["record_id"])
                            index.add_exact(normalized_kind, record["normalized_content_hash"], record["record_id"])
                            if near_policy.enabled:
                                index.add_near(f"accepted:{record['data_plane']}", record["record_id"], signature)
                    if reason is None and record is not None:
                        accepted_handle.write(_json_line(record))
                        accepted_count += 1
                        source_counts[source_name]["accepted"] += 1
                        plane_counts[record["data_plane"]] += 1
                    else:
                        reason = str(reason or "invalid_source_record")
                        reason_counts[reason] += 1
                        quarantine_count += 1
                        source_counts[source_name]["quarantine"] += 1
                        quarantine_handle.write(
                            _json_line(
                                {
                                    "source_name": source_name,
                                    "source_revision": source["revision"],
                                    "source_split": source["split"],
                                    "source_row_id": source_row_id,
                                    "reason_code": reason,
                                    "details": details,
                                    "record": record,
                                }
                            )
                        )
                    if (accepted_count + quarantine_count) % 1000 == 0:
                        index.connection.commit()
                if active_executor is not None:
                    active_executor.shutdown(wait=True, cancel_futures=True)
                    active_executor = None
        index.close()
        (temporary / ".governance.sqlite3").unlink(missing_ok=True)
        files = {
            filename: {"sha256": _sha256_file(temporary / filename), "rows": rows}
            for filename, rows in ((ACCEPTED_FILENAME, accepted_count), (QUARANTINE_FILENAME, quarantine_count))
        }
        source_registry = [
            {
                key: source.get(key)
                for key in (
                    "source_name", "kind", "split", "revision", "data_plane", "adapter", "fields", "license",
                    "allowed_uses", "acquired_at", "language", "domain", "strip_html", "quality_policy",
                    "quality_score_fields", "upstream_metadata"
                )
                if source.get(key) not in (None, "", [], {})
            }
            for source in sources
        ]
        manifest = {
            "format": CLEAN_RELEASE_FORMAT,
            "record_format": CLEAN_RECORD_FORMAT,
            "pipeline_version": pipeline_version,
            "intended_use": intended_use,
            "accepted_count": accepted_count,
            "quarantine_count": quarantine_count,
            "reason_counts": dict(sorted(reason_counts.items())),
            "data_plane_counts": dict(sorted(plane_counts.items())),
            "source_counts": {
                name: {key: int(counts.get(key, 0)) for key in ("scanned", "accepted", "quarantine")}
                for name, counts in sorted(source_counts.items())
            },
            "source_registry": source_registry,
            "quality_policy": quality_policy,
            "execution_policy": {"workers": workers, "chunk_size": worker_chunk_size, "ordered_output": True},
            "near_duplicate_policy": {
                "enabled": near_policy.enabled,
                "threshold": near_policy.threshold,
                "num_perm": near_policy.num_perm,
                "bands": near_policy.bands,
                "shingle_size": near_policy.shingle_size,
                "algorithm": "deterministic_minhash_lsh_v1",
            },
            "benchmark_count": benchmark_count,
            "files": files,
        }
        manifest["manifest_fingerprint"] = content_hash(manifest)
        (temporary / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report = validate_clean_release(temporary)
        if not report["ok"]:
            raise RuntimeError(f"clean release validation failed: {report['errors']}")
        os.replace(temporary, target)
        return manifest
    except Exception:
        if active_executor is not None:
            active_executor.shutdown(wait=True, cancel_futures=True)
        try:
            index.close()
        except Exception:
            pass
        if temporary.exists() and temporary.parent == target.parent and temporary.name.startswith(f".{target.name}.tmp-"):
            shutil.rmtree(temporary)
        raise


def validate_clean_release(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {MANIFEST_FILENAME}"], "manifest": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid manifest: {exc}"], "manifest": None}
    errors: list[str] = []
    if manifest.get("format") != CLEAN_RELEASE_FORMAT:
        errors.append(f"unsupported clean release format={manifest.get('format')!r}")
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


def iter_clean_records(release_dir: str | Path) -> Iterator[dict[str, Any]]:
    report = validate_clean_release(release_dir)
    if not report["ok"]:
        raise RuntimeError(f"invalid clean release: {report['errors']}")
    with (Path(release_dir).expanduser().resolve() / ACCEPTED_FILENAME).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("format") != CLEAN_RECORD_FORMAT:
                    raise RuntimeError("invalid clean record format")
                yield value
