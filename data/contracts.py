from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping

from .text_normalization import normalize_inline_text, normalize_text


CONTRACT_VERSION = "fitmotn_canonical_sample_v1"


class ReasonCode(str, Enum):
    MISSING_SOURCE_IDENTITY = "missing_source_identity"
    MISSING_CONTENT = "missing_content"
    INVALID_SCHEMA = "invalid_schema"
    DUPLICATE_SOURCE_IDENTITY = "duplicate_source_identity"
    EMPTY_SUPERVISION = "empty_supervision"
    OVERLONG = "overlong"
    TOKENIZATION_ERROR = "tokenization_error"
    MANIFEST_MISMATCH = "manifest_mismatch"


class DataContractError(ValueError):
    def __init__(self, reason_code: ReasonCode | str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))
        self.details = dict(details or {})


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SourceIdentity:
    source_name: str
    source_revision: str
    source_split: str
    source_row_id: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SourceIdentity":
        values = {
            "source_name": normalize_inline_text(row.get("source_name")),
            "source_revision": normalize_inline_text(row.get("source_revision")),
            "source_split": normalize_inline_text(row.get("source_split")),
            "source_row_id": normalize_inline_text(row.get("source_row_id")),
        }
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise DataContractError(
                ReasonCode.MISSING_SOURCE_IDENTITY,
                f"source identity is missing required field(s): {missing}",
                details={"missing_fields": missing},
            )
        return cls(**values)

    @property
    def source_sample_id(self) -> str:
        return content_hash(asdict(self))


@dataclass
class CanonicalSample:
    identity: SourceIdentity
    data_plane: str
    task_type: str
    pipeline_version: str
    prompt: str = ""
    target: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_hash: str = ""
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, pipeline_version: str) -> "CanonicalSample":
        identity = SourceIdentity.from_mapping(row)
        prompt = normalize_text(row.get("prompt"))
        target = normalize_text(row.get("target"))
        raw_messages = row.get("messages") or []
        messages: list[dict[str, str]] = []
        if raw_messages:
            if not isinstance(raw_messages, list):
                raise DataContractError(ReasonCode.INVALID_SCHEMA, "messages must be a list")
            for message in raw_messages:
                if not isinstance(message, Mapping):
                    raise DataContractError(ReasonCode.INVALID_SCHEMA, "each message must be a mapping")
                role = normalize_inline_text(message.get("role"))
                content = normalize_text(message.get("content"))
                if role not in {"system", "user", "assistant", "tool"} or not content:
                    raise DataContractError(ReasonCode.INVALID_SCHEMA, "message role/content is invalid")
                messages.append({"role": role, "content": content})
        if not ((prompt and target) or messages):
            raise DataContractError(
                ReasonCode.MISSING_CONTENT,
                "canonical SFT sample requires prompt/target or messages",
            )
        return cls(
            identity=identity,
            data_plane=normalize_inline_text(row.get("data_plane")) or "sft",
            task_type=normalize_inline_text(row.get("task_type")) or "single_turn",
            pipeline_version=str(pipeline_version),
            prompt=prompt,
            target=target,
            messages=messages,
            metadata=dict(row.get("metadata") or {}),
            raw_hash=content_hash(
                {"prompt": row.get("prompt"), "target": row.get("target"), "messages": row.get("messages") or []}
            ),
        )

    @property
    def raw_content_hash(self) -> str:
        return self.raw_hash

    @property
    def canonical_content_hash(self) -> str:
        return content_hash({"prompt": self.prompt, "target": self.target, "messages": self.messages})

    @property
    def derived_sample_id(self) -> str:
        return content_hash(
            {
                "source_sample_id": self.identity.source_sample_id,
                "pipeline_version": self.pipeline_version,
                "canonical_content_hash": self.canonical_content_hash,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            **asdict(self.identity),
            "source_sample_id": self.identity.source_sample_id,
            "raw_content_hash": self.raw_content_hash,
            "canonical_content_hash": self.canonical_content_hash,
            "derived_sample_id": self.derived_sample_id,
            "data_plane": self.data_plane,
            "task_type": self.task_type,
            "pipeline_version": self.pipeline_version,
            "prompt": self.prompt,
            "target": self.target,
            "messages": self.messages,
            "metadata": self.metadata,
        }
