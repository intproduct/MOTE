from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..tasks.answer_extraction import extract_final_answer, normalize_math_answer


@dataclass
class RewardExample:
    prompt: str
    completion: str
    final_answer: str
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierInput:
    question: str
    predicted_answer: str
    reference_answer: Optional[str] = None
    reasoning_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTraceRecord:
    prompt: str
    completion: str
    extracted_final_answer: str
    normalized_final_answer: str
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_reasoning_trace_record(prompt: str, completion: str, metadata: Mapping[str, Any] | None = None) -> ReasoningTraceRecord:
    final_answer = extract_final_answer(completion)
    return ReasoningTraceRecord(
        prompt=str(prompt),
        completion=str(completion),
        extracted_final_answer=final_answer,
        normalized_final_answer=normalize_math_answer(final_answer),
        dataset_name=None if metadata is None else metadata.get("dataset_name"),
        metadata=dict(metadata or {}),
    )
