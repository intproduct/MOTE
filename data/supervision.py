from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import ReasonCode


@dataclass(frozen=True)
class SupervisionDiagnostics:
    prompt_tokens: int
    target_tokens: int
    total_tokens: int
    supervised_tokens: int
    max_length: int
    overflow_tokens: int
    truncation_action: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupervisionError(ValueError):
    def __init__(self, reason_code: ReasonCode | str, message: str, diagnostics: SupervisionDiagnostics) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))
        self.diagnostics = diagnostics


def validate_supervision(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    max_length: int,
    prompt_tokens: int,
    target_tokens: int,
) -> SupervisionDiagnostics:
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have equal length")
    supervised_tokens = sum(int(label) != -100 for label in labels)
    diagnostics = SupervisionDiagnostics(
        prompt_tokens=int(prompt_tokens),
        target_tokens=int(target_tokens),
        total_tokens=len(input_ids),
        supervised_tokens=supervised_tokens,
        max_length=int(max_length),
        overflow_tokens=max(0, len(input_ids) - int(max_length)),
    )
    if supervised_tokens <= 0:
        raise SupervisionError(ReasonCode.EMPTY_SUPERVISION, "SFT example has no supervised tokens", diagnostics)
    if len(input_ids) > int(max_length):
        raise SupervisionError(
            ReasonCode.OVERLONG,
            f"SFT example has {len(input_ids)} tokens and exceeds max_length={max_length}; refusing silent truncation",
            diagnostics,
        )
    return diagnostics


def diagnostics_from_mapping(value: Mapping[str, Any]) -> SupervisionDiagnostics:
    return SupervisionDiagnostics(**{key: value[key] for key in SupervisionDiagnostics.__dataclass_fields__})
