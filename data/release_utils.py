"""Shared release primitives; keep serialized records and statistics stable."""
from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from typing import Any


def canonical_json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) + "\n"


def integer_distribution(values: Iterable[int]) -> dict[str, float | int | None]:
    """Summarize lengths using the existing nearest-rank percentile convention."""
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}

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
