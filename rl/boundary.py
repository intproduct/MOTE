from __future__ import annotations

from typing import Any, Mapping, Sequence


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / max(1, len(values)))


def summarize_rollout_rewards(
    rewards: Sequence[float],
    reward_debugs: Sequence[Mapping[str, Any]],
    response_lens: Sequence[float],
) -> dict[str, Any]:
    num_rollouts = int(len(rewards))
    num_correct = int(sum(1 for reward in rewards if float(reward) == 1.0))
    p_correct = float(num_correct / max(1, num_rollouts))
    strict_acc = _mean([1.0 if debug.get("strict_match") else 0.0 for debug in reward_debugs])
    fallback_acc = _mean([1.0 if debug.get("fallback_match") else 0.0 for debug in reward_debugs])
    return {
        "p_correct": p_correct,
        "num_correct": num_correct,
        "num_rollouts": num_rollouts,
        "avg_response_len": _mean([float(value) for value in response_lens]),
        "strict_acc": strict_acc,
        "fallback_acc": fallback_acc,
    }


def is_boundary_prompt(p_correct: float, min_correct_rate: float, max_correct_rate: float) -> bool:
    return float(min_correct_rate) <= float(p_correct) <= float(max_correct_rate)


def build_boundary_record(row: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    idx = row.get("idx")
    try:
        idx = int(idx)
    except Exception:
        idx = None if idx is None else str(idx)
    return {
        "idx": idx,
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "p_correct": float(summary.get("p_correct", 0.0)),
        "num_correct": int(summary.get("num_correct", 0)),
        "num_rollouts": int(summary.get("num_rollouts", 0)),
        "avg_response_len": float(summary.get("avg_response_len", 0.0)),
        "strict_acc": float(summary.get("strict_acc", 0.0)),
        "fallback_acc": float(summary.get("fallback_acc", 0.0)),
        "source": str(row.get("source", "gsm8k_train")),
    }


def build_verified_trace_record(
    row: Mapping[str, Any],
    generated_text: str,
    reward_debug: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    final_answer = reward_debug.get("gold_answer") or reward_debug.get("pred_answer") or reward_debug.get("final_answer") or ""
    return {
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "solution": str(generated_text),
        "final_answer": str(final_answer),
        "p_correct": float(summary.get("p_correct", 0.0)),
        "num_rollouts": int(summary.get("num_rollouts", 0)),
        "source": "boundary_rollout",
    }
