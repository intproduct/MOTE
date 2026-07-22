from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .mgpo import compute_mgpo_weights


@dataclass
class RolloutAuditAccumulator:
    total: int = 0
    reward_sum: float = 0.0
    match_counts: Counter = field(default_factory=Counter)
    length_counts: Counter = field(default_factory=Counter)
    length_sum: int = 0
    length_max: int = 0
    truncated_count: int = 0
    correct_length_sum: int = 0
    correct_count: int = 0
    incorrect_length_sum: int = 0
    incorrect_count: int = 0

    def add(self, item: Mapping[str, Any]) -> None:
        reward = float(item.get("reward", 0.0))
        length = int(item.get("response_token_count", 0))
        self.total += 1
        self.reward_sum += reward
        self.match_counts[str(item.get("reward_match_type", "invalid"))] += 1
        self.length_counts[length] += 1
        self.length_sum += length
        self.length_max = max(self.length_max, length)
        self.truncated_count += int(bool(item.get("truncated", False)))
        if reward == 1.0:
            self.correct_count += 1
            self.correct_length_sum += length
        else:
            self.incorrect_count += 1
            self.incorrect_length_sum += length

    def _median_length(self) -> float:
        if not self.total:
            return 0.0
        middle_left = (self.total - 1) // 2
        middle_right = self.total // 2
        seen = 0
        values = []
        for length, count in sorted(self.length_counts.items()):
            next_seen = seen + int(count)
            if seen <= middle_left < next_seen:
                values.append(int(length))
            if seen <= middle_right < next_seen:
                values.append(int(length))
            if len(values) == 2:
                break
            seen = next_seen
        return float(sum(values) / len(values)) if values else 0.0

    def diagnostics(self) -> tuple[dict[str, Any], dict[str, Any]]:
        reward = {
            "reward_accuracy": float(self.reward_sum / self.total) if self.total else 0.0,
            "strict_match_rate": float(self.match_counts["strict_hash_match"] / self.total) if self.total else 0.0,
            "fallback_only_rate": float(self.match_counts["fallback_last_number_only"] / self.total) if self.total else 0.0,
            "unparseable_rate": float(self.match_counts["unparseable"] / self.total) if self.total else 0.0,
            "reward_match_type_counts": dict(sorted(self.match_counts.items())),
        }
        length = {
            "average_response_tokens": float(self.length_sum / self.total) if self.total else 0.0,
            "median_response_tokens": self._median_length(),
            "maximum_response_tokens": self.length_max,
            "truncated_rollout_count": self.truncated_count,
            "truncated_rollout_rate": float(self.truncated_count / self.total) if self.total else 0.0,
            "average_correct_response_tokens": (
                float(self.correct_length_sum / self.correct_count) if self.correct_count else 0.0
            ),
            "average_incorrect_response_tokens": (
                float(self.incorrect_length_sum / self.incorrect_count) if self.incorrect_count else 0.0
            ),
        }
        return reward, length


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


def pass_at_k_estimate(num_rollouts: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator for a finite set of sampled completions."""
    n, c, k = int(num_rollouts), int(num_correct), int(k)
    if n <= 0:
        raise ValueError("num_rollouts must be > 0")
    if c < 0 or c > n:
        raise ValueError("num_correct must be in [0, num_rollouts]")
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, num_rollouts]")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - math.comb(n - c, k) / math.comb(n, k))


def classify_prompt(
    num_correct: int,
    num_rollouts: int,
    *,
    boundary_min: float,
    boundary_max: float,
) -> dict[str, Any]:
    n, c = int(num_rollouts), int(num_correct)
    if n <= 0 or c < 0 or c > n:
        raise ValueError("invalid correct/rollout counts")
    p = float(c / n)
    mixed = 0 < c < n
    return {
        "all_wrong": c == 0,
        "all_correct": c == n,
        "mixed": mixed,
        "boundary": is_boundary_prompt(p, boundary_min, boundary_max),
        "effective_rl_prompt": mixed,
        "zero_advantage_prompt": not mixed,
    }


def reward_match_type(debug: Mapping[str, Any], reward: float) -> str:
    """Return an exclusive match type without changing reward semantics."""
    explicit = str(debug.get("match_type", "") or "")
    aliases = {
        "strict_hash": "strict_hash_match",
        "final_answer": "final_answer_match",
        "answer_marker": "answer_marker_match",
        "fallback_last_number": "fallback_last_number_only",
        "none": "unparseable" if debug.get("pred_answer") is None else "no_match",
    }
    if explicit in aliases:
        return aliases[explicit]
    if debug.get("strict_match"):
        return "strict_hash_match"
    if debug.get("final_answer_match"):
        return "final_answer_match"
    if debug.get("answer_marker_match"):
        return "answer_marker_match"
    if float(reward) == 1.0 and debug.get("fallback_match"):
        return "fallback_last_number_only"
    return "unparseable" if debug.get("pred_answer") is None else "no_match"


def build_prompt_audit_record(
    row: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
    *,
    pass_k: Sequence[int],
    boundary_min: float,
    boundary_max: float,
    mgpo_cfg: Any | None = None,
) -> dict[str, Any]:
    n = len(rollouts)
    rewards = [float(item.get("reward", 0.0)) for item in rollouts]
    c = sum(value == 1.0 for value in rewards)
    token_counts = [int(item.get("response_token_count", 0)) for item in rollouts]
    truncated = [bool(item.get("truncated", False)) for item in rollouts]
    record = {
        "idx": row.get("idx"),
        "question": str(row.get("question", "")),
        "gold_answer": str(row.get("answer", "")),
        "answer": str(row.get("answer", "")),
        "source": str(row.get("source", "gsm8k_train")),
        "num_rollouts": n,
        "num_correct": int(c),
        "p_correct": float(c / n) if n else 0.0,
        **classify_prompt(c, n, boundary_min=boundary_min, boundary_max=boundary_max),
        "average_response_tokens": _mean(token_counts),
        "avg_response_len": _mean(token_counts),
        "maximum_response_tokens": max(token_counts, default=0),
        "truncated_rollout_count": sum(truncated),
        "truncated_rollout_ratio": _mean([float(value) for value in truncated]),
    }
    debugs = [dict(item.get("reward_debug") or {}) for item in rollouts]
    record["strict_acc"] = _mean([1.0 if item.get("strict_match") else 0.0 for item in debugs])
    record["fallback_acc"] = _mean([1.0 if item.get("fallback_match") else 0.0 for item in debugs])
    for k in pass_k:
        record[f"pass_at_{int(k)}"] = pass_at_k_estimate(n, c, int(k))
    if mgpo_cfg is not None and bool(getattr(mgpo_cfg, "mgpo_enabled", False)):
        weight = compute_mgpo_weights(
            torch.tensor([record["p_correct"]]),
            p0=float(getattr(mgpo_cfg, "mgpo_p0", 0.5)),
            gamma=float(getattr(mgpo_cfg, "mgpo_gamma", 2.0)),
            weight_min=float(getattr(mgpo_cfg, "mgpo_weight_min", 0.1)),
            weight_max=float(getattr(mgpo_cfg, "mgpo_weight_max", 1.0)),
            eps=float(getattr(mgpo_cfg, "mgpo_eps", 1e-6)),
        )
        record["mgpo_weight"] = float(weight.item())
    else:
        record["mgpo_weight"] = None
    return record


def aggregate_spectrum(
    prompt_records: Sequence[Mapping[str, Any]],
    rollout_records: Sequence[Mapping[str, Any]] | RolloutAuditAccumulator,
    *,
    pass_k: Sequence[int],
    mgpo_enabled: bool,
) -> dict[str, Any]:
    total_prompts = len(prompt_records)
    accuracies = [float(item["p_correct"]) for item in prompt_records]
    spectrum: dict[str, Any] = {
        "sampled_pass_at_1": _mean(accuracies),
        "mean_prompt_accuracy": _mean(accuracies),
        "prompt_accuracy_std": float(statistics.pstdev(accuracies)) if accuracies else 0.0,
        "correct_count_histogram": dict(sorted(Counter(str(int(item["num_correct"])) for item in prompt_records).items(), key=lambda x: int(x[0]))),
    }
    for k in pass_k:
        spectrum[f"pass_at_{int(k)}"] = _mean([float(item[f"pass_at_{int(k)}"]) for item in prompt_records])
    for name in ("all_wrong", "all_correct", "mixed", "effective_rl_prompt", "zero_advantage_prompt", "boundary"):
        count = sum(bool(item.get(name)) for item in prompt_records)
        spectrum[f"{name}_count"] = count
        spectrum[f"{name}_ratio"] = float(count / total_prompts) if total_prompts else 0.0

    if isinstance(rollout_records, RolloutAuditAccumulator):
        reward_diagnostics, length_diagnostics = rollout_records.diagnostics()
    else:
        accumulator = RolloutAuditAccumulator()
        for item in rollout_records:
            accumulator.add(item)
        reward_diagnostics, length_diagnostics = accumulator.diagnostics()
    weights = [float(item["mgpo_weight"]) for item in prompt_records if item.get("mgpo_weight") is not None]
    effective_weights = [float(item["mgpo_weight"]) for item in prompt_records if item.get("mgpo_weight") is not None and item.get("effective_rl_prompt")]
    mgpo_diagnostics: dict[str, Any] = {"available": bool(mgpo_enabled)}
    if weights:
        ordered = sorted(weights)
        def quantile(q: float) -> float:
            return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
        mgpo_diagnostics.update({
            "mgpo_weight_mean_all_prompts": _mean(weights),
            "mgpo_weight_mean_effective_prompts": _mean(effective_weights),
            "mgpo_weight_min": min(weights),
            "mgpo_weight_max": max(weights),
            "mgpo_weight_quantiles": {"p0": quantile(0), "p25": quantile(.25), "p50": quantile(.5), "p75": quantile(.75), "p100": quantile(1)},
        })
    return {
        "spectrum": spectrum,
        "reward_diagnostics": reward_diagnostics,
        "length_diagnostics": length_diagnostics,
        "mgpo_diagnostics": mgpo_diagnostics,
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
