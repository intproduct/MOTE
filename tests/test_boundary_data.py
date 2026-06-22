from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.rl" not in sys.modules:
    rl_pkg = types.ModuleType("MOTE.rl")
    rl_pkg.__path__ = [str(ROOT / "rl")]
    sys.modules["MOTE.rl"] = rl_pkg

from MOTE.rl.boundary import (
    build_boundary_record,
    build_verified_trace_record,
    is_boundary_prompt,
    summarize_rollout_rewards,
)


def test_boundary_summary_filtering_and_records_are_jsonable():
    rewards = [1.0, 0.0, 1.0, 0.0]
    reward_debugs = [
        {"strict_match": True, "fallback_match": True, "gold_answer": "42"},
        {"strict_match": False, "fallback_match": False, "gold_answer": "42"},
        {"strict_match": False, "fallback_match": True, "gold_answer": "42"},
        {"strict_match": False, "fallback_match": False, "gold_answer": "42"},
    ]
    summary = summarize_rollout_rewards(rewards, reward_debugs, [10.0, 20.0, 30.0, 40.0])
    assert summary["p_correct"] == 0.5
    assert summary["num_correct"] == 2
    assert summary["num_rollouts"] == 4
    assert summary["avg_response_len"] == 25.0
    assert summary["strict_acc"] == 0.25
    assert summary["fallback_acc"] == 0.5
    assert is_boundary_prompt(summary["p_correct"], 0.25, 0.75)
    assert not is_boundary_prompt(summary["p_correct"], 0.6, 0.75)

    row = {"idx": 12, "question": "q", "answer": "#### 42", "source": "gsm8k_train"}
    boundary_record = build_boundary_record(row, summary)
    trace_record = build_verified_trace_record(row, "work #### 42", reward_debugs[0], summary)
    assert trace_record["solution"] == "work #### 42"
    assert trace_record["final_answer"] == "42"
    assert trace_record["source"] == "boundary_rollout"
    json.dumps(boundary_record)
    json.dumps(trace_record)


def test_verified_trace_records_are_created_by_reward_one_filter():
    row = {"question": "q", "answer": "#### 7"}
    rewards = [0.0, 1.0, 0.0]
    texts = ["bad", "good #### 7", "also bad"]
    debugs = [
        {"gold_answer": "7", "strict_match": False, "fallback_match": False},
        {"gold_answer": "7", "strict_match": True, "fallback_match": True},
        {"gold_answer": "7", "strict_match": False, "fallback_match": False},
    ]
    summary = summarize_rollout_rewards(rewards, debugs, [3.0, 4.0, 5.0])
    records = [
        build_verified_trace_record(row, text, debug, summary)
        for text, reward, debug in zip(texts, rewards, debugs)
        if reward == 1.0
    ]
    assert len(records) == 1
    assert records[0]["solution"] == "good #### 7"
    assert records[0]["final_answer"] == "7"
