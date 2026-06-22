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
if "MOTE.data" not in sys.modules:
    data_pkg = types.ModuleType("MOTE.data")
    data_pkg.__path__ = [str(ROOT / "data")]
    sys.modules["MOTE.data"] = data_pkg
if "MOTE.tasks" not in sys.modules:
    tasks_pkg = types.ModuleType("MOTE.tasks")
    tasks_pkg.__path__ = [str(ROOT / "tasks")]
    sys.modules["MOTE.tasks"] = tasks_pkg

from MOTE.config.defaults import make_default_config
from MOTE.data.mixed_iterable import load_dataset_any
from MOTE.tasks.reasoning_specs import NormalizedReasoningTask, build_task_mixture_tasks


def test_custom_verified_reasoning_jsonl_loads_and_maps_full_trace(tmp_path):
    path = tmp_path / "verified_traces.jsonl"
    row = {
        "question": "What is 6 * 7?",
        "answer": "#### 42",
        "solution": "6 * 7 = 42. #### 42",
        "final_answer": "42",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = list(load_dataset_any("jsonl", str(path), "train"))
    assert loaded == [row]

    task = NormalizedReasoningTask(
        name="custom_reasoning_jsonl",
        path=str(path),
        split="train",
        weight=2.0,
        dataset_name="custom_verified_math",
        length_policy={"max_chars": 1000, "max_approx_tokens": 250, "prefer_short_reasoning": True},
        supervision_mode="full_trace",
        kind="jsonl",
        group="task",
        bucket="gsm8k_core",
        source_family="reasoning",
    )
    mapped = task.map_example(row)
    assert mapped is not None
    prompt, target, eval_type = mapped
    assert "What is 6 * 7?" in prompt
    assert "Solution:" in target
    assert "Final Answer:" in target
    assert "42" in target
    assert eval_type == "numeric"


def test_custom_reasoning_jsonl_registers_task(tmp_path):
    path = tmp_path / "verified_traces.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "q",
                "answer": "#### 1",
                "solution": "work #### 1",
                "final_answer": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = make_default_config()
    cfg.data.use_gsm8k_train = False
    cfg.data.use_math_train = False
    cfg.data.use_custom_reasoning_jsonl = True
    cfg.data.custom_reasoning_jsonl_path = str(path)
    cfg.data.reasoning_supervision_mode = "full_trace"

    tasks = build_task_mixture_tasks(cfg.data)
    assert [task.name for task in tasks] == ["custom_reasoning_jsonl"]
    assert tasks[0].kind == "jsonl"
