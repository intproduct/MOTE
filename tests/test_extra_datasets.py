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

from MOTE.config.loader import load_config_from_json
from MOTE.data.adapters import normalize_chat_like_example, normalize_text_example
from MOTE.data.mixed_iterable import load_dataset_any
from MOTE.data.specs import HFChatTask, HFTextTask
from MOTE.tasks.pretrain_specs import build_pretrain_tasks
from MOTE.tasks.reasoning_specs import build_task_mixture_tasks


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _base_payload(extra_datasets: list[dict]) -> dict:
    return {
        "data": {
            "use_wiki_local": False,
            "use_fineweb": False,
            "use_code": False,
            "use_gsm8k_train": False,
            "use_math_train": False,
            "extra_datasets": extra_datasets,
        }
    }


def test_extra_datasets_register_local_jsonl_text_and_chat(tmp_path):
    text_path = tmp_path / "text.jsonl"
    chat_path = tmp_path / "chat.jsonl"
    _write_jsonl(text_path, [{"text": "hello local text"}])
    _write_jsonl(chat_path, [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}]}])
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            _base_payload(
                [
                    {"name": "local_text", "source": "local_jsonl", "path": str(text_path), "format": "text"},
                    {"name": "local_chat", "source": "jsonl", "path": str(chat_path), "format": "chat_messages"},
                ]
            )
        ),
        encoding="utf-8",
    )

    cfg = load_config_from_json(cfg_path)
    pretrain_tasks = build_pretrain_tasks(cfg.data)
    task_tasks = build_task_mixture_tasks(cfg.data)

    text_task = next(task for task in pretrain_tasks if task.name == "local_text")
    chat_task = next(task for task in task_tasks if task.name == "local_chat")
    assert isinstance(text_task, HFTextTask)
    assert text_task.kind == "jsonl"
    assert isinstance(chat_task, HFChatTask)
    assert chat_task.kind == "jsonl"
    assert normalize_text_example(next(iter(load_dataset_any(text_task.kind, text_task.path, text_task.split))), text_field="text") == "hello local text"
    assert [m["role"] for m in normalize_chat_like_example(next(iter(load_dataset_any(chat_task.kind, chat_task.path, chat_task.split))), chat_task)] == [
        "user",
        "assistant",
    ]


def test_extra_dataset_adapters_cover_sharegpt_prompt_response_and_reasoning(tmp_path):
    sharegpt = HFChatTask(
        name="sharegpt",
        path="unused",
        dataset_format="chat_messages",
        messages_field="conversations",
        role_key="from",
        content_key="value",
    )
    messages = normalize_chat_like_example(
        {"conversations": [{"from": "human", "value": "Question?"}, {"from": "gpt", "value": "Answer."}]},
        sharegpt,
    )
    assert messages == [{"role": "user", "content": "Question?"}, {"role": "assistant", "content": "Answer."}]

    prompt_response = HFChatTask(name="alpaca", path="unused", dataset_format="prompt_response")
    messages = normalize_chat_like_example({"instruction": "Do it", "input": "Now", "output": "Done"}, prompt_response)
    assert messages == [{"role": "user", "content": "Do it\n\nNow"}, {"role": "assistant", "content": "Done"}]

    reasoning = HFChatTask(name="reasoning", path="unused", dataset_format="reasoning_qa", answer_extraction="xml_answer")
    messages = normalize_chat_like_example({"question": "2+2?", "solution": "Compute. <answer>4</answer>"}, reasoning)
    assert messages is not None
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "4" in messages[1]["content"]


def test_extra_dataset_loader_rejects_duplicate_names(tmp_path):
    path = tmp_path / "x.jsonl"
    _write_jsonl(path, [{"text": "x"}])
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            _base_payload(
                [
                    {"name": "dup", "source": "local_jsonl", "path": str(path), "format": "text"},
                    {"name": "dup", "source": "local_jsonl", "path": str(path), "format": "text"},
                ]
            )
        ),
        encoding="utf-8",
    )
    try:
        load_config_from_json(cfg_path)
    except ValueError as exc:
        assert "duplicate name" in str(exc)
    else:
        raise AssertionError("duplicate extra dataset names should fail")
