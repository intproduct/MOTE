from __future__ import annotations

from types import SimpleNamespace

from MOTE.data.mixed_iterable import StageAwareMixedTaskIterableDataset
from MOTE.data.release import build_frozen_sft_release
from MOTE.data.specs import FrozenSFTTask, TaskSpec


class TinyTokenizer:
    name_or_path = "tiny"
    vocab_size = 512
    eos_token = "<eos>"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 10 for char in str(text)]


class EchoTask(TaskSpec):
    def map_example(self, ex):
        return f"Q:{ex['id']}", f"A:{ex['id']}", "numeric"


class RejectTask(TaskSpec):
    def map_example(self, ex):
        self.metadata["last_reason"] = "missing_fields"
        return None


class StageState:
    def __init__(self):
        self.stage = SimpleNamespace(
            name="stage",
            pretrain_ratio=0.0,
            task_ratio=1.0,
            reasoning_focused=False,
            reasoning_boost=1.0,
            task_bucket_mode="flat",
            bucket_ratios={},
        )
        self.plan = SimpleNamespace(stages=[self.stage])
        self.runtime_state = {}

    def current_stage(self):
        return self.stage


def _cfg():
    return SimpleNamespace(
        source_sampling_mode="deterministic_auto",
        source_shuffle=True,
        source_max_epochs=0,
        reasoning_format="raw",
        reasoning_chat_enable_thinking=False,
        reasoning_chat_system_prompt=None,
        reasoning_chat_use_generation_prompt_for_labels=True,
    )


def _train_cfg():
    return SimpleNamespace(final_answer_weight_enabled=False)


def _dataset(task, *, samples=100):
    return StageAwareMixedTaskIterableDataset(
        tokenizer=TinyTokenizer(),
        pretrain_tasks=[],
        task_tasks=[task],
        stage_state=StageState(),
        max_len=64,
        samples_per_epoch=samples,
        seed=13,
        data_cfg=_cfg(),
        train_cfg=_train_cfg(),
    )


def _echo_task(count=20, max_samples=None):
    return EchoTask(
        name="echo",
        path="synthetic",
        kind="synthetic_reasoning",
        group="task",
        bucket="task",
        source_family="reasoning",
        max_samples=max_samples,
        metadata={"synthetic_dataset": [{"id": index} for index in range(count)]},
    )


def test_mixed_dataset_uses_deterministic_non_prefix_source_sampling():
    dataset = _dataset(_echo_task(count=50, max_samples=10), samples=10)
    rows = list(dataset)
    decoded_ids = [tuple(row["input_ids"].tolist()) for row in rows]
    assert len(set(decoded_ids)) == 10
    statistics = dataset.sampling_statistics()["sources"]["echo"]
    assert statistics["total_draws"] == 10
    assert statistics["repeat_count"] == 0
    assert statistics["selected_record_count"] == 10


def test_mixed_dataset_sampling_state_resumes_exact_sequence():
    first = _dataset(_echo_task(), samples=100)
    first_iter = iter(first)
    for _ in range(17):
        next(first_iter)
    state = first.sampling_state_dict()
    expected = [next(first_iter)["input_ids"].tolist() for _ in range(25)]

    resumed = _dataset(_echo_task(), samples=100)
    resumed.load_sampling_state_dict(state)
    resumed_iter = iter(resumed)
    assert [next(resumed_iter)["input_ids"].tolist() for _ in range(25)] == expected

    changed = _dataset(_echo_task(), samples=100)
    changed.data_cfg.source_max_epochs = 1
    with __import__("pytest").raises(RuntimeError, match="policy changed"):
        changed.load_sampling_state_dict(state)


def test_frozen_release_is_consumed_without_runtime_retokenization(tmp_path):
    tokenizer = TinyTokenizer()
    release = tmp_path / "release"
    build_frozen_sft_release(
        [
            {
                "source_name": "demo",
                "source_revision": "r1",
                "source_split": "train",
                "source_row_id": str(index),
                "prompt": f"Q{index}",
                "target": f"A{index}",
            }
            for index in range(5)
        ],
        release,
        tokenizer=tokenizer,
        max_length=64,
        pipeline_version="v1",
    )
    task = FrozenSFTTask(name="frozen", path=str(release), weight=1.0)
    dataset = _dataset(task, samples=5)
    rows = list(dataset)
    assert len(rows) == 5
    assert all(row["eval_type"] == "frozen_sft" for row in rows)
    assert all(row["sample_id"] for row in rows)


def test_dynamic_quality_skip_fails_closed_by_default():
    task = RejectTask(
        name="reject",
        path="synthetic",
        kind="synthetic_reasoning",
        group="task",
        bucket="task",
        metadata={"synthetic_dataset": [{"id": 1}]},
    )
    dataset = _dataset(task, samples=1)
    with __import__("pytest").raises(RuntimeError, match="runtime data rejection"):
        next(iter(dataset))
