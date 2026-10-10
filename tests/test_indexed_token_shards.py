from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fitmotn.data.indexed_token_shards import IndexedTokenShardDataset
from fitmotn.data.mixed_iterable import StageAwareMixedTaskIterableDataset
from fitmotn.data.specs import LocalTokenShardTask


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None


class StageState:
    def __init__(self):
        self.stage = SimpleNamespace(
            name="stage",
            pretrain_ratio=1.0,
            task_ratio=0.0,
            reasoning_focused=False,
            reasoning_boost=1.0,
            task_bucket_mode="flat",
            bucket_ratios={},
        )
        self.plan = SimpleNamespace(stages=[self.stage])
        self.runtime_state = {}

    def current_stage(self):
        return self.stage


def _data_cfg():
    return SimpleNamespace(
        source_sampling_mode="deterministic_auto",
        source_shuffle=False,
        source_max_epochs=1,
        reasoning_format="raw",
        reasoning_chat_enable_thinking=False,
        reasoning_chat_system_prompt=None,
        reasoning_chat_use_generation_prompt_for_labels=True,
        fail_on_dynamic_skip=True,
        rejection_quarantine_dir=None,
    )


def test_indexed_release_preserves_labels_and_numeric_metadata(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    torch.save(
        {
            "input_ids": torch.tensor([[10, 11, 12, 2], [20, 21, 22, 2]], dtype=torch.int32),
            "labels": torch.tensor([[-100, -100, 12, 2], [20, 21, 22, 2]], dtype=torch.int32),
            "source_id": torch.tensor([7, 8], dtype=torch.int16),
            "is_document_end": torch.tensor([1, 1], dtype=torch.uint8),
            "sequence_length": 4,
            "source": "demo",
        },
        release / "shard-00000.pt",
    )

    raw = IndexedTokenShardDataset(release)
    assert raw[0]["labels"] == [-100, -100, 12, 2]
    assert raw[0]["source_id"] == 7
    assert raw[1]["is_document_end"] == 1

    task = LocalTokenShardTask(name="v3", path=str(release), weight=1.0)
    dataset = StageAwareMixedTaskIterableDataset(
        tokenizer=TinyTokenizer(),
        pretrain_tasks=[task],
        task_tasks=[],
        stage_state=StageState(),
        max_len=4,
        samples_per_epoch=2,
        seed=1,
        data_cfg=_data_cfg(),
        train_cfg=SimpleNamespace(final_answer_weight_enabled=False),
    )
    rows = list(dataset)
    assert rows[0]["labels"].tolist() == [-100, -100, 12, 2]
    assert rows[0]["eval_type"] == "pretokenized_supervised"
    assert rows[0]["source_id"] == 7


def test_indexed_release_rejects_misaligned_labels(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    torch.save(
        {
            "input_ids": torch.ones((2, 4), dtype=torch.int32),
            "labels": torch.ones((2, 3), dtype=torch.int32),
        },
        release / "shard-00000.pt",
    )
    with pytest.raises(ValueError, match="labels shape mismatch"):
        IndexedTokenShardDataset(release)
