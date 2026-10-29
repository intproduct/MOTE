from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from MOTE.config.loader import load_config_from_json
from MOTE.data.clean_release import build_clean_release_from_registry
from MOTE.data.mixed_iterable import StageAwareMixedTaskIterableDataset
from MOTE.data.specs import FrozenTokenTask
from MOTE.data.training_release import (
    bind_clean_release,
    iter_bound_training_records,
    preflight_bound_training_release,
    validate_bound_training_release,
)
from MOTE.tasks.pretrain_specs import build_pretrain_tasks


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


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _registry(path, plane, adapter, fields):
    return {
        "format": "fitmotn_data_registry_v1",
        "pipeline_version": "clean-v1",
        "sources": [
            {
                "source_name": "demo",
                "kind": "jsonl",
                "path": str(path),
                "split": "train",
                "revision": "fixed-commit",
                "data_plane": plane,
                "adapter": adapter,
                "fields": fields,
                "license": "apache-2.0",
                "allowed_uses": ["training"],
                "acquired_at": "2026-07-29T00:00:00Z",
            }
        ],
    }


def test_bind_pretraining_release_adds_eos_buckets_and_quarantines_overlong(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"text": "short"}, {"text": "this document is much too long"}])
    clean = tmp_path / "clean"
    build_clean_release_from_registry(_registry(source, "pretraining", "text", {"text": "text"}), clean)
    bound = tmp_path / "bound"
    manifest = bind_clean_release(
        clean,
        bound,
        tokenizer=TinyTokenizer(),
        tokenizer_revision="tiny-commit",
        max_length=10,
        selected_planes=["pretraining"],
        length_buckets=[6, 10],
    )
    assert manifest["training_objective"] == "causal_lm"
    assert manifest["accepted_count"] == 1
    assert manifest["quarantine_count"] == 1
    record = next(iter_bound_training_records(bound))
    assert record["input_ids"][-1] == TinyTokenizer.eos_token_id
    assert record["labels"] == record["input_ids"]
    assert record["length_bucket"] == "le_6"
    assert validate_bound_training_release(bound)["ok"] is True
    assert preflight_bound_training_release(bound, min_records=1)["dynamic_tokenization"] is False
    with pytest.raises(RuntimeError, match="at least 2"):
        preflight_bound_training_release(bound, min_records=2)


def test_bind_sft_release_masks_prompt_and_refuses_causal_packing(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"prompt": "Q", "response": "A"}])
    clean = tmp_path / "clean"
    build_clean_release_from_registry(
        _registry(source, "general_sft", "prompt_response", {"prompt": "prompt", "response": "response"}), clean
    )
    bound = tmp_path / "bound"
    manifest = bind_clean_release(
        clean,
        bound,
        tokenizer=TinyTokenizer(),
        tokenizer_revision="tiny-commit",
        max_length=8,
        selected_planes=["general_sft"],
        length_buckets=[4, 8],
    )
    assert manifest["training_objective"] == "sft"
    record = next(iter_bound_training_records(bound))
    assert record["labels"][0] == -100
    assert record["labels"][-1] == TinyTokenizer.eos_token_id
    with pytest.raises(ValueError, match="packing"):
        bind_clean_release(
            clean,
            tmp_path / "packed",
            tokenizer=TinyTokenizer(),
            tokenizer_revision="tiny-commit",
            max_length=8,
            selected_planes=["general_sft"],
            packing_mode="pretrain_greedy",
        )


def test_pretrain_greedy_packing_preserves_member_boundaries(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"text": "aa"}, {"text": "bb"}, {"text": "cc"}])
    clean = tmp_path / "clean"
    build_clean_release_from_registry(_registry(source, "pretraining", "text", {"text": "text"}), clean)
    bound = tmp_path / "bound"
    manifest = bind_clean_release(
        clean,
        bound,
        tokenizer=TinyTokenizer(),
        tokenizer_revision="tiny-commit",
        max_length=6,
        selected_planes=["pretraining"],
        packing_mode="pretrain_greedy",
        tokenization_workers=2,
        prefetch_factor=2,
    )
    records = list(iter_bound_training_records(bound))
    assert manifest["accepted_source_records"] == 3
    assert len(records) == 2
    assert records[0]["packing"]["member_count"] == 2
    assert records[0]["input_ids"].count(TinyTokenizer.eos_token_id) == 2
    assert manifest["tokenization_execution"] == {"workers": 2, "prefetch_factor": 2, "ordered_output": True}


def test_frozen_token_release_can_be_registered_as_pretrain_extra_dataset(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"text": "hello"}])
    clean = tmp_path / "clean"
    build_clean_release_from_registry(_registry(source, "pretraining", "text", {"text": "text"}), clean)
    bound = tmp_path / "bound"
    bind_clean_release(
        clean,
        bound,
        tokenizer=TinyTokenizer(),
        tokenizer_revision="tiny-commit",
        max_length=16,
        selected_planes=["pretraining"],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "use_wiki_local": False,
                    "use_fineweb": False,
                    "use_code": False,
                    "use_gsm8k_train": False,
                    "use_math_train": False,
                    "source_max_epochs": 1,
                    "extra_datasets": [
                        {
                            "name": "frozen_docs",
                            "source": "frozen_release",
                            "path": str(bound),
                            "format": "text",
                            "group": "pretrain",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config_from_json(config_path)
    task = build_pretrain_tasks(cfg.data)[0]
    assert isinstance(task, FrozenTokenTask)
    assert task.kind == "frozen_token_release"
    stage_state = SimpleNamespace(plan=SimpleNamespace(stages=[]))
    dataset = StageAwareMixedTaskIterableDataset(
        TinyTokenizer(), [task], [], stage_state, max_len=16, samples_per_epoch=1, data_cfg=cfg.data
    )
    raw = dataset._load_raw_for_task(task, 0)
    supervised = dataset._to_supervised(task, raw[0])
    assert supervised["input_ids"].tolist()[-1] == TinyTokenizer.eos_token_id
    assert supervised["eval_type"] == "causal_lm"

    class OtherTokenizer(TinyTokenizer):
        name_or_path = "other"

    mismatch = StageAwareMixedTaskIterableDataset(
        OtherTokenizer(), [task], [], stage_state, max_len=16, samples_per_epoch=1, data_cfg=cfg.data
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        mismatch._load_raw_for_task(task, 0)


def test_frozen_token_extra_dataset_requires_exposure_bound(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "use_wiki_local": False,
                    "use_fineweb": False,
                    "use_code": False,
                    "use_gsm8k_train": False,
                    "use_math_train": False,
                    "extra_datasets": [
                        {"name": "frozen", "source": "frozen_release", "path": str(tmp_path / "bound"), "format": "text"}
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exposure bound"):
        load_config_from_json(config_path)
