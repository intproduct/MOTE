from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from MOTE.checkpointing import transactional_save_checkpoint
from MOTE.config.loader import load_config_from_json
from MOTE.train.checkpoint_selector import SELECTOR_FORMAT, select_checkpoint
from MOTE.train.controller import _validate_selected_metadata


def _write_candidate(path: Path, update: int, *, corrupt_restore: bool = False) -> None:
    def writer(prepared: Path) -> None:
        if corrupt_restore:
            (prepared / "fitmotn_state.pt").write_bytes(b"not-a-torch-checkpoint")
        else:
            torch.save({"patch_state_dict": {"model.layers.0.mlp.weight": torch.tensor([float(update)])}}, prepared / "fitmotn_state.pt")

    transactional_save_checkpoint(path, writer, checkpoint_kind="sft", update_step=update)


TASK_CONTRACTS = {
    "mmlu": {"backend": "lm_eval", "protocol": "legacy", "fewshot": 5, "limit": 64, "primary_metric": "acc"},
    "gsm8k": {"backend": "lm_eval", "protocol": "legacy", "fewshot": 8, "limit": 32, "primary_metric": "exact_match"},
}

def _task_result(score: float, contract: dict) -> dict:
    return {
        **contract,
        "primary_score": score,
        "warnings": [],
        "ignored_runtime_args": [],
        "unsupported_runtime_args": [],
        "unsupported_generation_args": [],
    }

def _entry(update: int, mmlu: float, gsm8k: float) -> dict:
    return {
        "update": update,
        "result": {
            "eval_mode": "mid",
            "primary_backend": "lm_eval",
            "warnings": [],
            "tasks": {
                "mmlu": _task_result(mmlu, TASK_CONTRACTS["mmlu"]),
                "gsm8k": _task_result(gsm8k, TASK_CONTRACTS["gsm8k"]),
            },
        },
    }


class CheckpointSelectorTests(unittest.TestCase):
    def test_selected_final_metadata_must_match_selected_update(self):
        metadata = {"updates_done": 20, "global_step": 20}
        self.assertIs(_validate_selected_metadata(metadata, 20), metadata)
        for field in ("updates_done", "global_step"):
            with self.subTest(field=field):
                invalid = dict(metadata)
                invalid[field] = 22
                with self.assertRaisesRegex(RuntimeError, field):
                    _validate_selected_metadata(invalid, 20)

    def test_selects_primary_then_secondary_then_earlier_and_audits_rejections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for update in (10, 20, 30, 40, 50):
                _write_candidate(run_dir / "checkpoints" / f"checkpoint-{update}", update, corrupt_restore=update == 50)
            summary = {
                "mid_evals": [
                    _entry(10, 0.49, 0.99),
                    _entry(20, 0.61, 0.10),
                    _entry(30, 0.61, 0.50),
                    _entry(40, 0.61, 0.50),
                    _entry(50, 0.99, 0.99),
                ]
            }
            result = select_checkpoint(
                run_dir,
                summary,
                candidate_updates=[10, 20, 30, 40, 50, 60],
                primary_floor=0.50,
                probe_restore=True,
                task_contracts=TASK_CONTRACTS,
            )

            self.assertEqual(result["format"], SELECTOR_FORMAT)
            self.assertEqual(result["selected_update"], 30)
            by_update = {item["update"]: item for item in result["candidates"]}
            self.assertTrue(any("below_floor" in reason for reason in by_update[10]["rejection_reasons"]))
            self.assertTrue(any("restore_probe_failed" in reason for reason in by_update[50]["rejection_reasons"]))
            self.assertTrue(any("checkpoint_invalid" in reason for reason in by_update[60]["rejection_reasons"]))
            record = json.loads((run_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(record["selected_update"], 30)

    def test_rejects_protocol_drift_warnings_and_missing_audit_fields(self):
        mutations = (
            ("protocol", "changed"),
            ("fewshot", 0),
            ("limit", 1),
            ("primary_metric", "wrong"),
            ("warnings", ["warning"]),
            ("unsupported_generation_args", ["top_k"]),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                _write_candidate(run_dir / "checkpoints" / "checkpoint-10", 10)
                entry = _entry(10, 0.6, 0.5)
                entry["result"]["tasks"]["mmlu"][field] = value
                with self.assertRaisesRegex(RuntimeError, "protocol contract mismatch|non-empty"):
                    select_checkpoint(
                        run_dir, {"mid_evals": [entry]}, candidate_updates=[10],
                        primary_floor=0.0, task_contracts=TASK_CONTRACTS,
                    )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_candidate(run_dir / "checkpoints" / "checkpoint-10", 10)
            entry = _entry(10, 0.6, 0.5)
            del entry["result"]["tasks"]["mmlu"]["warnings"]
            with self.assertRaisesRegex(RuntimeError, "audit field is missing"):
                select_checkpoint(
                    run_dir, {"mid_evals": [entry]}, candidate_updates=[10],
                    primary_floor=0.0, task_contracts=TASK_CONTRACTS,
                )

    def test_requires_complete_frozen_task_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "selector contract missing fields"):
                select_checkpoint(
                    run_dir, {"mid_evals": []}, candidate_updates=[10],
                    primary_floor=0.0, task_contracts={"mmlu": {}, "gsm8k": {}},
                )

    def test_config_selector_requires_frozen_aligned_candidates_and_floor(self):
        base = {
            "model": {"model_path": "/tmp/model"},
            "data": {
                "tok_shard_dir": "/tmp/data",
                "datas_dir": "/tmp/data",
                "fineweb_cache_path": "/tmp/data",
                "code_cache_path": "/tmp/data",
                "gsm8k_cache_path": "/tmp/data",
                "math_cache_root": "/tmp/data",
            },
            "output": {"root_dir": "/tmp/output"},
            "train": {
                "save_every_updates": 2000,
                "eval_every_updates": 2000,
                "checkpoint_selector_enabled": True,
                "checkpoint_selector_candidates": [24000, 20000, 22000],
                "checkpoint_selector_primary_floor": 0.45,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            cfg = load_config_from_json(path)
            self.assertEqual(cfg.train.checkpoint_selector_candidates, [20000, 22000, 24000])
            base["train"]["checkpoint_selector_candidates"] = [21000]
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "align"):
                load_config_from_json(path)


if __name__ == "__main__":
    unittest.main()
