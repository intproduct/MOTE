from __future__ import annotations

import json
import random
import tempfile
import types
import unittest
from pathlib import Path

import torch

from MOTE.checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    RL_TRAINING_STATE_NAME,
    cleanup_checkpoint_transactions,
    load_checkpoint_manifest,
    prune_checkpoints,
    transactional_save_checkpoint,
    transactional_update_checkpoint,
    validate_checkpoint,
    write_checkpoint_manifest,
)
from MOTE.train.rl_controller import (
    _capture_rl_training_state,
    _load_rl_training_state,
    _restore_rl_rng_state,
    _save_rl_model_artifacts,
)
from MOTE.rl.runtime import RLPolicyLoadInfo


def _write_sft_checkpoint(path: Path, *, step: int, boundary: bool = False) -> None:
    path.mkdir(parents=True)
    torch.save({"patch_state_dict": {}}, path / "fitmotn_state.pt")
    torch.save({}, path / "optimizer.pt")
    torch.save({}, path / "scheduler.pt")
    torch.save({}, path / "rng_state.pth")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    write_checkpoint_manifest(
        path,
        checkpoint_kind="sft",
        update_step=step,
        stage_name="stage_a_recover" if boundary else "stage_b_taskaware",
        stage_boundary=boundary,
    )


class CheckpointReliabilityTests(unittest.TestCase):
    def test_transactional_save_commits_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint-7"

            def writer(path: Path) -> None:
                torch.save({"patch_state_dict": {}}, path / "fitmotn_state.pt")
                torch.save(
                    {
                        "format": "fitmotn_rl_training_state_v1",
                        "optimizer_state_dict": {},
                        "update_step": 7,
                        "micro_step": 7,
                        "optimizer_micro_step": 7,
                        "data_pos": 0,
                        "python_random_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "reference_source": "/models/reference",
                    },
                    path / RL_TRAINING_STATE_NAME,
                )

            transactional_save_checkpoint(
                target,
                writer,
                checkpoint_kind="rl",
                update_step=7,
            )
            result = validate_checkpoint(target, require_exact_resume="rl")

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["manifest"]["update_step"], 7)
            self.assertTrue(result["capabilities"]["rl_exact_resume"])
            self.assertFalse(any(Path(tmpdir).glob(".checkpoint-*.tmp-*")))

    def test_failed_transaction_never_publishes_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint-3"

            def writer(path: Path) -> None:
                (path / "partial.bin").write_bytes(b"partial")
                raise OSError("simulated disk failure")

            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                transactional_save_checkpoint(target, writer, checkpoint_kind="sft", update_step=3)

            self.assertFalse(target.exists())
            self.assertFalse(any(Path(tmpdir).glob(".checkpoint-*.tmp-*")))

    def test_validation_rejects_incomplete_sampler_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint-2"

            def writer(path: Path) -> None:
                torch.save({"patch_state_dict": {}}, path / "fitmotn_state.pt")
                torch.save(
                    {
                        "format": "fitmotn_rl_training_state_v1",
                        "optimizer_state_dict": {},
                        "update_step": 2,
                        "micro_step": 2,
                        "optimizer_micro_step": 2,
                        "data_pos": 2,
                        "python_random_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "reference_source": "/models/reference",
                        "sampler_state": {"format": "fitmotn_rl_sampler_state_v1"},
                    },
                    path / RL_TRAINING_STATE_NAME,
                )

            transactional_save_checkpoint(target, writer, checkpoint_kind="rl", update_step=2)
            result = validate_checkpoint(target, require_exact_resume="rl")

            self.assertFalse(result["ok"])
            self.assertTrue(any("incomplete RL sampler state" in error for error in result["errors"]))

    def test_failed_transactional_update_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "final_model"

            def initial_writer(path: Path) -> None:
                torch.save({"patch_state_dict": {}}, path / "fitmotn_state.pt")
                (path / "stable.txt").write_text("stable", encoding="utf-8")

            transactional_save_checkpoint(target, initial_writer, checkpoint_kind="sft_final", update_step=2)
            original_manifest = (target / CHECKPOINT_MANIFEST_NAME).read_bytes()

            def broken_update(path: Path) -> None:
                (path / "stable.txt").write_text("partial", encoding="utf-8")
                raise OSError("simulated summary failure")

            with self.assertRaisesRegex(OSError, "simulated summary failure"):
                transactional_update_checkpoint(
                    target,
                    broken_update,
                    checkpoint_kind="sft_final",
                    update_step=2,
                )

            self.assertEqual((target / "stable.txt").read_text(encoding="utf-8"), "stable")
            self.assertEqual((target / CHECKPOINT_MANIFEST_NAME).read_bytes(), original_manifest)
            self.assertTrue(validate_checkpoint(target)["ok"])

    def test_cleanup_restores_orphaned_replacement_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            backup = root / ".checkpoint-final_model.backup-deadprocess"
            backup.mkdir()
            (backup / "stable.txt").write_text("stable", encoding="utf-8")

            cleanup_checkpoint_transactions(root)

            self.assertFalse(backup.exists())
            self.assertEqual((root / "final_model" / "stable.txt").read_text(encoding="utf-8"), "stable")

    def test_validation_detects_corruption_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint-1"
            _write_sft_checkpoint(target, step=1)
            (target / "optimizer.pt").write_bytes(b"corrupt")
            result = validate_checkpoint(target)
            self.assertFalse(result["ok"])
            self.assertTrue(any("optimizer.pt" in error for error in result["errors"]))

            manifest = load_checkpoint_manifest(target)
            manifest["files"].append({"path": "../outside", "size_bytes": 0, "sha256": None})
            (target / CHECKPOINT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
            escaped = validate_checkpoint(target, verify_hashes=False)
            self.assertFalse(escaped["ok"])
            self.assertTrue(any("unsafe checkpoint file path" in error for error in escaped["errors"]))

    def test_retention_keeps_recent_periodic_and_stage_boundary_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for step in range(1, 7):
                _write_sft_checkpoint(root / f"checkpoint-{step}", step=step, boundary=step == 2)

            result = prune_checkpoints(
                root,
                keep_last_n=2,
                keep_every_n=3,
                preserve_stage_boundaries=True,
            )
            kept = {Path(path).name for path in result["kept"]}

            self.assertEqual(kept, {"checkpoint-2", "checkpoint-3", "checkpoint-5", "checkpoint-6"})
            self.assertFalse((root / "checkpoint-1").exists())
            self.assertFalse(any(root.glob(".checkpoint-*.delete-*")))

    def test_rl_state_restores_optimizer_counters_and_rng(self):
        random.seed(17)
        torch.manual_seed(17)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        state = _capture_rl_training_state(
            optimizer=optimizer,
            update_step=4,
            micro_step=9,
            optimizer_micro_step=8,
            data_pos=13,
            zero_advantage_retry_count=2,
            reference_source="/models/reference",
            sampler_state={"format": "fitmotn_rl_sampler_state_v1", "position": 13},
        )
        expected_python = random.random()
        expected_torch = torch.rand(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            torch.save(state, path / RL_TRAINING_STATE_NAME)
            loaded = _load_rl_training_state(path)

        random.seed(999)
        torch.manual_seed(999)
        _restore_rl_rng_state(loaded)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

        restored_model = torch.nn.Linear(2, 1)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        self.assertEqual(loaded["update_step"], 4)
        self.assertEqual(loaded["data_pos"], 13)
        self.assertEqual(loaded["reference_source"], "/models/reference")
        self.assertEqual(loaded["sampler_state"]["position"], 13)
        self.assertTrue(restored_optimizer.state_dict()["state"])

    def test_cpu_tiny_rl_interruption_resume_matches_uninterrupted_training(self):
        def run_updates(model, optimizer, start: int, end: int) -> None:
            for _ in range(start, end):
                features = torch.rand(2, 2)
                target = torch.full((2, 1), random.random())
                loss = torch.nn.functional.mse_loss(model(features), target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        random.seed(31)
        torch.manual_seed(31)
        uninterrupted = torch.nn.Linear(2, 1)
        uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-2)
        run_updates(uninterrupted, uninterrupted_optimizer, 0, 4)

        random.seed(31)
        torch.manual_seed(31)
        interrupted = torch.nn.Linear(2, 1)
        interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1e-2)
        run_updates(interrupted, interrupted_optimizer, 0, 2)
        policy_state = {key: value.detach().clone() for key, value in interrupted.state_dict().items()}
        training_state = _capture_rl_training_state(
            optimizer=interrupted_optimizer,
            update_step=2,
            micro_step=2,
            optimizer_micro_step=2,
            data_pos=2,
            zero_advantage_retry_count=0,
            reference_source="/models/reference",
        )

        random.seed(999)
        torch.manual_seed(999)
        resumed = torch.nn.Linear(2, 1)
        resumed.load_state_dict(policy_state)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-2)
        resumed_optimizer.load_state_dict(training_state["optimizer_state_dict"])
        _restore_rl_rng_state(training_state)
        run_updates(resumed, resumed_optimizer, training_state["update_step"], 4)

        for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_rl_model_save_failure_is_not_silent(self):
        class FailingModel:
            def save_pretrained(self, output_dir):
                raise OSError("simulated model serialization failure")

        class Tokenizer:
            def __init__(self):
                self.saved = False

            def save_pretrained(self, output_dir):
                self.saved = True

        load_info = RLPolicyLoadInfo(
            is_fitmotn=False,
            metadata=None,
            layer_idxs=[],
            patch_cfg={},
            base_model_path="/models/base",
            tokenizer_path="/models/base",
            resolved_dtype=torch.float32,
            loaded_from="/models/base",
        )
        fit_cfg = types.SimpleNamespace(rl=types.SimpleNamespace(mode="gsm8k_grpo"))
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "model.save_pretrained failed"):
                _save_rl_model_artifacts(
                    model=FailingModel(),
                    tokenizer=tokenizer,
                    output_dir=Path(tmpdir) / "checkpoint-1",
                    load_info=load_info,
                    fit_cfg=fit_cfg,
                    update_step=1,
                    checkpoint_name="checkpoint-1",
                    trainable_mode_info={"requested_mode": "patch_only", "effective_mode": "patch_only"},
                    fail_on_model_save_error=True,
                )
        self.assertFalse(tokenizer.saved)

        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "checkpoint-1"
            _save_rl_model_artifacts(
                model=FailingModel(),
                tokenizer=tokenizer,
                output_dir=output_dir,
                load_info=load_info,
                fit_cfg=fit_cfg,
                update_step=1,
                checkpoint_name="checkpoint-1",
                trainable_mode_info={"requested_mode": "patch_only", "effective_mode": "patch_only"},
                fail_on_model_save_error=False,
            )
            status = json.loads((output_dir / "model_save_status.json").read_text(encoding="utf-8"))
        self.assertTrue(tokenizer.saved)
        self.assertFalse(status["ok"])


@unittest.skipUnless(__import__("importlib").util.find_spec("transformers"), "transformers is not installed")
class SFTExactResumeIntegrationTests(unittest.TestCase):
    class TinyOutput:
        def __init__(self, logits):
            self.logits = logits

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.projection = torch.nn.Linear(4, 8)

        def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
            return SFTExactResumeIntegrationTests.TinyOutput(self.projection(self.embedding(input_ids)))

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, index):
            input_ids = torch.tensor([(index % 4) + 1, 2, 3, 4])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids,
            }

    @staticmethod
    def _collate(rows):
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

    def _train(self, output_dir: Path, model, *, resume_from=None):
        from transformers import TrainingArguments

        from MOTE.train.trainer import FitMoTNTrainer

        policy = types.SimpleNamespace(
            checkpoint_fail_on_save_error=True,
            checkpoint_keep_last_n=3,
            checkpoint_keep_every_n=0,
        )
        args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=4,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            save_strategy="steps",
            save_steps=2,
            logging_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            use_cpu=True,
            disable_tqdm=True,
            seed=123,
            data_seed=123,
        )
        trainer = FitMoTNTrainer(
            model=model,
            args=args,
            train_dataset=self.TinyDataset(),
            data_collator=self._collate,
            fitmotn_metadata_builder=lambda checkpoint_name=None: {
                "checkpoint_format": "tiny_test",
                "checkpoint_name": checkpoint_name,
                "patch_state_dict": model.state_dict(),
            },
            checkpoint_policy=policy,
        )
        trainer.train(resume_from_checkpoint=resume_from)

    def test_cpu_interruption_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            torch.manual_seed(7)
            uninterrupted = self.TinyModel()
            self._train(root / "uninterrupted", uninterrupted)
            checkpoint = root / "uninterrupted" / "checkpoint-2"
            self.assertTrue(validate_checkpoint(checkpoint, require_exact_resume="sft")["ok"])

            torch.manual_seed(999)
            resumed = self.TinyModel()
            self._train(root / "resumed", resumed, resume_from=str(checkpoint))

            for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
