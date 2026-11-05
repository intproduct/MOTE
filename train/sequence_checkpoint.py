"""Coordinated DDP checkpoint publication using the existing transaction format."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import torch.distributed as dist
from transformers import Trainer

from ..checkpointing import commit_prepared_checkpoint, validate_checkpoint


def _signature(trainer):
    signature = trainer.sequence_parallel.signature()
    fields = ("per_device_train_batch_size", "gradient_accumulation_steps", "seed", "max_steps",
              "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps", "warmup_ratio",
              "lr_scheduler_type", "optim", "fp16", "bf16")
    signature["training"] = {name: getattr(trainer.args, name) for name in fields}
    signature["dataset_length"] = len(trainer.train_dataset)
    signature["dataset_fingerprint"] = trainer.train_dataset.fingerprint
    return signature


def _collect_errors(error):
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"Ulysses checkpoint failed; staging retained for inspection: {errors}")


def save_checkpoint(trainer, model, trial):
    if trial is not None:
        raise ValueError("Ulysses checkpointing does not support hyperparameter search trials")
    rank = dist.get_rank()
    root = Path(trainer.args.output_dir).resolve()
    name = f"checkpoint-{trainer.state.global_step}"
    token = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(token, src=0)
    staging = root / f".checkpoint-staging.tmp-{token[0]}"
    prepared = root / f".checkpoint-{name}.tmp-{token[0]}"
    error = None
    if rank == 0:
        try:
            staging.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error = repr(exc)
    _collect_errors(error)
    old_output, old_limit = trainer.args.output_dir, trainer.args.save_total_limit
    try:
        trainer.args.output_dir = str(staging)
        trainer.args.save_total_limit = None
        error = None
        try:
            # Every rank writes its RNG; HF writes model/optimizer/state on rank 0.
            Trainer._save_checkpoint(trainer, model, trial)
        except Exception as exc:
            error = repr(exc)
        _collect_errors(error)
        error = None
        if rank == 0:
            try:
                checkpoint = staging / name
                for index in range(dist.get_world_size()):
                    if not (checkpoint / f"rng_state_{index}.pth").is_file():
                        raise RuntimeError(f"Missing RNG state for rank {index}")
                signature = _signature(trainer)
                (checkpoint / "sequence_parallel.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
                checkpoint.replace(prepared)
                commit_prepared_checkpoint(prepared, root / name, checkpoint_kind="sft_ulysses",
                                           update_step=int(trainer.state.global_step), extra=signature)
                staging.rmdir()
            except Exception as exc:
                error = repr(exc)
        _collect_errors(error)
    finally:
        trainer.args.output_dir, trainer.args.save_total_limit = old_output, old_limit
    # Rotation only after all ranks observe a committed checkpoint.
    error = None
    if rank == 0:
        try:
            trainer._rotate_checkpoints(use_mtime=False, output_dir=str(root))
        except Exception as exc:
            error = repr(exc)
    _collect_errors(error)


def validate_resume(trainer, checkpoint):
    root = Path(checkpoint)
    report = validate_checkpoint(root, verify_hashes=True)
    if not report["ok"] or report["manifest"] is None:
        raise ValueError(f"Ulysses resume requires a committed checkpoint: {report['errors']}")
    actual = json.loads((root / "sequence_parallel.json").read_text(encoding="utf-8"))
    expected = _signature(trainer)
    if actual != expected:
        raise ValueError(f"Ulysses resume protocol changed: saved={actual}, requested={expected}")
    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json"]
    required += [f"rng_state_{i}.pth" for i in range(dist.get_world_size())]
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Ulysses checkpoint is missing optimizer/scheduler/rank RNG state")
