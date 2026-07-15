from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer import PREFIX_CHECKPOINT_DIR

from ..ADTN import TensorBlock
from ..checkpointing import (
    cleanup_checkpoint_transactions,
    commit_prepared_checkpoint,
    prune_checkpoints,
    save_fitmotn_metadata,
)
from ..gate import SoftGate, TopKGate, _SoftGate, _TopKGate
from .observability import register_microbatch


ROUTER_GATE_TYPES = (TopKGate, SoftGate, _TopKGate, _SoftGate)
KNOWN_BUCKETS = ("pretrain_general", "gsm8k_core", "aux_reasoning", "unknown_bucket")


def compute_loss_observability(logits, labels, *, buckets=None, tasks=None, loss_weights=None):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = int(shift_logits.shape[-1])
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    valid_tokens = valid.sum().clamp_min(1)
    standard_loss = token_loss.sum() / valid_tokens

    weighted_tokens = None
    final_answer_loss = None
    if loss_weights is not None:
        shift_weights = loss_weights[..., 1:].to(device=token_loss.device, dtype=token_loss.dtype).contiguous()
        shift_weights = torch.where(valid, shift_weights, torch.zeros_like(shift_weights))
        denom = shift_weights.sum()
        if float(denom.detach().cpu()) > 0.0:
            loss = (token_loss * shift_weights).sum() / denom
        else:
            loss = standard_loss
        final_mask = valid & shift_weights.gt(1.0)
        weighted_tokens = int(final_mask.sum().detach().cpu().item())
        if weighted_tokens > 0:
            final_answer_loss = float(token_loss[final_mask].mean().detach().cpu().item())
    else:
        loss = standard_loss
        shift_weights = None
        weighted_tokens = 0

    bucket_losses = {f"loss/{bucket}": None for bucket in KNOWN_BUCKETS}
    bucket_tokens = {f"tokens/{bucket}": 0 for bucket in KNOWN_BUCKETS}
    loss_by_task = {}
    if buckets is not None:
        bucket_names = list(buckets)
        for row_idx, bucket_name in enumerate(bucket_names[: token_loss.shape[0]]):
            key_bucket = str(bucket_name or "unknown_bucket")
            if key_bucket not in KNOWN_BUCKETS:
                key_bucket = "unknown_bucket"
            mask = valid[row_idx]
            count = int(mask.sum().detach().cpu().item())
            if count <= 0:
                continue
            loss_sum = token_loss[row_idx][mask].sum().detach()
            prev_tokens = int(bucket_tokens[f"tokens/{key_bucket}"])
            prev_loss = bucket_losses[f"loss/{key_bucket}"]
            prev_sum = 0.0 if prev_loss is None else float(prev_loss) * prev_tokens
            bucket_tokens[f"tokens/{key_bucket}"] = prev_tokens + count
            bucket_losses[f"loss/{key_bucket}"] = (prev_sum + float(loss_sum.cpu().item())) / (prev_tokens + count)
    if tasks is not None:
        for row_idx, task_name in enumerate(list(tasks)[: token_loss.shape[0]]):
            key_task = str(task_name or "unknown")
            mask = valid[row_idx]
            count = int(mask.sum().detach().cpu().item())
            if count <= 0:
                continue
            loss_sum = float(token_loss[row_idx][mask].sum().detach().cpu().item())
            item = loss_by_task.setdefault(key_task, {"loss_sum": 0.0, "tokens": 0})
            item["loss_sum"] += loss_sum
            item["tokens"] += count
    loss_by_task = {
        f"loss_by_task/{name}": values["loss_sum"] / max(1, values["tokens"])
        for name, values in loss_by_task.items()
    }
    metrics = {}
    metrics.update(bucket_losses)
    metrics.update(bucket_tokens)
    metrics.update(loss_by_task)
    metrics["final_answer_weighted_tokens"] = int(weighted_tokens or 0)
    metrics["final_answer_loss"] = final_answer_loss
    return loss, metrics


class FitMoTNTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_dataset_builder=None,
        fitmotn_metadata_builder=None,
        scheduler_builder=None,
        observability_state=None,
        optimizer_logger=None,
        checkpoint_policy=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_dataset_builder = train_dataset_builder
        self.fitmotn_metadata_builder = fitmotn_metadata_builder
        self.scheduler_builder = scheduler_builder
        self.observability_state = observability_state
        self.optimizer_logger = optimizer_logger or logging.getLogger(__name__)
        self.checkpoint_policy = checkpoint_policy
        if self.observability_state is not None:
            setattr(self.model, "fitmotn_runtime", self.observability_state)

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset if self.train_dataset_builder is None else self.train_dataset_builder()
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def create_optimizer(self):
        if self.optimizer is None:
            param_groups, group_summary, fallback_param_names = self._build_optimizer_param_groups()
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            optimizer_type = type(self.optimizer).__name__
            self._log_optimizer_param_groups(optimizer_type, group_summary, fallback_param_names)
            if self.observability_state is not None:
                self.observability_state["optimizer_type"] = optimizer_type
                self.observability_state["weight_decay"] = getattr(self.args, "weight_decay", None)
                self.observability_state["optimizer_param_group_count"] = len(param_groups)
                self.observability_state["optimizer_param_groups"] = group_summary
                self.observability_state["optimizer_fallback_param_names"] = list(fallback_param_names)
        return self.optimizer

    def _resolve_group_lrs(self) -> tuple[float, float]:
        train_cfg = None
        if self.observability_state is not None:
            fit_cfg = self.observability_state.get("fit_cfg")
            train_cfg = getattr(fit_cfg, "train", None) if fit_cfg is not None else None
        base_lr = float(getattr(self.args, "learning_rate"))
        block_lr = getattr(train_cfg, "block_lr", None)
        router_lr = getattr(train_cfg, "router_lr", None)
        block_lr = base_lr if block_lr is None else float(block_lr)
        router_lr = block_lr if router_lr is None else float(router_lr)
        return block_lr, router_lr

    def _build_optimizer_param_groups(self):
        block_lr, router_lr = self._resolve_group_lrs()
        router_param_ids = set()
        block_param_ids = set()
        for module in self.model.modules():
            if isinstance(module, ROUTER_GATE_TYPES):
                router_param_ids.update(id(p) for p in module.parameters(recurse=True))
            elif isinstance(module, TensorBlock):
                block_param_ids.update(id(p) for p in module.parameters(recurse=True))

        router_params = []
        router_names = []
        block_params = []
        block_names = []
        fallback_param_names = []
        trainable_param_ids = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in trainable_param_ids:
                raise RuntimeError(f"duplicate trainable parameter encountered while building optimizer groups: {name}")
            trainable_param_ids.add(param_id)
            if param_id in router_param_ids:
                router_params.append(param)
                router_names.append(name)
            else:
                block_params.append(param)
                block_names.append(name)
                if param_id not in block_param_ids:
                    fallback_param_names.append(name)

        router_group_ids = {id(p) for p in router_params}
        block_group_ids = {id(p) for p in block_params}
        overlap = router_group_ids & block_group_ids
        if overlap:
            raise RuntimeError(f"optimizer parameter groups overlap for {len(overlap)} parameter(s)")
        grouped_ids = router_group_ids | block_group_ids
        if grouped_ids != trainable_param_ids:
            missing = trainable_param_ids - grouped_ids
            extra = grouped_ids - trainable_param_ids
            raise RuntimeError(f"optimizer parameter grouping mismatch: missing={len(missing)} extra={len(extra)}")

        param_groups = [
            {"params": router_params, "lr": router_lr, "name": "router"},
            {"params": block_params, "lr": block_lr, "name": "blocks"},
        ]
        group_summary = [
            self._summarize_param_group("router", router_lr, router_params, router_names),
            self._summarize_param_group("blocks", block_lr, block_params, block_names),
        ]
        return param_groups, group_summary, fallback_param_names

    @staticmethod
    def _summarize_param_group(name: str, lr: float, params, param_names):
        return {
            "name": name,
            "lr": float(lr),
            "parameter_tensors": int(len(params)),
            "parameter_elements": int(sum(p.numel() for p in params)),
            "sample_param_names": list(param_names[:20]),
        }

    def _log_optimizer_param_groups(self, optimizer_type: str, group_summary, fallback_param_names) -> None:
        self.optimizer_logger.info("[Optimizer] type=%s param_groups=%s", optimizer_type, len(group_summary))
        for group in group_summary:
            self.optimizer_logger.info(
                "[Optimizer] group=%s lr=%s parameter_tensors=%s parameter_elements=%s",
                group["name"],
                group["lr"],
                group["parameter_tensors"],
                group["parameter_elements"],
            )
        if fallback_param_names:
            self.optimizer_logger.warning(
                "[Optimizer] fallback_to_blocks params=%s",
                ", ".join(fallback_param_names),
            )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            opt = optimizer or self.optimizer or self.create_optimizer()
            if self.scheduler_builder is not None:
                self.lr_scheduler = self.scheduler_builder(opt, int(num_training_steps))
            else:
                self.lr_scheduler = super().create_scheduler(num_training_steps=num_training_steps, optimizer=opt)
            if self.observability_state is not None:
                resolution = getattr(self.scheduler_builder, "last_resolution", {}) if self.scheduler_builder is not None else {}
                self.observability_state["scheduler_state_summary"] = {
                    "class_name": type(self.lr_scheduler).__name__ if self.lr_scheduler is not None else None,
                    "scheduler_type": resolution.get("lr_scheduler_type"),
                    "num_training_steps": int(num_training_steps),
                    "actual_training_steps": resolution.get("actual_training_steps", int(num_training_steps)),
                    "resolved_lr_warmup_steps": resolution.get("resolved_lr_warmup_steps"),
                    "resolved_lr_decay_steps": resolution.get("resolved_lr_decay_steps"),
                    "scheduler_total_steps": resolution.get("scheduler_total_steps"),
                    "has_state_dict": hasattr(self.lr_scheduler, "state_dict"),
                }
                self.observability_state["actual_training_steps"] = resolution.get("actual_training_steps", int(num_training_steps))
                self.observability_state["resolved_lr_warmup_steps"] = resolution.get(
                    "resolved_lr_warmup_steps",
                    self.observability_state.get("resolved_lr_warmup_steps"),
                )
                self.observability_state["resolved_lr_decay_steps"] = resolution.get("resolved_lr_decay_steps")
                self.observability_state["scheduler_total_steps"] = resolution.get("scheduler_total_steps")
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.observability_state is not None:
            register_microbatch(self.observability_state, inputs)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            use_cache=False,
        )
        labels = inputs.get("labels")
        if labels is None:
            loss = outputs.loss
        else:
            loss, loss_metrics = compute_loss_observability(
                outputs.logits,
                labels,
                buckets=inputs.get("bucket"),
                tasks=inputs.get("task"),
                loss_weights=inputs.get("loss_weights"),
            )
            if self.observability_state is not None:
                self.observability_state["loss_observability"] = loss_metrics
                self.observability_state["final_answer_weighted_tokens"] = loss_metrics.get("final_answer_weighted_tokens")
                self.observability_state["final_answer_loss"] = loss_metrics.get("final_answer_loss")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.observability_state is not None:
            scaler = getattr(self, "scaler", None)
            self.observability_state["loss_scale"] = float(scaler.get_scale()) if scaler is not None else None
        return loss

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.fitmotn_metadata_builder is not None:
            target_dir = output_dir or self.args.output_dir
            checkpoint_name = Path(target_dir).name
            try:
                metadata = self.fitmotn_metadata_builder(checkpoint_name=checkpoint_name)
            except TypeError:
                metadata = self.fitmotn_metadata_builder()
            save_fitmotn_metadata(target_dir, metadata)

    def _save_checkpoint(self, model, trial):
        policy = self.checkpoint_policy
        if policy is None:
            return super()._save_checkpoint(model, trial)
        if int(getattr(self.args, "world_size", 1)) > 1:
            raise RuntimeError(
                "Atomic FitMoTN SFT checkpoints currently support single-process Trainer runs only; "
                "multi-process/FSDP/DeepSpeed checkpoint publication requires coordinated rank state."
            )
        if not self.args.should_save:
            return None
        step = int(self.state.global_step)
        checkpoint_name = f"{PREFIX_CHECKPOINT_DIR}-{step}"
        real_root = Path(self._get_output_dir(trial=trial)).resolve()
        real_root.mkdir(parents=True, exist_ok=True)
        cleanup_checkpoint_transactions(
            real_root,
            max_age_sec=float(getattr(policy, "checkpoint_temp_max_age_sec", 3600.0)),
        )
        staging_root = real_root / f".checkpoint-staging.tmp-{uuid.uuid4().hex}"
        prepared = real_root / f".checkpoint-{checkpoint_name}.tmp-{uuid.uuid4().hex}"
        old_output_dir = self.args.output_dir
        runtime = self.observability_state or {}
        stage_name = runtime.get("checkpoint_stage_boundary_name") or runtime.get("current_stage")
        stage_boundary = bool(runtime.get("checkpoint_stage_boundary_pending", False))
        fail_on_error = bool(getattr(policy, "checkpoint_fail_on_save_error", True))
        try:
            staging_root.mkdir(parents=False, exist_ok=False)
            self.args.output_dir = str(staging_root)
            super()._save_checkpoint(model, trial)
            staged_checkpoint = staging_root / checkpoint_name
            if not staged_checkpoint.is_dir():
                raise RuntimeError(f"Trainer did not create expected staged checkpoint: {staged_checkpoint}")
            staged_checkpoint.replace(prepared)
            commit_prepared_checkpoint(
                prepared,
                real_root / checkpoint_name,
                checkpoint_kind="sft",
                update_step=step,
                stage_name=stage_name,
                stage_boundary=stage_boundary,
                extra={"trainer_global_step": step},
            )
            prune_checkpoints(
                real_root,
                keep_last_n=int(getattr(policy, "checkpoint_keep_last_n", 3)),
                keep_every_n=int(getattr(policy, "checkpoint_keep_every_n", 0)),
                preserve_stage_boundaries=True,
            )
            runtime["checkpoint_stage_boundary_pending"] = False
            runtime["checkpoint_stage_boundary_name"] = None
            runtime["last_checkpoint_path"] = str(real_root / checkpoint_name)
        except Exception as exc:
            self.optimizer_logger.error("[Checkpoint] atomic SFT checkpoint failed at step=%s: %s", step, exc)
            if fail_on_error:
                raise
        finally:
            self.args.output_dir = old_output_dir
            shutil.rmtree(staging_root, ignore_errors=True)
            shutil.rmtree(prepared, ignore_errors=True)
