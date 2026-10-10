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
from .observability import (
    DEFERRED_LOSS_METRICS_MARKER,
    accumulate_loss_observability,
    flush_loss_observability,
    materialize_loss_observability,
    register_microbatch,
)


ROUTER_GATE_TYPES = (TopKGate, SoftGate, _TopKGate, _SoftGate)
KNOWN_BUCKETS = ("pretrain_general", "gsm8k_core", "aux_reasoning", "unknown_bucket")


@torch.no_grad()
def build_teacher_topk_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    topk: int,
    token_chunk_size: int,
) -> dict:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"Expected logits [B,L,V] and labels [B,L], got {logits.shape}, {labels.shape}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Teacher logits/labels shape mismatch: {logits.shape[:2]} != {labels.shape}")
    vocab_size = int(logits.shape[-1])
    resolved_topk = min(int(topk), vocab_size)
    if resolved_topk < 1:
        raise ValueError(f"Invalid KD top-k={topk} for vocab_size={vocab_size}")

    shift_logits = logits[..., :-1, :]
    valid = labels[..., 1:].ne(-100)
    positions = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        raise ValueError("KD microbatch has no supervised target tokens")
    flat_logits = shift_logits.reshape(-1, vocab_size)
    top_indices = []
    top_log_probs = []
    residual_log_probs = []
    eps = torch.finfo(torch.float32).eps
    for start in range(0, int(positions.numel()), int(token_chunk_size)):
        chunk_positions = positions[start : start + int(token_chunk_size)]
        scaled = flat_logits.index_select(0, chunk_positions).float() / float(temperature)
        values, indices = torch.topk(scaled, k=resolved_topk, dim=-1)
        log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
        log_probs = values - log_z
        top_mass = log_probs.exp().sum(dim=-1).clamp(min=0.0, max=1.0 - eps)
        top_indices.append(indices)
        top_log_probs.append(log_probs)
        residual_log_probs.append(torch.log1p(-top_mass))
    all_top_log_probs = torch.cat(top_log_probs, dim=0)
    all_residual_log_probs = torch.cat(residual_log_probs, dim=0)
    return {
        "top_indices": torch.cat(top_indices, dim=0),
        "top_log_probs": all_top_log_probs,
        "residual_log_probs": all_residual_log_probs,
        "valid_tokens": int(positions.numel()),
        "topk": resolved_topk,
        "mean_topk_mass": all_top_log_probs.exp().sum(dim=-1).mean(),
    }


def compute_exact_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    token_chunk_size: int,
    divergence: str = "forward_kl",
    jsd_beta: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    if student_logits.ndim != 3 or teacher_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Exact KD expects student/teacher logits [B,L,V] and labels [B,L]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"Exact KD teacher/student shape mismatch: {teacher_logits.shape} != {student_logits.shape}")
    if student_logits.shape[:2] != labels.shape:
        raise ValueError(f"Exact KD logits/labels shape mismatch: {student_logits.shape[:2]} != {labels.shape}")

    vocab_size = int(student_logits.shape[-1])
    positions = labels[..., 1:].ne(-100).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    valid_tokens = int(positions.numel())
    if valid_tokens == 0:
        raise ValueError("Exact KD microbatch has no supervised target tokens")
    student_flat = student_logits[..., :-1, :].reshape(-1, vocab_size)
    teacher_flat = teacher_logits[..., :-1, :].reshape(-1, vocab_size)
    student_device = student_logits.device
    total_kl = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, valid_tokens, int(token_chunk_size)):
        chunk_positions = positions[start : start + int(token_chunk_size)]
        teacher_positions = chunk_positions.to(device=teacher_logits.device, non_blocking=True)
        teacher_scaled = teacher_flat.index_select(0, teacher_positions).float() / float(temperature)
        teacher_log_probs = F.log_softmax(teacher_scaled, dim=-1).to(
            device=student_device, non_blocking=True
        )
        student_scaled = student_flat.index_select(0, chunk_positions).float() / float(temperature)
        student_log_probs = F.log_softmax(student_scaled, dim=-1)
        resolved_divergence = str(divergence).strip().lower()
        if resolved_divergence == "forward_kl":
            chunk_loss = F.kl_div(student_log_probs, teacher_log_probs.exp(), reduction="sum")
        elif resolved_divergence == "reverse_kl":
            student_probs = student_log_probs.exp()
            chunk_loss = (student_probs * (student_log_probs - teacher_log_probs)).sum()
        elif resolved_divergence == "jsd":
            beta = float(jsd_beta)
            if not 0.0 < beta < 1.0:
                raise ValueError(f"jsd_beta must be in (0, 1), got {beta}")
            teacher_probs = teacher_log_probs.exp()
            student_probs = student_log_probs.exp()
            mixture_probs = beta * teacher_probs + (1.0 - beta) * student_probs
            mixture_log_probs = mixture_probs.clamp_min(torch.finfo(torch.float32).tiny).log()
            chunk_loss = beta * (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum()
            chunk_loss = chunk_loss + (1.0 - beta) * (
                student_probs * (student_log_probs - mixture_log_probs)
            ).sum()
        else:
            raise ValueError(
                f"Unsupported exact KD divergence={divergence!r}; expected forward_kl, reverse_kl, or jsd"
            )
        total_kl = total_kl + chunk_loss
    loss = total_kl * (float(temperature) ** 2) / valid_tokens
    return loss, {
        "kd_valid_tokens": valid_tokens,
        "kd_distribution": "exact",
        "kd_divergence": str(divergence).strip().lower(),
        "kd_jsd_beta": float(jsd_beta) if str(divergence).strip().lower() == "jsd" else None,
    }


def compute_topk_distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_targets: dict,
    *,
    temperature: float,
    token_chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"Expected student logits [B,L,V] and labels [B,L], got {student_logits.shape}, {labels.shape}"
        )
    if student_logits.shape[:2] != labels.shape:
        raise ValueError(f"Student logits/labels shape mismatch: {student_logits.shape[:2]} != {labels.shape}")
    vocab_size = int(student_logits.shape[-1])
    shift_logits = student_logits[..., :-1, :]
    positions = labels[..., 1:].ne(-100).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    valid_tokens = int(positions.numel())
    if valid_tokens != int(teacher_targets["valid_tokens"]):
        raise ValueError(
            f"Teacher/student KD token count mismatch: {teacher_targets['valid_tokens']} != {valid_tokens}"
        )
    if valid_tokens == 0:
        raise ValueError("KD microbatch has no supervised target tokens")

    device = student_logits.device
    teacher_indices = teacher_targets["top_indices"].to(device=device, non_blocking=True)
    teacher_top_log_probs = teacher_targets["top_log_probs"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    teacher_residual_log_probs = teacher_targets["residual_log_probs"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    if teacher_indices.shape[0] != valid_tokens:
        raise ValueError("Teacher top-k rows do not match supervised token count")
    if int(teacher_indices.max().item()) >= vocab_size:
        raise ValueError("Teacher top-k token id exceeds student vocabulary")

    flat_logits = shift_logits.reshape(-1, vocab_size)
    total_kl = student_logits.new_zeros((), dtype=torch.float32)
    eps = torch.finfo(torch.float32).eps
    for start in range(0, valid_tokens, int(token_chunk_size)):
        end = min(start + int(token_chunk_size), valid_tokens)
        chunk_positions = positions[start:end]
        scaled = flat_logits.index_select(0, chunk_positions).float() / float(temperature)
        student_log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
        indices = teacher_indices[start:end]
        student_top_log_probs = scaled.gather(dim=-1, index=indices) - student_log_z
        student_top_mass = student_top_log_probs.exp().sum(dim=-1).clamp(
            min=0.0, max=1.0 - eps
        )
        student_residual_log_probs = torch.log1p(-student_top_mass)

        teacher_chunk_log_probs = teacher_top_log_probs[start:end]
        teacher_chunk_probs = teacher_chunk_log_probs.exp()
        teacher_chunk_residual_log_probs = teacher_residual_log_probs[start:end]
        teacher_chunk_residual_probs = teacher_chunk_residual_log_probs.exp()

        cross_entropy = -(
            (teacher_chunk_probs * student_top_log_probs).sum(dim=-1)
            + teacher_chunk_residual_probs * student_residual_log_probs
        )
        teacher_entropy = -(
            (teacher_chunk_probs * teacher_chunk_log_probs).sum(dim=-1)
            + teacher_chunk_residual_probs * teacher_chunk_residual_log_probs
        )
        total_kl = total_kl + (cross_entropy - teacher_entropy).sum()

    loss = total_kl * (float(temperature) ** 2) / valid_tokens
    return loss, {
        "kd_valid_tokens": valid_tokens,
        "kd_topk": int(teacher_targets["topk"]),
        "kd_mean_topk_mass": teacher_targets["mean_topk_mass"],
    }


def compute_loss_observability(
    logits,
    labels,
    *,
    buckets=None,
    tasks=None,
    loss_weights=None,
    collect_metrics=True,
    defer_metrics=False,
):
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

    final_mask = torch.zeros_like(valid)
    if loss_weights is not None:
        shift_weights = loss_weights[..., 1:].to(device=token_loss.device, dtype=token_loss.dtype).contiguous()
        shift_weights = torch.where(valid, shift_weights, torch.zeros_like(shift_weights))
        denom = shift_weights.sum()
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        weighted_loss = (token_loss * shift_weights).sum() / safe_denom
        loss = torch.where(denom > 0, weighted_loss, standard_loss)
        final_mask = valid & shift_weights.gt(1.0)
    else:
        loss = standard_loss

    if not collect_metrics:
        return loss, {}

    token_loss_fp32 = token_loss.float()
    zero_loss = token_loss_fp32.new_zeros(())
    zero_count = valid.new_zeros((), dtype=torch.long)
    bucket_loss_sums = {bucket: zero_loss for bucket in KNOWN_BUCKETS}
    bucket_token_counts = {bucket: zero_count for bucket in KNOWN_BUCKETS}
    task_loss_sums = {}
    task_token_counts = {}
    row_loss_sums = token_loss_fp32.sum(dim=-1).detach()
    row_token_counts = valid.sum(dim=-1).detach()
    if buckets is not None:
        bucket_names = list(buckets)
        for row_idx, bucket_name in enumerate(bucket_names[: token_loss.shape[0]]):
            key_bucket = str(bucket_name or "unknown_bucket")
            if key_bucket not in KNOWN_BUCKETS:
                key_bucket = "unknown_bucket"
            bucket_loss_sums[key_bucket] = bucket_loss_sums[key_bucket] + row_loss_sums[row_idx]
            bucket_token_counts[key_bucket] = bucket_token_counts[key_bucket] + row_token_counts[row_idx]
    if tasks is not None:
        for row_idx, task_name in enumerate(list(tasks)[: token_loss.shape[0]]):
            key_task = str(task_name or "unknown")
            task_loss_sums[key_task] = task_loss_sums.get(key_task, zero_loss) + row_loss_sums[row_idx]
            task_token_counts[key_task] = task_token_counts.get(key_task, zero_count) + row_token_counts[row_idx]
    deferred = {
        DEFERRED_LOSS_METRICS_MARKER: True,
        "bucket_loss_sums": bucket_loss_sums,
        "bucket_token_counts": bucket_token_counts,
        "task_loss_sums": task_loss_sums,
        "task_token_counts": task_token_counts,
        "final_answer_loss_sum": (token_loss_fp32 * final_mask).sum().detach(),
        "final_answer_weighted_tokens": final_mask.sum().detach(),
    }
    return loss, deferred if defer_metrics else materialize_loss_observability(deferred)


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
        teacher_model=None,
        kd_config=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_dataset_builder = train_dataset_builder
        self.fitmotn_metadata_builder = fitmotn_metadata_builder
        self.scheduler_builder = scheduler_builder
        self.observability_state = observability_state
        self.optimizer_logger = optimizer_logger or logging.getLogger(__name__)
        self.checkpoint_policy = checkpoint_policy
        self.teacher_model = teacher_model
        self.kd_config = dict(kd_config or {})
        self.kd_enabled = bool(self.kd_config.get("enabled", False))
        if self.kd_enabled:
            if self.teacher_model is None:
                raise ValueError("KD is enabled but no frozen teacher model was provided")
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)
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
            use_cache=False,
        )
        labels = inputs.get("labels")
        if labels is None:
            if bool(getattr(self, "kd_enabled", False)):
                raise ValueError("KD requires labels to select supervised token positions")
            loss = outputs.loss
        else:
            hard_ce_loss, loss_metrics = compute_loss_observability(
                outputs.logits,
                labels,
                buckets=inputs.get("bucket"),
                tasks=inputs.get("task"),
                loss_weights=inputs.get("loss_weights"),
                collect_metrics=self.observability_state is not None,
                defer_metrics=True,
            )
            if self.observability_state is not None:
                accumulate_loss_observability(self.observability_state, loss_metrics)
            if not bool(getattr(self, "kd_enabled", False)):
                loss = hard_ce_loss
            else:
                temperature = float(self.kd_config["temperature"])
                token_chunk_size = int(self.kd_config["token_chunk_size"])
                teacher_device = next(self.teacher_model.parameters()).device
                teacher_input_ids = inputs["input_ids"].to(teacher_device, non_blocking=True)
                attention_mask = inputs.get("attention_mask")
                teacher_attention_mask = (
                    None
                    if attention_mask is None
                    else attention_mask.to(teacher_device, non_blocking=True)
                )
                teacher_labels = labels.to(teacher_device, non_blocking=True)
                distribution = str(self.kd_config.get("distribution", "topk"))
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        use_cache=False,
                    )
                    teacher_targets = (
                        build_teacher_topk_targets(
                            teacher_outputs.logits,
                            teacher_labels,
                            temperature=temperature,
                            topk=int(self.kd_config["topk"]),
                            token_chunk_size=token_chunk_size,
                        )
                        if distribution == "topk"
                        else None
                    )
                if distribution == "exact":
                    kd_loss, kd_metrics = compute_exact_distillation_loss(
                        outputs.logits,
                        teacher_outputs.logits,
                        labels,
                        temperature=temperature,
                        token_chunk_size=token_chunk_size,
                        divergence=str(self.kd_config.get("divergence", "forward_kl")),
                        jsd_beta=float(self.kd_config.get("jsd_beta", 0.5)),
                    )
                else:
                    kd_loss, kd_metrics = compute_topk_distillation_loss(
                        outputs.logits,
                        labels,
                        teacher_targets,
                        temperature=temperature,
                        token_chunk_size=token_chunk_size,
                    )
                del teacher_outputs
                kd_weight = float(self.kd_config["weight"])
                hard_ce_weight = float(self.kd_config["hard_ce_weight"])
                loss = kd_weight * kd_loss + hard_ce_weight * hard_ce_loss
                if self.observability_state is not None:
                    runtime = self.observability_state
                    runtime["kd_microbatches"] = int(runtime.get("kd_microbatches", 0)) + 1
                    runtime["kd_last_loss"] = float(kd_loss.detach().item())
                    runtime["kd_last_hard_ce_loss"] = float(hard_ce_loss.detach().item())
                    topk_mass = kd_metrics.get("kd_mean_topk_mass")
                    runtime["kd_last_topk_mass"] = (
                        None if topk_mass is None else float(topk_mass.detach().item())
                    )
                    runtime["kd_last_valid_tokens"] = int(kd_metrics["kd_valid_tokens"])
                    runtime["kd_last_topk"] = kd_metrics.get("kd_topk")
                    runtime["kd_distribution"] = distribution
                    runtime["kd_divergence"] = kd_metrics.get("kd_divergence", "forward_kl")
                    runtime["kd_jsd_beta"] = kd_metrics.get("kd_jsd_beta")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.observability_state is not None:
            scaler = getattr(self, "scaler", None)
            self.observability_state["loss_scale"] = float(scaler.get_scale()) if scaler is not None else None
        return loss

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if self.observability_state is not None:
            flush_loss_observability(self.observability_state)
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
