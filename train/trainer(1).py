from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import Trainer

from ..ADTN import TensorBlock
from ..checkpointing import save_fitmotn_metadata
from ..gate import SoftGate, TopKGate, _SoftGate, _TopKGate
from .observability import register_microbatch


ROUTER_GATE_TYPES = (TopKGate, SoftGate, _TopKGate, _SoftGate)


class FitMoTNTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_dataset_builder=None,
        fitmotn_metadata_builder=None,
        scheduler_builder=None,
        observability_state=None,
        optimizer_logger=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_dataset_builder = train_dataset_builder
        self.fitmotn_metadata_builder = fitmotn_metadata_builder
        self.scheduler_builder = scheduler_builder
        self.observability_state = observability_state
        self.optimizer_logger = optimizer_logger or logging.getLogger(__name__)
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
                self.observability_state["scheduler_state_summary"] = {
                    "class_name": type(self.lr_scheduler).__name__ if self.lr_scheduler is not None else None,
                    "num_training_steps": int(num_training_steps),
                    "has_state_dict": hasattr(self.lr_scheduler, "state_dict"),
                }
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
        loss = outputs.loss
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
