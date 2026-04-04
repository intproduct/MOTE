from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import Trainer

from ..checkpointing import save_fitmotn_metadata
from .observability import register_microbatch


class FitMoTNTrainer(Trainer):
    def __init__(self, *args, train_dataset_builder=None, fitmotn_metadata_builder=None, scheduler_builder=None, observability_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset_builder = train_dataset_builder
        self.fitmotn_metadata_builder = fitmotn_metadata_builder
        self.scheduler_builder = scheduler_builder
        self.observability_state = observability_state
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
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(params, **optimizer_kwargs)
            if self.observability_state is not None:
                self.observability_state["optimizer_type"] = type(self.optimizer).__name__
                self.observability_state["weight_decay"] = getattr(self.args, "weight_decay", None)
        return self.optimizer

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
