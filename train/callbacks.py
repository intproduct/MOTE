from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState

from ..audit import jsonl_append, module_norm, grad_norm
from ..patching import set_motn_gate_trainable, set_motn_temperature, update_motn_expert_warmup_scaling
from .observability import (
    build_mid_eval_record,
    build_train_record,
    build_usage_report_record,
    capture_usage_light_snapshot,
    finalize_update_batch_meta,
    flush_pending_usage_light_records,
    snapshot_cuda,
    update_step_runtime,
)
from .stages import resolve_stage_temperature, set_optimizer_stage_lrs


class MOTNScheduleCallback(TrainerCallback):
    def __init__(self, fit_cfg, stage_state, train_jsonl_path: str | Path, train_light_jsonl_path: str | Path, usage_jsonl_path: str | Path, eval_jsonl_path: str | Path, eval_fn=None, logger=None):
        self.fit_cfg = fit_cfg
        self.stage_state = stage_state
        self.train_jsonl_path = Path(train_jsonl_path)
        self.train_light_jsonl_path = Path(train_light_jsonl_path)
        self.usage_jsonl_path = Path(usage_jsonl_path)
        self.eval_jsonl_path = Path(eval_jsonl_path)
        self.eval_fn = eval_fn
        self.logger = logger
        self._gate_state = None
        self._last_lr_stage = None

    def _patch_backend(self) -> str:
        return str(getattr(self.fit_cfg.model, "patch_backend", "motn") or "motn").lower()

    @staticmethod
    def _capture_param_group_lrs(runtime, optimizer) -> dict[str, float] | None:
        if runtime is None or optimizer is None:
            return runtime.get("current_param_group_lrs") if runtime is not None else None
        group_lrs = {}
        for idx, group in enumerate(getattr(optimizer, "param_groups", [])):
            group_name = str(group.get("name") or f"group_{idx}")
            group_lrs[group_name] = float(group["lr"])
        runtime["current_param_group_lrs"] = group_lrs or None
        return runtime.get("current_param_group_lrs")

    @staticmethod
    def _should_run(step: int, every: int | None) -> bool:
        if every is None:
            return False
        every = int(every)
        return every > 0 and step > 0 and step % every == 0

    def _apply_stage_lrs(self, model, optimizer, stage_name: str):
        runtime = getattr(model, "fitmotn_runtime", None) if model is not None else None
        resolved = set_optimizer_stage_lrs(optimizer, self.fit_cfg.train, stage_name)
        if runtime is not None:
            runtime["current_param_group_lrs"] = dict(resolved)
        if self._last_lr_stage != stage_name:
            if self.logger is not None:
                self.logger.info(
                    "[LR] stage transition old_stage=%s new_stage=%s blocks_lr=%s router_lr=%s",
                    self._last_lr_stage,
                    stage_name,
                    resolved.get("blocks"),
                    resolved.get("router"),
                )
            self._last_lr_stage = stage_name
        return resolved

    def _apply_schedule(self, model, global_step: int):
        train_cfg = self.fit_cfg.train
        self.stage_state.set_global_step(global_step)
        stage = self.stage_state.current_stage()
        patch_backend = self._patch_backend()
        if patch_backend == "motn":
            gate_trainable = int(global_step) >= int(train_cfg.gate_freeze_steps)
            if gate_trainable != self._gate_state:
                set_motn_gate_trainable(model, gate_trainable)
                self._gate_state = gate_trainable
                if self.logger is not None:
                    self.logger.info(f"[Gate] trainable={gate_trainable} at global_step={global_step}")
            t = resolve_stage_temperature(train_cfg, stage, global_step)
            set_motn_temperature(model, t)
        else:
            gate_trainable = None
            t = None
        runtime = getattr(model, "fitmotn_runtime", None)
        if runtime is not None:
            runtime["current_T"] = None if t is None else float(t)
            runtime["current_temperature"] = None if t is None else float(t)
            runtime["gate_trainable"] = None if gate_trainable is None else bool(gate_trainable)
            runtime["current_stage"] = stage.name
            runtime["current_stage_update"] = max(0, int(global_step) - int(stage.start_update))
            if bool(getattr(self.fit_cfg.approx_init, "expert_warmup_scaling_enabled", False)):
                runtime["expert_warmup_scaling_summary"] = update_motn_expert_warmup_scaling(model, self.fit_cfg.approx_init, int(global_step))
        return t

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self._apply_schedule(model, int(state.global_step))
        self._apply_stage_lrs(model, kwargs.get("optimizer"), self.stage_state.current_stage().name)
        self._capture_param_group_lrs(getattr(model, "fitmotn_runtime", None), kwargs.get("optimizer"))
        return control

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self.stage_state.set_global_step(int(state.global_step))
        self._apply_stage_lrs(model, kwargs.get("optimizer"), self.stage_state.current_stage().name)
        self._capture_param_group_lrs(getattr(model, "fitmotn_runtime", None), kwargs.get("optimizer"))
        return control

    def on_log(self, args, state: TrainerState, control: TrainerControl, model=None, logs=None, **kwargs):
        logs = logs or {}
        runtime = getattr(model, "fitmotn_runtime", None)
        if runtime is None:
            return control
        if not any(key in logs for key in ("loss", "learning_rate", "grad_norm")):
            return control
        if any(str(key).startswith("eval_") for key in logs.keys()):
            return control
        step = int(state.global_step)
        runtime["current_lr"] = logs.get("learning_rate", runtime.get("current_lr"))
        runtime["current_loss"] = logs.get("loss", runtime.get("current_loss"))
        runtime["grad_norm"] = logs.get("grad_norm", runtime.get("grad_norm"))
        group_lrs = self._capture_param_group_lrs(runtime, kwargs.get("optimizer"))
        run_heavy = bool(getattr(self.fit_cfg.train, "enable_heavy_runtime_stats", True)) and self._should_run(step, getattr(self.fit_cfg.train, "heavy_log_every", 500))
        if run_heavy and bool(getattr(self.fit_cfg.train, "enable_grad_param_norm", False)):
            runtime["param_norm"] = module_norm(model, trainable_only=True)
            grad_stats = grad_norm(model)
            runtime.update(grad_stats)
            runtime["overflow_or_nan_detected"] = bool((grad_stats.get("num_nan_grads") or 0) > 0 or (grad_stats.get("num_inf_grads") or 0) > 0)
        else:
            runtime["param_norm"] = None
            runtime["grad_norm"] = logs.get("grad_norm")
            runtime["num_nan_grads"] = None
            runtime["num_inf_grads"] = None
            runtime["overflow_or_nan_detected"] = None
        runtime["optimizer_step_skipped"] = None
        if run_heavy and bool(getattr(self.fit_cfg.train, "enable_cuda_snapshot", False)):
            try:
                snapshot_cuda(runtime, next(model.parameters()).device)
            except Exception:
                snapshot_cuda(runtime, getattr(args, "device", None))
        else:
            runtime["cuda_mem_alloc_mb"] = None
            runtime["cuda_mem_reserved_mb"] = None
            runtime["cuda_mem_peak_alloc_mb"] = None
            runtime["cuda_mem_peak_reserved_mb"] = None
            runtime["cpu_ram_used_mb"] = None
            runtime["host_ram_used_mb"] = None
        record = build_train_record(runtime, {"loss": logs.get("loss"), "learning_rate": logs.get("learning_rate"), "epoch": logs.get("epoch"), "step": int(state.global_step), "grad_norm": logs.get("grad_norm")})
        if self._should_run(step, getattr(self.fit_cfg.train, "train_jsonl_every", getattr(self.fit_cfg.train, "log_every", 50))):
            jsonl_append(self.train_jsonl_path, record)
        if self.logger is not None and self._should_run(step, getattr(self.fit_cfg.train, "log_every", 50)):
            if run_heavy:
                self.logger.info(
                    "[Train] stage=%s step=%s loss=%s lr=%s param_group_lrs=%s T=%s gate=%s tokens/s=%s step_time=%s grad_norm=%s param_norm=%s cuda_peak=%sMB",
                    record.get("stage"),
                    record.get("update_step"),
                    record.get("train_loss"),
                    record.get("lr"),
                    group_lrs,
                    record.get("T"),
                    record.get("gate_trainable"),
                    record.get("tokens_per_sec"),
                    record.get("step_time_sec"),
                    record.get("grad_norm"),
                    record.get("param_norm"),
                    record.get("cuda_mem_peak_alloc_mb"),
                )
            else:
                self.logger.info(
                    "[Train] stage=%s step=%s loss=%s lr=%s param_group_lrs=%s T=%s gate=%s tokens/s=%s step_time=%s",
                    record.get("stage"),
                    record.get("update_step"),
                    record.get("train_loss"),
                    record.get("lr"),
                    group_lrs,
                    record.get("T"),
                    record.get("gate_trainable"),
                    record.get("tokens_per_sec"),
                    record.get("step_time_sec"),
                )
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        step = int(state.global_step)
        current_t = self._apply_schedule(model, step)
        stage = self.stage_state.current_stage()
        runtime = getattr(model, "fitmotn_runtime", None)
        if runtime is not None:
            self._apply_stage_lrs(model, kwargs.get("optimizer"), stage.name)
            self._capture_param_group_lrs(runtime, kwargs.get("optimizer"))
            update_step_runtime(
                runtime,
                global_step=step,
                stage_name=stage.name,
                stage_start=stage.start_update,
                current_t=current_t,
                gate_trainable=None if self._patch_backend() != "motn" else (step >= int(self.fit_cfg.train.gate_freeze_steps)),
                lr=runtime.get("current_lr"),
                epoch=getattr(state, "epoch", None),
                loss=runtime.get("current_loss"),
            )
            finalize_update_batch_meta(runtime)

        usage_tracking_enabled = bool(getattr(self.fit_cfg.train, "enable_usage_runtime_tracking", True))
        if usage_tracking_enabled and self._should_run(step, getattr(self.fit_cfg.train, "usage_light_every", getattr(self.fit_cfg.train, "log_every", 50))):
            capture_usage_light_snapshot(runtime, model)
        if usage_tracking_enabled and self._should_run(step, getattr(self.fit_cfg.train, "usage_light_jsonl_every", getattr(self.fit_cfg.train, "log_every", 50))):
            for record in flush_pending_usage_light_records(runtime):
                jsonl_append(self.train_light_jsonl_path, record)

        usage_report_enabled = bool(getattr(self.fit_cfg.train, "enable_usage_report", True))
        if usage_report_enabled and self._should_run(step, getattr(self.fit_cfg.train, "usage_report_every", getattr(self.fit_cfg.train, "usage_dump_every", 500))):
            dump_rec = build_usage_report_record(runtime, model)
            jsonl_append(self.usage_jsonl_path, dump_rec)
            if self.logger is not None:
                self.logger.info("[Usage] stage=%s step=%s layers=%s", dump_rec.get("stage"), dump_rec.get("update_step"), len([k for k in dump_rec.keys() if k.startswith("layer_")]))

        if self.eval_fn is not None and self.fit_cfg.train.eval_every_updates > 0 and step > 0 and step % int(self.fit_cfg.train.eval_every_updates) == 0:
            result = self.eval_fn(model=model, step=step, stage_name=stage.name)
            mid_eval_rec = build_mid_eval_record(runtime, result.get("result", result), result.get("vs_baseline", {}))
            jsonl_append(self.eval_jsonl_path, mid_eval_rec)
            if result.get("stop_training"):
                control.should_training_stop = True
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        runtime = getattr(model, "fitmotn_runtime", None)
        if runtime is None:
            return control
        if bool(getattr(self.fit_cfg.train, "enable_usage_runtime_tracking", True)) and int(getattr(self.fit_cfg.train, "usage_light_jsonl_every", 0)) > 0:
            for record in flush_pending_usage_light_records(runtime):
                jsonl_append(self.train_light_jsonl_path, record)
        return control
