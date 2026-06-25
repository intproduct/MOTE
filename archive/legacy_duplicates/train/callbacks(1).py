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
from .stages import temperature_schedule


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

    @staticmethod
    def _should_run(step: int, every: int | None) -> bool:
        if every is None:
            return False
        every = int(every)
        return every > 0 and step > 0 and step % every == 0

    def _apply_schedule(self, model, global_step: int):
        train_cfg = self.fit_cfg.train
        self.stage_state.set_global_step(global_step)
        gate_trainable = int(global_step) >= int(train_cfg.gate_freeze_steps)
        if gate_trainable != self._gate_state:
            set_motn_gate_trainable(model, gate_trainable)
            self._gate_state = gate_trainable
            if self.logger is not None:
                self.logger.info(f"[Gate] trainable={gate_trainable} at global_step={global_step}")
        total_steps = self.stage_state.plan.total_updates
        t = temperature_schedule(
            global_step,
            t_start=float(train_cfg.begin_t),
            t_end=float(train_cfg.end_t),
            total_steps=total_steps,
            hold_steps=int(total_steps * 0.10),
            decay_steps=int(total_steps * 0.70),
            mode="cosine",
        )
        set_motn_temperature(model, t)
        runtime = getattr(model, "fitmotn_runtime", None)
        if runtime is not None:
            runtime["current_T"] = float(t)
            runtime["gate_trainable"] = bool(gate_trainable)
            runtime["current_stage"] = self.stage_state.current_stage().name
            runtime["current_stage_update"] = max(0, int(global_step) - int(self.stage_state.current_stage().start_update))
            if bool(getattr(self.fit_cfg.approx_init, "expert_warmup_scaling_enabled", False)):
                runtime["expert_warmup_scaling_summary"] = update_motn_expert_warmup_scaling(model, self.fit_cfg.approx_init, int(global_step))
        return t

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self._apply_schedule(model, int(state.global_step))
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
                    "[Train] stage=%s step=%s loss=%s lr=%s T=%s gate=%s tokens/s=%s step_time=%s grad_norm=%s param_norm=%s cuda_peak=%sMB",
                    record.get("stage"),
                    record.get("update_step"),
                    record.get("train_loss"),
                    record.get("lr"),
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
                    "[Train] stage=%s step=%s loss=%s lr=%s T=%s gate=%s tokens/s=%s step_time=%s",
                    record.get("stage"),
                    record.get("update_step"),
                    record.get("train_loss"),
                    record.get("lr"),
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
            update_step_runtime(
                runtime,
                global_step=step,
                stage_name=stage.name,
                stage_start=stage.start_update,
                current_t=current_t,
                gate_trainable=step >= int(self.fit_cfg.train.gate_freeze_steps),
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
