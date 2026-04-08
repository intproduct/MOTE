from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState

from ..audit import jsonl_append, module_norm, grad_norm
from ..patching import set_motn_gate_trainable, set_motn_temperature, update_motn_expert_warmup_scaling
from .observability import build_mid_eval_record, build_train_record, build_usage_record, finalize_update_batch_meta, snapshot_cuda, update_step_runtime
from .stages import temperature_schedule


class MOTNScheduleCallback(TrainerCallback):
    def __init__(self, fit_cfg, stage_state, train_jsonl_path: str | Path, usage_jsonl_path: str | Path, eval_jsonl_path: str | Path, eval_fn=None, logger=None):
        self.fit_cfg = fit_cfg
        self.stage_state = stage_state
        self.train_jsonl_path = Path(train_jsonl_path)
        self.usage_jsonl_path = Path(usage_jsonl_path)
        self.eval_jsonl_path = Path(eval_jsonl_path)
        self.eval_fn = eval_fn
        self.logger = logger
        self._gate_state = None

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
        runtime["current_lr"] = logs.get("learning_rate", runtime.get("current_lr"))
        runtime["current_loss"] = logs.get("loss", runtime.get("current_loss"))
        runtime["grad_norm"] = logs.get("grad_norm", runtime.get("grad_norm"))
        runtime["param_norm"] = module_norm(model, trainable_only=True)
        grad_stats = grad_norm(model)
        runtime.update(grad_stats)
        runtime["overflow_or_nan_detected"] = bool((grad_stats.get("num_nan_grads") or 0) > 0 or (grad_stats.get("num_inf_grads") or 0) > 0)
        # We only mark "step skipped" when we can prove it; NaN/Inf detection alone is not enough.
        runtime["optimizer_step_skipped"] = None
        try:
            snapshot_cuda(runtime, next(model.parameters()).device)
        except Exception:
            snapshot_cuda(runtime, getattr(args, "device", None))
        record = build_train_record(runtime, {"loss": logs.get("loss"), "learning_rate": logs.get("learning_rate"), "epoch": logs.get("epoch"), "step": int(state.global_step), "grad_norm": logs.get("grad_norm")})
        jsonl_append(self.train_jsonl_path, record)
        if self.logger is not None:
            latest_mid = runtime.get("latest_mid_eval_summary") or {}
            latest_mid_update = runtime.get("latest_mid_eval_update")
            gsm8k = None
            mmlu = None
            if isinstance(latest_mid, dict):
                gsm8k = (latest_mid.get("gsm8k") or {}).get("primary_score")
                mmlu = (latest_mid.get("mmlu") or {}).get("primary_score")
            self.logger.info(
                "[Train] stage=%s step=%s loss=%s lr=%s T=%s gate=%s reasoning_mode=%s reasoning_focused=%s tokens/s=%s step_time=%s "
                "cuda_alloc=%.1fMB cuda_peak=%.1fMB cuda_reserved=%.1fMB cuda_peak_reserved=%.1fMB "
                "mid_eval(u=%s gsm8k=%s mmlu=%s)",
                record.get("stage"),
                record.get("update_step"),
                record.get("train_loss"),
                record.get("lr"),
                record.get("T"),
                record.get("gate_trainable"),
                record.get("reasoning_supervision_mode"),
                record.get("stage_reasoning_focused"),
                record.get("tokens_per_sec"),
                record.get("step_time_sec"),
                record.get("cuda_mem_alloc_mb") or 0.0,
                record.get("cuda_mem_peak_alloc_mb") or 0.0,
                record.get("cuda_mem_reserved_mb") or 0.0,
                record.get("cuda_mem_peak_reserved_mb") or 0.0,
                latest_mid_update,
                gsm8k,
                mmlu,
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

        if self.fit_cfg.train.usage_dump_every > 0 and step > 0 and step % int(self.fit_cfg.train.usage_dump_every) == 0:
            dump_rec = build_usage_record(runtime, model)
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
