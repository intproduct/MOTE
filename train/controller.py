from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch as tc
from transformers import TrainingArguments

from ..audit import build_environment_snapshot, json_dump, jsonl_append, parameter_snapshot, to_jsonable
from ..checkpointing import extract_patch_state_dict, save_fitmotn_metadata
from ..data.builders import build_stage_aware_train_dataset
from ..data.collate import pad_collate
from ..eval.lm_eval_hf import run_lm_eval_tasks
from ..eval.metrics import build_early_stop_record
from ..init.approx import collect_dense_ffn_targets, run_approx_init
from ..patching import (
    build_motn_model_config,
    configure_motn_expert_warmup_scaling,
    patch_qwen_ffn_layers,
    reset_motn_expert_warmup_scaling,
    resolve_layer_idxs,
    set_trainable_motn_only,
)
from ..runtime import dtype_to_name, load_causal_lm_and_tokenizer
from ..tasks.registry import build_pretrain_tasks, build_task_mixture_tasks
from .callbacks import MOTNScheduleCallback
from .observability import build_checkpoint_metadata, build_run_summary, make_runtime_state
from .runtime import build_scheduler_builder
from .stages import MutableStageState, build_stage_plan
from .trainer import FitMoTNTrainer


def build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"fitmotn.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def make_run_name(fit_cfg) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(fit_cfg.model.model_path).name.replace("/", "_")
    return fit_cfg.output.run_name or f"fitmotn_{model_tag}_L{fit_cfg.model.layers_to_patch}_E{fit_cfg.model.E}_K{fit_cfg.model.topk}_{ts}"


def run_fitmotn_training(fit_cfg):
    stage_b_mode = str(getattr(fit_cfg.train, "stage_b_mode", "mixed"))
    reasoning_supervision_mode = str(getattr(fit_cfg.data, "reasoning_supervision_mode", "answer_only"))
    if stage_b_mode == "reasoning_recovery" and reasoning_supervision_mode != "full_trace":
        raise ValueError(
            "stage_b_mode=reasoning_recovery requires data.reasoning_supervision_mode=full_trace; "
            f"got {reasoning_supervision_mode!r}"
        )
    run_name = make_run_name(fit_cfg)
    root_dir = Path(fit_cfg.output.root_dir).resolve()
    run_dir = root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir)
    logger.info("[Run] start %s", run_name)
    logger.info("[Cfg] %s", json.dumps(to_jsonable(asdict(fit_cfg)), ensure_ascii=False))
    logger.info(
        "[Reasoning] supervision_mode=%s stage_b_mode=%s stage_b_disable_pretrain=%s stage_b_reasoning_boost=%s",
        reasoning_supervision_mode,
        stage_b_mode,
        bool(getattr(fit_cfg.train, "stage_b_disable_pretrain", False)),
        float(getattr(fit_cfg.train, "stage_b_reasoning_boost", 1.0)),
    )
    run_start_time = time.time()

    jsonl_path = run_dir / "train.jsonl"
    usage_jsonl_path = run_dir / "usage.jsonl"
    eval_jsonl_path = run_dir / "mid_eval.jsonl"
    eval_summary_path = run_dir / "eval_summary.json"
    run_summary_path = run_dir / "run_summary.json"
    lm_eval_root = run_dir / "lm_eval_outputs"
    lm_eval_root.mkdir(parents=True, exist_ok=True)

    device = tc.device(fit_cfg.model.device)
    model, tokenizer, model_dtype = load_causal_lm_and_tokenizer(
        fit_cfg.model.model_path,
        device=device,
        trust_remote_code=fit_cfg.model.trust_remote_code,
        torch_dtype=fit_cfg.model.torch_dtype,
        use_cache=False,
    )
    logger.info("[Runtime] resolved_model_dtype=%s", dtype_to_name(model_dtype))

    pretrain_tasks = build_pretrain_tasks(fit_cfg.data, logger=logger)
    task_tasks = build_task_mixture_tasks(fit_cfg.data, logger=logger)
    if bool(getattr(fit_cfg.data, "use_mmlu_train", False)) and str(getattr(fit_cfg.data, "mmlu_train_split", "auxiliary_train")) != "auxiliary_train":
        logger.warning(
            "[Data] skip MMLU training split=%s because only auxiliary_train is allowed for training; eval split=%s remains unchanged",
            getattr(fit_cfg.data, "mmlu_train_split", None),
            getattr(fit_cfg.data, "mmlu_split", None),
        )
    logger.info("[Data] pretrain_tasks=%s task_tasks=%s", [task.name for task in pretrain_tasks], [task.name for task in task_tasks])
    stage_plan = build_stage_plan(fit_cfg.train)
    stage_state = MutableStageState(stage_plan)
    scheduler_builder, warmup_updates = build_scheduler_builder(fit_cfg, stage_plan.total_updates, logger=logger)

    baseline_small = None
    baseline_final = None
    if fit_cfg.eval.run_baseline_eval:
        baseline_small = run_lm_eval_tasks(model, tokenizer, fit_cfg, tasks=fit_cfg.eval.baseline_small_tasks, logger=logger, eval_name="baseline_small", out_root=lm_eval_root, eval_mode="baseline_small")
        baseline_final = run_lm_eval_tasks(model, tokenizer, fit_cfg, tasks=fit_cfg.eval.final_tasks, logger=logger, eval_name="baseline_final", out_root=lm_eval_root, eval_mode="final")
        jsonl_append(eval_jsonl_path, {"kind": "baseline_small", "result": baseline_small, "time": time.time(), "run_name": run_name, "seed": int(fit_cfg.train.seed), "stage": "baseline", "global_step": 0, "resolved_model_dtype": dtype_to_name(model_dtype), "amp_enabled": bool(fit_cfg.model.use_amp and device.type == "cuda")})
        jsonl_append(eval_jsonl_path, {"kind": "baseline_final", "result": baseline_final, "time": time.time(), "run_name": run_name, "seed": int(fit_cfg.train.seed), "stage": "baseline", "global_step": 0, "resolved_model_dtype": dtype_to_name(model_dtype), "amp_enabled": bool(fit_cfg.model.use_amp and device.type == "cuda")})
    else:
        baseline_small = {"tasks": {task: {"primary_score": 0.0, "primary_metric": None} for task in fit_cfg.eval.baseline_small_tasks}}
        baseline_final = {"tasks": {task: {"primary_score": 0.0, "primary_metric": None} for task in fit_cfg.eval.final_tasks}}

    n_layers = int(model.config.num_hidden_layers)
    layer_idxs = resolve_layer_idxs(n_layers, fit_cfg.model.layers_to_patch)
    dense_targets = collect_dense_ffn_targets(model, layer_idxs)
    motn_cfg = build_motn_model_config(fit_cfg)
    model = patch_qwen_ffn_layers(model, layer_idxs, motn_cfg, device=device, dtype=tc.float32, log=logger)
    set_trainable_motn_only(model, log=logger)

    env_snapshot = build_environment_snapshot()
    param_snapshot = parameter_snapshot(model)
    runtime_state = make_runtime_state(
        run_name=run_name,
        run_dir=run_dir,
        fit_cfg=fit_cfg,
        stage_plan=stage_plan,
        model_dtype=model_dtype,
        amp_enabled=bool(fit_cfg.model.use_amp and device.type == "cuda"),
        warmup_updates=warmup_updates if fit_cfg.train.lr_warmup else 0,
        env_snapshot=env_snapshot,
        param_snapshot=param_snapshot,
    )
    runtime_state["start_time"] = run_start_time
    runtime_state["layers_to_patch"] = layer_idxs
    runtime_state["patched_layer_count"] = len(layer_idxs)
    runtime_state["baseline_small_summary"] = None if baseline_small is None else to_jsonable(baseline_small.get("summary"))
    runtime_state["baseline_final_summary"] = None if baseline_final is None else to_jsonable(baseline_final.get("summary"))
    runtime_state["resolved_model_dtype"] = dtype_to_name(model_dtype)
    runtime_state["amp_enabled"] = bool(fit_cfg.model.use_amp and device.type == "cuda")
    runtime_state["approx_init_summary"] = None
    setattr(model, "fitmotn_runtime", runtime_state)

    if bool(fit_cfg.approx_init.enabled):
        approx_summary = run_approx_init(model, layer_idxs, dense_targets, fit_cfg.approx_init, logger, run_dir)
        runtime_state["approx_init_summary"] = to_jsonable(approx_summary)
        set_trainable_motn_only(model, log=logger)
        configure_motn_expert_warmup_scaling(model, fit_cfg.approx_init, global_step=0)
    else:
        reset_motn_expert_warmup_scaling(model)

    dataset = build_stage_aware_train_dataset(tokenizer, pretrain_tasks, task_tasks, stage_state, fit_cfg.data, fit_cfg.train, logger=logger)
    collator = lambda batch: pad_collate(batch, pad_id=dataset.pad_id)

    eval_summary = {
        "run_name": run_name,
        "seed": int(fit_cfg.train.seed),
        "cfg": to_jsonable(asdict(fit_cfg)),
        "baseline_small": baseline_small,
        "baseline_final": baseline_final,
        "mid_evals": [],
        "final_full": None,
        "compare_vs_baseline": None,
        "approx_init_summary": runtime_state.get("approx_init_summary"),
    }
    with eval_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(eval_summary), f, ensure_ascii=False, indent=2)
    final_eval_info = {
        "final_full_summary": None,
        "compare_vs_baseline": None,
    }

    def mid_eval_fn(model, step: int, stage_name: str):
        mid_eval = run_lm_eval_tasks(model, tokenizer, fit_cfg, tasks=fit_cfg.eval.baseline_small_tasks, logger=logger, eval_name=f"mid_eval_u{step}", out_root=lm_eval_root, eval_mode="mid")
        early_rec = build_early_stop_record(mid_eval, baseline_small, fit_cfg)
        payload = {"stage": stage_name, "update": step, "result": mid_eval, "vs_baseline": early_rec, "time": time.time()}
        eval_summary["mid_evals"].append(payload)
        runtime_state["latest_mid_eval_summary"] = to_jsonable(mid_eval.get("summary"))
        runtime_state["latest_mid_eval_update"] = int(step)
        with eval_summary_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(eval_summary), f, ensure_ascii=False, indent=2)
        return {"result": mid_eval, "vs_baseline": early_rec, "stop_training": bool(fit_cfg.train.early_stop and early_rec["passed_abs_gate"])}

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=bool(fit_cfg.output.overwrite_output_dir),
        per_device_train_batch_size=int(fit_cfg.train.batch_size),
        gradient_accumulation_steps=int(fit_cfg.train.grad_accum),
        learning_rate=float(fit_cfg.train.lr),
        max_steps=int(stage_plan.total_updates),
        num_train_epochs=1.0,
        logging_steps=int(fit_cfg.train.log_every),
        save_steps=int(fit_cfg.train.save_every_updates),
        save_strategy="steps",
        eval_strategy="no",
        dataloader_num_workers=int(fit_cfg.data.dataloader_num_workers),
        report_to=list(fit_cfg.train.report_to),
        bf16=False,
        fp16=bool(fit_cfg.model.use_amp and device.type == "cuda"),
        remove_unused_columns=False,
        dataloader_pin_memory=(device.type == "cuda"),
        max_grad_norm=float(fit_cfg.train.max_grad_norm),
        seed=int(fit_cfg.train.seed),
        save_safetensors=False,
    )

    def metadata_builder(checkpoint_name: str | None = None):
        runtime_state["baseline_small_summary"] = None if baseline_small is None else to_jsonable(baseline_small.get("summary"))
        runtime_state["baseline_final_summary"] = None if baseline_final is None else to_jsonable(baseline_final.get("summary"))
        runtime_state["final_full_summary"] = final_eval_info["final_full_summary"]
        runtime_state["compare_vs_baseline"] = final_eval_info["compare_vs_baseline"]
        return build_checkpoint_metadata(
            runtime_state,
            layers_to_patch=layer_idxs,
            motn_cfg=motn_cfg,
            fit_cfg=fit_cfg,
            checkpoint_name=checkpoint_name,
            patch_state_dict=extract_patch_state_dict(model, layer_idxs),
        )

    trainer = FitMoTNTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        train_dataset_builder=lambda: dataset,
        fitmotn_metadata_builder=metadata_builder,
        scheduler_builder=scheduler_builder,
        observability_state=runtime_state,
    )
    trainer.add_callback(
        MOTNScheduleCallback(
            fit_cfg,
            stage_state,
            train_jsonl_path=jsonl_path,
            usage_jsonl_path=usage_jsonl_path,
            eval_jsonl_path=eval_jsonl_path,
            eval_fn=mid_eval_fn,
            logger=logger,
        )
    )
    if device.type == "cuda" and tc.cuda.is_available():
        try:
            tc.cuda.reset_peak_memory_stats(device)
            logger.info("[Runtime] reset CUDA peak memory stats before training")
        except Exception as exc:
            logger.warning("[Runtime] failed to reset CUDA peak memory stats: %s", exc)
    trainer.train()

    final_model_dir = run_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(final_model_dir)
    save_fitmotn_metadata(final_model_dir, metadata_builder(checkpoint_name=final_model_dir.name))

    model.eval()
    final_full = run_lm_eval_tasks(model, tokenizer, fit_cfg, tasks=fit_cfg.eval.final_tasks, logger=logger, eval_name="final_full", out_root=lm_eval_root, eval_mode="final")
    compare_summary = {}
    for task_name in fit_cfg.eval.final_tasks:
        b = baseline_final["tasks"][task_name]["primary_score"]
        f = final_full["tasks"][task_name]["primary_score"]
        compare_summary[task_name] = {"baseline": b, "final": f, "delta": None if (b is None or f is None) else f - b}
    eval_summary["final_full"] = final_full
    eval_summary["compare_vs_baseline"] = compare_summary
    final_eval_info["final_full_summary"] = to_jsonable(final_full.get("summary"))
    final_eval_info["compare_vs_baseline"] = to_jsonable(compare_summary)
    runtime_state["final_full_summary"] = final_eval_info["final_full_summary"]
    runtime_state["compare_vs_baseline"] = final_eval_info["compare_vs_baseline"]
    with eval_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(eval_summary), f, ensure_ascii=False, indent=2)
    jsonl_append(eval_jsonl_path, {
        "kind": "final_full",
        "time": time.time(),
        "run_name": run_name,
        "seed": int(fit_cfg.train.seed),
        "stage": runtime_state.get("current_stage"),
        "global_step": runtime_state.get("scheduler_step"),
        "lr": runtime_state.get("current_lr"),
        "T": runtime_state.get("current_T"),
        "gate_trainable": runtime_state.get("gate_trainable"),
        "resolved_model_dtype": runtime_state.get("resolved_model_dtype"),
        "amp_enabled": runtime_state.get("amp_enabled"),
        "result": final_full,
        "compare_vs_baseline": compare_summary,
    })
    metadata = metadata_builder(checkpoint_name=final_model_dir.name)
    save_fitmotn_metadata(final_model_dir, metadata)
    trainer_control = getattr(trainer, "control", None)
    stop_reason = "early_stop" if bool(getattr(trainer_control, "should_training_stop", False)) else None
    run_summary = build_run_summary(runtime_state, eval_summary, final_model_dir=str(final_model_dir), stop_reason=stop_reason)
    json_dump(run_summary_path, run_summary)
    json_dump(final_model_dir / "run_summary.json", run_summary)
    logger.info("[Run] finished %s", run_name)
    return {"run_dir": str(run_dir), "final_model_dir": str(final_model_dir), "eval_summary": str(eval_summary_path), "run_summary": str(run_summary_path)}
