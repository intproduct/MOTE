from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch as tc
from transformers import TrainingArguments

from ..audit import build_environment_snapshot, json_dump, jsonl_append, parameter_snapshot, to_jsonable
from ..checkpointing import (
    build_fitmotn_json_metadata,
    extract_patch_state_dict,
    get_restore_state_dict,
    load_fitmotn_metadata,
    save_fitmotn_metadata,
    transactional_save_checkpoint,
    transactional_update_checkpoint,
    validate_checkpoint,
)
from ..gate import normalize_legacy_gate_state_dict_for_model
from ..data.builders import build_stage_aware_train_dataset
from ..data.collate import pad_collate
from ..eval.runner import run_eval_tasks
from ..eval.config_utils import resolve_task_eval_settings
from ..eval.metrics import build_early_stop_record
from ..init.approx import collect_dense_ffn_targets, run_approx_init
from ..patching import (
    build_patch_model_config,
    configure_motn_expert_warmup_scaling,
    iter_patched_layers,
    patch_qwen_ffn_layers,
    reset_motn_expert_warmup_scaling,
    resolve_layer_idxs,
    set_motn_usage_tracking,
    set_trainable_patch_only,
    summarize_motn_gate_routers,
)
from ..runtime import dtype_to_name, load_causal_lm_and_tokenizer
from ..tasks.registry import build_pretrain_tasks, build_task_mixture_tasks
from ..utils.paths import assert_no_unsafe_paths
from .callbacks import MOTNScheduleCallback
from .checkpoint_selector import select_checkpoint, write_checkpoint_selection
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


def _is_resume_training(fit_cfg) -> bool:
    return bool(_resume_source(fit_cfg))


def _resume_source(fit_cfg) -> str | None:
    return (
        getattr(fit_cfg.train, "resume_checkpoint_from", None)
        or getattr(fit_cfg.train, "resume_weights_from", None)
        or getattr(fit_cfg.train, "resume_fitmotn_from", None)
    )


def _validate_resume_source(path_value: str | Path) -> tuple[Path, Path]:
    ckpt_dir = Path(path_value).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"train.resume_fitmotn_from path does not exist: {ckpt_dir}")
    state_path = ckpt_dir / "fitmotn_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            "train.resume_fitmotn_from is not a valid FitMoTN final_model/checkpoint; "
            f"missing fitmotn_state.pt at {state_path}"
        )
    return ckpt_dir, state_path


def _path_is_same_or_inside(child: Path, parent: Path) -> bool:
    child = child.expanduser().resolve()
    parent = parent.expanduser().resolve()
    return child == parent or parent in child.parents


def _validate_resume_output_paths(fit_cfg, run_dir: Path) -> None:
    resume_from = _resume_source(fit_cfg)
    if not resume_from:
        return
    resume_dir = Path(resume_from).expanduser().resolve()
    final_model_dir = run_dir / "final_model"
    for label, candidate in [("run_dir", run_dir), ("final_model_dir", final_model_dir)]:
        if _path_is_same_or_inside(candidate, resume_dir):
            raise ValueError(
                f"resume continuation output {label}={candidate.resolve()} must not be the same as or inside "
                f"train.resume_fitmotn_from={resume_dir}; choose a separate output run directory."
            )


def _warn_resume_continuation_risks(fit_cfg, metadata: Dict[str, Any] | None, logger) -> None:
    if not getattr(fit_cfg.train, "resume_fitmotn_from", None) or getattr(fit_cfg.train, "resume_checkpoint_from", None):
        return
    if int(getattr(fit_cfg.train, "gate_freeze_steps", 0)) > 0:
        logger.warning(
            "resume continuation restarts global_step from 0; gate will be frozen again for gate_freeze_steps steps. "
            "Set gate_freeze_steps=0 for normal continuation."
        )
    stage_b_begin_set = getattr(fit_cfg.train, "stage_b_begin_t", None) is not None
    stage_b_end_set = getattr(fit_cfg.train, "stage_b_end_t", None) is not None
    global_constant = float(getattr(fit_cfg.train, "begin_t")) == float(getattr(fit_cfg.train, "end_t"))
    if not (stage_b_begin_set or stage_b_end_set or global_constant):
        logger.warning(
            "[Resume] temperature schedule restarts from begin_t for this continuation; set stage_b_begin_t/stage_b_end_t "
            "or use begin_t=end_t for a normal fixed-temperature continuation."
        )
    if metadata:
        ckpt_base_model_path = metadata.get("base_model_path")
        if ckpt_base_model_path and str(ckpt_base_model_path) != str(fit_cfg.model.model_path):
            logger.warning(
                "[Resume] checkpoint base_model_path=%s differs from current model.model_path=%s; continuing with current base model.",
                ckpt_base_model_path,
                fit_cfg.model.model_path,
            )


def _metadata_model_cfg(metadata: Dict[str, Any]) -> Dict[str, Any]:
    fit_cfg = metadata.get("fit_cfg") or {}
    model_cfg = fit_cfg.get("model") if isinstance(fit_cfg, dict) else getattr(fit_cfg, "model", {})
    if model_cfg is None:
        return {}
    if isinstance(model_cfg, dict):
        return model_cfg
    return vars(model_cfg) if hasattr(model_cfg, "__dict__") else {}


def _metadata_structure_summary(metadata: Dict[str, Any], layer_idxs: List[int], motn_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = _metadata_model_cfg(metadata)
    return {
        "patch_backend": metadata.get("patch_backend", motn_cfg.get("patch_backend", model_cfg.get("patch_backend", "motn"))),
        "k_in": motn_cfg.get("k_in", model_cfg.get("k_in")),
        "topk": motn_cfg.get("topk", model_cfg.get("topk")),
        "num_experts": motn_cfg.get("E", model_cfg.get("E")),
        "target_layers": list(layer_idxs),
        "use_global_expert": motn_cfg.get("global_expert_enabled", model_cfg.get("global_expert_enabled")),
        "global_alpha": motn_cfg.get("global_expert_weight", model_cfg.get("global_expert_weight")),
        "sparse_mixt": motn_cfg.get("sparse_mixt", model_cfg.get("sparse_mixt")),
    }


def _current_structure_summary(fit_cfg, layer_idxs: List[int]) -> Dict[str, Any]:
    return {
        "patch_backend": getattr(fit_cfg.model, "patch_backend", "motn"),
        "k_in": getattr(fit_cfg.model, "k_in", None),
        "topk": getattr(fit_cfg.model, "topk", None),
        "num_experts": getattr(fit_cfg.model, "E", None),
        "target_layers": list(layer_idxs),
        "use_global_expert": getattr(fit_cfg.model, "global_expert_enabled", None),
        "global_alpha": getattr(fit_cfg.model, "global_expert_weight", None),
        "sparse_mixt": getattr(fit_cfg.model, "sparse_mixt", None),
    }


def _validate_resume_structure(
    fit_cfg,
    metadata: Dict[str, Any],
    ckpt_layer_idxs: List[int],
    motn_cfg: Dict[str, Any],
    n_layers: int,
    *,
    allow_layer_expansion: bool = False,
) -> None:
    explicit_model_keys = set(getattr(fit_cfg, "_explicit_model_keys", set()) or set())
    checks = [
        ("patch_backend", "patch_backend", getattr(fit_cfg.model, "patch_backend", "motn"), metadata.get("patch_backend", motn_cfg.get("patch_backend", "motn"))),
        ("k_in", "k_in", getattr(fit_cfg.model, "k_in", None), motn_cfg.get("k_in")),
        ("topk", "topk", getattr(fit_cfg.model, "topk", None), motn_cfg.get("topk")),
        ("E", "num_experts", getattr(fit_cfg.model, "E", None), motn_cfg.get("E")),
        (
            "global_expert_enabled",
            "use_global_expert",
            getattr(fit_cfg.model, "global_expert_enabled", None),
            motn_cfg.get("global_expert_enabled"),
        ),
        (
            "global_expert_weight",
            "global_alpha",
            getattr(fit_cfg.model, "global_expert_weight", None),
            motn_cfg.get("global_expert_weight"),
        ),
        (
            "global_expert_init_scale",
            "global_expert_init_scale",
            getattr(fit_cfg.model, "global_expert_init_scale", None),
            motn_cfg.get("global_expert_init_scale"),
        ),
        (
            "global_expert_pos_strategy",
            "global_expert_pos_strategy",
            getattr(fit_cfg.model, "global_expert_pos_strategy", None),
            motn_cfg.get("global_expert_pos_strategy"),
        ),
        (
            "sparse_mixt",
            "sparse_mixt",
            getattr(fit_cfg.model, "sparse_mixt", None),
            motn_cfg.get("sparse_mixt"),
        ),
    ]
    conflicts = []
    for config_key, label, current_value, ckpt_value in checks:
        if config_key in explicit_model_keys and ckpt_value is not None and current_value != ckpt_value:
            conflicts.append(f"{label}: config={current_value!r} checkpoint={ckpt_value!r}")

    if "layers_to_patch" in explicit_model_keys:
        current_layers = resolve_layer_idxs(n_layers, fit_cfg.model.layers_to_patch)
        if list(current_layers) != list(ckpt_layer_idxs):
            is_strict_superset = (
                allow_layer_expansion
                and set(ckpt_layer_idxs).issubset(set(current_layers))
                and set(current_layers) != set(ckpt_layer_idxs)
            )
            if not is_strict_superset:
                conflicts.append(f"target_layers: config={list(current_layers)!r} checkpoint={list(ckpt_layer_idxs)!r}")

    if conflicts:
        raise ValueError(
            "resume_fitmotn_from checkpoint structure conflicts with explicit model config: "
            + "; ".join(conflicts)
        )


def _load_resume_metadata(fit_cfg, logger) -> Dict[str, Any]:
    source = _resume_source(fit_cfg)
    ckpt_dir, state_path = _validate_resume_source(source)
    if getattr(fit_cfg.train, "resume_checkpoint_from", None):
        report = validate_checkpoint(ckpt_dir, require_exact_resume="sft")
        if not report["ok"]:
            raise RuntimeError(f"SFT exact-resume checkpoint validation failed: {report['errors']}")
    metadata = load_fitmotn_metadata(ckpt_dir)
    metadata["_resume_ckpt_dir"] = str(ckpt_dir)
    metadata["_resume_state_path"] = str(state_path)
    if not getattr(fit_cfg.train, "resume_checkpoint_from", None) and getattr(fit_cfg.train, "extra_updates", None) is None:
        logger.warning("[Resume] extra_updates is not set; using resolved total_updates for continuation.")
    logger.info("[Resume] resumed_from=%s state_path=%s", ckpt_dir, state_path)
    return metadata


def _load_resume_state_into_model(
    model,
    state_dict: Dict[str, Any],
    layer_idxs: List[int],
    logger,
    *,
    required_layer_idxs: List[int] | None = None,
) -> Dict[str, Any]:
    state_dict = normalize_legacy_gate_state_dict_for_model(model, state_dict)
    model_keys = set(model.state_dict().keys())
    unexpected_before_load = sorted(set(state_dict.keys()) - model_keys)
    if unexpected_before_load:
        sample = unexpected_before_load[:20]
        raise RuntimeError(
            f"resume checkpoint contains {len(unexpected_before_load)} parameter key(s) not present in patched model; "
            f"sample={sample}"
        )
    required_layer_idxs = layer_idxs if required_layer_idxs is None else required_layer_idxs
    prefixes = tuple(f"model.layers.{int(idx)}.mlp." for idx in sorted(set(int(idx) for idx in required_layer_idxs)))
    required_patch_keys = sorted(key for key in model_keys if key.startswith(prefixes))
    missing_patch_keys = sorted(set(required_patch_keys) - set(state_dict.keys()))
    if missing_patch_keys:
        raise RuntimeError(
            f"resume checkpoint is missing {len(missing_patch_keys)} patched MoTN parameter key(s); "
            f"sample={missing_patch_keys[:20]}"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
    if unexpected_keys:
        raise RuntimeError(f"resume checkpoint produced unexpected model keys during load: {unexpected_keys[:20]}")
    summary = {
        "loaded_tensor_count": int(len(state_dict)),
        "required_patch_tensor_count": int(len(required_patch_keys)),
        "missing_key_count": int(len(missing_keys)),
        "unexpected_key_count": int(len(unexpected_keys)),
        "missing_key_sample": missing_keys[:20],
        "unexpected_key_sample": unexpected_keys[:20],
    }
    logger.info("[Resume] load_state_dict summary=%s", json.dumps(to_jsonable(summary), ensure_ascii=False))
    return summary


def _set_trainable_patch_layers(model, selected_layer_idxs: List[int], all_layer_idxs: List[int], logger) -> Dict[str, Any]:
    selected = set(int(idx) for idx in selected_layer_idxs)
    patched = set(int(idx) for idx in all_layer_idxs)
    if not selected:
        raise ValueError("trainable patched layer selection must not be empty")
    if not selected.issubset(patched):
        raise ValueError(
            f"trainable patched layers must be a subset of patched layers: selected={sorted(selected)} patched={sorted(patched)}"
        )
    seen = set()
    for layer_idx, module in iter_patched_layers(model):
        layer_idx = int(layer_idx)
        seen.add(layer_idx)
        requires_grad = layer_idx in selected
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)
    if seen != patched:
        raise RuntimeError(f"patched layer discovery mismatch: discovered={sorted(seen)} expected={sorted(patched)}")
    summary = {
        "patched_layers": sorted(patched),
        "trainable_patch_layers": sorted(selected),
        "frozen_patch_layers": sorted(patched - selected),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    logger.info("[Freeze] progressive_selection=%s", json.dumps(summary, ensure_ascii=False))
    return summary


def _ensure_phase_end_exact_checkpoint(trainer, model, checkpoint_root: Path, expected_updates: int, logger) -> Path:
    completed_updates = int(getattr(trainer.state, "global_step", -1))
    if completed_updates != int(expected_updates):
        raise RuntimeError(
            f"phase ended at global_step={completed_updates}, expected={int(expected_updates)}; "
            "refusing to publish a resume checkpoint for an incomplete phase"
        )
    target = Path(checkpoint_root) / f"checkpoint-{completed_updates}"
    report = validate_checkpoint(target, require_exact_resume="sft")
    if not report["ok"]:
        logger.info("[Checkpoint] forcing exact phase-end checkpoint at global_step=%s", completed_updates)
        trainer._save_checkpoint(model, trial=None)
        report = validate_checkpoint(target, require_exact_resume="sft")
    if not report["ok"]:
        raise RuntimeError(f"phase-end exact checkpoint validation failed: {report['errors']}")
    logger.info("[Checkpoint] validated exact phase-end checkpoint=%s", target)
    return target


def _restore_exact_data_sampling_state(dataset, fit_cfg, resume_metadata, runtime_state, logger) -> bool:
    """Restore explicit sampler state, or select deterministic replay for legacy sequential data."""
    if not getattr(fit_cfg.train, "resume_checkpoint_from", None) or resume_metadata is None:
        return False
    if bool(getattr(fit_cfg.train, "resume_reset_data_sampling_state", False)):
        runtime_state["data_sampling_resume"] = "intentionally_reset_for_new_curriculum_phase"
        logger.info(
            "[Resume] data sampler intentionally reset for a new curriculum phase; "
            "model/optimizer/global-step resume remains exact and Trainer data skip is disabled"
        )
        return True
    saved_sampling_state = resume_metadata.get("data_sampling_state")
    if not isinstance(saved_sampling_state, dict) or not bool(saved_sampling_state.get("initialized", False)):
        runtime_state["data_sampling_resume"] = "checkpoint_has_no_initialized_state"
        return False
    if int(fit_cfg.data.dataloader_num_workers) != 0:
        raise RuntimeError(
            "exact SFT data resume currently requires data.dataloader_num_workers=0; "
            "worker-local sampler state cannot be recovered from the parent process"
        )
    uncheckpointable = list(saved_sampling_state.get("uncheckpointable_tasks") or [])
    sampling_mode = str(getattr(fit_cfg.data, "source_sampling_mode", "deterministic_auto"))
    if uncheckpointable and sampling_mode == "legacy_sequential":
        # With one parent-process worker, Trainer's normal resume skip deterministically
        # replays the seeded task/source choices to the saved global step. Marking the
        # state as explicitly restored would set ignore_data_skip=True and be incorrect.
        runtime_state["data_sampling_resume"] = "restored_via_legacy_replay_skip"
        logger.info(
            "[Resume] data sampling restored via deterministic legacy replay/skip; "
            "uncheckpointable_source_count=%s",
            len(uncheckpointable),
        )
        return False
    dataset.load_sampling_state_dict(saved_sampling_state)
    runtime_state["data_sampling_resume"] = "restored"
    logger.info("[Resume] data sampling restored from explicit checkpoint state")
    return True


def _prepare_run_dir(fit_cfg, run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not _is_resume_training(fit_cfg) and not bool(fit_cfg.output.overwrite_output_dir):
            raise FileExistsError(
                f"non-resume output directory is not empty and output.overwrite_output_dir=false: {run_dir}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)


def _training_arguments_from_kwargs(kwargs: Dict[str, Any]) -> TrainingArguments:
    supported = set(inspect.signature(TrainingArguments).parameters)
    compatible = dict(kwargs)
    for optional_name in ("overwrite_output_dir", "save_safetensors"):
        if optional_name not in supported:
            compatible.pop(optional_name, None)
    return TrainingArguments(**compatible)


def _resolved_config_sha256(fit_cfg) -> str:
    payload = json.dumps(to_jsonable(asdict(fit_cfg)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_selected_metadata(metadata: Dict[str, Any], selected_update: int) -> Dict[str, Any]:
    expected = int(selected_update)
    for field in ("updates_done", "global_step"):
        actual = metadata.get(field)
        if actual is None or int(actual) != expected:
            raise RuntimeError(
                f"selected checkpoint metadata {field} mismatch: expected={expected} actual={actual!r}"
            )
    return metadata


def _validate_kd_tokenizer_identity(student_tokenizer, teacher_tokenizer) -> Dict[str, Any]:
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab != teacher_vocab:
        raise RuntimeError("KD teacher/student token-id vocabularies are not identical")
    special_fields = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    )
    special_ids = {}
    for field in special_fields:
        student_value = getattr(student_tokenizer, field, None)
        teacher_value = getattr(teacher_tokenizer, field, None)
        if student_value != teacher_value:
            raise RuntimeError(
                f"KD tokenizer special-id mismatch for {field}: {student_value} != {teacher_value}"
            )
        special_ids[field] = student_value
    frozen_prompts = (
        "Question: What is 17 + 25?\nAnswer:",
        "#### 42",
        "请计算并给出最终答案。",
    )
    for prompt in frozen_prompts:
        student_ids = student_tokenizer(prompt, add_special_tokens=True)["input_ids"]
        teacher_ids = teacher_tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if student_ids != teacher_ids:
            raise RuntimeError(f"KD tokenizer frozen-prompt mismatch for {prompt!r}")
    if student_tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise RuntimeError("KD teacher/student chat templates are not identical")
    return {
        "vocab_size": len(student_vocab),
        "special_token_ids": special_ids,
        "frozen_prompt_count": len(frozen_prompts),
        "chat_template_equal": True,
    }


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
    _validate_resume_output_paths(fit_cfg, run_dir)
    _prepare_run_dir(fit_cfg, run_dir)
    logger = build_logger(run_dir)
    logger.info("[Run] start %s", run_name)
    assert_no_unsafe_paths(fit_cfg, context="train", logger=logger)
    logger.info("[Cfg] %s", json.dumps(to_jsonable(asdict(fit_cfg)), ensure_ascii=False))
    is_resume_training = _is_resume_training(fit_cfg)
    resume_metadata = _load_resume_metadata(fit_cfg, logger) if is_resume_training else None
    _warn_resume_continuation_risks(fit_cfg, resume_metadata, logger)
    patch_backend = str(getattr(fit_cfg.model, "patch_backend", "motn") or "motn").lower()
    explicit_train_keys = set(getattr(fit_cfg, "_explicit_train_keys", set()) or set())
    if patch_backend == "adtn_fixed":
        if "enable_usage_runtime_tracking" not in explicit_train_keys:
            fit_cfg.train.enable_usage_runtime_tracking = False
        if "enable_usage_report" not in explicit_train_keys:
            fit_cfg.train.enable_usage_report = False
        logger.info(
            "[Patch] backend=adtn_fixed usage_runtime_tracking=%s usage_report=%s",
            bool(getattr(fit_cfg.train, "enable_usage_runtime_tracking", False)),
            bool(getattr(fit_cfg.train, "enable_usage_report", False)),
        )
    logger.info(
        "[Reasoning] supervision_mode=%s task_bucket_mode=%s stage_b_mode=%s stage_b_disable_pretrain=%s stage_b_reasoning_boost=%s",
        reasoning_supervision_mode,
        str(getattr(fit_cfg.train, "task_bucket_mode", "flat")),
        stage_b_mode,
        bool(getattr(fit_cfg.train, "stage_b_disable_pretrain", False)),
        float(getattr(fit_cfg.train, "stage_b_reasoning_boost", 1.0)),
    )
    run_start_time = time.time()

    jsonl_path = run_dir / "train.jsonl"
    train_light_jsonl_path = run_dir / "train_light.jsonl"
    usage_jsonl_path = run_dir / "usage.jsonl"
    eval_jsonl_path = run_dir / "mid_eval.jsonl"
    eval_summary_path = run_dir / "eval_summary.json"
    run_summary_path = run_dir / "run_summary.json"
    lm_eval_root = run_dir / "lm_eval_outputs"
    lm_eval_root.mkdir(parents=True, exist_ok=True)
    allow_non_primary_backend_skip = str(fit_cfg.eval.eval_backend) == "both" and str(fit_cfg.eval.primary_eval_backend) != "evalscope"

    device = tc.device(fit_cfg.model.device)
    model, tokenizer, model_dtype = load_causal_lm_and_tokenizer(
        fit_cfg.model.model_path,
        device=device,
        trust_remote_code=fit_cfg.model.trust_remote_code,
        torch_dtype=fit_cfg.model.torch_dtype,
        use_cache=False,
    )
    logger.info("[Runtime] resolved_model_dtype=%s", dtype_to_name(model_dtype))
    logger.info(
        "[DataFormat] reasoning_format=%s reasoning_chat_enable_thinking=%s "
        "reasoning_chat_system_prompt_present=%s tokenizer_chat_template_present=%s",
        str(getattr(fit_cfg.data, "reasoning_format", "raw") or "raw"),
        bool(getattr(fit_cfg.data, "reasoning_chat_enable_thinking", False)),
        bool(getattr(fit_cfg.data, "reasoning_chat_system_prompt", None)),
        bool(getattr(tokenizer, "chat_template", None)),
    )

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
    logger.info(
        "[StagePlan] actual_total_updates=%s stage_a_updates=%s stage_b_updates=%s resume=%s stage2_only_on_resume=%s",
        int(stage_plan.total_updates),
        int(stage_plan.stage_a_updates),
        int(stage_plan.stage_b_updates),
        bool(is_resume_training),
        bool(getattr(fit_cfg.train, "stage2_only_on_resume", True)),
    )
    for stage in stage_plan.stages:
        logger.info(
            "[Stage] name=%s task_bucket_mode=%s bucket_ratios=%s",
            stage.name,
            getattr(stage, "task_bucket_mode", "flat"),
            dict(getattr(stage, "bucket_ratios", {}) or {}),
        )
    for task in pretrain_tasks + task_tasks:
        logger.info(
            "[Data] task=%s group=%s bucket=%s source_family=%s weight=%s",
            task.name,
            task.group,
            getattr(task, "bucket", "task"),
            task.source_family,
            float(task.weight),
        )
    scheduler_builder, scheduler_metadata = build_scheduler_builder(fit_cfg, stage_plan.total_updates, logger=logger)

    baseline_small = None
    baseline_final = None
    if fit_cfg.eval.run_baseline_eval:
        baseline_small = run_eval_tasks(
            fit_cfg,
            tasks=fit_cfg.eval.baseline_small_tasks,
            logger=logger,
            eval_name="baseline_small",
            out_root=lm_eval_root,
            eval_mode="baseline_small",
            model=model,
            tokenizer=tokenizer,
            model_or_path=fit_cfg.model.model_path,
            allow_backend_skip=allow_non_primary_backend_skip,
        )
        baseline_final = run_eval_tasks(
            fit_cfg,
            tasks=fit_cfg.eval.final_tasks,
            logger=logger,
            eval_name="baseline_final",
            out_root=lm_eval_root,
            eval_mode="final",
            model=model,
            tokenizer=tokenizer,
            model_or_path=fit_cfg.model.model_path,
            allow_backend_skip=allow_non_primary_backend_skip,
        )
        jsonl_append(eval_jsonl_path, {"kind": "baseline_small", "result": baseline_small, "time": time.time(), "run_name": run_name, "seed": int(fit_cfg.train.seed), "stage": "baseline", "global_step": 0, "resolved_model_dtype": dtype_to_name(model_dtype), "amp_enabled": bool(fit_cfg.model.use_amp and device.type == "cuda")})
        jsonl_append(eval_jsonl_path, {"kind": "baseline_final", "result": baseline_final, "time": time.time(), "run_name": run_name, "seed": int(fit_cfg.train.seed), "stage": "baseline", "global_step": 0, "resolved_model_dtype": dtype_to_name(model_dtype), "amp_enabled": bool(fit_cfg.model.use_amp and device.type == "cuda")})
    else:
        baseline_small = {
            "primary_backend": str(fit_cfg.eval.primary_eval_backend),
            "tasks": {task: {"primary_score": 0.0, "primary_metric": None} for task in fit_cfg.eval.baseline_small_tasks},
            "summary": {task: {"primary_score": 0.0, "primary_metric": None} for task in fit_cfg.eval.baseline_small_tasks},
        }
        baseline_final = {
            "primary_backend": str(fit_cfg.eval.primary_eval_backend),
            "tasks": {task: {"primary_score": 0.0, "primary_metric": None} for task in fit_cfg.eval.final_tasks},
            "summary": {task: {"primary_score": 0.0, "primary_metric": None} for task in fit_cfg.eval.final_tasks},
        }

    n_layers = int(model.config.num_hidden_layers)
    resume_load_summary = None
    loaded_checkpoint_metadata = None
    resume_layer_idxs: List[int] = []
    new_layer_idxs: List[int] = []
    resume_expand_layers = bool(getattr(fit_cfg.train, "resume_expand_layers", False))
    if is_resume_training:
        assert resume_metadata is not None
        checkpoint_layer_idxs = [int(idx) for idx in resume_metadata.get("layers_to_patch", [])]
        if not checkpoint_layer_idxs:
            raise ValueError("resume checkpoint metadata is missing non-empty layers_to_patch")
        out_of_range_layers = [idx for idx in checkpoint_layer_idxs if idx < 0 or idx >= n_layers]
        if out_of_range_layers:
            raise ValueError(
                f"resume checkpoint target layer(s) are outside current base model range 0..{n_layers - 1}: "
                f"{out_of_range_layers}"
            )
        motn_cfg = dict(resume_metadata.get("motn_cfg") or {})
        if not motn_cfg:
            raise ValueError("resume checkpoint metadata is missing motn_cfg")
        if "dtype" not in motn_cfg:
            motn_cfg["dtype"] = tc.float32
        patch_backend = str(resume_metadata.get("patch_backend", motn_cfg.get("patch_backend", patch_backend)) or "motn").lower()
        motn_cfg["patch_backend"] = patch_backend
        _validate_resume_structure(
            fit_cfg,
            resume_metadata,
            checkpoint_layer_idxs,
            motn_cfg,
            n_layers,
            allow_layer_expansion=resume_expand_layers,
        )
        loaded_checkpoint_metadata = build_fitmotn_json_metadata(resume_metadata)
        logger.info(
            "[Resume] checkpoint_structure=%s",
            json.dumps(to_jsonable(_metadata_structure_summary(resume_metadata, checkpoint_layer_idxs, motn_cfg)), ensure_ascii=False),
        )
        if resume_expand_layers:
            layer_idxs = resolve_layer_idxs(n_layers, fit_cfg.model.layers_to_patch)
            new_layer_idxs = [
                idx
                for idx in layer_idxs
                if idx not in set(checkpoint_layer_idxs)
            ]
            if not new_layer_idxs:
                raise ValueError(
                    "train.resume_expand_layers=true requires model.layers_to_patch to be a strict superset of checkpoint layers"
                )
            dense_targets = collect_dense_ffn_targets(model, new_layer_idxs)
            logger.info(
                "[Resume] progressive expansion checkpoint_layers=%s target_layers=%s new_layers=%s",
                checkpoint_layer_idxs,
                layer_idxs,
                new_layer_idxs,
            )
        else:
            layer_idxs = checkpoint_layer_idxs
            dense_targets = None
        resume_layer_idxs = list(checkpoint_layer_idxs)
    else:
        layer_idxs = resolve_layer_idxs(n_layers, fit_cfg.model.layers_to_patch)
        dense_targets = collect_dense_ffn_targets(model, layer_idxs)
        motn_cfg = build_patch_model_config(fit_cfg)
        patch_backend = str(motn_cfg.get("patch_backend", patch_backend) or "motn").lower()
    if bool(getattr(fit_cfg.train, "gradient_checkpointing", False)):
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("gradient_checkpointing=true but model does not support it")
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        logger.info("[Checkpointing] gradient_checkpointing=true use_cache=false")
    model = patch_qwen_ffn_layers(model, layer_idxs, motn_cfg, device=device, dtype=tc.float32, log=logger)
    if is_resume_training:
        resume_state_dict = get_restore_state_dict(resume_metadata)
        resume_load_summary = _load_resume_state_into_model(
            model,
            resume_state_dict,
            layer_idxs,
            logger,
            required_layer_idxs=resume_layer_idxs,
        )
    set_motn_usage_tracking(model, bool(getattr(fit_cfg.train, "enable_usage_runtime_tracking", True)))
    set_trainable_patch_only(model, log=logger)
    progressive_trainable_summary = None
    trainable_patch_spec = getattr(fit_cfg.train, "trainable_patch_layers", None)
    if trainable_patch_spec is not None:
        selected_layer_idxs = resolve_layer_idxs(n_layers, trainable_patch_spec)
    teacher_model = None
    kd_config = {
        "enabled": bool(getattr(fit_cfg.train, "kd_enabled", False)),
        "distribution": str(getattr(fit_cfg.train, "kd_distribution", "topk")),
        "temperature": float(getattr(fit_cfg.train, "kd_temperature", 2.0)),
        "divergence": str(getattr(fit_cfg.train, "kd_divergence", "forward_kl")),
        "jsd_beta": float(getattr(fit_cfg.train, "kd_jsd_beta", 0.5)),
        "weight": float(getattr(fit_cfg.train, "kd_weight", 1.0)),
        "hard_ce_weight": float(getattr(fit_cfg.train, "kd_hard_ce_weight", 0.0)),
        "topk": int(getattr(fit_cfg.train, "kd_topk", 64)),
        "token_chunk_size": int(getattr(fit_cfg.train, "kd_token_chunk_size", 64)),
    }
    kd_runtime = {"enabled": kd_config["enabled"]}
    if kd_config["enabled"]:
        teacher_device = tc.device(fit_cfg.train.kd_teacher_device)
        if teacher_device == device:
            logger.warning("[KD] teacher and student share device=%s", teacher_device)
        teacher_model, teacher_tokenizer, teacher_dtype = load_causal_lm_and_tokenizer(
            fit_cfg.train.kd_teacher_model_path,
            device=teacher_device,
            trust_remote_code=fit_cfg.model.trust_remote_code,
            torch_dtype=fit_cfg.model.torch_dtype,
            use_cache=False,
        )
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        tokenizer_contract = _validate_kd_tokenizer_identity(tokenizer, teacher_tokenizer)
        if int(teacher_model.config.vocab_size) != int(model.config.vocab_size):
            raise RuntimeError(
                f"KD model vocabulary mismatch: {teacher_model.config.vocab_size} != {model.config.vocab_size}"
            )
        kd_runtime.update(
            {
                "teacher_model_path": str(fit_cfg.train.kd_teacher_model_path),
                "teacher_device": str(teacher_device),
                "teacher_dtype": dtype_to_name(teacher_dtype),
                "teacher_parameter_count": int(sum(p.numel() for p in teacher_model.parameters())),
                "tokenizer_contract": tokenizer_contract,
                **kd_config,
            }
        )
        logger.info("[KD] %s", json.dumps(to_jsonable(kd_runtime), ensure_ascii=False))


    env_snapshot = build_environment_snapshot()
    param_snapshot = parameter_snapshot(model)
    runtime_state = make_runtime_state(
        run_name=run_name,
        run_dir=run_dir,
        fit_cfg=fit_cfg,
        stage_plan=stage_plan,
        model_dtype=model_dtype,
        amp_enabled=bool(fit_cfg.model.use_amp and device.type == "cuda"),
        scheduler_metadata=scheduler_metadata,
        env_snapshot=env_snapshot,
        param_snapshot=param_snapshot,
    )
    runtime_state["start_time"] = run_start_time
    runtime_state["layers_to_patch"] = layer_idxs
    runtime_state["patched_layer_count"] = len(layer_idxs)
    runtime_state["patch_backend"] = patch_backend
    runtime_state["patch_cfg"] = to_jsonable(motn_cfg)
    gate_router_summary = summarize_motn_gate_routers(model, patch_backend)
    runtime_state.update(to_jsonable(gate_router_summary))
    runtime_state["baseline_small_summary"] = None if baseline_small is None else to_jsonable(baseline_small.get("summary"))
    runtime_state["baseline_final_summary"] = None if baseline_final is None else to_jsonable(baseline_final.get("summary"))
    runtime_state["resolved_model_dtype"] = dtype_to_name(model_dtype)
    runtime_state["amp_enabled"] = bool(fit_cfg.model.use_amp and device.type == "cuda")
    runtime_state["approx_init_summary"] = None
    runtime_state["is_resume_training"] = bool(is_resume_training)
    runtime_state["resume_fitmotn_from"] = getattr(fit_cfg.train, "resume_fitmotn_from", None)
    runtime_state["resume_weights_from"] = getattr(fit_cfg.train, "resume_weights_from", None)
    runtime_state["resume_checkpoint_from"] = getattr(fit_cfg.train, "resume_checkpoint_from", None)
    runtime_state["resume_mode"] = "exact" if getattr(fit_cfg.train, "resume_checkpoint_from", None) else ("weights" if is_resume_training else "none")
    runtime_state["resume_stage"] = getattr(fit_cfg.train, "resume_stage", "auto")
    runtime_state["stage2_only_on_resume"] = bool(getattr(fit_cfg.train, "stage2_only_on_resume", True))
    runtime_state["resume_expand_layers"] = bool(resume_expand_layers)
    runtime_state["resume_layer_idxs"] = list(resume_layer_idxs)
    runtime_state["new_layer_idxs"] = list(new_layer_idxs)
    runtime_state["progressive_trainable_summary"] = to_jsonable(progressive_trainable_summary)
    runtime_state["extra_updates"] = getattr(fit_cfg.train, "extra_updates", None)
    runtime_state["kd"] = to_jsonable(kd_runtime)
    runtime_state["actual_total_updates"] = int(stage_plan.total_updates)
    runtime_state["stage_a_updates"] = int(stage_plan.stage_a_updates)
    runtime_state["stage_b_updates"] = int(stage_plan.stage_b_updates)
    runtime_state["approx_init_skipped_due_to_resume"] = bool(is_resume_training and not resume_expand_layers)
    runtime_state["loaded_fitmotn_state_path"] = None if resume_metadata is None else resume_metadata.get("_resume_state_path")
    runtime_state["loaded_checkpoint_metadata"] = loaded_checkpoint_metadata
    runtime_state["resume_load_summary"] = resume_load_summary
    runtime_state["current_model_structure_summary"] = _metadata_structure_summary(resume_metadata, layer_idxs, motn_cfg) if is_resume_training else _current_structure_summary(fit_cfg, layer_idxs)
    setattr(model, "fitmotn_runtime", runtime_state)
    setattr(stage_state, "runtime_state", runtime_state)

    if resume_expand_layers:
        if bool(fit_cfg.approx_init.enabled):
            approx_summary = run_approx_init(model, new_layer_idxs, dense_targets, fit_cfg.approx_init, logger, run_dir)
            runtime_state["approx_init_summary"] = to_jsonable(approx_summary)
            set_trainable_patch_only(model, log=logger)
            configure_motn_expert_warmup_scaling(model, fit_cfg.approx_init, global_step=0)
        else:
            reset_motn_expert_warmup_scaling(model)
    elif is_resume_training:
        logger.info("[ApproxInit] skipped because resume_fitmotn_from is set")
        reset_motn_expert_warmup_scaling(model)
    elif bool(fit_cfg.approx_init.enabled):
        approx_summary = run_approx_init(model, layer_idxs, dense_targets, fit_cfg.approx_init, logger, run_dir)
        runtime_state["approx_init_summary"] = to_jsonable(approx_summary)
        set_trainable_patch_only(model, log=logger)
        configure_motn_expert_warmup_scaling(model, fit_cfg.approx_init, global_step=0)
    else:
        reset_motn_expert_warmup_scaling(model)
    if trainable_patch_spec is not None:
        progressive_trainable_summary = _set_trainable_patch_layers(model, selected_layer_idxs, layer_idxs, logger)

    dataset = build_stage_aware_train_dataset(tokenizer, pretrain_tasks, task_tasks, stage_state, fit_cfg.data, fit_cfg.train, logger=logger)
    collator = lambda batch: pad_collate(
        batch,
        pad_id=dataset.pad_id,
        pad_to_length_buckets=getattr(fit_cfg.data, "pad_to_length_buckets", []),
    )
    data_sampling_state_restored = _restore_exact_data_sampling_state(
        dataset,
        fit_cfg,
        resume_metadata,
        runtime_state,
        logger,
    )

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
        "patch_backend": patch_backend,
        "patch_cfg": to_jsonable(motn_cfg),
    }
    with eval_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(eval_summary), f, ensure_ascii=False, indent=2)
    final_eval_info = {
        "final_full_summary": None,
        "compare_vs_baseline": None,
    }

    def mid_eval_fn(model, step: int, stage_name: str):
        mid_eval = run_eval_tasks(
            fit_cfg,
            tasks=fit_cfg.eval.baseline_small_tasks,
            logger=logger,
            eval_name=f"mid_eval_u{step}",
            out_root=lm_eval_root,
            eval_mode="mid",
            model=model,
            tokenizer=tokenizer,
            model_or_path=None,
            allow_backend_skip=True,
        )
        early_rec = build_early_stop_record(mid_eval, baseline_small, fit_cfg)
        payload = {"stage": stage_name, "update": step, "result": mid_eval, "vs_baseline": early_rec, "time": time.time()}
        eval_summary["mid_evals"].append(payload)
        runtime_state["latest_mid_eval_summary"] = to_jsonable(mid_eval.get("summary"))
        runtime_state["latest_mid_eval_update"] = int(step)
        with eval_summary_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(eval_summary), f, ensure_ascii=False, indent=2)
        return {"result": mid_eval, "vs_baseline": early_rec, "stop_training": bool(fit_cfg.train.early_stop and early_rec["passed_abs_gate"])}

    if bool(getattr(fit_cfg.train, "benchmark_train_only", False)):
        logger.info("[Benchmark] benchmark_train_only enabled: usage tracking and heavy runtime stats defaults are disabled unless explicitly overridden")

    use_bf16 = bool(fit_cfg.model.use_amp and device.type == "cuda" and model_dtype == tc.bfloat16)
    use_fp16 = bool(fit_cfg.model.use_amp and device.type == "cuda" and model_dtype == tc.float16)
    logging_steps = min(v for v in [int(fit_cfg.train.log_every), int(fit_cfg.train.train_jsonl_every)] if v > 0)
    training_args = _training_arguments_from_kwargs(dict(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=bool(fit_cfg.output.overwrite_output_dir),
        per_device_train_batch_size=int(fit_cfg.train.batch_size),
        gradient_accumulation_steps=int(fit_cfg.train.grad_accum),
        learning_rate=float(fit_cfg.train.block_lr),
        lr_scheduler_type=str(fit_cfg.train.lr_scheduler_type),
        warmup_steps=int(fit_cfg.train.lr_warmup_steps) if fit_cfg.train.lr_warmup else 0,
        max_steps=int(stage_plan.total_updates),
        num_train_epochs=1.0,
        logging_steps=logging_steps,
        save_steps=int(fit_cfg.train.save_every_updates),
        save_strategy="steps",
        eval_strategy="no",
        dataloader_num_workers=int(fit_cfg.data.dataloader_num_workers),
        report_to=list(fit_cfg.train.report_to),
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
        dataloader_pin_memory=(device.type == "cuda"),
        max_grad_norm=float(fit_cfg.train.max_grad_norm),
        seed=int(fit_cfg.train.seed),
        save_safetensors=False,
        ignore_data_skip=bool(data_sampling_state_restored),
    ))

    def metadata_builder(checkpoint_name: str | None = None):
        runtime_state["data_sampling_state"] = dataset.sampling_state_dict()
        runtime_state["data_sampling_statistics"] = dataset.sampling_statistics()
        runtime_state["baseline_small_summary"] = None if baseline_small is None else to_jsonable(baseline_small.get("summary"))
        runtime_state["baseline_final_summary"] = None if baseline_final is None else to_jsonable(baseline_final.get("summary"))
        runtime_state["final_full_summary"] = final_eval_info["final_full_summary"]
        runtime_state["compare_vs_baseline"] = final_eval_info["compare_vs_baseline"]
        return build_checkpoint_metadata(
            runtime_state,
            layers_to_patch=layer_idxs,
            motn_cfg=motn_cfg,
            patch_cfg=motn_cfg,
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
        optimizer_logger=logger,
        teacher_model=teacher_model,
        kd_config=kd_config,
        checkpoint_policy=fit_cfg.train,
    )
    trainer.add_callback(
        MOTNScheduleCallback(
            fit_cfg,
            stage_state,
            train_jsonl_path=jsonl_path,
            train_light_jsonl_path=train_light_jsonl_path,
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
    trainer.train(resume_from_checkpoint=getattr(fit_cfg.train, "resume_checkpoint_from", None))

    # Throughput ladders must exercise the real Trainer/optimizer path without
    # serializing a multi-gigabyte final model or running downstream evals.
    if bool(getattr(fit_cfg.train, "benchmark_train_only", False)):
        completed_updates = int(getattr(trainer.state, "global_step", 0))
        run_summary = build_run_summary(
            runtime_state, eval_summary, final_model_dir=None, stop_reason="benchmark_train_only"
        )
        json_dump(run_summary_path, run_summary)
        logger.info("[Benchmark] finished train-only run=%s updates=%s", run_name, completed_updates)
        return {
            "run_dir": str(run_dir),
            "final_model_dir": None,
            "eval_summary": str(eval_summary_path),
            "run_summary": str(run_summary_path),
        }

    _ensure_phase_end_exact_checkpoint(
        trainer,
        model,
        run_dir / "checkpoints",
        int(stage_plan.total_updates),
        logger,
    )

    final_model_dir = run_dir / "final_model"
    def save_final_model(prepared_dir: Path, metadata_override: Dict[str, Any] | None = None) -> None:
        trainer.save_model(str(prepared_dir))
        tokenizer.save_pretrained(prepared_dir)
        metadata_to_save = (
            metadata_override
            if metadata_override is not None
            else metadata_builder(checkpoint_name=final_model_dir.name)
        )
        save_fitmotn_metadata(prepared_dir, metadata_to_save)

    transactional_save_checkpoint(
        final_model_dir,
        save_final_model,
        checkpoint_kind="sft_final",
        update_step=int(getattr(trainer.state, "global_step", stage_plan.total_updates)),
        stage_name=runtime_state.get("current_stage"),
        temp_max_age_sec=float(getattr(fit_cfg.train, "checkpoint_temp_max_age_sec", 3600.0)),
    )

    checkpoint_selection = None
    selected_metadata = None
    if bool(getattr(fit_cfg.train, "checkpoint_selector_enabled", False)):
        selector_tasks = (
            str(fit_cfg.train.checkpoint_selector_primary_task),
            str(fit_cfg.train.checkpoint_selector_secondary_task),
        )
        selector_primary_metrics = {
            "mmlu": "acc,none",
            "gsm8k": "exact_match,strict-match",
        }
        selector_contracts = {}
        for selector_task in selector_tasks:
            if selector_task not in selector_primary_metrics:
                raise ValueError(f"no frozen selector primary metric for task {selector_task!r}")
            selector_settings = resolve_task_eval_settings(
                fit_cfg, selector_task, "mid", str(fit_cfg.eval.primary_eval_backend)
            )
            selector_contracts[selector_task] = {
                "backend": selector_settings["backend"],
                "protocol": selector_settings["protocol"],
                "fewshot": selector_settings["fewshot"],
                "limit": selector_settings["limit"],
                "primary_metric": selector_primary_metrics[selector_task],
            }
        checkpoint_selection = select_checkpoint(
            run_dir,
            eval_summary,
            candidate_updates=fit_cfg.train.checkpoint_selector_candidates,
            primary_task=fit_cfg.train.checkpoint_selector_primary_task,
            secondary_task=fit_cfg.train.checkpoint_selector_secondary_task,
            primary_floor=fit_cfg.train.checkpoint_selector_primary_floor,
            verify_hashes=fit_cfg.train.checkpoint_selector_verify_hashes,
            probe_restore=fit_cfg.train.checkpoint_selector_probe_restore,
            config_audit={
                "source_path": getattr(fit_cfg, "_config_source_path", None),
                "source_sha256": getattr(fit_cfg, "_config_source_sha256", None),
                "resolved_sha256": _resolved_config_sha256(fit_cfg),
            },
            task_contracts=selector_contracts,
        )
        selected_checkpoint = Path(checkpoint_selection["selected_checkpoint"])
        selected_metadata = _validate_selected_metadata(
            load_fitmotn_metadata(selected_checkpoint), checkpoint_selection["selected_update"]
        )
        selection_load_summary = _load_resume_state_into_model(
            model, get_restore_state_dict(selected_metadata), layer_idxs, logger
        )
        checkpoint_selection["load_summary"] = to_jsonable(selection_load_summary)
        write_checkpoint_selection(checkpoint_selection["selection_record"], checkpoint_selection)
        eval_summary["checkpoint_selection"] = to_jsonable(checkpoint_selection)
        runtime_state["checkpoint_selection"] = to_jsonable(checkpoint_selection)
        logger.info(
            "[Selector] selected update=%s checkpoint=%s; republishing final_model with selected weights",
            checkpoint_selection["selected_update"],
            selected_checkpoint,
        )
        transactional_save_checkpoint(
            final_model_dir,
            lambda prepared_dir: save_final_model(prepared_dir, selected_metadata),
            checkpoint_kind="sft_final_selected",
            update_step=int(checkpoint_selection["selected_update"]),
            stage_name=runtime_state.get("current_stage"),
            extra={
                "selected_from": str(selected_checkpoint),
                "selection_record": checkpoint_selection["selection_record"],
            },
            temp_max_age_sec=float(getattr(fit_cfg.train, "checkpoint_temp_max_age_sec", 3600.0)),
        )

    model.eval()
    final_full = run_eval_tasks(
        fit_cfg,
        tasks=fit_cfg.eval.final_tasks,
        logger=logger,
        eval_name="final_full",
        out_root=lm_eval_root,
        eval_mode="final",
        model=model,
        tokenizer=tokenizer,
        model_or_path=final_model_dir,
        allow_backend_skip=allow_non_primary_backend_skip,
    )
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
    metadata = (
        selected_metadata
        if checkpoint_selection is not None
        else metadata_builder(checkpoint_name=final_model_dir.name)
    )
    trainer_control = getattr(trainer, "control", None)
    completed_updates = int(getattr(trainer.state, "global_step", 0))
    stop_reason = (
        "early_stop"
        if bool(getattr(trainer_control, "should_training_stop", False))
        and completed_updates < int(stage_plan.total_updates)
        else None
    )
    run_summary = build_run_summary(runtime_state, eval_summary, final_model_dir=str(final_model_dir), stop_reason=stop_reason)
    json_dump(run_summary_path, run_summary)
    def finalize_sft_model(prepared_dir: Path) -> None:
        save_fitmotn_metadata(prepared_dir, metadata)
        json_dump(prepared_dir / "run_summary.json", run_summary)
    final_manifest_update = (
        int(checkpoint_selection["selected_update"])
        if checkpoint_selection is not None
        else int(getattr(trainer.state, "global_step", stage_plan.total_updates))
    )
    final_manifest_kind = "sft_final_selected" if checkpoint_selection is not None else "sft_final"
    final_manifest_extra = {"final_evaluation_complete": True}
    if checkpoint_selection is not None:
        final_manifest_extra.update({
            "selected_from": checkpoint_selection["selected_checkpoint"],
            "selected_update": checkpoint_selection["selected_update"],
            "selection_record": checkpoint_selection["selection_record"],
            "selection_record_sha256": _sha256_path(checkpoint_selection["selection_record"]),
            "source_checkpoint_manifest_sha256": _sha256_path(
                Path(checkpoint_selection["selected_checkpoint"]) / "fitmotn_checkpoint_manifest.json"
            ),
            "source_fitmotn_state_sha256": _sha256_path(
                Path(checkpoint_selection["selected_checkpoint"]) / "fitmotn_state.pt"
            ),
        })
    transactional_update_checkpoint(
        final_model_dir,
        finalize_sft_model,
        checkpoint_kind=final_manifest_kind,
        update_step=final_manifest_update,
        stage_name=runtime_state.get("current_stage"),
        extra=final_manifest_extra,
        temp_max_age_sec=float(getattr(fit_cfg.train, "checkpoint_temp_max_age_sec", 3600.0)),
    )
    logger.info("[Run] finished %s", run_name)
    return {"run_dir": str(run_dir), "final_model_dir": str(final_model_dir), "eval_summary": str(eval_summary_path), "run_summary": str(run_summary_path)}
