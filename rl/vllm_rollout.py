from __future__ import annotations

import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .rollout_backends import (
    HFRolloutBackend,
    RolloutBatch,
    RolloutGenerationConfig,
    RolloutSyncResult,
    rollout_pad_token_id,
)
from .vllm_sync import SavePolicyCheckpointFn, VLLMPolicySyncManager
from .vllm_actor import VLLMActorClient
from .vllm_integrity import prompt_batch_fingerprint, sampling_fingerprint
from .device_topology import rollout_actor_specs, rollout_topology_config


def _visible_device_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in list(value)]


def _row_index_summary(indices: Sequence[int]) -> Dict[str, Any]:
    values = [int(index) for index in indices]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "first": values[:8],
        "last": values[-8:] if len(values) > 8 else values,
    }


def _output_statistics(outputs: Sequence[Any]) -> Dict[str, Any]:
    lengths: list[int] = []
    finish_reasons: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    completion_count = 0
    for output in outputs:
        completions = list(getattr(output, "outputs", []) or [])
        completion_count += len(completions)
        for completion in completions:
            lengths.append(len(list(getattr(completion, "token_ids", []) or [])))
            reason = getattr(completion, "finish_reason", None)
            finish_reasons["unknown" if reason is None else str(reason).lower()] += 1
            stop_reason = getattr(completion, "stop_reason", None)
            if stop_reason is not None:
                stop_reasons[str(stop_reason)] += 1
    total = int(sum(lengths))
    return {
        "output_row_count": len(outputs),
        "completion_count": int(completion_count),
        "total_generated_tokens": total,
        "mean_generated_tokens_per_response": (
            float(total) / len(lengths) if lengths else 0.0
        ),
        "max_generated_tokens": max(lengths) if lengths else 0,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
    }


def _actor_consistency_metadata(actors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    active = [item for item in actors.values() if int(item.get("row_count", 0)) > 0]
    sampling = []
    for item in active:
        per_prompt = list(item.get("sampling_kwargs_by_prompt") or [])
        candidates = per_prompt or [dict(item.get("sampling_kwargs") or {})]
        sampling.extend(
            {key: value for key, value in dict(candidate).items() if key != "seed"}
            for candidate in candidates
        )
    engine_keys = ("model", "tokenizer", "dtype", "max_model_len", "max_num_seqs", "model_impl")
    engine_views = [
        {key: dict(item.get("engine_config") or {}).get(key) for key in engine_keys}
        for item in active
    ]
    observed_devices = [
        tuple(_visible_device_list(item.get("observed_cuda_visible_devices")))
        for item in active
    ]
    return {
        "sampling_params_consistent": not sampling or all(value == sampling[0] for value in sampling[1:]),
        "engine_config_consistent": not engine_views or all(
            value == engine_views[0] for value in engine_views[1:]
        ),
        "engine_config_comparison_keys": list(engine_keys),
        "observed_cuda_device_sets_unique": len(set(observed_devices)) == len(observed_devices),
    }


def assert_multi_actor_dispatch(
    dispatch: Dict[str, Any],
    *,
    expected_row_count: Optional[int] = None,
    expected_prompt_count: Optional[int] = None,
    expected_num_samples: Optional[int] = None,
    require_overlap: bool = True,
) -> None:
    """Strict audit-only validation; it never changes actor assignment."""
    actor_count = int(dispatch.get("actor_count", 0))
    actors = dict(dispatch.get("actors") or {})
    active_actor_count = sum(1 for item in actors.values() if int(item.get("row_count", 0)) > 0)
    actual_assigned_rows = sum(int(item.get("row_count", 0)) for item in actors.values())
    reported_rows = int(dispatch.get("row_count", 0))
    expected_rows = reported_rows if expected_row_count is None else int(expected_row_count)
    reported_prompt_count = int(
        dispatch.get("prompt_count", dispatch.get("prompt_batch_size", reported_rows))
    )
    expected_prompts = (
        reported_prompt_count if expected_prompt_count is None else int(expected_prompt_count)
    )
    num_samples = int(
        dispatch.get("num_samples", dispatch.get("num_rollouts", 1))
        if expected_num_samples is None
        else expected_num_samples
    )
    actual_assigned_prompts = sum(
        int(item.get("prompt_count", item.get("prompt_group_count", 0)))
        for item in actors.values()
    )
    expected_active_actor_count = min(actor_count, expected_prompts) if expected_prompts > 0 else 0
    errors = []
    if actor_count > 1 and active_actor_count < expected_active_actor_count:
        errors.append(
            f"active_actor_count={active_actor_count} < expected active actors={expected_active_actor_count}"
        )
    if len(actors) != actor_count:
        errors.append(f"actor metadata count={len(actors)} != actor_count={actor_count}")
    if int(dispatch.get("active_actor_count", active_actor_count)) != active_actor_count:
        errors.append(
            "reported active_actor_count="
            f"{dispatch.get('active_actor_count')} != observed active_actor_count={active_actor_count}"
        )
    if actual_assigned_rows != reported_rows:
        errors.append(
            f"actor row_count sum={actual_assigned_rows} != dispatch row_count={reported_rows}"
        )
    if actual_assigned_rows != expected_rows:
        errors.append(f"actor row_count sum={actual_assigned_rows} != expected rows={expected_rows}")
    static_prompt_sharding = str(dispatch.get("dispatch_path", "")).startswith(
        "static_multi_sample"
    )
    if static_prompt_sharding:
        if actual_assigned_prompts != reported_prompt_count:
            errors.append(
                "actor prompt_count sum="
                f"{actual_assigned_prompts} != dispatch prompt_count={reported_prompt_count}"
            )
        if actual_assigned_prompts != expected_prompts:
            errors.append(
                f"actor prompt_count sum={actual_assigned_prompts} != expected prompts={expected_prompts}"
            )
    if actor_count > 1 and expected_prompts >= actor_count:
        empty = sorted(name for name, item in actors.items() if int(item.get("row_count", 0)) == 0)
        if empty:
            errors.append(f"configured actors received zero rows: {empty}")
    for name, item in actors.items():
        configured = _visible_device_list(item.get("configured_cuda_visible_devices"))
        observed = _visible_device_list(item.get("observed_cuda_visible_devices"))
        if configured and configured != observed:
            errors.append(
                f"actor {name!r} CUDA_VISIBLE_DEVICES mismatch: configured={configured}, observed={observed}"
            )
        if int(item.get("row_count", 0)) > 0:
            completion_count = item.get("completion_count")
            if completion_count is not None and int(completion_count) != int(item["row_count"]):
                errors.append(
                    f"actor {name!r} completion_count={completion_count} != row_count={item['row_count']}"
                )
            if item.get("llm_generate_called") is False:
                errors.append(f"actor {name!r} did not report an LLM.generate call")
    completion_counts_by_prompt = dispatch.get("completion_counts_by_prompt")
    if completion_counts_by_prompt is not None:
        counts = list(dict(completion_counts_by_prompt).values())
        if len(counts) != expected_prompts:
            errors.append(
                f"per-prompt completion metadata count={len(counts)} != expected prompts={expected_prompts}"
            )
        bad_counts = [int(value) for value in counts if int(value) != num_samples]
        if bad_counts:
            errors.append(
                f"per-prompt completion counts differ from num_samples={num_samples}: {bad_counts[:8]}"
            )
    if require_overlap and actor_count > 1 and expected_active_actor_count > 1:
        intervals = [
            (str(name), float(item["generate_start_time"]), float(item["generate_end_time"]))
            for name, item in actors.items()
            if int(item.get("row_count", 0)) > 0
            and item.get("generate_start_time") is not None
            and item.get("generate_end_time") is not None
        ]
        overlap = False
        for index, (_, start, end) in enumerate(intervals):
            for _, other_start, other_end in intervals[index + 1 :]:
                if min(end, other_end) - max(start, other_start) > 0.0:
                    overlap = True
                    break
            if overlap:
                break
        if len(intervals) != expected_active_actor_count:
            errors.append("missing generate timing interval for an active actor")
        elif not overlap:
            errors.append("active actor generate intervals do not overlap (serial execution)")
    if errors:
        raise RuntimeError("multi-actor dispatch audit failed: " + "; ".join(errors))


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def _format_oom_message(exc: BaseException) -> str:
    return (
        "vLLM failed with a CUDA out-of-memory error. Stage 4D does not silently move, unload, "
        "or offload the HF training model. Configure vLLM placement externally, lower "
        "rl.vllm_gpu_memory_utilization, reduce rl.vllm_max_model_len/rl.vllm_max_num_seqs, "
        "or set rl.vllm_fallback_to_hf=true for explicit HF rollout fallback. "
        f"Original error: {exc}"
    )


class VLLMRolloutBackend:
    name = "vllm"

    def __init__(
        self,
        *,
        fit_cfg,
        rl_dir: Path,
        save_policy_checkpoint: SavePolicyCheckpointFn,
        logger=None,
        transfer_training_names: Optional[Sequence[str]] = None,
    ):
        self.fit_cfg = fit_cfg
        self.rl_dir = Path(rl_dir)
        self.logger = logger
        self.sync_manager = VLLMPolicySyncManager(
            fit_cfg=fit_cfg,
            rl_dir=self.rl_dir,
            save_policy_checkpoint=save_policy_checkpoint,
            logger=logger,
            transfer_training_names=transfer_training_names,
        )
        self.llm = None
        self._vllm = None
        self._sampling_params_cls = None
        self._actor_specs = rollout_actor_specs(fit_cfg.rl) if self.uses_subprocess_actor else []
        self._actors: Dict[str, VLLMActorClient] = {}
        self._actor_resource_snapshot: Dict[str, Dict[str, Any]] = {}
        self._engine_policy_descriptor: Dict[str, Any] = {}
        self._rollout_session_id = uuid.uuid4().hex[:12]
        self._rollout_request_sequence = 0
        self._last_actor_dispatch_metadata: Dict[str, Any] = {}
        self._last_sync = RolloutSyncResult(synced=False, policy_version=-1, policy_lag_updates=0)
        self._hf_fallback = HFRolloutBackend(logger=logger)

    @property
    def uses_subprocess_actor(self) -> bool:
        return str(getattr(self.fit_cfg.rl, "vllm_execution_mode", "in_process") or "in_process").strip().lower() == "subprocess"

    @property
    def last_actor_dispatch_metadata(self) -> Dict[str, Any]:
        return dict(self._last_actor_dispatch_metadata)

    def _actor_observability(self, name: str) -> Dict[str, Any]:
        spec = next(item for item in self._actor_specs if str(item["name"]) == str(name))
        snapshot = dict(self._actor_resource_snapshot.get(name) or {})
        startup = dict(snapshot.get("startup") or {})
        engine = dict(snapshot.get("engine") or {})
        latest = dict(snapshot.get("latest") or {})
        startup_resources = dict(startup.get("actor_resources") or {})
        engine_resources = dict(engine.get("actor_resources") or {})
        latest_resources = dict(latest.get("actor_resources") or {})
        observed_resources = latest_resources or engine_resources or startup_resources
        client = self._actors.get(name)
        generate_info = dict(getattr(client, "last_generate_info", {}) or {})
        generate_resources = dict(generate_info.get("actor_resources") or {})
        if generate_resources:
            observed_resources = generate_resources
        engine_resources_at_load = dict(
            generate_info.get("engine_resources_at_load") or engine_resources
        )
        return {
            "actor_name": str(name),
            "configured_cuda_visible_devices": list(spec["cuda_visible_devices"]),
            "configured_cuda_visible_devices_raw": list(
                spec.get("cuda_visible_devices_configured") or spec["cuda_visible_devices"]
            ),
            "observed_cuda_visible_devices": observed_resources.get("cuda_visible_devices"),
            "actor_pid": observed_resources.get("process_id") or startup_resources.get("process_id"),
            "torch_cuda_device_count": engine_resources_at_load.get("cuda_device_count"),
            "torch_cuda_current_device": engine_resources_at_load.get("cuda_current_device"),
            "enginecore_pids": list(engine_resources_at_load.get("enginecore_process_ids") or []),
            "os_child_process_ids": list(engine_resources_at_load.get("os_child_process_ids") or []),
            "engine_config": dict(generate_info.get("engine_config") or engine.get("engine_config") or {}),
            "engine_topology": dict(engine.get("engine_topology") or latest.get("engine_topology") or {}),
            "sampling_kwargs": dict(generate_info.get("sampling_kwargs") or {}),
            "sampling_kwargs_by_prompt": [
                dict(item) for item in list(generate_info.get("sampling_kwargs_by_prompt") or [])
            ],
            "actor_generate_start_time": generate_info.get("generate_start_time"),
            "actor_generate_end_time": generate_info.get("generate_end_time"),
            "actor_generate_sec": generate_info.get("generate_sec"),
            "llm_generate_called": bool(generate_info.get("llm_generate_called", False)),
        }

    def _actor_client(self, name: Optional[str] = None) -> VLLMActorClient:
        if not self._actor_specs:
            raise RuntimeError("No subprocess vLLM actor topology is configured")
        actor_name = str(name or self._actor_specs[0]["name"])
        spec = next((item for item in self._actor_specs if item["name"] == actor_name), None)
        if spec is None:
            raise KeyError(f"Unknown vLLM rollout actor {actor_name!r}")
        if actor_name not in self._actors:
            self._actors[actor_name] = VLLMActorClient(
                start_method=str(getattr(self.fit_cfg.rl, "vllm_actor_start_method", "spawn") or "spawn"),
                request_timeout_sec=float(getattr(self.fit_cfg.rl, "vllm_actor_request_timeout_sec", 600.0)),
                shutdown_timeout_sec=float(getattr(self.fit_cfg.rl, "vllm_actor_shutdown_timeout_sec", 30.0)),
                cuda_visible_devices=list(spec["cuda_visible_devices"]),
            )
        return self._actors[actor_name]

    def _import_vllm(self):
        if self._vllm is not None:
            return self._vllm, self._sampling_params_cls
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:
            raise ImportError("vLLM is required when rl.rollout_backend='vllm'. Install vllm or use rollout_backend='hf'.") from exc
        self._vllm = LLM
        self._sampling_params_cls = SamplingParams
        return LLM, SamplingParams

    def _llm_kwargs(self, export_dir: str, actor_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rl_cfg = self.fit_cfg.rl
        kwargs: Dict[str, Any] = {
            "model": str(export_dir),
            "tokenizer": str(export_dir),
            "trust_remote_code": True,
            "model_impl": str(getattr(rl_cfg, "vllm_model_impl", "transformers") or "transformers"),
            "enforce_eager": bool(getattr(rl_cfg, "vllm_enforce_eager", True)),
            "dtype": str(getattr(rl_cfg, "vllm_dtype", "auto") or "auto"),
            "tensor_parallel_size": int(
                actor_spec["tensor_parallel_size"] if actor_spec else getattr(rl_cfg, "vllm_tensor_parallel_size", 1)
            ),
            "gpu_memory_utilization": float(
                actor_spec["gpu_memory_utilization"] if actor_spec else getattr(rl_cfg, "vllm_gpu_memory_utilization", 0.85)
            ),
        }
        optional_values = [
            ("max_model_len", int(actor_spec["max_model_len"] if actor_spec else getattr(rl_cfg, "vllm_max_model_len", 0))),
            ("max_num_seqs", int(actor_spec["max_num_seqs"] if actor_spec else getattr(rl_cfg, "vllm_max_num_seqs", 0))),
            ("seed", actor_spec["engine_seed"] if actor_spec else getattr(rl_cfg, "seed", None)),
        ]
        for key, value in optional_values:
            if key == "seed" and value is not None:
                kwargs[key] = int(value)
            elif value is not None and int(value) > 0:
                kwargs[key] = value
        if bool(getattr(rl_cfg, "vllm_disable_log_stats", True)):
            kwargs["disable_log_stats"] = True
        # Subprocess actors select their device through an isolated
        # CUDA_VISIBLE_DEVICES and see the first assigned GPU as cuda:0.
        # Do not pass ``device`` to LLM: vLLM 0.19 no longer accepts it in
        # EngineArgs.  Keep the legacy option only for in-process runtimes
        # that still expose that argument.
        device = None if actor_spec else getattr(rl_cfg, "vllm_device", None)
        if device is not None and str(device).strip():
            kwargs["device"] = str(device).strip()
        if bool(getattr(rl_cfg, "vllm_enable_sleep_mode", False)):
            kwargs["enable_sleep_mode"] = True
        if str(getattr(rl_cfg, "vllm_sync_strategy", "export_reload") or "export_reload").strip().lower() in {
            "weight_transfer_nccl",
            "weight_transfer_ipc",
        }:
            try:
                from vllm.config import WeightTransferConfig  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "vLLM WeightTransferConfig is required for native weight-transfer sync. "
                    "Use rl.vllm_sync_strategy='export_reload' or install a vLLM build with weight-transfer support."
                ) from exc
            backend = str(getattr(rl_cfg, "vllm_weight_transfer_backend", "nccl") or "nccl").strip().lower()
            kwargs["weight_transfer_config"] = WeightTransferConfig(backend=backend)
            # A policy update must not reuse prefix-cache entries computed by
            # the previous policy.  Keep prefix caching disabled on the native
            # sync path in addition to the explicit post-update reset call.
            kwargs["enable_prefix_caching"] = False
        return kwargs

    def _build_engine(self, export_dir: str, *, policy_descriptor: Optional[Dict[str, Any]] = None) -> float:
        descriptor = dict(policy_descriptor or {})
        if self.uses_subprocess_actor:
            wall_start = time.perf_counter()
            failures = []
            snapshots: Dict[str, Dict[str, Any]] = {}

            def load_actor(spec: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
                name = str(spec["name"])
                client = self._actor_client(name)
                load_sec = client.load_engine(
                    self._llm_kwargs(export_dir, actor_spec=spec),
                    policy_descriptor=descriptor,
                )
                ping = client.ping()
                loaded = dict(ping.get("policy_descriptor") or {})
                if bool(getattr(self.fit_cfg.rl, "vllm_verify_engine_policy", True)) and loaded != descriptor:
                    raise RuntimeError(
                        f"actor {name!r} loaded unexpected policy descriptor: expected={descriptor}, loaded={loaded}"
                    )
                actor_info = dict(client.last_engine_info or {})
                engine_topology = dict(actor_info.get("engine_topology") or {})
                observed = engine_topology.get("observed_tensor_parallel_size")
                if observed is not None and int(observed) != int(spec["tensor_parallel_size"]):
                    raise RuntimeError(
                        f"actor {name!r} TP mismatch: expected={spec['tensor_parallel_size']}, observed={observed}"
                    )
                return name, {
                    "configured_topology": dict(spec),
                    "startup": dict(client.startup_info or {}),
                    "engine": actor_info,
                    "latest": ping,
                    "load_sec": float(load_sec),
                    "policy_verified": loaded == descriptor,
                }

            for spec in self._actor_specs:
                self._actor_client(str(spec["name"]))
            with ThreadPoolExecutor(max_workers=len(self._actor_specs), thread_name_prefix="fitmotn-vllm-load") as executor:
                future_map = {executor.submit(load_actor, spec): spec["name"] for spec in self._actor_specs}
                for future in as_completed(future_map):
                    try:
                        name, snapshot = future.result()
                        snapshots[name] = snapshot
                    except Exception as exc:
                        failures.append(f"{future_map[future]!r}: {exc}")
            if failures:
                self.close()
                raise RuntimeError(f"Failed to build all vLLM rollout actors: {failures}")
            self._actor_resource_snapshot = snapshots
            self._engine_policy_descriptor = descriptor
            return max(0.0, time.perf_counter() - wall_start)
        LLM, _ = self._import_vllm()
        if bool(getattr(self.fit_cfg.rl, "vllm_empty_cache_before_engine_init", False)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = time.perf_counter()
        llm_kwargs = self._llm_kwargs(export_dir)
        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as exc:
            # vLLM 0.19 removed the legacy EngineArgs.device keyword.  An
            # in-process caller may still have vllm_device configured; retry
            # only for this exact compatibility failure and otherwise retain
            # the original exception.
            if "device" in llm_kwargs and "unexpected keyword argument 'device'" in str(exc):
                llm_kwargs = dict(llm_kwargs)
                llm_kwargs.pop("device", None)
                if self.logger is not None:
                    self.logger.warning(
                        "[VLLMCompat] runtime rejected EngineArgs.device; retrying with CUDA_VISIBLE_DEVICES placement"
                    )
                self.llm = LLM(**llm_kwargs)
            else:
                raise
        except Exception as exc:
            if _is_cuda_oom(exc) and bool(getattr(self.fit_cfg.rl, "vllm_fail_on_cuda_oom", True)):
                raise RuntimeError(_format_oom_message(exc)) from exc
            raise
        self._engine_policy_descriptor = descriptor
        return max(0.0, time.perf_counter() - start)

    def sync_policy(
        self,
        *,
        model,
        tokenizer,
        update_step: int,
        force: bool = False,
    ) -> RolloutSyncResult:
        try:
            if (
                bool(getattr(self.fit_cfg.rl, "vllm_enable_sleep_mode", False))
                and self.sync_manager.strategy in {
                    "export_reload",
                    "weight_transfer_dryrun_static",
                    "weight_transfer_dryrun_runtime",
                }
                and self.sync_manager.sync_due(update_step, force=force)
            ):
                sleep_level = max(1, int(getattr(self.fit_cfg.rl, "vllm_sleep_level_before_sync", 1)))
                if self.uses_subprocess_actor and self._actors:
                    with ThreadPoolExecutor(max_workers=len(self._actors)) as executor:
                        futures = [
                            executor.submit(client.sleep, level=sleep_level)
                            for client in self._actors.values()
                            if client.is_alive
                        ]
                        for future in futures:
                            future.result()
                elif self.llm is not None and callable(getattr(self.llm, "sleep", None)):
                    self.llm.sleep(level=sleep_level)
            sync_result = self.sync_manager.sync(
                model=model,
                tokenizer=tokenizer,
                update_step=update_step,
                force=force,
                llm=self.llm,
                runtime_params=self._runtime_params(),
                actors=(
                    [
                        (str(spec["name"]), self._actor_client(str(spec["name"])), dict(spec))
                        for spec in self._actor_specs
                        if str(spec["name"]) in self._actors and self._actors[str(spec["name"])].is_alive
                    ]
                    if self.uses_subprocess_actor
                    else None
                ),
            )
            native_sync = bool(sync_result.metadata.get("vllm_weight_transfer_native_sync", False))
            if sync_result.synced and not native_sync:
                self._unload_engine()
                descriptor = {
                    "policy_version": int(sync_result.policy_version),
                    "policy_fingerprint": sync_result.metadata.get("vllm_policy_fingerprint"),
                    "export_dir": str(sync_result.export_dir),
                }
                rebuild_sec = self._build_engine(
                    str(sync_result.export_dir),
                    policy_descriptor=descriptor,
                )
                sync_result.engine_rebuild_sec = rebuild_sec
                sync_result.metadata["vllm_engine_rebuild_sec"] = rebuild_sec
                sync_result.metadata["vllm_engine_policy_verified"] = bool(
                    getattr(self.fit_cfg.rl, "vllm_verify_engine_policy", True)
                )
                sync_result.metadata["vllm_engine_policy_descriptor"] = descriptor
                if self.uses_subprocess_actor:
                    sync_result.metadata["vllm_actor_resources"] = dict(self._actor_resource_snapshot)
                    sync_result.metadata["vllm_rollout_actor_count"] = len(self._actor_specs)
                    sync_result.metadata["vllm_all_actors_policy_verified"] = all(
                        bool(snapshot.get("policy_verified"))
                        for snapshot in self._actor_resource_snapshot.values()
                    )
            elif sync_result.synced and native_sync:
                self._engine_policy_descriptor = {
                    "policy_version": int(sync_result.policy_version),
                    "policy_fingerprint": sync_result.metadata.get("vllm_policy_fingerprint"),
                    "export_dir": None if sync_result.export_dir is None else str(sync_result.export_dir),
                    "weight_transfer_scope": sync_result.metadata.get("vllm_weight_transfer_scope"),
                    "weight_transfer_plan_fingerprint": sync_result.metadata.get(
                        "weight_transfer_plan_fingerprint"
                    ),
                }
                if self.uses_subprocess_actor:
                    actor_results = dict(
                        sync_result.metadata.get("weight_transfer_actor_results") or {}
                    )
                    for name in self._actors:
                        actor_result = dict(actor_results.get(name) or {})
                        commit = dict(actor_result.get("commit") or {})
                        loaded = dict(commit.get("policy_descriptor") or {})
                        if (
                            actor_result.get("committed") is not True
                            or commit.get("committed") is not True
                            or loaded != self._engine_policy_descriptor
                            or commit.get("actor_status") != "READY"
                        ):
                            raise RuntimeError(
                                f"actor {name!r} failed post-commit policy barrier: "
                                f"expected={self._engine_policy_descriptor}, commit={commit}"
                            )
                        snapshot = self._actor_resource_snapshot.setdefault(name, {})
                        snapshot["latest"] = {
                            "policy_descriptor": loaded,
                            "actor_status": commit.get("actor_status"),
                            "source": "native_sync_commit_receipt",
                        }
                        snapshot["policy_verified"] = True
                    if self.logger is not None:
                        self.logger.info(
                            "[VLLMSync] update=%s phase=post_commit_barrier_complete actors=%s",
                            int(update_step),
                            len(self._actors),
                        )
                    sync_result.metadata["vllm_engine_rebuilt"] = False
                    sync_result.metadata["vllm_all_actors_policy_verified"] = True
                    sync_result.metadata["vllm_engine_policy_descriptor"] = dict(
                        self._engine_policy_descriptor
                    )
            self._last_sync = sync_result
            return sync_result
        except Exception:
            self.close()
            raise

    def _runtime_params(self) -> Optional[Dict[str, Any]]:
        if self.uses_subprocess_actor:
            return None
        llm = self.llm
        if llm is None:
            return None
        for obj in [
            llm,
            getattr(llm, "model", None),
            getattr(getattr(llm, "llm_engine", None), "model", None),
            getattr(getattr(getattr(llm, "llm_engine", None), "model_executor", None), "driver_worker", None),
        ]:
            if obj is None:
                continue
            named_parameters = getattr(obj, "named_parameters", None)
            if callable(named_parameters):
                try:
                    return {str(name): param for name, param in named_parameters()}
                except Exception:
                    pass
        candidates = [
            "llm_engine.model_executor.driver_worker.model_runner.model",
            "llm_engine.model_executor.driver_worker.model_runner.model_runner.model",
        ]
        for candidate in candidates:
            obj = llm
            try:
                for part in candidate.split("."):
                    obj = getattr(obj, part)
                named_parameters = getattr(obj, "named_parameters", None)
                if callable(named_parameters):
                    return {str(name): param for name, param in named_parameters()}
            except Exception:
                continue
        return None

    def _sampling_params_kwargs(self, tokenizer, generation_config: RolloutGenerationConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_tokens": int(generation_config.max_new_tokens),
            "temperature": float(generation_config.temperature),
            "top_p": float(generation_config.top_p),
        }
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["stop_token_ids"] = [int(eos_token_id)]
        return kwargs

    def _sampling_params(self, tokenizer, generation_config: RolloutGenerationConfig):
        kwargs = self._sampling_params_kwargs(tokenizer, generation_config)
        if self.uses_subprocess_actor:
            return kwargs
        _, SamplingParams = self._import_vllm()
        return SamplingParams(**kwargs)

    def generate_static_samples(
        self,
        *,
        prompt_token_ids: Sequence[int],
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        prompt_index: int = 0,
        eos_token_id: Optional[int] = None,
        decoding: str = "sample",
        stop: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible single-prompt wrapper around static batch generation."""
        return self.generate_static_samples_batch(
            prompt_token_ids=[prompt_token_ids],
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=[seed],
            prompt_indices=[prompt_index],
            eos_token_id=eos_token_id,
            decoding=decoding,
            stop=stop,
            top_k=top_k,
        )[0]

    def generate_static_samples_batch(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seeds: Sequence[int],
        prompt_indices: Optional[Sequence[int]] = None,
        eos_token_id: Optional[int] = None,
        decoding: str = "sample",
        stop: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Generate n samples for a batch of unique prompts without policy sync."""
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be > 0")
        decoding = str(decoding).strip().lower()
        if decoding not in {"greedy", "sample"}:
            raise ValueError("decoding must be 'greedy' or 'sample'")
        if decoding == "greedy" and int(num_samples) != 1:
            raise ValueError("greedy static generation requires num_samples=1")
        ids_batch = [[int(value) for value in row] for row in prompt_token_ids]
        if not ids_batch or any(not row for row in ids_batch):
            raise ValueError("prompt_token_ids must contain non-empty prompt rows")
        prompt_count = len(ids_batch)
        seed_values = [int(value) for value in seeds]
        if len(seed_values) != prompt_count:
            raise ValueError(
                f"seeds must match prompt count: seeds={len(seed_values)}, prompts={prompt_count}"
            )
        global_prompt_indices = (
            list(range(prompt_count))
            if prompt_indices is None
            else [int(value) for value in prompt_indices]
        )
        if len(global_prompt_indices) != prompt_count:
            raise ValueError(
                "prompt_indices must match prompt count: "
                f"indices={len(global_prompt_indices)}, prompts={prompt_count}"
            )
        base_sampling_kwargs: Dict[str, Any] = {
            "n": int(num_samples),
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0 if decoding == "greedy" else float(temperature),
        }
        if decoding == "sample":
            base_sampling_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                base_sampling_kwargs["top_k"] = int(top_k)
        if stop:
            base_sampling_kwargs["stop"] = [str(value) for value in stop]
        if eos_token_id is not None:
            base_sampling_kwargs["stop_token_ids"] = [int(eos_token_id)]
        sampling_kwargs_by_prompt = [
            {**base_sampling_kwargs, "seed": seed_value} for seed_value in seed_values
        ]
        expected = dict(self._engine_policy_descriptor)
        wall_start = time.perf_counter()

        if not self.uses_subprocess_actor:
            if self.llm is None:
                raise RuntimeError("vLLM engine is not initialized")
            if bool(getattr(self.fit_cfg.rl, "vllm_verify_engine_policy", True)) and not expected:
                raise RuntimeError("vLLM static engine has no policy provenance descriptor")
            _, SamplingParams = self._import_vllm()
            params = [SamplingParams(**kwargs) for kwargs in sampling_kwargs_by_prompt]
            try:
                outputs = self.llm.generate(
                    prompts=[{"prompt_token_ids": ids} for ids in ids_batch],
                    sampling_params=params,
                )
            except TypeError:
                outputs = self.llm.generate(
                    prompt_token_ids=ids_batch,
                    sampling_params=params,
                )
            merged_outputs = list(outputs)
        else:
            actor_names = [str(spec["name"]) for spec in self._actor_specs]
            if not actor_names:
                raise RuntimeError("No subprocess vLLM actors are configured")
            assignments: Dict[str, list[int]] = {name: [] for name in actor_names}
            for local_index, global_index in enumerate(global_prompt_indices):
                assignments[actor_names[global_index % len(actor_names)]].append(local_index)
            merged_outputs: list[Any] = [None] * prompt_count
            actors: Dict[str, Dict[str, Any]] = {}
            for spec in self._actor_specs:
                name = str(spec["name"])
                local_indices = assignments[name]
                assigned_prompt_indices = [global_prompt_indices[index] for index in local_indices]
                conceptual_rows = [
                    prompt_index * int(num_samples) + sample_index
                    for prompt_index in assigned_prompt_indices
                    for sample_index in range(int(num_samples))
                ]
                actors[name] = {
                    **self._actor_observability(name),
                    "prompt_indices": assigned_prompt_indices,
                    "prompt_indices_summary": _row_index_summary(assigned_prompt_indices),
                    "prompt_count": len(local_indices),
                    "prompt_group_count": len(local_indices),
                    "group_indices": assigned_prompt_indices,
                    "row_indices": conceptual_rows,
                    "row_indices_summary": _row_index_summary(conceptual_rows),
                    "row_count": len(local_indices) * int(num_samples),
                    "dispatch_start_time": None,
                    "generate_start_time": None,
                    "generate_end_time": None,
                    "generate_sec": 0.0,
                    "output_row_count": 0,
                    "completion_count": 0,
                    "prompt_completion_counts": {},
                    "total_generated_tokens": 0,
                    "mean_generated_tokens_per_response": 0.0,
                    "max_generated_tokens": 0,
                    "generated_tokens_per_sec": None,
                    "finish_reason_counts": {},
                    "stop_reason_counts": {},
                    "prompt_seeds": [seed_values[index] for index in local_indices],
                    "engine_seed": int(spec["engine_seed"]),
                    "policy_descriptor": expected,
                }

            def generate_actor(name: str, local_indices: list[int], dispatch_start_time: float):
                client = self._actor_client(name)
                generate_start_time = time.perf_counter()
                outputs = client.generate(
                    prompt_token_ids=[ids_batch[index] for index in local_indices],
                    sampling_kwargs_by_prompt=[
                        sampling_kwargs_by_prompt[index] for index in local_indices
                    ],
                    expected_policy_descriptor=expected,
                )
                generate_end_time = time.perf_counter()
                return (
                    name,
                    local_indices,
                    outputs,
                    dispatch_start_time,
                    generate_start_time,
                    generate_end_time,
                )

            failures = []
            with ThreadPoolExecutor(
                max_workers=len(actor_names), thread_name_prefix="fitmotn-vllm-static"
            ) as executor:
                future_map = {}
                for name, local_indices in assignments.items():
                    if not local_indices:
                        continue
                    dispatch_start_time = time.perf_counter()
                    actors[name]["dispatch_start_time"] = float(dispatch_start_time)
                    future_map[
                        executor.submit(
                            generate_actor, name, local_indices, dispatch_start_time
                        )
                    ] = name
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        (
                            _,
                            local_indices,
                            outputs,
                            dispatch_start_time,
                            generate_start_time,
                            generate_end_time,
                        ) = future.result()
                        if len(outputs) != len(local_indices):
                            raise RuntimeError(
                                f"actor {name!r} returned {len(outputs)} prompt rows for "
                                f"{len(local_indices)} assigned prompts"
                            )
                        completion_counts = {}
                        for local_index, output in zip(local_indices, outputs):
                            completion_count = len(list(getattr(output, "outputs", []) or []))
                            if completion_count != int(num_samples):
                                raise RuntimeError(
                                    f"actor {name!r} prompt_index={global_prompt_indices[local_index]} "
                                    f"returned {completion_count} completions; expected {int(num_samples)}"
                                )
                            completion_counts[str(global_prompt_indices[local_index])] = completion_count
                            merged_outputs[local_index] = output
                        elapsed = max(0.0, generate_end_time - generate_start_time)
                        statistics = _output_statistics(outputs)
                        actors[name].update(self._actor_observability(name))
                        actors[name].update(statistics)
                        actors[name].update(
                            {
                                "prompt_completion_counts": completion_counts,
                                "dispatch_start_time": float(dispatch_start_time),
                                "generate_start_time": float(generate_start_time),
                                "generate_end_time": float(generate_end_time),
                                "generate_sec": float(elapsed),
                                "generated_tokens_per_sec": (
                                    float(statistics["total_generated_tokens"]) / elapsed
                                    if elapsed > 0.0
                                    else None
                                ),
                            }
                        )
                    except Exception as exc:
                        failures.append(f"{name!r}: {exc}")
            if failures or any(output is None for output in merged_outputs):
                if not failures:
                    failures.append("merged static batch contains missing prompt outputs")
                cleanup_error = None
                try:
                    self.close()
                except Exception as exc:
                    cleanup_error = str(exc)
                raise RuntimeError(
                    "Multi-actor vLLM static batch failed; all actors were closed: "
                    f"failures={failures}, cleanup_error={cleanup_error}"
                )

            active_intervals = [
                (float(item["generate_start_time"]), float(item["generate_end_time"]))
                for item in actors.values()
                if item.get("generate_start_time") is not None
                and item.get("generate_end_time") is not None
            ]
            overlap_sec = (
                max(
                    0.0,
                    min(end for _, end in active_intervals)
                    - max(start for start, _ in active_intervals),
                )
                if len(active_intervals) > 1
                else 0.0
            )
            completion_counts_by_prompt = {
                str(global_prompt_indices[index]): len(
                    list(getattr(output, "outputs", []) or [])
                )
                for index, output in enumerate(merged_outputs)
            }
            total_generated_tokens = sum(
                int(item["total_generated_tokens"]) for item in actors.values()
            )
            wall_sec = max(0.0, time.perf_counter() - wall_start)
            self._last_actor_dispatch_metadata = {
                "dispatch_path": "static_multi_sample_batch",
                "actor_count": len(actor_names),
                "active_actor_count": sum(
                    1 for item in actors.values() if int(item["prompt_count"]) > 0
                ),
                "prompt_count": prompt_count,
                "prompt_batch_size": prompt_count,
                "prompt_indices": global_prompt_indices,
                "num_samples": int(num_samples),
                "num_rollouts": int(num_samples),
                "decoding": decoding,
                "stochastic_sampling": decoding == "sample",
                "sampling_kwargs": dict(base_sampling_kwargs),
                "completion_count": prompt_count * int(num_samples),
                "completion_counts_by_prompt": completion_counts_by_prompt,
                "row_count": prompt_count * int(num_samples),
                "assigned_row_count": sum(int(item["row_count"]) for item in actors.values()),
                "backend_generate_wall_sec": wall_sec,
                "total_generated_tokens": total_generated_tokens,
                "generated_tokens_per_sec": (
                    float(total_generated_tokens) / wall_sec if wall_sec > 0.0 else None
                ),
                "generate_intervals_overlap": overlap_sec > 0.0,
                "generate_overlap_sec": float(overlap_sec),
                "seed_strategy": "global_seed_plus_dataset_prompt_index",
                "time_source": "time.perf_counter",
                "actors": actors,
                **_actor_consistency_metadata(actors),
            }

        if len(merged_outputs) != prompt_count:
            raise RuntimeError(
                f"vLLM static batch returned {len(merged_outputs)} prompts; expected {prompt_count}"
            )
        serialized_batch: List[List[Dict[str, Any]]] = []
        for global_index, output in zip(global_prompt_indices, merged_outputs):
            completions = list(getattr(output, "outputs", []) or [])
            if len(completions) != int(num_samples):
                raise RuntimeError(
                    f"vLLM static prompt_index={global_index} returned {len(completions)} "
                    f"completions; expected {int(num_samples)}"
                )
            serialized_batch.append(
                [
                    {
                        "token_ids": [
                            int(token)
                            for token in list(getattr(item, "token_ids", []) or [])
                        ],
                        "text": str(getattr(item, "text", "") or ""),
                        "finish_reason": (
                            None
                            if getattr(item, "finish_reason", None) is None
                            else str(getattr(item, "finish_reason"))
                        ),
                        "stop_reason": getattr(item, "stop_reason", None),
                    }
                    for item in completions
                ]
            )
        return serialized_batch

    def _generate_token_ids(
        self,
        *,
        prompts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_params,
        expected_policy_descriptor: Optional[Dict[str, Any]] = None,
    ):
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape for vLLM token prompts")
        prompt_token_ids: List[List[int]] = []
        for row, mask in zip(input_ids, attention_mask):
            effective = row[mask.to(device=row.device, dtype=torch.bool)]
            ids = effective.detach().cpu().to(dtype=torch.long).tolist()
            if not ids:
                raise ValueError("vLLM token prompt contains no unmasked tokens")
            prompt_token_ids.append(ids)
        if self.uses_subprocess_actor:
            if not isinstance(sampling_params, dict):
                raise TypeError("subprocess vLLM actor requires serializable sampling parameter kwargs")
            return self._generate_subprocess_actors(
                prompt_token_ids=prompt_token_ids,
                sampling_kwargs=sampling_params,
                expected_policy_descriptor=expected_policy_descriptor,
            )
        if self.llm is None:
            raise RuntimeError("vLLM engine is not initialized; call sync_policy(..., force=True) before rollout generation")
        if bool(getattr(self.fit_cfg.rl, "vllm_verify_engine_policy", True)):
            expected = dict(expected_policy_descriptor or {})
            if expected and expected != self._engine_policy_descriptor:
                raise RuntimeError(
                    "in-process vLLM policy provenance mismatch: "
                    f"expected={expected}, loaded={self._engine_policy_descriptor}"
                )
        try:
            return self.llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
                sampling_params=sampling_params,
            )
        except TypeError as first_exc:
            try:
                return self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
            except TypeError as second_exc:
                if not bool(getattr(self.fit_cfg.rl, "vllm_allow_text_prompt_fallback", False)):
                    raise RuntimeError(
                        "This vLLM version does not appear to accept token-id prompts via prompts="
                        "[{'prompt_token_ids': ...}] or prompt_token_ids=. Set "
                        "rl.vllm_allow_text_prompt_fallback=true to allow text prompt fallback, or upgrade vLLM. "
                        f"Errors: {first_exc}; {second_exc}"
                    ) from second_exc
                return self.llm.generate(prompts, sampling_params)

    def _generate_subprocess_actors(
        self,
        *,
        prompt_token_ids: list[list[int]],
        sampling_kwargs: Dict[str, Any],
        expected_policy_descriptor: Optional[Dict[str, Any]],
    ):
        actor_names = [str(spec["name"]) for spec in self._actor_specs]
        if not actor_names:
            raise RuntimeError("No vLLM rollout actors are configured")
        group_size = max(1, int(getattr(self.fit_cfg.rl, "group_size", 1)))
        if len(prompt_token_ids) % group_size != 0:
            raise RuntimeError(
                "expanded rollout row count must be divisible by rl.group_size for group-aware sharding: "
                f"rows={len(prompt_token_ids)}, group_size={group_size}"
            )
        assignments: Dict[str, list[int]] = {name: [] for name in actor_names}
        for row_index in range(len(prompt_token_ids)):
            group_index = row_index // group_size
            sample_index = row_index % group_size
            actor_index = (group_index + sample_index) % len(actor_names)
            assignments[actor_names[actor_index]].append(row_index)
        assigned_rows = sorted(index for indices in assignments.values() for index in indices)
        if assigned_rows != list(range(len(prompt_token_ids))):
            raise RuntimeError(f"multi-actor rollout sharding lost or duplicated rows: {assignments}")

        merged: list[Any] = [None] * len(prompt_token_ids)
        dispatch_wall_start = time.perf_counter()
        dispatch: Dict[str, Any] = {
            name: {
                **self._actor_observability(name),
                "row_indices": list(assignments[name]),
                "row_indices_summary": _row_index_summary(assignments[name]),
                "row_count": len(assignments[name]),
                "prompt_group_count": len(
                    {index // group_size for index in assignments[name]}
                ),
                "group_indices": sorted({index // group_size for index in assignments[name]}),
                "sample_indices": [index % group_size for index in assignments[name]],
                "dispatch_start_time": None,
                "generate_start_time": None,
                "generate_end_time": None,
                "generate_sec": 0.0,
                "output_row_count": 0,
                "completion_count": 0,
                "total_generated_tokens": 0,
                "mean_generated_tokens_per_response": 0.0,
                "max_generated_tokens": 0,
                "generated_tokens_per_sec": None,
                "finish_reason_counts": {},
                "stop_reason_counts": {},
                "engine_seed": int(
                    next(spec["engine_seed"] for spec in self._actor_specs if spec["name"] == name)
                ),
                "policy_descriptor": dict(expected_policy_descriptor or {}),
            }
            for name in actor_names
        }
        resource_every = int(getattr(self.fit_cfg.rl, "vllm_actor_resource_log_every", 1))

        def generate_actor(name: str, row_indices: list[int], dispatch_start_time: float):
            if not row_indices:
                return name, [], [], dispatch_start_time, None, None, None
            client = self._actor_client(name)
            generate_start_time = time.perf_counter()
            outputs = client.generate(
                prompt_token_ids=[prompt_token_ids[index] for index in row_indices],
                sampling_kwargs=sampling_kwargs,
                expected_policy_descriptor=expected_policy_descriptor,
            )
            generate_end_time = time.perf_counter()
            resources = None
            if resource_every > 0 and self._rollout_request_sequence % resource_every == 0:
                resources = client.ping()
            return (
                name,
                row_indices,
                outputs,
                dispatch_start_time,
                generate_start_time,
                generate_end_time,
                resources,
            )

        failures = []
        with ThreadPoolExecutor(max_workers=len(actor_names), thread_name_prefix="fitmotn-vllm-generate") as executor:
            future_map = {}
            for name, indices in assignments.items():
                if not indices:
                    continue
                dispatch_start_time = time.perf_counter()
                dispatch[name]["dispatch_start_time"] = float(dispatch_start_time)
                future_map[
                    executor.submit(generate_actor, name, indices, dispatch_start_time)
                ] = name
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    (
                        _,
                        row_indices,
                        outputs,
                        dispatch_start_time,
                        generate_start_time,
                        generate_end_time,
                        resources,
                    ) = future.result()
                    if len(outputs) != len(row_indices):
                        raise RuntimeError(
                            f"actor {name!r} returned {len(outputs)} rows for {len(row_indices)} assigned rows"
                        )
                    for row_index, output in zip(row_indices, outputs):
                        merged[row_index] = output
                    if resources is not None:
                        self._actor_resource_snapshot[name]["latest"] = resources
                    elapsed = max(0.0, float(generate_end_time - generate_start_time))
                    statistics = _output_statistics(outputs)
                    dispatch[name].update(self._actor_observability(name))
                    dispatch[name].update(statistics)
                    dispatch[name].update(
                        {
                            "dispatch_start_time": float(dispatch_start_time),
                            "generate_start_time": float(generate_start_time),
                            "generate_end_time": float(generate_end_time),
                            "generate_sec": elapsed,
                            "generated_tokens_per_sec": (
                                float(statistics["total_generated_tokens"]) / elapsed
                                if elapsed > 0.0
                                else None
                            ),
                        }
                    )
                except Exception as exc:
                    failures.append(f"{name!r}: {exc}")
        if failures:
            raise RuntimeError(f"Multi-actor vLLM rollout failed; the entire rollout batch is invalid: {failures}")
        if any(output is None for output in merged):
            raise RuntimeError("Multi-actor vLLM rollout merge contains missing rows")
        active_intervals = [
            (float(item["generate_start_time"]), float(item["generate_end_time"]))
            for item in dispatch.values()
            if item.get("generate_start_time") is not None and item.get("generate_end_time") is not None
        ]
        overlap_sec = (
            max(0.0, min(end for _, end in active_intervals) - max(start for start, _ in active_intervals))
            if len(active_intervals) > 1
            else 0.0
        )
        self._last_actor_dispatch_metadata = {
            "dispatch_path": "expanded_rows_generate",
            "actor_count": len(actor_names),
            "active_actor_count": sum(1 for value in dispatch.values() if value["row_count"] > 0),
            "group_size": group_size,
            "prompt_batch_size": len(prompt_token_ids) // group_size,
            "num_rollouts": group_size,
            "row_count": len(prompt_token_ids),
            "assigned_row_count": sum(int(value["row_count"]) for value in dispatch.values()),
            "backend_generate_wall_sec": max(0.0, time.perf_counter() - dispatch_wall_start),
            "generate_intervals_overlap": overlap_sec > 0.0,
            "generate_overlap_sec": float(overlap_sec),
            "time_source": "time.perf_counter",
            "actors": dispatch,
            **_actor_consistency_metadata(dispatch),
        }
        return merged

    @staticmethod
    def _extract_generated_ids(output: Any) -> List[int]:
        completions = list(getattr(output, "outputs", []) or [])
        if not completions:
            return []
        token_ids = getattr(completions[0], "token_ids", None)
        if token_ids is None:
            raise RuntimeError("vLLM output did not include generated token_ids; Stage 4D does not decode/re-tokenize text")
        return [int(tok) for tok in list(token_ids)]

    @staticmethod
    def build_full_sequences_from_token_outputs(
        *,
        input_ids: torch.Tensor,
        generated_token_ids: Sequence[Sequence[int]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, List[int]]:
        if int(input_ids.shape[0]) != len(generated_token_ids):
            raise ValueError("input_ids and generated_token_ids must have the same batch size")
        full_rows: List[torch.Tensor] = []
        original_lens: List[int] = []
        max_len = int(input_ids.shape[1]) + max((len(ids) for ids in generated_token_ids), default=0)
        for row_idx, ids in enumerate(generated_token_ids):
            suffix = torch.tensor(list(ids), dtype=input_ids.dtype, device=input_ids.device)
            full = torch.cat([input_ids[row_idx], suffix], dim=0)
            original_lens.append(int(full.numel()))
            if int(full.numel()) < max_len:
                pad = torch.full(
                    (max_len - int(full.numel()),),
                    int(pad_token_id),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                full = torch.cat([full, pad], dim=0)
            full_rows.append(full)
        if not full_rows:
            raise RuntimeError("No vLLM rollout rows were produced")
        return torch.stack(full_rows, dim=0), original_lens

    def generate(
        self,
        *,
        model,
        tokenizer,
        prompts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: RolloutGenerationConfig,
        update_step: int,
    ) -> RolloutBatch:
        start = time.perf_counter()
        metadata: Dict[str, Any] = {
            "rollout_backend": self.name,
            "vllm_sync_strategy": self.sync_manager.strategy,
            "vllm_fallback_used": False,
            "vllm_error": None,
            "vllm_execution_mode": "subprocess" if self.uses_subprocess_actor else "in_process",
            "vllm_actor_resources": dict(self._actor_resource_snapshot) if self.uses_subprocess_actor else None,
        }
        try:
            lag = self.sync_manager.assert_fresh_or_allowed(update_step)
            sampling_kwargs = self._sampling_params_kwargs(tokenizer, generation_config)
            sampling_params = sampling_kwargs if self.uses_subprocess_actor else self._sampling_params(
                tokenizer, generation_config
            )
            effective_prompt_ids = []
            for row, mask in zip(input_ids, attention_mask):
                effective = row[mask.to(device=row.device, dtype=torch.bool)]
                effective_prompt_ids.append(effective.detach().cpu().to(dtype=torch.long).tolist())
            self._rollout_request_sequence += 1
            request_id = (
                f"{self._rollout_session_id}-u{int(update_step)}-r{self._rollout_request_sequence}"
            )
            manager_descriptor = {
                "policy_version": int(self.sync_manager.policy_version),
                "policy_fingerprint": self._last_sync.metadata.get("vllm_policy_fingerprint"),
                "export_dir": None if self.sync_manager.export_dir is None else str(self.sync_manager.export_dir),
            }
            # Native patch sync extends the bootstrap descriptor with transfer
            # scope and plan provenance. Preserve that complete authoritative
            # descriptor for the next update's generation request instead of
            # reconstructing only the three bootstrap fields.
            policy_descriptor = dict(self._engine_policy_descriptor or manager_descriptor)
            inconsistent = {
                key: {"manager": expected, "engine": policy_descriptor.get(key)}
                for key, expected in manager_descriptor.items()
                if policy_descriptor.get(key) != expected
            }
            if inconsistent:
                raise RuntimeError(
                    "rollout engine policy descriptor is inconsistent with current sync state: "
                    f"{inconsistent}"
                )
            outputs = self._generate_token_ids(
                prompts=prompts,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sampling_params=sampling_params,
                expected_policy_descriptor=policy_descriptor,
            )
            if self.uses_subprocess_actor:
                metadata["vllm_actor_resources"] = dict(self._actor_resource_snapshot)
            if len(outputs) != int(input_ids.shape[0]):
                raise RuntimeError(
                    "vLLM returned an unexpected number of rollout rows: "
                    f"request_id={request_id}, expected={int(input_ids.shape[0])}, actual={len(outputs)}"
                )
            generated_ids = [self._extract_generated_ids(output) for output in outputs]
            sequences, original_seq_lens = self.build_full_sequences_from_token_outputs(
                input_ids=input_ids,
                generated_token_ids=generated_ids,
                pad_token_id=rollout_pad_token_id(tokenizer),
            )
            generate_sec = max(0.0, time.perf_counter() - start)
            generated_tokens = int(sum(len(ids) for ids in generated_ids))
            if self.uses_subprocess_actor:
                self._last_actor_dispatch_metadata["backend_generate_wall_sec"] = float(generate_sec)
                metadata["vllm_actor_dispatch"] = dict(self._last_actor_dispatch_metadata)
            metadata.update(
                {
                    "vllm_policy_version": int(self.sync_manager.policy_version),
                    "vllm_policy_fingerprint": policy_descriptor["policy_fingerprint"],
                    "policy_lag_updates": int(lag),
                    "rollout_request_id": request_id,
                    "rollout_prompt_fingerprint": prompt_batch_fingerprint(effective_prompt_ids),
                    "rollout_sampling_fingerprint": sampling_fingerprint(
                        {
                            **sampling_kwargs,
                            "seed": generation_config.seed,
                        }
                    ),
                    "rollout_output_row_count": int(len(outputs)),
                    "vllm_engine_policy_verified": bool(
                        getattr(self.fit_cfg.rl, "vllm_verify_engine_policy", True)
                    ) and (
                        not self.uses_subprocess_actor
                        or all(
                            bool(snapshot.get("policy_verified"))
                            for snapshot in self._actor_resource_snapshot.values()
                        )
                    ),
                    "vllm_rollout_actor_count": len(self._actor_specs) if self.uses_subprocess_actor else 0,
                    "vllm_active_rollout_actor_count": (
                        int(self._last_actor_dispatch_metadata.get("active_actor_count", 0))
                        if self.uses_subprocess_actor
                        else 0
                    ),
                    "vllm_export_dir": None if self.sync_manager.export_dir is None else str(self.sync_manager.export_dir),
                    "rollout_policy_sync_sec": float(self._last_sync.sync_sec),
                    "rollout_policy_engine_rebuild_sec": float(self._last_sync.engine_rebuild_sec),
                    "vllm_generate_sec": generate_sec,
                    "vllm_num_prompts": int(len(prompts)),
                    "vllm_num_generated_tokens": generated_tokens,
                    "vllm_generated_tok_per_sec": (
                        float(generated_tokens) / generate_sec if generate_sec > 0.0 else None
                    ),
                }
            )
            return RolloutBatch(
                sequences=sequences,
                original_seq_lens=original_seq_lens,
                response_start=int(input_ids.shape[1]),
                metadata=metadata,
            )
        except Exception as exc:
            if _is_cuda_oom(exc) and bool(getattr(self.fit_cfg.rl, "vllm_fail_on_cuda_oom", True)):
                exc = RuntimeError(_format_oom_message(exc))
            if not bool(getattr(self.fit_cfg.rl, "vllm_fallback_to_hf", False)):
                self.close()
                raise exc
            if self.logger is not None:
                self.logger.warning("[VLLMRollout] falling back to HF rollout after vLLM error: %s", exc)
            fallback = self._hf_fallback.generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                update_step=update_step,
            )
            fallback.metadata.update(metadata)
            fallback.metadata.update(
                {
                    "rollout_backend": self.name,
                    "vllm_fallback_used": True,
                    "vllm_error": str(exc),
                    "vllm_generate_sec": max(0.0, time.perf_counter() - start),
                    "vllm_policy_version": int(self.sync_manager.policy_version),
                    "policy_lag_updates": int(self.sync_manager.policy_lag(update_step)),
                }
            )
            return fallback

    def _unload_engine(self) -> None:
        if self.uses_subprocess_actor:
            failures = []
            with ThreadPoolExecutor(max_workers=max(1, len(self._actors))) as executor:
                future_map = {
                    executor.submit(client.unload_engine): name
                    for name, client in self._actors.items()
                    if client.is_alive
                }
                for future in as_completed(future_map):
                    try:
                        future.result()
                    except Exception as exc:
                        name = str(future_map[future])
                        # An update-only receiver may still be blocked inside
                        # update_weights after a trainer-side transfer error.
                        # It cannot service unload_engine on the control pipe;
                        # force-discard it so export_reload recovery can start
                        # a clean process.
                        try:
                            self._actors[name].close(force=True)
                        except Exception as close_exc:
                            failures.append(f"{name!r}: unload={exc}; force_close={close_exc}")
            trainer_group_error = None
            try:
                # Receiver engines no longer own the peer communicator, so it
                # is now safe to release the trainer TCPStore servers.
                self.sync_manager.mark_engines_discarded()
            except Exception as exc:
                trainer_group_error = str(exc)
            if failures:
                raise RuntimeError(
                    "Failed to unload all vLLM rollout actors: "
                    f"actors={failures}, trainer_groups={trainer_group_error}"
                )
            if trainer_group_error is not None:
                raise RuntimeError(trainer_group_error)
            return
        llm = self.llm
        self.llm = None
        self._engine_policy_descriptor = {}
        shutdown_complete = llm is None
        if llm is not None:
            for attr in ("shutdown", "close"):
                fn = getattr(llm, attr, None)
                if callable(fn):
                    try:
                        fn()
                        shutdown_complete = True
                        break
                    except Exception:
                        pass
            if not shutdown_complete:
                engine = getattr(llm, "llm_engine", None)
                for attr in ("shutdown", "close"):
                    fn = getattr(engine, attr, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:
                            pass
        self.sync_manager.mark_engines_discarded()

    def close(self) -> Dict[str, Any]:
        if self.uses_subprocess_actor:
            clients = dict(self._actors)
            reports: Dict[str, Any] = {}
            failures: Dict[str, str] = {}
            if self.logger is not None:
                self.logger.info(
                    "[VLLMTeardown] phase=actor_close_begin actors=%s",
                    len(clients),
                )
            with ThreadPoolExecutor(max_workers=max(1, len(clients))) as executor:
                future_map = {
                    executor.submit(client.close): name
                    for name, client in clients.items()
                }
                for future in as_completed(future_map):
                    name = str(future_map[future])
                    try:
                        reports[name] = future.result()
                    except Exception as exc:
                        failures[name] = str(exc)
            if self.logger is not None:
                self.logger.info(
                    "[VLLMTeardown] phase=actor_close_complete actors=%s failures=%s",
                    len(reports),
                    len(failures),
                )
            trainer_groups: Dict[str, Any] = {}
            trainer_group_error: Optional[str] = None
            try:
                trainer_groups = self.sync_manager.mark_engines_discarded()
            except Exception as exc:
                trainer_group_error = str(exc)
            self._actors = {
                name: client
                for name, client in clients.items()
                if name in failures
            }
            self._actor_resource_snapshot = {}
            self._engine_policy_descriptor = {}
            if failures or trainer_group_error is not None:
                raise RuntimeError(
                    "Failed to close vLLM rollout resources: "
                    f"actor_failures={failures}, completed_actors={reports}, "
                    f"trainer_group_error={trainer_group_error}"
                )
            if self.logger is not None:
                self.logger.info(
                    "[VLLMTeardown] phase=complete actors=%s trainer_groups=%s",
                    len(reports),
                    int(trainer_groups.get("group_count", 0)),
                )
            return {
                "closed": True,
                "actors": reports,
                "trainer_nccl_groups": trainer_groups,
            }
        self._unload_engine()
        return {"closed": True, "actors": {}}
