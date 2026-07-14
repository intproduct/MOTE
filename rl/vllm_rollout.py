from __future__ import annotations

import time
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
    ):
        self.fit_cfg = fit_cfg
        self.rl_dir = Path(rl_dir)
        self.logger = logger
        self.sync_manager = VLLMPolicySyncManager(
            fit_cfg=fit_cfg,
            rl_dir=self.rl_dir,
            save_policy_checkpoint=save_policy_checkpoint,
            logger=logger,
        )
        self.llm = None
        self._vllm = None
        self._sampling_params_cls = None
        self._actor: Optional[VLLMActorClient] = None
        self._last_sync = RolloutSyncResult(synced=False, policy_version=-1, policy_lag_updates=0)
        self._hf_fallback = HFRolloutBackend(logger=logger)

    @property
    def uses_subprocess_actor(self) -> bool:
        return str(getattr(self.fit_cfg.rl, "vllm_execution_mode", "in_process") or "in_process").strip().lower() == "subprocess"

    def _actor_client(self) -> VLLMActorClient:
        if self._actor is None:
            self._actor = VLLMActorClient(
                start_method=str(getattr(self.fit_cfg.rl, "vllm_actor_start_method", "spawn") or "spawn"),
                request_timeout_sec=float(getattr(self.fit_cfg.rl, "vllm_actor_request_timeout_sec", 600.0)),
                shutdown_timeout_sec=float(getattr(self.fit_cfg.rl, "vllm_actor_shutdown_timeout_sec", 30.0)),
            )
        return self._actor

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

    def _llm_kwargs(self, export_dir: str) -> Dict[str, Any]:
        rl_cfg = self.fit_cfg.rl
        kwargs: Dict[str, Any] = {
            "model": str(export_dir),
            "tokenizer": str(export_dir),
            "trust_remote_code": True,
            "model_impl": str(getattr(rl_cfg, "vllm_model_impl", "transformers") or "transformers"),
            "enforce_eager": bool(getattr(rl_cfg, "vllm_enforce_eager", True)),
            "dtype": str(getattr(rl_cfg, "vllm_dtype", "auto") or "auto"),
            "tensor_parallel_size": int(getattr(rl_cfg, "vllm_tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(getattr(rl_cfg, "vllm_gpu_memory_utilization", 0.85)),
        }
        optional_values = [
            ("max_model_len", int(getattr(rl_cfg, "vllm_max_model_len", 0))),
            ("max_num_seqs", int(getattr(rl_cfg, "vllm_max_num_seqs", 0))),
            ("seed", getattr(rl_cfg, "seed", None)),
        ]
        for key, value in optional_values:
            if value is not None and int(value) > 0:
                kwargs[key] = value
        if bool(getattr(rl_cfg, "vllm_disable_log_stats", True)):
            kwargs["disable_log_stats"] = True
        device = getattr(rl_cfg, "vllm_device", None)
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
        return kwargs

    def _build_engine(self, export_dir: str) -> float:
        if self.uses_subprocess_actor:
            return self._actor_client().load_engine(self._llm_kwargs(export_dir))
        LLM, _ = self._import_vllm()
        if bool(getattr(self.fit_cfg.rl, "vllm_empty_cache_before_engine_init", False)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = time.perf_counter()
        try:
            self.llm = LLM(**self._llm_kwargs(export_dir))
        except Exception as exc:
            if _is_cuda_oom(exc) and bool(getattr(self.fit_cfg.rl, "vllm_fail_on_cuda_oom", True)):
                raise RuntimeError(_format_oom_message(exc)) from exc
            raise
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
                if self.uses_subprocess_actor and self._actor is not None and self._actor.is_alive:
                    self._actor.sleep(level=sleep_level)
                elif self.llm is not None and callable(getattr(self.llm, "sleep", None)):
                    self.llm.sleep(level=sleep_level)
            sync_result = self.sync_manager.sync(
                model=model,
                tokenizer=tokenizer,
                update_step=update_step,
                force=force,
                llm=self.llm,
                runtime_params=self._runtime_params(),
            )
            native_sync = bool(sync_result.metadata.get("vllm_weight_transfer_native_sync", False))
            if sync_result.synced and not native_sync:
                self._unload_engine()
                rebuild_sec = self._build_engine(str(sync_result.export_dir))
                sync_result.engine_rebuild_sec = rebuild_sec
                sync_result.metadata["vllm_engine_rebuild_sec"] = rebuild_sec
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

    def _generate_token_ids(
        self,
        *,
        prompts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_params,
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
            return self._actor_client().generate(
                prompt_token_ids=prompt_token_ids,
                sampling_kwargs=sampling_params,
            )
        if self.llm is None:
            raise RuntimeError("vLLM engine is not initialized; call sync_policy(..., force=True) before rollout generation")
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
        }
        try:
            lag = self.sync_manager.assert_fresh_or_allowed(update_step)
            sampling_params = self._sampling_params(tokenizer, generation_config)
            outputs = self._generate_token_ids(
                prompts=prompts,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sampling_params=sampling_params,
            )
            generated_ids = [self._extract_generated_ids(output) for output in outputs]
            sequences, original_seq_lens = self.build_full_sequences_from_token_outputs(
                input_ids=input_ids,
                generated_token_ids=generated_ids,
                pad_token_id=rollout_pad_token_id(tokenizer),
            )
            generate_sec = max(0.0, time.perf_counter() - start)
            generated_tokens = int(sum(len(ids) for ids in generated_ids))
            metadata.update(
                {
                    "vllm_policy_version": int(self.sync_manager.policy_version),
                    "policy_lag_updates": int(lag),
                    "vllm_export_dir": None if self.sync_manager.export_dir is None else str(self.sync_manager.export_dir),
                    "vllm_sync_sec": float(self._last_sync.sync_sec),
                    "vllm_engine_rebuild_sec": float(self._last_sync.engine_rebuild_sec),
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
            if self._actor is not None:
                self._actor.unload_engine()
            return
        llm = self.llm
        self.llm = None
        if llm is None:
            return
        for attr in ("shutdown", "close"):
            fn = getattr(llm, attr, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass
        engine = getattr(llm, "llm_engine", None)
        for attr in ("shutdown", "close"):
            fn = getattr(engine, attr, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass

    def close(self) -> None:
        if self._actor is not None:
            self._actor.close()
            self._actor = None
        self._unload_engine()
