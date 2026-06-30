from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .rollout_backends import RolloutSyncResult
from .vllm_weight_mapping import (
    build_weight_mapping_report,
    iter_transfer_tensors,
    selected_tensor_checksums,
)
from .vllm_weight_transfer_capabilities import (
    VLLMWeightTransferCapabilityReport,
    probe_vllm_weight_transfer_capabilities,
)


SavePolicyCheckpointFn = Callable[[Path, int, str, Dict[str, Any]], Path]


class VLLMPolicySyncManager:
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
        self.save_policy_checkpoint = save_policy_checkpoint
        self.logger = logger
        self.policy_version = -1
        self.export_dir: Optional[Path] = None
        self.last_result = RolloutSyncResult(synced=False, policy_version=-1, policy_lag_updates=0)
        self.capability_report: Optional[VLLMWeightTransferCapabilityReport] = None
        self.weight_transfer_initialized = False

    @property
    def strategy(self) -> str:
        return str(getattr(self.fit_cfg.rl, "vllm_sync_strategy", "export_reload") or "export_reload").strip().lower()

    @property
    def sync_every_updates(self) -> int:
        return max(1, int(getattr(self.fit_cfg.rl, "vllm_sync_every_updates", 1)))

    def policy_lag(self, update_step: int) -> int:
        if self.policy_version < 0:
            return int(update_step)
        return max(0, int(update_step) - int(self.policy_version))

    def sync_due(self, update_step: int, *, force: bool = False) -> bool:
        if force or self.policy_version < 0:
            return True
        lag = self.policy_lag(update_step)
        return lag >= self.sync_every_updates

    def assert_fresh_or_allowed(self, update_step: int) -> int:
        lag = self.policy_lag(update_step)
        if lag > 0 and not bool(getattr(self.fit_cfg.rl, "allow_stale_vllm_policy", False)):
            raise RuntimeError(
                "vLLM rollout policy is stale: "
                f"policy_version={self.policy_version}, update_step={int(update_step)}, policy_lag_updates={lag}. "
                "Keep rl.vllm_sync_every_updates=1 or set rl.allow_stale_vllm_policy=true to allow stale rollouts."
            )
        return lag

    def sync(
        self,
        *,
        model,
        tokenizer,
        update_step: int,
        force: bool = False,
        llm=None,
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> RolloutSyncResult:
        del tokenizer
        if self.strategy == "export_reload":
            return self._sync_export_reload(model=model, update_step=update_step, force=force)
        if self.strategy in {"weight_transfer_dryrun_static", "weight_transfer_dryrun_runtime"}:
            return self._sync_dryrun(
                model=model,
                update_step=update_step,
                force=force,
                runtime_params=runtime_params if self.strategy == "weight_transfer_dryrun_runtime" else None,
            )
        if self.strategy == "weight_transfer_nccl":
            return self._sync_weight_transfer_nccl(
                model=model,
                update_step=update_step,
                force=force,
                llm=llm,
                runtime_params=runtime_params,
            )
        if self.strategy == "weight_transfer_ipc":
            return self._sync_weight_transfer_ipc(update_step=update_step, force=force)
        raise ValueError(f"Unsupported rl.vllm_sync_strategy={self.strategy!r}")

    def _sync_export_reload(self, *, model, update_step: int, force: bool = False) -> RolloutSyncResult:
        del model
        lag_before = self.policy_lag(update_step)
        if not self.sync_due(update_step, force=force):
            lag = self.assert_fresh_or_allowed(update_step)
            self.last_result = RolloutSyncResult(
                synced=False,
                policy_version=int(self.policy_version),
                policy_lag_updates=int(lag),
                export_dir=None if self.export_dir is None else str(self.export_dir),
                metadata={"vllm_sync_strategy": self.strategy},
            )
            return self.last_result

        from ..export.hf_export import export_fitmotn_hf_roundtrip
        from ..export.roundtrip import validate_hf_roundtrip
        from ..export.validate import validate_export_layout

        root = Path(getattr(self.fit_cfg.rl, "vllm_export_root", None) or (self.rl_dir / "vllm_sync")).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = root / f"raw-policy-u{int(update_step)}"
        export_dir = root / f"hf-policy-u{int(update_step)}"
        sync_start = time.perf_counter()
        self.save_policy_checkpoint(
            checkpoint_dir,
            int(update_step),
            checkpoint_dir.name,
            {
                "vllm_sync_strategy": self.strategy,
                "vllm_policy_version": int(update_step),
                "policy_lag_updates_before_sync": int(lag_before),
            },
        )
        result = export_fitmotn_hf_roundtrip(
            checkpoint_dir,
            export_dir,
            base_model=getattr(self.fit_cfg.model, "model_path", None),
            torch_dtype=str(getattr(self.fit_cfg.model, "torch_dtype", "auto") or "auto"),
            roundtrip_device="cpu",
            base_trust_remote_code=bool(getattr(self.fit_cfg.model, "trust_remote_code", True)),
        )
        layout = validate_export_layout(result.output_dir)
        if not layout.ok:
            raise RuntimeError(f"Stage 4C export layout validation failed for vLLM sync: {layout.errors}")
        roundtrip = validate_hf_roundtrip(result.output_dir, device="cpu", torch_dtype=str(getattr(self.fit_cfg.model, "torch_dtype", "auto") or "auto"))
        if not roundtrip.ok:
            raise RuntimeError(f"Stage 4C HF roundtrip validation failed for vLLM sync: {roundtrip.errors}")
        sync_sec = max(0.0, time.perf_counter() - sync_start)
        self.policy_version = int(update_step)
        self.export_dir = Path(result.output_dir)
        self._prune_exports(root)
        self.last_result = RolloutSyncResult(
            synced=True,
            policy_version=int(self.policy_version),
            policy_lag_updates=0,
            export_dir=str(self.export_dir),
            sync_sec=sync_sec,
            metadata={
                "vllm_sync_strategy": self.strategy,
                "vllm_export_dir": str(self.export_dir),
                "vllm_export_root": str(root),
                "vllm_export_layout_warnings": layout.warnings,
                "vllm_export_roundtrip_warnings": roundtrip.warnings,
            },
        )
        if self.logger is not None:
            self.logger.info(
                "[VLLMSync] update=%s export_dir=%s sync_sec=%.3f",
                int(update_step),
                self.export_dir,
                sync_sec,
            )
        return self.last_result

    def _capabilities(self) -> VLLMWeightTransferCapabilityReport:
        if self.capability_report is None:
            self.capability_report = probe_vllm_weight_transfer_capabilities()
        return self.capability_report

    def _mapping_metadata(self, *, model, runtime_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        report = build_weight_mapping_report(model, runtime_params=runtime_params)
        checksums = selected_tensor_checksums(report, model)
        return {
            "weight_mapping_report": report.to_dict(),
            "weight_transfer_tensor_count": int(report.tensor_count),
            "weight_transfer_bytes": int(report.num_bytes),
            "weight_transfer_coverage_ratio": float(report.coverage),
            "weight_transfer_motn_coverage_ratio": float(report.motn_coverage),
            "weight_transfer_runtime_inspection_available": bool(report.runtime_inspection_available),
            "motn_required_key_count": int(len(report.required_motn_keys)),
            "motn_transferred_key_count": int(len(report.transferred_motn_keys)),
            "motn_required_keys": list(report.required_motn_keys),
            "motn_transferred_keys": list(report.transferred_motn_keys),
            "checksum_validation": {
                "mode": "hf_selected_tensor_checksum",
                "ok": True,
                "checksums": checksums,
            },
        }

    def _enforce_mapping_coverage(self, metadata: Dict[str, Any]) -> None:
        if not bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_validate_coverage", True)):
            return
        report = metadata.get("weight_mapping_report", {})
        coverage = float(metadata.get("weight_transfer_coverage_ratio", 0.0))
        motn_coverage = float(metadata.get("weight_transfer_motn_coverage_ratio", 0.0))
        if coverage >= 1.0 and motn_coverage >= 1.0 and not report.get("shape_mismatches") and not report.get("dtype_mismatches"):
            return
        message = (
            "vLLM native weight-transfer coverage is incomplete: "
            f"coverage={coverage:.6f}, motn_coverage={motn_coverage:.6f}, "
            f"missing_in_vllm={report.get('missing_in_vllm', [])}, "
            f"shape_mismatches={report.get('shape_mismatches', [])}, "
            f"dtype_mismatches={report.get('dtype_mismatches', [])}"
        )
        if bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_fail_on_partial", True)):
            raise RuntimeError(message)
        if self.logger is not None:
            self.logger.warning("[VLLMSync] %s", message)

    def _sync_dryrun(
        self,
        *,
        model,
        update_step: int,
        force: bool,
        runtime_params: Optional[Dict[str, Any]],
    ) -> RolloutSyncResult:
        del force
        lag = self.policy_lag(update_step)
        mode = "runtime" if runtime_params is not None else "static"
        start = time.perf_counter()
        metadata = {
            "vllm_sync_strategy": self.strategy,
            "vllm_weight_transfer_dryrun": True,
            "vllm_weight_transfer_dryrun_mode": mode,
            "native_transfer_level": self._capabilities().native_transfer_level,
            "native_transfer_capability_report": self._capabilities().to_dict(),
            "rollout_facing_validation": {
                "mode": "not_run",
                "ok": None,
                "reason": "dryrun does not mutate or validate vLLM policy freshness",
            },
            **self._mapping_metadata(model=model, runtime_params=runtime_params),
        }
        self.last_result = RolloutSyncResult(
            synced=False,
            policy_version=int(self.policy_version),
            policy_lag_updates=int(lag),
            export_dir=None if self.export_dir is None else str(self.export_dir),
            sync_sec=max(0.0, time.perf_counter() - start),
            metadata=metadata,
        )
        return self.last_result

    def _native_unavailable_error(self, report: VLLMWeightTransferCapabilityReport) -> RuntimeError:
        required = str(getattr(self.fit_cfg.rl, "vllm_native_transfer_required_level", "four_phase") or "four_phase")
        return RuntimeError(
            "rl.vllm_sync_strategy='weight_transfer_nccl' requires native_transfer_level='four_phase' "
            f"for real RL training, but this vLLM runtime reports {report.native_transfer_level!r} "
            f"(required={required!r}). Missing APIs: {report.missing}. "
            "Use rl.vllm_sync_strategy='export_reload' or set "
            "rl.vllm_weight_transfer_fallback_to_export_reload=true for explicit fallback."
        )

    def _fallback_or_raise(self, *, exc: Exception, model, update_step: int, force: bool) -> RolloutSyncResult:
        if not bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_fallback_to_export_reload", False)):
            raise exc
        result = self._sync_export_reload(model=model, update_step=update_step, force=force)
        result.metadata.update(
            {
                "vllm_weight_transfer_fallback_used": True,
                "vllm_weight_transfer_error": str(exc),
                "vllm_sync_strategy_requested": self.strategy,
            }
        )
        return result

    def _sync_weight_transfer_nccl(
        self,
        *,
        model,
        update_step: int,
        force: bool,
        llm,
        runtime_params: Optional[Dict[str, Any]],
    ) -> RolloutSyncResult:
        if self.policy_version < 0 or self.export_dir is None or int(update_step) == 0:
            result = self._sync_export_reload(model=model, update_step=update_step, force=True)
            result.metadata.update(
                {
                    "vllm_sync_strategy_requested": self.strategy,
                    "vllm_weight_transfer_bootstrap": True,
                }
            )
            return result
        lag_before = self.policy_lag(update_step)
        if not self.sync_due(update_step, force=force):
            lag = self.assert_fresh_or_allowed(update_step)
            self.last_result = RolloutSyncResult(
                synced=False,
                policy_version=int(self.policy_version),
                policy_lag_updates=int(lag),
                export_dir=None if self.export_dir is None else str(self.export_dir),
                metadata={"vllm_sync_strategy": self.strategy},
            )
            return self.last_result
        start = time.perf_counter()
        report = self._capabilities()
        metadata = {
            "vllm_sync_strategy": self.strategy,
            "native_transfer_level": report.native_transfer_level,
            "native_transfer_capability_report": report.to_dict(),
            **self._mapping_metadata(model=model, runtime_params=runtime_params),
        }
        try:
            self._enforce_mapping_coverage(metadata)
            if report.native_transfer_level != "four_phase":
                raise self._native_unavailable_error(report)
            if llm is None:
                raise RuntimeError("vLLM native weight transfer requires an initialized vLLM engine")
            transfer_result = self._run_four_phase_nccl_update(llm=llm, model=model, update_step=update_step)
            metadata.update(transfer_result)
            metadata.update(self._post_sync_validation_metadata(model=model, llm=llm, update_step=update_step))
        except Exception as exc:
            return self._fallback_or_raise(exc=exc, model=model, update_step=update_step, force=force)

        self.policy_version = int(update_step)
        self.last_result = RolloutSyncResult(
            synced=True,
            policy_version=int(self.policy_version),
            policy_lag_updates=0,
            export_dir=None if self.export_dir is None else str(self.export_dir),
            sync_sec=max(0.0, time.perf_counter() - start),
            metadata={
                **metadata,
                "vllm_weight_transfer_native_sync": True,
                "policy_lag_updates_before_sync": int(lag_before),
            },
        )
        return self.last_result

    def _sync_weight_transfer_ipc(self, *, update_step: int, force: bool) -> RolloutSyncResult:
        del force
        report = self._capabilities()
        raise RuntimeError(
            "rl.vllm_sync_strategy='weight_transfer_ipc' is explicit opt-in but not enabled in Stage 4E "
            "without verified IPC safe lifetime handling. "
            f"Capability ipc_available={report.ipc_available}. Use weight_transfer_nccl or export_reload."
        )

    def _run_four_phase_nccl_update(self, *, llm, model, update_step: int) -> Dict[str, Any]:
        from vllm.distributed.weight_transfer.nccl_engine import (  # type: ignore
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
            NCCLWeightTransferInitInfo,
            NCCLWeightTransferUpdateInfo,
        )

        mapping = build_weight_mapping_report(model)
        names = [entry.transfer_name for entry in mapping.entries]
        dtype_names = [entry.dtype for entry in mapping.entries]
        shapes = [entry.shape for entry in mapping.entries]
        packed = bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_packed", True))
        timeout = float(getattr(self.fit_cfg.rl, "vllm_weight_transfer_timeout_sec", 300.0))
        timings: Dict[str, float] = {}
        init_start = time.perf_counter()
        if not self.weight_transfer_initialized:
            init_fn = getattr(llm, "init_weight_transfer_engine", None)
            if not callable(init_fn):
                raise RuntimeError("vLLM LLM.init_weight_transfer_engine is unavailable")
            init_info = NCCLWeightTransferInitInfo(
                master_address=str(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_addr", "127.0.0.1") or "127.0.0.1"),
                master_port=int(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_port", 0)),
                rank_offset=1,
                world_size=int(getattr(self.fit_cfg.rl, "vllm_tensor_parallel_size", 1)) + 1,
            )
            init_fn(init_info)
            self.weight_transfer_initialized = True
        timings["weight_transfer_init_sec"] = max(0.0, time.perf_counter() - init_start)

        start = time.perf_counter()
        update_info = llm.start_weight_update(is_checkpoint_format=True)
        if update_info is None:
            update_info = NCCLWeightTransferUpdateInfo(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=packed,
                is_checkpoint_format=True,
            )
        timings["weight_transfer_start_sec"] = max(0.0, time.perf_counter() - start)

        update_start = time.perf_counter()
        group = NCCLWeightTransferEngine.trainer_init(
            NCCLWeightTransferInitInfo(
                master_address=str(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_addr", "127.0.0.1") or "127.0.0.1"),
                master_port=int(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_port", 0)),
                rank_offset=0,
                world_size=int(getattr(self.fit_cfg.rl, "vllm_tensor_parallel_size", 1)) + 1,
            )
        )
        trainer_args = NCCLTrainerSendWeightsArgs(group=group, packed=packed)
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vllm_weight_transfer")
        try:
            receive_future = executor.submit(llm.update_weights, update_info)
            send_future = executor.submit(
                NCCLWeightTransferEngine.trainer_send_weights,
                iter_transfer_tensors(mapping, model),
                trainer_args,
            )
            done, pending = wait([receive_future, send_future], timeout=timeout)
            if pending:
                for future in pending:
                    future.cancel()
                raise TimeoutError(
                    "Timed out during vLLM NCCL weight transfer. "
                    "Receiver-side update_weights and trainer_send_weights must both complete; "
                    f"timeout_sec={timeout}."
                )
            for future in done:
                future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        timings["weight_transfer_update_sec"] = max(0.0, time.perf_counter() - update_start)

        finish_start = time.perf_counter()
        llm.finish_weight_update()
        timings["weight_transfer_finish_sec"] = max(0.0, time.perf_counter() - finish_start)
        return {
            **timings,
            "weight_transfer_update_step": int(update_step),
            "weight_transfer_packed": bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_packed", True)),
        }

    def _post_sync_validation_metadata(self, *, model, llm, update_step: int) -> Dict[str, Any]:
        every = max(1, int(getattr(self.fit_cfg.rl, "vllm_sync_validation_every", 1)))
        if not bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_validate_after_sync", True)) or int(update_step) % every != 0:
            return {
                "rollout_facing_validation": {
                    "mode": "skipped",
                    "ok": None,
                }
            }
        runtime_params = self._collect_llm_named_parameters(llm)
        if not runtime_params:
            raise RuntimeError("Post-sync validation could not inspect vLLM tensor names/checksums")
        mapping = build_weight_mapping_report(model, runtime_params=runtime_params)
        hf_checksums = selected_tensor_checksums(mapping, model)
        vllm_checksums: Dict[str, float] = {}
        for name in hf_checksums:
            tensor = runtime_params.get(name)
            if tensor is None:
                raise RuntimeError(f"Post-sync checksum validation missing vLLM tensor {name!r}")
            vllm_checksums[name] = float(tensor.detach().float().cpu().sum().item())
        checksum_ok = all(abs(hf_checksums[name] - vllm_checksums[name]) <= 1e-3 for name in hf_checksums)
        if not checksum_ok:
            raise RuntimeError(
                "Post-sync checksum validation failed for vLLM native transfer: "
                f"hf={hf_checksums}, vllm={vllm_checksums}"
            )
        rollout_validation = self._rollout_facing_validation(llm)
        if not rollout_validation.get("ok"):
            raise RuntimeError(f"Post-sync rollout-facing validation failed: {rollout_validation}")
        return {
            "checksum_validation": {
                "mode": "selected_tensor_checksum",
                "ok": True,
                "hf_checksums": hf_checksums,
                "vllm_checksums": vllm_checksums,
            },
            "rollout_facing_validation": {
                **rollout_validation,
            }
        }

    def _collect_llm_named_parameters(self, llm) -> Dict[str, Any]:
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
        return {}

    def _rollout_facing_validation(self, llm) -> Dict[str, Any]:
        generate = getattr(llm, "generate", None)
        if not callable(generate):
            return {"mode": "greedy_generation", "ok": False, "error": "vLLM engine does not expose generate"}
        try:
            from vllm import SamplingParams  # type: ignore

            outputs = generate(["1 + 1 ="], SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0))
            return {
                "mode": "greedy_generation",
                "ok": bool(outputs),
                "prompt": "1 + 1 =",
            }
        except Exception as exc:
            return {
                "mode": "greedy_generation",
                "ok": False,
                "error": str(exc),
            }

    def _prune_exports(self, root: Path) -> None:
        keep = int(getattr(self.fit_cfg.rl, "vllm_keep_sync_exports", 1))
        if keep <= 0:
            return
        exports = sorted(root.glob("hf-policy-u*"), key=lambda p: p.stat().st_mtime, reverse=True)
        raws = sorted(root.glob("raw-policy-u*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in exports[keep:] + raws[keep:]:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning("[VLLMSync] failed to prune old sync artifact %s: %s", path, exc)
