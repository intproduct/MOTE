from __future__ import annotations

import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .rollout_backends import RolloutSyncResult
from .vllm_weight_mapping import (
    build_weight_mapping_report,
    selected_tensor_checksums,
)
from .vllm_weight_transfer_capabilities import (
    VLLMWeightTransferCapabilityReport,
    probe_vllm_weight_transfer_capabilities,
)
from .vllm_weight_transfer_adapters import NCCLTransferSettings, select_nccl_transfer_adapter
from .vllm_integrity import (
    SYNC_MANIFEST_NAME,
    SYNC_STATE_NAME,
    atomic_write_json,
    directory_size_bytes,
    load_json_object,
    model_policy_fingerprint,
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
        self.weight_transfer_master_port: Optional[int] = None

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
        sync_start = time.perf_counter()
        cleanup_start = time.perf_counter()
        stale_temps_removed = self._cleanup_incomplete_exports(root)
        cleanup_sec = max(0.0, time.perf_counter() - cleanup_start)
        fingerprint_start = time.perf_counter()
        policy_fingerprint = model_policy_fingerprint(
            model,
            sample_elements_per_tensor=int(
                getattr(self.fit_cfg.rl, "vllm_policy_fingerprint_samples_per_tensor", 16)
            ),
        )
        fingerprint_sec = max(0.0, time.perf_counter() - fingerprint_start)
        artifact_id = f"u{int(update_step)}-{policy_fingerprint[:12]}"
        checkpoint_dir = root / f"raw-policy-{artifact_id}"
        export_dir = root / f"hf-policy-{artifact_id}"
        existing_manifest = load_json_object(export_dir / SYNC_MANIFEST_NAME)
        reused = bool(
            export_dir.is_dir()
            and existing_manifest
            and existing_manifest.get("complete") is True
            and int(existing_manifest.get("policy_version", -1)) == int(update_step)
            and existing_manifest.get("policy_fingerprint") == policy_fingerprint
        )
        checkpoint_sec = 0.0
        export_sec = 0.0
        commit_sec = 0.0
        validation_start = time.perf_counter()
        roundtrip_every = max(
            1,
            int(getattr(self.fit_cfg.rl, "vllm_export_roundtrip_validation_every", 1)),
        )
        run_roundtrip = bool(getattr(self.fit_cfg.rl, "vllm_export_validate_roundtrip", True)) and (
            int(update_step) == 0 or int(update_step) % roundtrip_every == 0
        )
        temporary_checkpoint: Optional[Path] = None
        temporary_export: Optional[Path] = None
        try:
            if not reused:
                transaction = uuid.uuid4().hex
                temporary_checkpoint = root / f".raw-policy-{artifact_id}.tmp-{transaction}"
                temporary_export = root / f".hf-policy-{artifact_id}.tmp-{transaction}"
                temporary_checkpoint.mkdir(parents=True, exist_ok=False)
                checkpoint_start = time.perf_counter()
                self.save_policy_checkpoint(
                    temporary_checkpoint,
                    int(update_step),
                    checkpoint_dir.name,
                    {
                        "vllm_sync_strategy": self.strategy,
                        "vllm_policy_version": int(update_step),
                        "vllm_policy_fingerprint": policy_fingerprint,
                        "policy_lag_updates_before_sync": int(lag_before),
                    },
                )
                checkpoint_sec = max(0.0, time.perf_counter() - checkpoint_start)
                export_start = time.perf_counter()
                result = export_fitmotn_hf_roundtrip(
                    temporary_checkpoint,
                    temporary_export,
                    base_model=getattr(self.fit_cfg.model, "model_path", None),
                    torch_dtype=str(getattr(self.fit_cfg.model, "torch_dtype", "auto") or "auto"),
                    roundtrip_device="cpu",
                    base_trust_remote_code=bool(getattr(self.fit_cfg.model, "trust_remote_code", True)),
                )
                export_sec = max(0.0, time.perf_counter() - export_start)
                candidate_export = Path(result.output_dir)
            else:
                candidate_export = export_dir

            layout = validate_export_layout(candidate_export)
            if not layout.ok:
                raise RuntimeError(f"Stage 4C export layout validation failed for vLLM sync: {layout.errors}")
            if run_roundtrip:
                roundtrip = validate_hf_roundtrip(
                    candidate_export,
                    device="cpu",
                    torch_dtype=str(getattr(self.fit_cfg.model, "torch_dtype", "auto") or "auto"),
                )
                if not roundtrip.ok:
                    raise RuntimeError(f"Stage 4C HF roundtrip validation failed for vLLM sync: {roundtrip.errors}")
                roundtrip_warnings = list(roundtrip.warnings)
            else:
                roundtrip_warnings = []
            validation_sec = max(0.0, time.perf_counter() - validation_start)

            if not reused:
                manifest = {
                    "format_version": 1,
                    "complete": True,
                    "policy_version": int(update_step),
                    "policy_fingerprint": policy_fingerprint,
                    "raw_checkpoint_name": checkpoint_dir.name,
                    "export_name": export_dir.name,
                    "created_at_unix": time.time(),
                    "layout_warnings": list(layout.warnings),
                    "roundtrip_validated": bool(run_roundtrip),
                    "roundtrip_warnings": roundtrip_warnings,
                }
                atomic_write_json(candidate_export / SYNC_MANIFEST_NAME, manifest)
                commit_start = time.perf_counter()
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                if export_dir.exists():
                    shutil.rmtree(export_dir)
                if temporary_checkpoint is None or temporary_export is None:
                    raise RuntimeError("vLLM export transaction paths were not initialized")
                temporary_checkpoint.replace(checkpoint_dir)
                temporary_checkpoint = None
                temporary_export.replace(export_dir)
                temporary_export = None
                commit_sec = max(0.0, time.perf_counter() - commit_start)
        except Exception:
            for path in (temporary_checkpoint, temporary_export):
                if path is not None and path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            raise

        self.policy_version = int(update_step)
        self.export_dir = export_dir
        # The rollout backend rebuilds its engine from this export. Any native
        # transfer engine previously attached to the old vLLM instance is no
        # longer initialized.
        self.weight_transfer_initialized = False
        prune_start = time.perf_counter()
        pruned_artifacts = self._prune_exports(root)
        prune_sec = max(0.0, time.perf_counter() - prune_start)
        state = {
            "format_version": 1,
            "policy_version": int(self.policy_version),
            "policy_fingerprint": policy_fingerprint,
            "export_dir": str(self.export_dir),
            "raw_checkpoint_dir": str(checkpoint_dir),
            "updated_at_unix": time.time(),
        }
        atomic_write_json(root / SYNC_STATE_NAME, state)
        artifact_bytes = directory_size_bytes(checkpoint_dir) + directory_size_bytes(export_dir)
        sync_sec = max(0.0, time.perf_counter() - sync_start)
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
                "vllm_policy_fingerprint": policy_fingerprint,
                "vllm_export_reused": bool(reused),
                "vllm_export_transaction_committed": True,
                "vllm_export_roundtrip_validated": bool(run_roundtrip),
                "vllm_export_layout_warnings": layout.warnings,
                "vllm_export_roundtrip_warnings": roundtrip_warnings,
                "vllm_fingerprint_sec": fingerprint_sec,
                "vllm_checkpoint_save_sec": checkpoint_sec,
                "vllm_export_convert_sec": export_sec,
                "vllm_export_validation_sec": validation_sec,
                "vllm_export_commit_sec": commit_sec,
                "vllm_export_cleanup_sec": cleanup_sec + prune_sec,
                "vllm_sync_artifact_bytes": int(artifact_bytes),
                "vllm_stale_temp_artifacts_removed": int(stale_temps_removed),
                "vllm_pruned_artifacts": int(pruned_artifacts),
            },
        )
        if self.logger is not None:
            self.logger.info(
                "[VLLMSync] update=%s fingerprint=%s reused=%s export_dir=%s sync_sec=%.3f",
                int(update_step),
                policy_fingerprint[:12],
                reused,
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
        lag = self.policy_lag(update_step)
        mode = "runtime" if runtime_params is not None else "static"
        start = time.perf_counter()
        diagnostic_metadata = {
            "vllm_sync_strategy": self.strategy,
            "vllm_weight_transfer_dryrun": True,
            "vllm_weight_transfer_dryrun_mode": mode,
            "vllm_weight_transfer_dryrun_requested_mode": (
                "runtime" if self.strategy == "weight_transfer_dryrun_runtime" else "static"
            ),
            "native_transfer_level": self._capabilities().native_transfer_level,
            "native_transfer_capability_report": self._capabilities().to_dict(),
            "rollout_facing_validation": {
                "mode": "not_run",
                "ok": None,
                "reason": "dryrun validates mapping but uses export_reload for policy freshness",
            },
            **self._mapping_metadata(model=model, runtime_params=runtime_params),
        }
        diagnostic_sec = max(0.0, time.perf_counter() - start)
        diagnostic_metadata["vllm_weight_transfer_dryrun_sec"] = diagnostic_sec
        # A dryrun is a diagnostic for native in-place transfer, not a stale
        # rollout policy mode. Keep the rollout engine usable and fresh through
        # the proven export/reload path while attaching the mapping report.
        result = self._sync_export_reload(model=model, update_step=update_step, force=force)
        result.sync_sec += diagnostic_sec
        result.metadata.update(diagnostic_metadata)
        result.metadata["policy_lag_updates_before_sync"] = int(lag)
        self.last_result = result
        return result

    def _native_unavailable_error(self, report: VLLMWeightTransferCapabilityReport) -> RuntimeError:
        required = str(getattr(self.fit_cfg.rl, "vllm_native_transfer_required_level", "four_phase") or "four_phase")
        return RuntimeError(
            "rl.vllm_sync_strategy='weight_transfer_nccl' cannot satisfy the configured native transfer gate "
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
            if llm is None:
                raise RuntimeError("vLLM native weight transfer requires an initialized vLLM engine")
            required_level = str(
                getattr(self.fit_cfg.rl, "vllm_native_transfer_required_level", "four_phase") or "four_phase"
            ).strip().lower()
            try:
                adapter = select_nccl_transfer_adapter(report, required_level=required_level)
            except Exception as exc:
                raise self._native_unavailable_error(report) from exc
            transfer_result = self._run_nccl_update(
                adapter=adapter,
                llm=llm,
                model=model,
                update_step=update_step,
            )
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

    def _resolve_weight_transfer_master_port(self) -> int:
        configured = int(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_port", 0))
        if configured > 0:
            self.weight_transfer_master_port = configured
            return configured
        if self.weight_transfer_master_port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((str(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_addr", "127.0.0.1")), 0))
                self.weight_transfer_master_port = int(sock.getsockname()[1])
        return int(self.weight_transfer_master_port)

    def _run_nccl_update(self, *, adapter, llm, model, update_step: int) -> Dict[str, Any]:
        mapping = build_weight_mapping_report(model)
        start = time.perf_counter()
        result, initialized = adapter.transfer(
            llm=llm,
            model=model,
            mapping=mapping,
            settings=NCCLTransferSettings(
                master_address=str(
                    getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_addr", "127.0.0.1") or "127.0.0.1"
                ),
                master_port=self._resolve_weight_transfer_master_port(),
                tensor_parallel_size=int(getattr(self.fit_cfg.rl, "vllm_tensor_parallel_size", 1)),
                packed=bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_packed", True)),
                timeout_sec=float(getattr(self.fit_cfg.rl, "vllm_weight_transfer_timeout_sec", 300.0)),
            ),
            initialized=bool(self.weight_transfer_initialized),
            update_step=int(update_step),
        )
        self.weight_transfer_initialized = bool(initialized)
        result["weight_transfer_total_sec"] = max(0.0, time.perf_counter() - start)
        return result

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

    def _cleanup_incomplete_exports(self, root: Path) -> int:
        max_age = max(0.0, float(getattr(self.fit_cfg.rl, "vllm_export_temp_max_age_sec", 3600.0)))
        now = time.time()
        removed = 0
        for path in root.glob(".*policy-*.tmp-*"):
            try:
                age = max(0.0, now - path.stat().st_mtime)
                if max_age > 0.0 and age < max_age:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def _prune_exports(self, root: Path) -> int:
        keep = int(getattr(self.fit_cfg.rl, "vllm_keep_sync_exports", 1))
        if keep <= 0:
            return 0
        exports = sorted(root.glob("hf-policy-u*"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        retained_raw_names = set()
        for path in exports[:keep]:
            manifest = load_json_object(path / SYNC_MANIFEST_NAME) or {}
            raw_name = manifest.get("raw_checkpoint_name")
            if raw_name:
                retained_raw_names.add(str(raw_name))
        removal_paths = list(exports[keep:])
        for path in exports[keep:]:
            manifest = load_json_object(path / SYNC_MANIFEST_NAME) or {}
            raw_name = manifest.get("raw_checkpoint_name")
            if raw_name:
                removal_paths.append(root / str(raw_name))
        for raw in root.glob("raw-policy-u*"):
            if raw.name not in retained_raw_names and raw not in removal_paths:
                removal_paths.append(raw)
        seen = set()
        for path in removal_paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                removed += 1
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning("[VLLMSync] failed to prune old sync artifact %s: %s", path, exc)
        return removed
