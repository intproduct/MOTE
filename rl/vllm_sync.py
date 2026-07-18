from __future__ import annotations

import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .rollout_backends import RolloutSyncResult
from .vllm_weight_mapping import (
    build_weight_mapping_report,
    selected_tensor_checksums,
    validate_trainable_patch_transfer_selection,
)
from .vllm_weight_transfer_capabilities import (
    VLLMWeightTransferCapabilityReport,
    probe_vllm_weight_transfer_capabilities,
)
from .vllm_weight_transfer_adapters import (
    NCCLTransferSettings,
    nccl_init_request_payload,
    nccl_update_request_payload,
    select_nccl_transfer_adapter,
    shutdown_trainer_nccl_group,
    trainer_initialize_nccl_group,
    trainer_send_weights_to_actor,
)
from .vllm_integrity import (
    SYNC_MANIFEST_NAME,
    SYNC_STATE_NAME,
    atomic_write_json,
    directory_size_bytes,
    load_json_object,
    model_named_parameter_fingerprint,
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
        transfer_training_names: Optional[Sequence[str]] = None,
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
        self.actor_weight_transfer_initialized: Dict[str, bool] = {}
        self.actor_weight_transfer_master_ports: Dict[str, int] = {}
        self.actor_trainer_nccl_groups: Dict[str, Any] = {}
        self.actor_capability_reports: Dict[str, Dict[str, Any]] = {}
        self.transfer_training_names = tuple(str(name) for name in (transfer_training_names or ()))
        self.patch_transfer_validation: Optional[Dict[str, Any]] = None
        self.frozen_parameter_fingerprint: Optional[str] = None
        self.frozen_parameter_versions: Dict[str, int] = {}

    @property
    def strategy(self) -> str:
        return str(getattr(self.fit_cfg.rl, "vllm_sync_strategy", "export_reload") or "export_reload").strip().lower()

    @property
    def transfer_scope(self) -> str:
        return str(
            getattr(self.fit_cfg.rl, "vllm_weight_transfer_scope", "full_policy") or "full_policy"
        ).strip().lower()

    def _mapping(self, model, *, runtime_params: Optional[Dict[str, Any]] = None):
        selected_names = None
        if self.transfer_scope == "trainable_patch":
            if not self.transfer_training_names:
                raise RuntimeError("trainable_patch native sync requires an explicit immutable transfer allowlist")
            self.patch_transfer_validation = validate_trainable_patch_transfer_selection(
                model,
                transfer_training_names=self.transfer_training_names,
            )
            selected_names = self.transfer_training_names
        return build_weight_mapping_report(
            model,
            runtime_params=runtime_params,
            selected_training_names=selected_names,
            transfer_scope=self.transfer_scope,
        )

    def _frozen_drift_metadata(self, model, *, initialize: bool = False) -> Dict[str, Any]:
        if self.transfer_scope != "trainable_patch":
            return {}
        transfer_names = set(self.transfer_training_names)
        frozen_params = {name: param for name, param in model.named_parameters() if name not in transfer_names}
        frozen_names = list(frozen_params)
        if initialize or self.frozen_parameter_fingerprint is None:
            self.frozen_parameter_fingerprint = model_named_parameter_fingerprint(
                model,
                parameter_names=frozen_names,
                sample_elements_per_tensor=int(
                    getattr(self.fit_cfg.rl, "vllm_policy_fingerprint_samples_per_tensor", 16)
                ),
                selection_label="frozen_policy_guard",
            )
            self.frozen_parameter_versions = {
                name: int(getattr(param, "_version", 0))
                for name, param in frozen_params.items()
            }
        changed_versions = sorted(
            name
            for name, param in frozen_params.items()
            if int(getattr(param, "_version", 0)) != self.frozen_parameter_versions.get(name)
        )
        if changed_versions:
            raise RuntimeError(
                "frozen policy parameters changed during trainable_patch native sync; "
                "refusing a patch-only update because the Actor would become stale; "
                f"changed={changed_versions[:20]}"
            )
        return {
            "frozen_parameter_count": len(frozen_names),
            "frozen_parameter_fingerprint": self.frozen_parameter_fingerprint,
            "frozen_parameter_guard": "torch_parameter_version",
            "frozen_parameter_drift": False,
        }

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
        actors: Optional[Sequence[tuple[str, Any, Dict[str, Any]]]] = None,
    ) -> RolloutSyncResult:
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
                actors=actors,
                tokenizer=tokenizer,
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
        from ..diagnostics.vllm_export_preflight import validate_vllm_export_preflight

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
        validation_sec = 0.0
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

            validation_start = time.perf_counter()
            layout = validate_export_layout(candidate_export)
            if not layout.ok:
                raise RuntimeError(f"Stage 4C export layout validation failed for vLLM sync: {layout.errors}")
            repository_remote_code = (
                Path(__file__).resolve().parents[1] / "export" / "remote_code" / "modeling_fitmotn.py"
            )
            vllm_preflight = validate_vllm_export_preflight(
                candidate_export,
                expected_remote_code=repository_remote_code,
                require_vllm=False,
            )
            if not vllm_preflight["ok"]:
                raise RuntimeError(
                    "vLLM static weight contract failed during policy sync: "
                    + "; ".join(str(error) for error in vllm_preflight["errors"])
                )
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
                    "vllm_preflight_validated": True,
                    "vllm_preflight_warnings": list(vllm_preflight["warnings"]),
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
        self.mark_engines_discarded()
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

    def mark_engines_discarded(self) -> None:
        """Forget receiver state after an engine/actor has been destroyed."""
        self.weight_transfer_initialized = False
        self.actor_weight_transfer_initialized.clear()
        self.actor_weight_transfer_master_ports.clear()
        for group in self.actor_trainer_nccl_groups.values():
            shutdown_trainer_nccl_group(group)
        self.actor_trainer_nccl_groups.clear()
        self.actor_capability_reports.clear()

    def _capabilities(self) -> VLLMWeightTransferCapabilityReport:
        if self.capability_report is None:
            self.capability_report = probe_vllm_weight_transfer_capabilities()
        return self.capability_report

    def _mapping_metadata(self, *, model, runtime_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        report = self._mapping(model, runtime_params=runtime_params)
        checksums = selected_tensor_checksums(report, model)
        return {
            "weight_mapping_report": report.to_dict(),
            "weight_transfer_tensor_count": int(report.tensor_count),
            "weight_transfer_bytes": int(report.num_bytes),
            "weight_transfer_coverage_ratio": float(report.coverage),
            "weight_transfer_motn_coverage_ratio": float(report.motn_coverage),
            "weight_transfer_runtime_inspection_available": bool(report.runtime_inspection_available),
            "vllm_weight_transfer_scope": report.transfer_scope,
            "weight_transfer_plan_fingerprint": report.transfer_plan_fingerprint,
            "full_policy_tensor_count": int(report.full_tensor_count),
            "full_policy_bytes": int(report.full_num_bytes),
            "weight_transfer_payload_ratio": float(report.payload_ratio),
            "weight_transfer_payload_reduction_ratio": float(1.0 - report.payload_ratio),
            "patch_transfer_validation": self.patch_transfer_validation,
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
        if (
            coverage >= 1.0
            and motn_coverage >= 1.0
            and not report.get("missing_in_training_model")
            and not report.get("shape_mismatches")
            and not report.get("dtype_mismatches")
        ):
            return
        message = (
            "vLLM native weight-transfer coverage is incomplete: "
            f"coverage={coverage:.6f}, motn_coverage={motn_coverage:.6f}, "
            f"missing_in_vllm={report.get('missing_in_vllm', [])}, "
            f"missing_in_training_model={report.get('missing_in_training_model', [])}, "
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
        actors: Optional[Sequence[tuple[str, Any, Dict[str, Any]]]],
        tokenizer,
    ) -> RolloutSyncResult:
        if self.policy_version < 0 or self.export_dir is None or int(update_step) == 0:
            result = self._sync_export_reload(model=model, update_step=update_step, force=True)
            result.metadata.update(
                {
                    "vllm_sync_strategy_requested": self.strategy,
                    "vllm_weight_transfer_bootstrap": True,
                    "vllm_weight_transfer_scope": self.transfer_scope,
                    **self._frozen_drift_metadata(model, initialize=True),
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
            **self._frozen_drift_metadata(model),
        }
        try:
            self._enforce_mapping_coverage(metadata)
            required_level = str(
                getattr(self.fit_cfg.rl, "vllm_native_transfer_required_level", "four_phase") or "four_phase"
            ).strip().lower()
            try:
                adapter = select_nccl_transfer_adapter(report, required_level=required_level)
            except Exception as exc:
                raise self._native_unavailable_error(report) from exc
            fingerprint_start = time.perf_counter()
            policy_fingerprint = model_policy_fingerprint(
                model,
                sample_elements_per_tensor=int(
                    getattr(self.fit_cfg.rl, "vllm_policy_fingerprint_samples_per_tensor", 16)
                ),
            )
            metadata["vllm_policy_fingerprint"] = policy_fingerprint
            metadata["vllm_fingerprint_sec"] = max(0.0, time.perf_counter() - fingerprint_start)
            if actors:
                transfer_result = self._run_subprocess_nccl_update(
                    actors=actors,
                    model=model,
                    tokenizer=tokenizer,
                    update_step=update_step,
                    policy_fingerprint=policy_fingerprint,
                    required_level=required_level,
                )
                metadata.update(transfer_result)
            else:
                if llm is None:
                    raise RuntimeError("vLLM native weight transfer requires an initialized vLLM engine")
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

    def _resolve_actor_weight_transfer_master_port(self, actor_name: str, actor_index: int) -> int:
        if actor_name in self.actor_weight_transfer_master_ports:
            return int(self.actor_weight_transfer_master_ports[actor_name])
        configured = int(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_port", 0))
        if configured > 0:
            port = configured + int(actor_index)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((str(getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_addr", "127.0.0.1")), 0))
                port = int(sock.getsockname()[1])
        self.actor_weight_transfer_master_ports[actor_name] = int(port)
        return int(port)

    @staticmethod
    def _validation_prompt_token_ids(tokenizer) -> Optional[list[int]]:
        if tokenizer is None:
            return None
        try:
            return [int(token) for token in tokenizer.encode("1 + 1 =", add_special_tokens=False)]
        except Exception:
            return None

    def _run_subprocess_nccl_update(
        self,
        *,
        actors: Sequence[tuple[str, Any, Dict[str, Any]]],
        model,
        tokenizer,
        update_step: int,
        policy_fingerprint: str,
        required_level: str,
    ) -> Dict[str, Any]:
        """Update persistent subprocess actors and commit only after all validate.

        Actor updates are deliberately sequential in the first production
        implementation.  This avoids driving the same trainer tensors through
        multiple NCCL process groups concurrently before that path has a GPU
        soak test.  Rollout remains blocked by the synchronous caller for the
        entire transaction.
        """
        if not actors:
            raise RuntimeError("subprocess native transfer requires at least one rollout actor")
        mapping = self._mapping(model)
        expected_checksums = selected_tensor_checksums(mapping, model)
        descriptor = {
            "policy_version": int(update_step),
            "policy_fingerprint": str(policy_fingerprint),
            "export_dir": None if self.export_dir is None else str(self.export_dir),
            "weight_transfer_scope": self.transfer_scope,
            "weight_transfer_plan_fingerprint": mapping.transfer_plan_fingerprint,
        }
        prompt_ids = self._validation_prompt_token_ids(tokenizer)
        require_checksums = bool(
            getattr(self.fit_cfg.rl, "vllm_weight_transfer_require_runtime_checksums", False)
        )
        actor_results: Dict[str, Any] = {}
        transaction_start = time.perf_counter()
        for actor_index, (actor_name, client, actor_spec) in enumerate(actors):
            if int(actor_spec.get("tensor_parallel_size", 1)) != 1:
                raise RuntimeError(
                    "subprocess native NCCL transfer currently requires TP1 actors; "
                    f"actor={actor_name!r}, tensor_parallel_size={actor_spec.get('tensor_parallel_size')}"
                )
            if actor_name not in self.actor_capability_reports:
                self.actor_capability_reports[actor_name] = dict(client.weight_transfer_capabilities())
            capabilities = dict(self.actor_capability_reports[actor_name])
            actual_level = str(capabilities.get("native_transfer_level", "none"))
            level_order = {"none": 0, "update_only": 1, "four_phase": 2}
            if level_order.get(actual_level, 0) < level_order.get(required_level, 0):
                raise RuntimeError(
                    f"actor {actor_name!r} native transfer level {actual_level!r} "
                    f"is below required {required_level!r}: {capabilities}"
                )
            settings = NCCLTransferSettings(
                master_address=str(
                    getattr(self.fit_cfg.rl, "vllm_weight_transfer_master_addr", "127.0.0.1") or "127.0.0.1"
                ),
                master_port=self._resolve_actor_weight_transfer_master_port(actor_name, actor_index),
                tensor_parallel_size=1,
                packed=bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_packed", True)),
                timeout_sec=float(getattr(self.fit_cfg.rl, "vllm_weight_transfer_timeout_sec", 300.0)),
            )
            if not self.actor_weight_transfer_initialized.get(actor_name, False):
                init_request_id = client.begin_init_weight_transfer(
                    init_info=nccl_init_request_payload(settings)
                )
                # NCCL rendezvous is collective. The actor receiver and
                # trainer rank must initialize concurrently, matching the
                # official vLLM 0.19 RLHF NCCL lifecycle.
                trainer_group = trainer_initialize_nccl_group(settings)
                init_result = client.finish_init_weight_transfer(init_request_id)
                if not bool(init_result.get("initialized")):
                    raise RuntimeError(f"actor {actor_name!r} failed to initialize NCCL receiver: {init_result}")
                self.actor_trainer_nccl_groups[actor_name] = trainer_group
                self.actor_weight_transfer_initialized[actor_name] = True
            request_id = client.begin_weight_update(
                update_info=nccl_update_request_payload(mapping, packed=settings.packed),
                policy_descriptor=descriptor,
                transfer_scope=self.transfer_scope,
                transfer_plan_fingerprint=mapping.transfer_plan_fingerprint,
                expected_checksums=expected_checksums,
                validation_prompt_token_ids=prompt_ids,
                require_runtime_checksums=require_checksums,
            )
            send_result = trainer_send_weights_to_actor(
                model=model,
                mapping=mapping,
                settings=settings,
                group=self.actor_trainer_nccl_groups[actor_name],
            )
            receive_result = client.finish_weight_update(request_id)
            if receive_result.get("actor_status") != "UPDATED_PENDING_COMMIT":
                raise RuntimeError(
                    f"actor {actor_name!r} did not reach the commit barrier: {receive_result}"
                )
            actor_results[actor_name] = {
                "capabilities": capabilities,
                "master_port": int(settings.master_port),
                "send": send_result,
                "receive": receive_result,
                "committed": False,
            }

        # No actor is allowed to generate until every receiver has completed.
        # A failure here is fail-closed: the rollout backend discards all
        # actors because update_only has no rollback primitive.
        for actor_name, client, _actor_spec in actors:
            commit_result = client.commit_weight_update(policy_descriptor=descriptor)
            if not bool(commit_result.get("committed")):
                raise RuntimeError(f"actor {actor_name!r} failed policy commit: {commit_result}")
            actor_results[actor_name]["commit"] = commit_result
            actor_results[actor_name]["committed"] = True

        return {
            "weight_transfer_adapter": "nccl_update_only_subprocess",
            "weight_transfer_capability_level": "update_only",
            "weight_transfer_update_step": int(update_step),
            "weight_transfer_packed": bool(getattr(self.fit_cfg.rl, "vllm_weight_transfer_packed", True)),
            "weight_transfer_total_sec": max(0.0, time.perf_counter() - transaction_start),
            "weight_transfer_actor_count": len(actors),
            "weight_transfer_commit_barrier": True,
            "weight_transfer_commit_atomic": False,
            "weight_transfer_partial_failure_policy": "discard_all_actors",
            "weight_transfer_actor_mode": "sequential",
            "weight_transfer_actor_results": actor_results,
            "rollout_facing_validation": {
                "mode": "actor_greedy_token_generation" if prompt_ids else "skipped",
                "ok": all(
                    result["receive"].get("rollout_facing_validation", {}).get("ok") is not False
                    for result in actor_results.values()
                ),
            },
            "checksum_validation": {
                "mode": "actor_selected_tensor_checksum",
                "required": require_checksums,
                "ok": all(
                    result["receive"].get("checksum_validation", {}).get("ok") is True
                    for result in actor_results.values()
                ) if require_checksums else None,
                "expected": expected_checksums,
            },
        }

    def _run_nccl_update(self, *, adapter, llm, model, update_step: int) -> Dict[str, Any]:
        mapping = self._mapping(model)
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
        mapping = self._mapping(model, runtime_params=runtime_params)
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
