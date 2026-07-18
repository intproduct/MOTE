from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from typing import Any, Dict, Optional


_SPAWN_ENV_LOCK = threading.Lock()


def _cuda_resource_snapshot(*, probe_cuda_runtime: bool = True) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "process_id": int(os.getpid()),
        "process_group_id": int(os.getpgrp()),
        "child_processes": [],
        "os_child_process_ids": [],
    }
    try:
        if not probe_cuda_runtime:
            raise StopIteration
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_device_count"] = int(torch.cuda.device_count())
        devices = []
        for index in range(int(torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "logical_index": int(index),
                    "name": str(props.name),
                    "total_memory_bytes": int(props.total_memory),
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
        snapshot["cuda_devices"] = devices
    except StopIteration:
        snapshot["cuda_runtime_probe_deferred"] = True
    except Exception as exc:
        snapshot["cuda_probe_error"] = str(exc)
    try:
        snapshot["child_processes"] = [
            {"pid": child.pid, "name": child.name, "alive": child.is_alive()}
            for child in mp.active_children()
        ]
    except Exception as exc:
        snapshot["child_process_probe_error"] = str(exc)
    try:
        children_path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
        if children_path.is_file():
            snapshot["os_child_process_ids"] = [
                int(value)
                for value in children_path.read_text(encoding="utf-8").split()
                if value.isdigit()
            ]
    except Exception as exc:
        snapshot["os_child_process_probe_error"] = str(exc)
    return snapshot


@dataclass
class ActorCompletion:
    token_ids: list[int]
    text: str = ""


@dataclass
class ActorRequestOutput:
    outputs: list[ActorCompletion]


def _shutdown_engine(llm: Any) -> None:
    if llm is None:
        return
    for obj in (llm, getattr(llm, "llm_engine", None)):
        for attr in ("shutdown", "close"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass


def _engine_topology_snapshot(llm: Any, llm_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    requested_tp = int(llm_kwargs.get("tensor_parallel_size", 1))
    candidates = [
        ("llm.llm_engine.vllm_config.parallel_config", getattr(getattr(getattr(llm, "llm_engine", None), "vllm_config", None), "parallel_config", None)),
        ("llm.llm_engine.parallel_config", getattr(getattr(llm, "llm_engine", None), "parallel_config", None)),
        ("llm.vllm_config.parallel_config", getattr(getattr(llm, "vllm_config", None), "parallel_config", None)),
    ]
    for source, parallel in candidates:
        if parallel is None:
            continue
        observed = getattr(parallel, "tensor_parallel_size", None)
        if observed is None:
            continue
        return {
            "requested_tensor_parallel_size": requested_tp,
            "observed_tensor_parallel_size": int(observed),
            "tensor_parallel_verified": int(observed) == requested_tp,
            "pipeline_parallel_size": int(getattr(parallel, "pipeline_parallel_size", 1)),
            "data_parallel_size": int(getattr(parallel, "data_parallel_size", 1)),
            "device": llm_kwargs.get("device"),
            "source": source,
        }
    return {
        "requested_tensor_parallel_size": requested_tp,
        "observed_tensor_parallel_size": None,
        "tensor_parallel_verified": False,
        "device": llm_kwargs.get("device"),
        "source": "requested_only",
    }


def _actor_status(state: Dict[str, Any]) -> str:
    return str(state.get("status") or ("READY" if state.get("llm") is not None else "LOADING"))


def _collect_engine_named_parameters(llm: Any) -> Dict[str, Any]:
    candidates = [
        llm,
        getattr(llm, "model", None),
        getattr(getattr(llm, "llm_engine", None), "model", None),
        getattr(getattr(getattr(llm, "llm_engine", None), "model_executor", None), "driver_worker", None),
    ]
    for obj in candidates:
        named_parameters = getattr(obj, "named_parameters", None) if obj is not None else None
        if callable(named_parameters):
            try:
                return {str(name): tensor for name, tensor in named_parameters()}
            except Exception:
                continue
    return {}


def _invalidate_generation_caches(llm: Any) -> Dict[str, Any]:
    """Invalidate reusable prefix caches after an in-place policy update."""
    for source, obj in (
        ("llm", llm),
        ("llm.llm_engine", getattr(llm, "llm_engine", None)),
    ):
        reset = getattr(obj, "reset_prefix_cache", None) if obj is not None else None
        if callable(reset):
            result = reset()
            serialized_result = result if isinstance(result, (str, int, float, bool, type(None))) else repr(result)
            return {"attempted": True, "source": source, "result": serialized_result}
    return {"attempted": False, "source": None, "reason": "reset_prefix_cache_unavailable"}


def _actor_rollout_validation(llm: Any, prompt_token_ids: Optional[list[int]]) -> Dict[str, Any]:
    if not prompt_token_ids:
        return {"mode": "skipped", "ok": None}
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0)
    try:
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": [int(token) for token in prompt_token_ids]}],
            sampling_params=params,
        )
    except TypeError:
        outputs = llm.generate(
            prompt_token_ids=[[int(token) for token in prompt_token_ids]],
            sampling_params=params,
        )
    completions = list(getattr(outputs[0], "outputs", []) or []) if outputs else []
    token_ids = list(getattr(completions[0], "token_ids", []) or []) if completions else []
    return {
        "mode": "greedy_token_generation",
        "ok": bool(outputs and completions and token_ids),
        "generated_token_ids": [int(token) for token in token_ids[:4]],
    }


def _handle_actor_request(state: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    command = str(request.get("command", ""))
    if command == "ping":
        return {
            "ready": _actor_status(state) == "READY",
            "engine_loaded": state.get("llm") is not None,
            "actor_status": _actor_status(state),
            "policy_descriptor": state.get("policy_descriptor"),
            "pending_policy_descriptor": state.get("pending_policy_descriptor"),
            "engine_process_id": int(os.getpid()),
            "actor_resources": _cuda_resource_snapshot(),
            "engine_topology": state.get("engine_topology"),
        }
    if command == "load_engine":
        _shutdown_engine(state.get("llm"))
        from vllm import LLM  # type: ignore

        start = time.perf_counter()
        llm_kwargs = dict(request["llm_kwargs"])
        # Actor placement is already enforced by the actor-private
        # CUDA_VISIBLE_DEVICES.  vLLM 0.19 removed ``device`` from EngineArgs,
        # so forwarding even ``cuda:0`` now fails before engine startup.  Pop
        # it here as a compatibility guard for older callers as well as at the
        # kwargs construction site.
        requested_device = llm_kwargs.pop("device", None)
        state["llm"] = LLM(**llm_kwargs)
        state["policy_descriptor"] = dict(request.get("policy_descriptor") or {})
        state["pending_policy_descriptor"] = None
        state["weight_transfer_initialized"] = False
        state["status"] = "READY"
        topology_kwargs = dict(llm_kwargs)
        if requested_device is not None:
            topology_kwargs["device"] = requested_device
        state["engine_topology"] = _engine_topology_snapshot(state["llm"], topology_kwargs)
        return {
            "load_sec": max(0.0, time.perf_counter() - start),
            "policy_descriptor": state["policy_descriptor"],
            "actor_resources": _cuda_resource_snapshot(),
            "engine_topology": state["engine_topology"],
        }
    if command == "unload_engine":
        _shutdown_engine(state.get("llm"))
        state["llm"] = None
        state["policy_descriptor"] = None
        state["engine_topology"] = None
        state["pending_policy_descriptor"] = None
        state["weight_transfer_initialized"] = False
        state["status"] = "LOADING"
        return {"unloaded": True}
    if command == "weight_transfer_capabilities":
        from .vllm_weight_transfer_capabilities import probe_vllm_weight_transfer_capabilities

        return probe_vllm_weight_transfer_capabilities().to_dict()
    if command == "init_weight_transfer":
        llm = state.get("llm")
        if llm is None:
            raise RuntimeError("vLLM actor engine is not loaded")
        if _actor_status(state) != "READY":
            raise RuntimeError(f"cannot initialize weight transfer while actor status={_actor_status(state)!r}")
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferInitInfo  # type: ignore
        from .vllm_weight_transfer_adapters import _wrap_init_request

        info = NCCLWeightTransferInitInfo(**dict(request["init_info"]))
        llm.init_weight_transfer_engine(_wrap_init_request(info))
        state["weight_transfer_initialized"] = True
        return {"initialized": True, "actor_status": _actor_status(state)}
    if command == "receive_weight_update":
        llm = state.get("llm")
        if llm is None:
            raise RuntimeError("vLLM actor engine is not loaded")
        if _actor_status(state) != "READY":
            raise RuntimeError(f"cannot receive weights while actor status={_actor_status(state)!r}")
        if not bool(state.get("weight_transfer_initialized")):
            raise RuntimeError("vLLM actor weight transfer engine is not initialized")
        state["status"] = "UPDATING"
        state["pending_policy_descriptor"] = dict(request.get("policy_descriptor") or {})
        update_payload = dict(request.get("update_info") or {})
        transfer_scope = str(request.get("transfer_scope") or "full_policy")
        transfer_names = [str(name) for name in update_payload.get("names", [])]
        transfer_plan_fingerprint = str(request.get("transfer_plan_fingerprint") or "")
        descriptor_plan_fingerprint = str(
            state["pending_policy_descriptor"].get("weight_transfer_plan_fingerprint") or ""
        )
        if transfer_scope == "trainable_patch" and not transfer_names:
            state["status"] = "FAILED"
            raise RuntimeError("trainable_patch update contains no tensors")
        if transfer_plan_fingerprint != descriptor_plan_fingerprint:
            state["status"] = "FAILED"
            raise RuntimeError(
                "weight-transfer plan fingerprint does not match pending policy descriptor: "
                f"request={transfer_plan_fingerprint!r}, descriptor={descriptor_plan_fingerprint!r}"
            )
        start = time.perf_counter()
        try:
            from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferUpdateInfo  # type: ignore
            from .vllm_weight_transfer_adapters import _wrap_update_request

            update_info = NCCLWeightTransferUpdateInfo(**update_payload)
            llm.update_weights(_wrap_update_request(update_info))
            state["status"] = "VALIDATING"
            cache = _invalidate_generation_caches(llm)
            expected_checksums = dict(request.get("expected_checksums") or {})
            runtime_params = _collect_engine_named_parameters(llm)
            observed_checksums: Dict[str, float] = {}
            for name in expected_checksums:
                tensor = runtime_params.get(name)
                if tensor is not None:
                    observed_checksums[name] = float(tensor.detach().float().cpu().sum().item())
            checksum_ok = None
            if observed_checksums:
                checksum_ok = len(observed_checksums) == len(expected_checksums) and all(
                    abs(float(expected_checksums[name]) - value) <= 1e-3
                    for name, value in observed_checksums.items()
                )
                if not checksum_ok:
                    raise RuntimeError(
                        "actor post-update checksum mismatch: "
                        f"expected={expected_checksums}, observed={observed_checksums}"
                    )
            if bool(request.get("require_runtime_checksums", False)) and checksum_ok is not True:
                raise RuntimeError("actor runtime parameter inspection is unavailable or incomplete")
            rollout_validation = _actor_rollout_validation(
                llm,
                request.get("validation_prompt_token_ids"),
            )
            if rollout_validation.get("ok") is False:
                raise RuntimeError(f"actor post-update rollout validation failed: {rollout_validation}")
            state["status"] = "UPDATED_PENDING_COMMIT"
            return {
                "receive_sec": max(0.0, time.perf_counter() - start),
                "actor_status": state["status"],
                "cache_invalidation": cache,
                "runtime_inspection_available": bool(runtime_params),
                "checksum_validation": {
                    "ok": checksum_ok,
                    "expected": expected_checksums,
                    "observed": observed_checksums,
                },
                "rollout_facing_validation": rollout_validation,
                "engine_process_id": int(os.getpid()),
                "engine_resources": _cuda_resource_snapshot(),
                "received_weight_update_contract": {
                    "transfer_scope": transfer_scope,
                    "transfer_plan_fingerprint": transfer_plan_fingerprint,
                    "tensor_count": len(transfer_names),
                    "first_names": transfer_names[:8],
                },
            }
        except Exception:
            state["status"] = "FAILED"
            raise
    if command == "commit_weight_update":
        if _actor_status(state) != "UPDATED_PENDING_COMMIT":
            raise RuntimeError(f"cannot commit weights while actor status={_actor_status(state)!r}")
        expected = dict(request.get("policy_descriptor") or {})
        pending = dict(state.get("pending_policy_descriptor") or {})
        if expected != pending:
            state["status"] = "FAILED"
            raise RuntimeError(f"pending policy mismatch during commit: expected={expected}, pending={pending}")
        state["policy_descriptor"] = pending
        state["pending_policy_descriptor"] = None
        state["status"] = "READY"
        return {
            "committed": True,
            "actor_status": state["status"],
            "policy_descriptor": state["policy_descriptor"],
        }
    if command == "abort_weight_update":
        state["status"] = "FAILED"
        return {"aborted": True, "actor_status": state["status"]}
    if command == "generate":
        llm = state.get("llm")
        if llm is None:
            raise RuntimeError("vLLM actor engine is not loaded")
        if _actor_status(state) != "READY":
            raise RuntimeError(f"vLLM actor cannot generate while status={_actor_status(state)!r}")
        expected_descriptor = dict(request.get("expected_policy_descriptor") or {})
        loaded_descriptor = dict(state.get("policy_descriptor") or {})
        if expected_descriptor and expected_descriptor != loaded_descriptor:
            raise RuntimeError(
                "vLLM actor policy provenance mismatch: "
                f"expected={expected_descriptor}, loaded={loaded_descriptor}"
            )
        from vllm import SamplingParams  # type: ignore

        sampling_params = SamplingParams(**dict(request["sampling_kwargs"]))
        prompt_token_ids = [list(map(int, row)) for row in request["prompt_token_ids"]]
        try:
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
                sampling_params=sampling_params,
            )
        except TypeError:
            outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        serialized = []
        for output in outputs:
            completions = list(getattr(output, "outputs", []) or [])
            serialized.append(
                [
                    {
                        "token_ids": [int(token) for token in list(getattr(completion, "token_ids", []) or [])],
                        "text": str(getattr(completion, "text", "") or ""),
                    }
                    for completion in completions
                ]
            )
        return {"outputs": serialized}
    if command == "sleep":
        llm = state.get("llm")
        if llm is None:
            return {"slept": False, "reason": "engine_not_loaded"}
        sleep_fn = getattr(llm, "sleep", None)
        if not callable(sleep_fn):
            raise RuntimeError("vLLM actor engine does not expose sleep()")
        sleep_fn(level=int(request.get("level", 1)))
        return {"slept": True, "level": int(request.get("level", 1))}
    if command == "wake_up":
        llm = state.get("llm")
        if llm is None:
            return {"woke": False, "reason": "engine_not_loaded"}
        wake_fn = getattr(llm, "wake_up", None)
        if not callable(wake_fn):
            raise RuntimeError("vLLM actor engine does not expose wake_up()")
        tags = request.get("tags")
        wake_fn(tags=None if tags is None else list(tags))
        return {"woke": True, "tags": tags}
    if command == "close":
        # Acknowledge the control-plane close before tearing down vLLM.  In
        # vLLM 0.19 an EngineCore can block indefinitely while destroying an
        # update-only NCCL receiver.  _actor_main sends this response first,
        # then performs the best-effort shutdown from its finally block.  The
        # parent can therefore enforce a bounded join/TERM/KILL sequence.
        state["closed"] = True
        state["status"] = "CLOSED"
        return {"closed": True}
    raise ValueError(f"Unsupported vLLM actor command {command!r}")


def _actor_main(connection, environment: Optional[Dict[str, str]] = None) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    for key, value in dict(environment or {}).items():
        os.environ[str(key)] = str(value)
    state: Dict[str, Any] = {
        "llm": None,
        "closed": False,
        "policy_descriptor": None,
        "engine_topology": None,
        "pending_policy_descriptor": None,
        "weight_transfer_initialized": False,
        "status": "LOADING",
    }
    # Do not initialize CUDA before vLLM chooses and starts its worker model.
    connection.send({"kind": "ready", "actor_resources": _cuda_resource_snapshot(probe_cuda_runtime=False)})
    try:
        while not state["closed"]:
            request = connection.recv()
            request_id = request.get("request_id")
            try:
                payload = _handle_actor_request(state, request)
                connection.send({"request_id": request_id, "ok": True, "payload": payload})
            except Exception as exc:
                connection.send(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                )
    except EOFError:
        pass
    finally:
        _shutdown_engine(state.get("llm"))
        connection.close()


class VLLMActorClient:
    def __init__(
        self,
        *,
        start_method: str = "spawn",
        request_timeout_sec: float = 600.0,
        shutdown_timeout_sec: float = 30.0,
        cuda_visible_devices: Optional[list[str]] = None,
    ) -> None:
        self.start_method = str(start_method)
        self.request_timeout_sec = float(request_timeout_sec)
        self.shutdown_timeout_sec = float(shutdown_timeout_sec)
        self.cuda_visible_devices = [str(item) for item in list(cuda_visible_devices or [])]
        self._connection = None
        self._process = None
        self._request_id = 0
        self.startup_info: Dict[str, Any] = {}
        self.last_engine_info: Dict[str, Any] = {}
        self._pending_request: Optional[tuple[int, str]] = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        context = mp.get_context(self.start_method)
        parent, child = context.Pipe(duplex=True)
        # vLLM may create its own worker processes; a daemon multiprocessing
        # process is not allowed to create children.
        environment = {}
        if self.cuda_visible_devices:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(self.cuda_visible_devices)
        process = context.Process(
            target=_actor_main,
            args=(child, environment),
            name="fitmotn-vllm-actor",
            daemon=False,
        )
        # multiprocessing has no per-child env argument. Hold a process-local
        # lock while spawn snapshots the environment, then restore the trainer.
        with _SPAWN_ENV_LOCK:
            previous = {key: os.environ.get(key) for key in environment}
            try:
                os.environ.update(environment)
                process.start()
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        child.close()
        self._connection = parent
        self._process = process
        if not parent.poll(self.request_timeout_sec):
            self.close(force=True)
            raise TimeoutError("Timed out waiting for the vLLM actor process to start")
        try:
            message = parent.recv()
        except EOFError as exc:
            exitcode = process.exitcode
            self.close(force=True)
            raise RuntimeError(f"vLLM actor exited during startup with code {exitcode}") from exc
        if message.get("kind") != "ready":
            self.close(force=True)
            raise RuntimeError(f"Unexpected vLLM actor startup response: {message}")
        self.startup_info = dict(message)

    def _begin_request(self, command: str, **payload: Any) -> int:
        self.start()
        if self._connection is None or self._process is None:
            raise RuntimeError("vLLM actor process is unavailable")
        if not self._process.is_alive():
            raise RuntimeError(f"vLLM actor process exited unexpectedly with code {self._process.exitcode}")
        if self._pending_request is not None:
            raise RuntimeError(f"vLLM actor already has an outstanding request: {self._pending_request}")
        self._request_id += 1
        request_id = self._request_id
        self._connection.send({"request_id": request_id, "command": command, **payload})
        self._pending_request = (request_id, command)
        return request_id

    def _finish_request(
        self,
        request_id: int,
        command: str,
        *,
        timeout_sec: Optional[float] = None,
        force_close_on_timeout: bool = True,
    ) -> Dict[str, Any]:
        if self._connection is None or self._process is None:
            raise RuntimeError("vLLM actor process is unavailable")
        effective_timeout = self.request_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        if not self._connection.poll(effective_timeout):
            self._pending_request = None
            if force_close_on_timeout:
                self.close(force=True)
            raise TimeoutError(f"Timed out waiting for vLLM actor command={command!r}")
        try:
            response = self._connection.recv()
        except EOFError as exc:
            exitcode = self._process.exitcode
            self.close(force=True)
            raise RuntimeError(
                f"vLLM actor exited during command={command!r} with code {exitcode}"
            ) from exc
        if response.get("request_id") != request_id:
            self.close(force=True)
            raise RuntimeError(f"Mismatched vLLM actor response id: expected={request_id}, response={response}")
        self._pending_request = None
        if not response.get("ok"):
            raise RuntimeError(
                f"vLLM actor command={command!r} failed: {response.get('error_type')}: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return dict(response.get("payload") or {})

    def _request(self, command: str, **payload: Any) -> Dict[str, Any]:
        request_id = self._begin_request(command, **payload)
        return self._finish_request(request_id, command)

    def load_engine(
        self,
        llm_kwargs: Dict[str, Any],
        *,
        policy_descriptor: Optional[Dict[str, Any]] = None,
    ) -> float:
        result = self._request(
            "load_engine",
            llm_kwargs=dict(llm_kwargs),
            policy_descriptor=dict(policy_descriptor or {}),
        )
        self.last_engine_info = dict(result)
        resources = dict(result.get("actor_resources") or {})
        if self.cuda_visible_devices and int(resources.get("cuda_device_count", 0)) != len(self.cuda_visible_devices):
            self.close(force=True)
            raise RuntimeError(
                "vLLM actor CUDA isolation mismatch after engine load: "
                f"configured={self.cuda_visible_devices}, observed={resources}"
            )
        return float(result.get("load_sec", 0.0))

    def ping(self) -> Dict[str, Any]:
        return self._request("ping")

    def weight_transfer_capabilities(self) -> Dict[str, Any]:
        return self._request("weight_transfer_capabilities")

    def init_weight_transfer(self, *, init_info: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("init_weight_transfer", init_info=dict(init_info))

    def begin_init_weight_transfer(self, *, init_info: Dict[str, Any]) -> int:
        return self._begin_request("init_weight_transfer", init_info=dict(init_info))

    def finish_init_weight_transfer(self, request_id: int) -> Dict[str, Any]:
        return self._finish_request(request_id, "init_weight_transfer")

    def begin_weight_update(
        self,
        *,
        update_info: Dict[str, Any],
        policy_descriptor: Dict[str, Any],
        expected_checksums: Dict[str, float],
        validation_prompt_token_ids: Optional[list[int]],
        transfer_scope: str = "full_policy",
        transfer_plan_fingerprint: str = "",
        require_runtime_checksums: bool = False,
    ) -> int:
        return self._begin_request(
            "receive_weight_update",
            update_info=dict(update_info),
            policy_descriptor=dict(policy_descriptor),
            transfer_scope=str(transfer_scope),
            transfer_plan_fingerprint=str(transfer_plan_fingerprint),
            expected_checksums=dict(expected_checksums),
            validation_prompt_token_ids=validation_prompt_token_ids,
            require_runtime_checksums=bool(require_runtime_checksums),
        )

    def finish_weight_update(self, request_id: int) -> Dict[str, Any]:
        return self._finish_request(request_id, "receive_weight_update")

    def commit_weight_update(self, *, policy_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("commit_weight_update", policy_descriptor=dict(policy_descriptor))

    def abort_weight_update(self) -> Dict[str, Any]:
        return self._request("abort_weight_update")

    def unload_engine(self) -> None:
        if self.is_alive:
            self._request("unload_engine")

    def sleep(self, *, level: int = 1) -> Dict[str, Any]:
        return self._request("sleep", level=int(level))

    def wake_up(self, *, tags: Optional[list[str]] = None) -> Dict[str, Any]:
        return self._request("wake_up", tags=tags)

    def generate(
        self,
        *,
        prompt_token_ids: list[list[int]],
        sampling_kwargs: Dict[str, Any],
        expected_policy_descriptor: Optional[Dict[str, Any]] = None,
    ) -> list[ActorRequestOutput]:
        result = self._request(
            "generate",
            prompt_token_ids=prompt_token_ids,
            sampling_kwargs=dict(sampling_kwargs),
            expected_policy_descriptor=dict(expected_policy_descriptor or {}),
        )
        return [
            ActorRequestOutput(
                outputs=[
                    ActorCompletion(token_ids=list(item.get("token_ids") or []), text=str(item.get("text") or ""))
                    for item in completions
                ]
            )
            for completions in result.get("outputs", [])
        ]

    def close(self, *, force: bool = False) -> None:
        process = self._process
        connection = self._connection
        if process is None:
            return
        graceful_acknowledged = False
        if process.is_alive() and not force:
            try:
                request_id = self._begin_request("close")
                self._finish_request(
                    request_id,
                    "close",
                    timeout_sec=min(self.request_timeout_sec, self.shutdown_timeout_sec),
                    force_close_on_timeout=False,
                )
                graceful_acknowledged = True
            except Exception:
                force = True
        # Only grant a graceful EngineCore teardown window after the actor has
        # acknowledged close.  If the control pipe itself timed out, move
        # directly to terminating the actor process group.
        if graceful_acknowledged:
            process.join(timeout=self.shutdown_timeout_sec)
        if process.is_alive():
            self._terminate_actor_group(process, signal.SIGTERM)
            process.join(timeout=self.shutdown_timeout_sec)
        if process.is_alive():
            self._terminate_actor_group(process, signal.SIGKILL)
            process.join(timeout=min(5.0, self.shutdown_timeout_sec))
        if connection is not None:
            connection.close()
        self._connection = None
        self._process = None
        self.startup_info = {}
        self.last_engine_info = {}
        self._pending_request = None

    @staticmethod
    def _terminate_actor_group(process, sig: signal.Signals) -> None:
        pid = getattr(process, "pid", None)
        if pid:
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, sig)
                    return
            except (OSError, ProcessLookupError):
                pass
        if sig == signal.SIGTERM:
            process.terminate()
        elif hasattr(process, "kill"):
            process.kill()
        else:
            process.terminate()
