from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, Protocol, Tuple

import torch

from .vllm_weight_mapping import VLLMWeightMappingReport, iter_transfer_tensors
from .vllm_weight_transfer_capabilities import VLLMWeightTransferCapabilityReport


TRANSFER_LEVEL_ORDER = {"none": 0, "update_only": 1, "four_phase": 2}


@dataclass(frozen=True)
class NCCLTransferSettings:
    master_address: str
    master_port: int
    tensor_parallel_size: int
    packed: bool
    timeout_sec: float

    @property
    def world_size(self) -> int:
        return int(self.tensor_parallel_size) + 1


class NCCLTransferAdapter(Protocol):
    name: str
    capability_level: str

    def transfer(
        self,
        *,
        llm: Any,
        model: torch.nn.Module,
        mapping: VLLMWeightMappingReport,
        settings: NCCLTransferSettings,
        initialized: bool,
        update_step: int,
    ) -> Tuple[Dict[str, Any], bool]: ...


def _payload(value: Any) -> Dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return dict(vars(value))


def _wrap_init_request(init_info: Any) -> Any:
    try:
        from vllm.distributed.weight_transfer.base import WeightTransferInitRequest  # type: ignore

        return WeightTransferInitRequest(init_info=_payload(init_info))
    except Exception:
        return {"init_info": _payload(init_info)}


def _wrap_update_request(update_info: Any) -> Any:
    try:
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest  # type: ignore

        return WeightTransferUpdateRequest(update_info=_payload(update_info))
    except Exception:
        return {"update_info": _payload(update_info)}


def nccl_init_request_payload(settings: NCCLTransferSettings) -> Dict[str, Any]:
    """Return a serialization-safe receiver initialization payload.

    Subprocess rollout actors cannot receive vLLM request/dataclass instances
    over the control pipe reliably across vLLM releases.  The actor recreates
    the concrete vLLM request from this plain dictionary.
    """
    return {
        "master_address": str(settings.master_address),
        "master_port": int(settings.master_port),
        "rank_offset": 1,
        "world_size": int(settings.world_size),
    }


def nccl_update_request_payload(
    mapping: VLLMWeightMappingReport,
    *,
    packed: bool,
) -> Dict[str, Any]:
    """Return the checkpoint-format receiver contract for one full update."""
    return {
        "names": [entry.transfer_name for entry in mapping.entries],
        "dtype_names": [entry.dtype for entry in mapping.entries],
        "shapes": [list(entry.shape) for entry in mapping.entries],
        "packed": bool(packed),
    }


def trainer_send_weights_to_actor(
    *,
    model: torch.nn.Module,
    mapping: VLLMWeightMappingReport,
    settings: NCCLTransferSettings,
    group: Any,
) -> Dict[str, Any]:
    """Send the mapping-selected checkpoint-format tensors to a waiting actor.

    The receiver must have entered ``LLM.update_weights`` before this function
    is called.  Keeping the trainer half here avoids importing the training
    model into the actor and is the cross-process equivalent of
    ``_run_sender_receiver``.
    """
    from vllm.distributed.weight_transfer.nccl_engine import (  # type: ignore
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    start = __import__("time").perf_counter()
    trainer_args = NCCLTrainerSendWeightsArgs(group=group, packed=bool(settings.packed))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm_actor_weight_send")
    future = executor.submit(
        NCCLWeightTransferEngine.trainer_send_weights,
        iter_transfer_tensors(mapping, model),
        trainer_args,
    )
    done, pending = wait([future], timeout=float(settings.timeout_sec))
    if pending:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(
            "Timed out sending selected policy tensors to subprocess vLLM actor; "
            "the actor and trainer NCCL group must be discarded. "
            f"timeout_sec={settings.timeout_sec}"
        )
    try:
        future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return {
        "trainer_send_sec": max(0.0, __import__("time").perf_counter() - start),
        "weight_transfer_tensor_count": int(mapping.tensor_count),
        "weight_transfer_bytes": int(mapping.num_bytes),
    }


def trainer_initialize_nccl_group(settings: NCCLTransferSettings) -> Any:
    """Initialize the trainer rank once and reuse it for every actor update."""
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine  # type: ignore

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm_actor_group_init")
    future = executor.submit(
        NCCLWeightTransferEngine.trainer_init,
        {
            "master_address": settings.master_address,
            "master_port": int(settings.master_port),
            "world_size": settings.world_size,
        },
    )
    done, pending = wait([future], timeout=float(settings.timeout_sec))
    if pending:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(
            "Timed out initializing the trainer side of the subprocess vLLM NCCL group; "
            f"timeout_sec={settings.timeout_sec}"
        )
    try:
        return future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def shutdown_trainer_nccl_group(group: Any) -> None:
    """Best-effort teardown for vLLM StatelessProcessGroup variants."""
    for name in ("destroy", "shutdown", "close"):
        method = getattr(group, name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            return


def _run_sender_receiver(
    *,
    receiver,
    receiver_request: Any,
    tensor_iterator: Iterable[tuple[str, torch.Tensor]],
    trainer_args: Any,
    timeout_sec: float,
) -> None:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine  # type: ignore

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vllm_weight_transfer")
    try:
        receive_future = executor.submit(receiver, receiver_request)
        send_future = executor.submit(NCCLWeightTransferEngine.trainer_send_weights, tensor_iterator, trainer_args)
        done, pending = wait([receive_future, send_future], timeout=float(timeout_sec))
        if pending:
            for future in pending:
                future.cancel()
            raise TimeoutError(
                "Timed out during vLLM NCCL weight transfer; the vLLM engine must be discarded before retry. "
                f"timeout_sec={float(timeout_sec)}"
            )
        for future in done:
            future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class UpdateOnlyNCCLTransferAdapter:
    name = "nccl_update_only"
    capability_level = "update_only"

    @staticmethod
    def _initialize(llm: Any, settings: NCCLTransferSettings, initialized: bool) -> bool:
        if initialized:
            return True
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferInitInfo  # type: ignore

        init_info = NCCLWeightTransferInitInfo(
            master_address=settings.master_address,
            master_port=int(settings.master_port),
            rank_offset=1,
            world_size=settings.world_size,
        )
        llm.init_weight_transfer_engine(_wrap_init_request(init_info))
        return True

    @staticmethod
    def _make_update_info(mapping: VLLMWeightMappingReport, packed: bool) -> Any:
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferUpdateInfo  # type: ignore

        return NCCLWeightTransferUpdateInfo(**nccl_update_request_payload(mapping, packed=packed))

    @staticmethod
    def _send(
        *,
        llm: Any,
        model: torch.nn.Module,
        mapping: VLLMWeightMappingReport,
        settings: NCCLTransferSettings,
        update_info: Any,
        wrap_request: bool = True,
    ) -> None:
        from vllm.distributed.weight_transfer.nccl_engine import (  # type: ignore
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
            NCCLWeightTransferInitInfo,
        )

        group = NCCLWeightTransferEngine.trainer_init(
            NCCLWeightTransferInitInfo(
                master_address=settings.master_address,
                master_port=int(settings.master_port),
                rank_offset=0,
                world_size=settings.world_size,
            )
        )
        trainer_args = NCCLTrainerSendWeightsArgs(group=group, packed=bool(settings.packed))
        _run_sender_receiver(
            receiver=llm.update_weights,
            receiver_request=_wrap_update_request(update_info) if wrap_request else update_info,
            tensor_iterator=iter_transfer_tensors(mapping, model),
            trainer_args=trainer_args,
            timeout_sec=settings.timeout_sec,
        )

    def transfer(
        self,
        *,
        llm: Any,
        model: torch.nn.Module,
        mapping: VLLMWeightMappingReport,
        settings: NCCLTransferSettings,
        initialized: bool,
        update_step: int,
    ) -> Tuple[Dict[str, Any], bool]:
        initialized = self._initialize(llm, settings, initialized)
        self._send(
            llm=llm,
            model=model,
            mapping=mapping,
            settings=settings,
            update_info=self._make_update_info(mapping, settings.packed),
        )
        return {
            "weight_transfer_adapter": self.name,
            "weight_transfer_capability_level": self.capability_level,
            "weight_transfer_update_step": int(update_step),
            "weight_transfer_packed": bool(settings.packed),
            "weight_transfer_master_port": int(settings.master_port),
        }, initialized


class FourPhaseNCCLTransferAdapter(UpdateOnlyNCCLTransferAdapter):
    name = "nccl_four_phase"
    capability_level = "four_phase"

    @staticmethod
    def _initialize(llm: Any, settings: NCCLTransferSettings, initialized: bool) -> bool:
        if initialized:
            return True
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferInitInfo  # type: ignore

        llm.init_weight_transfer_engine(
            NCCLWeightTransferInitInfo(
                master_address=settings.master_address,
                master_port=int(settings.master_port),
                rank_offset=1,
                world_size=settings.world_size,
            )
        )
        return True

    def transfer(
        self,
        *,
        llm: Any,
        model: torch.nn.Module,
        mapping: VLLMWeightMappingReport,
        settings: NCCLTransferSettings,
        initialized: bool,
        update_step: int,
    ) -> Tuple[Dict[str, Any], bool]:
        if not callable(getattr(llm, "start_weight_update", None)) or not callable(
            getattr(llm, "finish_weight_update", None)
        ):
            raise RuntimeError("four_phase adapter requires start_weight_update and finish_weight_update")
        initialized = self._initialize(llm, settings, initialized)
        update_info = llm.start_weight_update(is_checkpoint_format=True)
        if update_info is None:
            update_info = self._make_update_info(mapping, settings.packed)
        try:
            self._send(
                llm=llm,
                model=model,
                mapping=mapping,
                settings=settings,
                update_info=update_info,
                wrap_request=False,
            )
        finally:
            llm.finish_weight_update()
        return {
            "weight_transfer_adapter": self.name,
            "weight_transfer_capability_level": self.capability_level,
            "weight_transfer_update_step": int(update_step),
            "weight_transfer_packed": bool(settings.packed),
            "weight_transfer_master_port": int(settings.master_port),
        }, initialized


def select_nccl_transfer_adapter(
    report: VLLMWeightTransferCapabilityReport,
    *,
    required_level: str,
) -> NCCLTransferAdapter:
    required_level = str(required_level or "four_phase").strip().lower()
    actual_level = str(report.native_transfer_level or "none").strip().lower()
    if required_level not in {"update_only", "four_phase"}:
        raise ValueError(f"NCCL transfer required_level must be 'update_only' or 'four_phase', got {required_level!r}")
    if TRANSFER_LEVEL_ORDER.get(actual_level, 0) < TRANSFER_LEVEL_ORDER[required_level]:
        raise RuntimeError(
            "vLLM native weight transfer capability is below the configured safety gate: "
            f"actual={actual_level!r}, required={required_level!r}, missing={report.missing}"
        )
    if actual_level == "four_phase":
        return FourPhaseNCCLTransferAdapter()
    if actual_level == "update_only":
        return UpdateOnlyNCCLTransferAdapter()
    raise RuntimeError(f"vLLM runtime does not expose a usable NCCL transfer adapter: actual={actual_level!r}")
