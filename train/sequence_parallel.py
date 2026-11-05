"""Optional HF/DeepSpeed Ulysses integration; core MoTE algorithms stay unchanged.

The initial contract is fixed-length, unpacked, unpadded causal text with DDP.
Attention communication and sequence sharding are provided by DeepSpeed. Only
router logits (N x experts), never FFN hidden states, are gathered for global
MoTE capacity/auxiliary statistics. Imports of DeepSpeed are intentionally lazy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_tensor_autograd
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from ..gate import SoftGate, TopKGate, _SoftGate, _TopKGate


@dataclass(frozen=True)
class UlyssesConfig:
    size: int
    sequence_length: int
    # Existing FitMoTN SFT optimizes CE alone. Explicit opt-in adds sum(l_aux).
    auxiliary_loss_weight: float = 0.0


def validate_model(model, config: UlyssesConfig) -> None:
    if config.size < 2 or config.sequence_length < 2 or config.sequence_length % config.size:
        raise ValueError("Ulysses requires size >= 2 and sequence_length divisible by size")
    if not 0 <= config.auxiliary_loss_weight < float("inf"):
        raise ValueError("auxiliary_loss_weight must be finite and nonnegative")
    cfg = model.config
    if cfg.model_type not in {"qwen3", "llama"}:
        raise ValueError(f"Ulysses adapter currently supports full-attention Qwen3/Llama; got {cfg.model_type}. "
                         "Gemma3 windows and Qwen3.5 recurrent attention need separate adapters.")
    if getattr(cfg, "use_sliding_window", False) or any(
        kind != "full_attention" for kind in (getattr(cfg, "layer_types", None) or [])
    ):
        raise ValueError("Sliding/window/recurrent attention is not supported by this SDPA adapter")
    if cfg._attn_implementation != "sdpa":
        raise ValueError("Load the model with attn_implementation='sdpa'")
    if cfg.num_attention_heads % config.size:
        raise ValueError("Attention head count must be divisible by Ulysses size")
    kv = cfg.num_key_value_heads
    if kv % config.size and config.size % kv:
        raise ValueError("KV head count must divide or be divisible by Ulysses size")
    if getattr(cfg, "attention_dropout", 0):
        raise ValueError("Attention dropout must be zero for the current deterministic SP contract")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise ValueError("Nonzero dropout requires a sequence-aware RNG adapter")
        if isinstance(module, (TopKGate, _TopKGate)) and module.jitter_eps:
            raise ValueError("Router jitter requires a sequence-aware RNG adapter")


def prepare_batch(batch, sequence_length):
    """Validate BEFORE Ulysses discards masks; shift weights alongside labels."""
    batch = dict(batch)
    ids, labels = batch["input_ids"], batch["labels"]
    if ids.ndim != 2 or ids.shape != labels.shape or ids.shape[1] != sequence_length:
        raise ValueError("Expected input_ids/labels [batch, fixed sequence_length]")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("input_ids and labels must be int64")
    mask = batch.pop("attention_mask", None)
    if mask is not None and (mask.shape != ids.shape or not bool(mask.eq(1).all())):
        raise ValueError("Padded or custom attention masks are unsupported; supply full unpadded sequences")
    positions = torch.arange(sequence_length, device=ids.device).expand_as(ids)
    supplied = batch.get("position_ids")
    if supplied is not None and not torch.equal(supplied, positions):
        raise ValueError("Packed/reset position IDs are unsupported by the SDPA adapter")
    batch["position_ids"] = positions.contiguous()
    weights = batch.pop("loss_weights", None)
    if weights is not None:
        if weights.shape != ids.shape or not bool(torch.isfinite(weights).all()) or bool(weights.lt(0).any()):
            raise ValueError("loss_weights must be finite, nonnegative and match input_ids")
        batch["shift_loss_weights"] = F.pad(weights, (0, 1), value=0)[..., 1:].contiguous()
    supported = {"input_ids", "labels", "position_ids", "shift_loss_weights", "bucket", "task"}
    if set(batch) - supported:
        raise ValueError(f"Unsupported Ulysses batch fields: {sorted(set(batch) - supported)}")
    return batch


class _ValidatedBatches:
    def __init__(self, loader, runtime):
        self.loader, self.runtime = loader, runtime

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for raw in self.loader:
            error, batch = None, None
            try:
                batch = prepare_batch(raw, self.runtime.config.sequence_length)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            # All ranks fail together before any rank enters a data collective.
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
            if any(errors):
                raise ValueError(f"Invalid Ulysses batch: {errors}")
            yield batch


class _SequenceBatches(IterableDataset):
    def __init__(self, loader, runtime):
        self.loader, self.runtime = loader, runtime

    def __len__(self):
        return len(self.loader) * self.runtime.config.size

    def __iter__(self):
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter
        rt = self.runtime
        yield from UlyssesSPDataLoaderAdapter(
            _ValidatedBatches(self.loader, rt), rt.rank, rt.group, rt.config.size, rt.device)


class SequenceDataLoader(DataLoader):
    """Iterable outer loader preserves Accelerate's resume-by-skipping behavior."""
    def set_epoch(self, epoch):
        self.dataset.loader.sampler.set_epoch(epoch)


class UlyssesRuntime:
    def __init__(self, model, args, config: UlyssesConfig):
        validate_model(model, config)
        if not dist.is_initialized() or dist.get_world_size() % config.size:
            raise ValueError("Launch with torchrun; world size must be divisible by Ulysses size")
        if args.deepspeed or args.fsdp:
            raise ValueError("This adapter uses DDP + official Ulysses attention; ZeRO/FSDP are not yet integrated")
        if args.gradient_checkpointing or getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("Gradient checkpointing needs a separate SP recomputation validation")
        if args.dataloader_num_workers or args.ignore_data_skip or args.save_only_model or args.save_on_each_node:
            raise ValueError("Ulysses requires workers=0, ignore_data_skip=False, save_only_model=False, save_on_each_node=False")
        if args.eval_strategy != "no" or args.load_best_model_at_end or args.torch_compile:
            raise ValueError("Ulysses currently supports training without Trainer evaluation/compilation")
        if args.ddp_find_unused_parameters is False:
            raise ValueError("MoTE routing requires ddp_find_unused_parameters=True")
        for package, expected in {"transformers": "4.57.6", "deepspeed": "0.19.9", "accelerate": "1.15.0"}.items():
            if version(package) != expected:
                raise ValueError(f"Validated Ulysses integration requires {package}=={expected}")
        if version("torch").split("+")[0] != "2.10.0":
            raise ValueError("The subgroup autograd collective is validated with torch==2.10.0")
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        import deepspeed
        # Reuse torchrun/Accelerate's process group; never create a second WORLD.
        deepspeed.comm.init_distributed(dist_backend=dist.get_backend(), auto_mpi_discovery=False)
        self.original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.mpu = UlyssesSPAttentionHF.register_with_transformers(
            model, "sdpa", config.size, args.per_device_train_batch_size,
            seq_length=config.sequence_length, seq_length_is_variable=False)
        self.config, self.device = config, args.device
        self.group = self.mpu.get_sequence_parallel_group()
        self.rank = dist.get_rank(self.group)
        self.batch_size = args.per_device_train_batch_size
        self.auxiliary = []
        self.handles = []
        for gate in model.modules():
            if isinstance(gate, (TopKGate, SoftGate, _TopKGate, _SoftGate)):
                router = gate.router if hasattr(gate, "router") else gate.fc
                self.handles.append(router.register_forward_hook(self._gather_logits))
                self.handles.append(gate.register_forward_hook(self._slice_routes))

    def _gather_logits(self, module, inputs, logits):
        local = self.config.sequence_length // self.config.size
        if logits.shape[0] != self.batch_size * local or logits.ndim != 2:
            raise ValueError("Router token layout differs from [batch * local_sequence, experts]")
        # Autograd's official distributed gather sums remote contributions back
        # into each rank's local router graph. No hidden-state replication.
        # torch.distributed.nn.functional's Gloo fallback uses global src=i in
        # subgroup scatter (incorrect for groups not containing rank 0). The
        # pinned official functional collective uses the group's registered name.
        gathered = all_gather_tensor_autograd(logits.contiguous(), gather_dim=0, group=self.group)
        return gathered.reshape(self.config.size, self.batch_size, local, -1).permute(
            1, 0, 2, 3).reshape(-1, logits.shape[-1])

    def _slice_routes(self, module, inputs, output):
        probs, mask, aux = output
        local = self.config.sequence_length // self.config.size
        def shard(value):
            if value is None:
                return None
            return value.reshape(self.batch_size, self.config.sequence_length, -1)[
                :, self.rank * local:(self.rank + 1) * local].reshape(self.batch_size * local, -1)
        if aux is not None and "l_aux" in aux:
            self.auxiliary.append(aux["l_aux"])
        return shard(probs), shard(mask), aux

    def dataloader(self, dataset, collator, args):
        if isinstance(dataset, IterableDataset):
            raise ValueError("Ulysses requires a finite indexable dataset for deterministic resume")
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
        base = DataLoader(dataset, sampler=sampler, batch_size=args.per_device_train_batch_size,
                          collate_fn=collator, drop_last=True, num_workers=0)
        if not len(base):
            raise ValueError("Dataset must contain at least world_size * micro_batch_size records")
        return SequenceDataLoader(_SequenceBatches(base, self), batch_size=None, num_workers=0)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.auxiliary.clear()
        out = model(input_ids=inputs["input_ids"], position_ids=inputs["position_ids"], use_cache=False)
        labels = inputs["shift_labels"]
        token_loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                                     labels.reshape(-1), reduction="none").reshape_as(labels)
        valid = labels.ne(-100)
        weights = inputs.get("shift_loss_weights", valid).to(token_loss.dtype) * valid
        denominators = torch.stack((weights.sum(), valid.sum().to(token_loss.dtype)))
        dist.all_reduce(denominators, group=self.group)
        numerator = torch.where(denominators[0] > 0, (token_loss * weights).sum(), token_loss.sum())
        denominator = torch.where(denominators[0] > 0, denominators[0], denominators[1]).clamp_min(1)
        # DDP averages gradients across SP x DP. Multiplication by SP size gives
        # the full-sequence mean within each SP group, averaged across DP groups.
        loss = numerator / denominator * self.config.size
        if self.auxiliary and self.config.auxiliary_loss_weight:
            # Each rank has the same global aux scalar. DDP averaging counts it
            # once; the autograd gather supplies all local router contributions.
            loss = loss + torch.stack(self.auxiliary).sum() * self.config.auxiliary_loss_weight
        self.auxiliary.clear()
        return (loss, out) if return_outputs else loss

    def close(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.original_sdpa
        dist.destroy_process_group(self.group)
        dist.destroy_process_group(self.mpu.get_sequence_data_parallel_group())
        self.mpu._release_sequence_parallel_groups()

    def signature(self):
        return {**asdict(self.config), "world_size": dist.get_world_size(), "backend": "ulysses-ddp-v1"}
