"""Two-process CPU/Gloo compatibility probe using the installed official Ulysses.

This is an isolated diagnostic, not a production training launcher. No weights
are downloaded. Tiny random HF models exercise real forward/backward collectives.
Run with an isolated environment containing DeepSpeed 0.19.9 and Transformers
4.57.6. Outputs distinguish numerical parity from unsupported combinations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F


def build_model(family, mote):
    from transformers import (
        Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM,
        Gemma3TextConfig, Gemma3ForCausalLM,
    )
    from fitmotn.model import MOTNFFNLayer
    classes = {"qwen3": (Qwen3Config, Qwen3ForCausalLM),
               "llama": (LlamaConfig, LlamaForCausalLM),
               "gemma3": (Gemma3TextConfig, Gemma3ForCausalLM)}
    cfg_cls, model_cls = classes["gemma3" if family == "gemma3_full_control" else family]
    extras = {}
    if family.startswith("gemma3"):
        extras = dict(sliding_window=4, layer_types=["sliding_attention", "full_attention"])
        if family == "gemma3_full_control":
            extras["layer_types"] = ["full_attention", "full_attention"]
    if family == "llama":
        extras = dict(rope_theta=500000., max_position_embeddings=131072,
                      rope_scaling=dict(rope_type="llama3", factor=8., low_freq_factor=1.,
                                        high_freq_factor=4., original_max_position_embeddings=8192))
    cfg_args = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
    cfg_args.update(extras)
    cfg = cfg_cls(**cfg_args)
    cfg._attn_implementation = "sdpa"
    model = model_cls(cfg).float().train()
    if mote:
        patch = dict(E=4, d=2, k_in=2, topk=2, gate_type="topk",
                     min_capacity=1, capacity_factor=0.5, drop_tokens=mote == "drop",
                     pos_strategy="sliding", global_expert_enabled=False,
                     block_init_mode="gamma_normal")
        for i, layer in enumerate(model.model.layers):
            layer.mlp = MOTNFFNLayer(layer.mlp, patch, device=torch.device("cpu"), layer_idx=i)
    return model


def worker(rank, init_method, output, gloo_only):
    torch.set_num_threads(1)
    import deepspeed
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter,
    )
    import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from fitmotn.train.trainer import FitMoTNTrainer
    from types import SimpleNamespace

    if gloo_only:
        # Only disable the optional Linux shared-memory C++ optimization. The
        # installed official Ulysses still uses REAL Gloo all-to-all operations.
        # This avoids requiring a system C++ toolchain for the CPU diagnostic.
        deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.comm.init_distributed(dist_backend="gloo", init_method=init_method,
                                   rank=rank, world_size=2, timeout=timedelta(seconds=45),
                                   auto_mpi_discovery=False)
    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    result = {"rank": rank, "cases": []}
    try:
        # all_to_all_single is the actual transport used by the official module.
        src = torch.tensor([10 * rank, 10 * rank + 1])
        dst = torch.empty_like(src)
        dist.all_to_all_single(dst, src)
        torch.testing.assert_close(dst, torch.tensor([rank, 10 + rank]))
        result["gloo_all_to_all_single"] = "pass"
        for family, mote in [("qwen3", None), ("llama", None), ("gemma3", None),
                             ("gemma3_full_control", None),
                             ("qwen3", "no_drop"), ("qwen3", "drop")]:
            name = ("llama3_1" if family == "llama" else family) + ("_mote_" + mote if mote else "_dense")
            torch.manual_seed(1701)
            model = build_model(family, mote)
            reference = copy.deepcopy(model)
            ids = torch.arange(1, 17).unsqueeze(0)
            labels = ids.clone()
            labels[:, :5] = -100
            positions = torch.arange(16).unsqueeze(0)
            ref_logits = reference(input_ids=ids, position_ids=positions, use_cache=False).logits
            ref_loss = F.cross_entropy(ref_logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
            ref_loss.backward()
            # Each rank starts with a different example, as the official adapter
            # expects. Its first microbatch must reconstruct rank 0's example.
            local_ids = ids + rank
            local_labels = local_ids.clone()
            local_labels[:, :5] = -100
            batches = [dict(input_ids=local_ids, labels=local_labels, position_ids=positions)]
            adapter = UlyssesSPDataLoaderAdapter(batches, rank, dist.group.WORLD, 2,
                                                torch.device("cpu"))
            shard = next(iter(adapter))
            torch.testing.assert_close(shard["input_ids"], ids[:, rank * 8:(rank + 1) * 8])
            shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
            torch.testing.assert_close(shard["shift_labels"], shifted[:, rank * 8:(rank + 1) * 8])
            torch.testing.assert_close(shard["position_ids"], positions[:, rank * 8:(rank + 1) * 8])
            UlyssesSPAttentionHF.register_with_transformers(
                model, "sdpa", 2, 1, seq_length=16, seq_length_is_variable=False)
            inputs = {k: v for k, v in shard.items() if k != "shift_labels"}
            out = model(**inputs, use_cache=False)
            local_loss = F.cross_entropy(out.logits.reshape(-1, 64),
                                         shard["shift_labels"].reshape(-1), reduction="sum")
            local_loss = local_loss / shifted.ne(-100).sum()
            local_loss.backward()
            total_loss = local_loss.detach().clone()
            dist.all_reduce(total_loss)
            gathered = [torch.empty_like(out.logits) for _ in range(2)]
            dist.all_gather(gathered, out.logits.detach())
            logits = torch.cat(gathered, dim=1)
            grad_error = 0.0
            for (n, p), (rn, rp) in zip(model.named_parameters(), reference.named_parameters()):
                assert n == rn and (p.grad is None) == (rp.grad is None), n
                if p.grad is not None:
                    # Each replica contributes a disjoint set of target tokens;
                    # SUM restores the full-sequence gradient in this probe.
                    dist.all_reduce(p.grad)
                    grad_error = max(grad_error, (p.grad - rp.grad).abs().max().item())
            case = dict(name=name, data_positions_shift_labels="pass",
                        logits_max_abs=(logits - ref_logits.detach()).abs().max().item(),
                        loss_abs=abs(total_loss.item() - ref_loss.item()),
                        gradients_max_abs=grad_error)
            case["parity"] = all(case[k] < 2e-5 for k in
                                 ("logits_max_abs", "loss_abs", "gradients_max_abs"))
            # Exercise the real custom Trainer method with the official batch.
            try:
                loss = FitMoTNTrainer.compute_loss(
                    SimpleNamespace(observability_state=None, kd_enabled=False), model, shard)
                case["current_trainer_returns_loss"] = torch.is_tensor(loss)
            except (AssertionError, ValueError, AttributeError) as exc:
                case["current_trainer_error"] = str(exc)
            result["cases"].append(case)
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
            # Each independent model registration owns new groups. Release both
            # only after all its forward/backward operations have completed.
            dist.destroy_process_group(mpu.get_sequence_parallel_group())
            dist.destroy_process_group(mpu.get_sequence_data_parallel_group())
            mpu._release_sequence_parallel_groups()
            if rank == 0:
                print(json.dumps(case), flush=True)
    except Exception as exc:
        import traceback
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
        Path(output, f"rank_{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gloo-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(versions={n: importlib.metadata.version(n) for n in
                             ["torch", "transformers", "deepspeed", "accelerate"]},
                    backend="cpu/gloo", world_size=2, sequence_length=16,
                    optional_shm_extension_disabled=args.gloo_only,
                    precision="float32", model_weights="random tiny configs",
                    note="Forward/backward integration probe; not full Trainer or GPU training.")
    import deepspeed.runtime.sequence_parallel.ulysses_sp as usp
    import deepspeed.sequence.layer as layer
    metadata["library_sha256"] = {
        mod.__name__: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() for mod in [usp, layer]
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="mote-sp-") as tmp:
        init_method = (Path(tmp) / "rendezvous").as_uri()
        torch.multiprocessing.spawn(worker, args=(init_method, str(args.output), args.gloo_only),
                                    nprocs=2, join=True)


if __name__ == "__main__":
    main()
