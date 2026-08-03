from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitmotn.init.teacher import fit_teacher_ffn
from fitmotn.model import MixedMiXTFFNLayer, SparseMiXTFFNLayer


class _ShapeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)


class _ShapeOnlyQwenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _ShapeLinear(2560, 9216)
        self.up_proj = _ShapeLinear(2560, 9216)
        self.down_proj = _ShapeLinear(9216, 2560)
        self.act_fn = nn.SiLU()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _parse_csv(value: str, cast=str) -> List:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _sample_wiki_tokens(tokenizer, wiki_jsonl: Path, *, num_sequences: int, seq_len: int, seed: int) -> torch.Tensor:
    rng = random.Random(int(seed))
    documents = []
    with wiki_jsonl.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if line_no >= 512:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(record.get("text") or "").strip()
            if len(text) < 256:
                continue
            title = str(record.get("title") or "").strip()
            documents.append((title + "\n\n" + text[:12000]).strip())
    if not documents:
        raise RuntimeError(f"no usable Wiki documents found in {wiki_jsonl}")
    rng.shuffle(documents)
    required = int(num_sequences) * int(seq_len)
    token_ids: List[int] = []
    eos = int(tokenizer.eos_token_id)
    for document in documents:
        token_ids.extend(tokenizer.encode(document, add_special_tokens=False))
        token_ids.append(eos)
        if len(token_ids) >= required:
            break
    if len(token_ids) < required:
        raise RuntimeError(f"Wiki sample produced {len(token_ids)} tokens, need {required}")
    return torch.tensor(token_ids[:required], dtype=torch.long).reshape(int(num_sequences), int(seq_len))


def _install_qwen35_torch_fallback(text_model: nn.Module) -> int:
    """Replace Windows-incompatible FLA kernels without changing site-packages."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5RMSNormGated,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    count = 0
    for layer in text_model.layers:
        if str(layer.layer_type) != "linear_attention":
            continue
        attention = layer.linear_attn
        attention.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        attention.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        if not isinstance(attention.norm, Qwen3_5RMSNormGated):
            old_norm = attention.norm
            new_norm = Qwen3_5RMSNormGated(attention.head_v_dim, eps=attention.layer_norm_epsilon).to(
                device=old_norm.weight.device,
                dtype=old_norm.weight.dtype,
            )
            new_norm.load_state_dict(old_norm.state_dict(), strict=True)
            attention.norm = new_norm
        count += 1
    return count


def _capture_teacher_cache(args, cache_path: Path) -> Dict[str, object]:
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    model_path = Path(args.model_path).resolve()
    wiki_jsonl = Path(args.wiki_jsonl).resolve()
    print(f"[capture] loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    input_ids = _sample_wiki_tokens(
        tokenizer,
        wiki_jsonl,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(f"[capture] input_ids={tuple(input_ids.shape)} range=({int(input_ids.min())},{int(input_ids.max())})", flush=True)

    load_started = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    text_model = model.model.language_model
    fallback_layers = _install_qwen35_torch_fallback(text_model)
    print(
        f"[capture] model_loaded_sec={time.perf_counter()-load_started:.3f} "
        f"allocated_gib={torch.cuda.memory_allocated()/2**30:.3f} torch_fallback_layers={fallback_layers}",
        flush=True,
    )

    layers = [int(layer) for layer in args.layers]
    buffers: Dict[int, Dict[str, List[torch.Tensor]]] = {
        layer: {"inputs": [], "targets": []} for layer in layers
    }
    handles = []
    for layer_idx in layers:
        mlp = text_model.layers[layer_idx].mlp

        def pre_hook(_module, positional, layer_idx=layer_idx):
            buffers[layer_idx]["inputs"].append(positional[0].detach().to(device="cpu", dtype=torch.bfloat16))

        def post_hook(_module, _positional, output, layer_idx=layer_idx):
            value = output[0] if isinstance(output, tuple) else output
            buffers[layer_idx]["targets"].append(value.detach().to(device="cpu", dtype=torch.bfloat16))

        handles.append(mlp.register_forward_pre_hook(pre_hook))
        handles.append(mlp.register_forward_hook(post_hook))

    torch.cuda.reset_peak_memory_stats()
    capture_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, input_ids.shape[0], int(args.capture_batch_size)):
                stop = min(input_ids.shape[0], start + int(args.capture_batch_size))
                batch = input_ids[start:stop].to(device="cuda:0")
                mask = torch.ones_like(batch)
                _ = text_model(input_ids=batch, attention_mask=mask, use_cache=False, return_dict=True)
                torch.cuda.synchronize()
                print(f"[capture] sequences={stop}/{input_ids.shape[0]}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    layer_payload = {}
    for layer_idx in layers:
        inputs = torch.cat(buffers[layer_idx]["inputs"], dim=0).reshape(-1, 2560).contiguous()
        targets = torch.cat(buffers[layer_idx]["targets"], dim=0).reshape(-1, 2560).contiguous()
        if inputs.shape != targets.shape:
            raise RuntimeError(f"layer {layer_idx} capture mismatch: {inputs.shape} vs {targets.shape}")
        layer_payload[int(layer_idx)] = {"inputs": inputs, "targets": targets}
        print(
            f"[capture] layer={layer_idx} tokens={inputs.shape[0]} "
            f"input_rms={float(inputs.float().square().mean().sqrt()):.6g} "
            f"target_rms={float(targets.float().square().mean().sqrt()):.6g}",
            flush=True,
        )

    metadata = {
        "format_version": 1,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "model_index_sha256": _sha256(model_path / "model.safetensors.index.json"),
        "wiki_jsonl": str(wiki_jsonl),
        "layers": layers,
        "num_sequences": int(args.num_sequences),
        "seq_len": int(args.seq_len),
        "tokens": int(input_ids.numel()),
        "seed": int(args.seed),
        "capture_batch_size": int(args.capture_batch_size),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "torch_fallback_layers": int(fallback_layers),
        "capture_elapsed_sec": time.perf_counter() - capture_started,
        "capture_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "input_ids_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
    }
    payload: Dict[str, object] = {"metadata": metadata, "layers": layer_payload}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"[capture] saved {cache_path} size_mib={cache_path.stat().st_size/1024**2:.1f}", flush=True)

    del text_model, model, tokenizer, buffers, layer_payload
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _load_or_capture(args, cache_path: Path) -> Dict[str, object]:
    if cache_path.exists() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        expected_layers = [int(layer) for layer in args.layers]
        checks = {
            "model_path": str(Path(args.model_path).resolve()),
            "wiki_jsonl": str(Path(args.wiki_jsonl).resolve()),
            "layers": expected_layers,
            "num_sequences": int(args.num_sequences),
            "seq_len": int(args.seq_len),
            "seed": int(args.seed),
        }
        conflicts = {key: (metadata.get(key), value) for key, value in checks.items() if metadata.get(key) != value}
        if conflicts:
            raise ValueError(f"activation cache conflicts with requested experiment: {conflicts}; use --rebuild-cache")
        print(f"[capture] reused {cache_path}", flush=True)
        return payload
    return _capture_teacher_cache(args, cache_path)


def _base_cfg(args, *, seed: int) -> Dict[str, object]:
    return {
        "E": int(args.experts),
        "d": 2,
        "k_in": 8,
        "topk": int(args.topk),
        "gate_type": "topk",
        "temperature": 1.0,
        "use_ste": True,
        "jitter_eps": 0.0,
        "capacity_factor": 100.0,
        "min_capacity": 1,
        "drop_tokens": False,
        "pos_strategy": "random",
        "seed": int(seed),
        "global_expert_enabled": bool(args.global_expert),
        "global_expert_weight": 1.0,
        "dtype": torch.bfloat16,
        "block_init_mode": "gamma_normal",
        "init_gamma": 0.6,
    }


def _build_backend(backend: str, args, *, layer_idx: int, device: torch.device) -> nn.Module:
    cfg = _base_cfg(args, seed=int(args.seed) + int(layer_idx))
    source = _ShapeOnlyQwenMLP()
    if backend == "sparse_mixt":
        cfg.update(
            patch_backend="sparse_mixt",
            sparse_mixt={
                "backend_version": 1,
                "hidden_main": 2048,
                "intermediate_main": 8192,
                "router_input_policy": "full_real",
                "boundary_init": "zero",
                "gate_ranks": {"01": 128, "10": 64, "11": 64},
                "up_ranks": {"01": 128, "10": 64, "11": 64},
                "down_ranks": {"01": 64, "10": 128, "11": 64},
            },
        )
        return SparseMiXTFFNLayer(source, cfg, device=device, dtype=torch.bfloat16, layer_idx=layer_idx).to(device)
    if backend == "mixed_mixt":
        cfg.update(
            patch_backend="mixed_mixt",
            mixed_mixt={
                "backend_version": 1,
                "hidden_main": 2048,
                "intermediate_main": 8192,
                "router_input_policy": "full_real",
                "dense_init": "none",
                "gate_bonds": {"m00": 8, "m01": 5, "m10": 6, "m11": 5},
                "up_bonds": {"m00": 8, "m01": 5, "m10": 6, "m11": 5},
                "down_bonds": {"m00": 10, "m01": 6, "m10": 8, "m11": 6},
            },
        )
        return MixedMiXTFFNLayer(source, cfg, device=device, dtype=torch.bfloat16, layer_idx=layer_idx).to(device)
    raise ValueError(f"unsupported backend={backend!r}")


def _run_one(args, payload: Dict[str, object], *, backend: str, layer_idx: int, output_dir: Path):
    result_path = output_dir / "runs" / f"{backend}_layer{layer_idx:02d}.json"
    if result_path.exists() and not args.force:
        print(f"[run] reuse completed {result_path}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    layer_data = payload["layers"][int(layer_idx)]
    inputs = layer_data["inputs"]
    targets = layer_data["targets"]
    tokens_per_sequence = int(payload["metadata"]["seq_len"])
    train_tokens = int(args.train_sequences) * tokens_per_sequence
    if not (0 < train_tokens < inputs.shape[0]):
        raise ValueError(f"train split must leave validation tokens, got {train_tokens}/{inputs.shape[0]}")
    train_inputs, val_inputs = inputs[:train_tokens], inputs[train_tokens:]
    train_targets, val_targets = targets[:train_tokens], targets[train_tokens:]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(args.seed) + int(layer_idx))
    torch.cuda.manual_seed_all(int(args.seed) + int(layer_idx))
    device = torch.device("cuda:0")
    module = _build_backend(backend, args, layer_idx=layer_idx, device=device)
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    print(
        f"[run] backend={backend} layer={layer_idx} params={parameter_count:,} "
        f"train_tokens={train_inputs.shape[0]} val_tokens={val_inputs.shape[0]}",
        flush=True,
    )

    trace_lines = [f"backend={backend} layer={layer_idx} real_teacher_input"]
    print(f"[trace] {trace_lines[0]}", flush=True)

    def record_trace(line):
        trace_lines.append(str(line))
        print(f"[trace] {line}", flush=True)

    with torch.inference_mode():
        trace_input = train_inputs[:1].to(device=device, non_blocking=True)
        _ = module.forward_with_trace(trace_input, trace=record_trace)
    del trace_input

    def progress(record):
        val = record["val"]
        print(
            f"[fit] backend={backend} layer={layer_idx} step={record['step']} "
            f"loss={record['train_loss']:.8e} rel_l2={val['rel_l2']:.6f} "
            f"cos={val['cosine']:.6f} grad={record['grad_norm']:.6f}",
            flush=True,
        )

    result, best_state = fit_teacher_ffn(
        module,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        device=device,
        steps=int(args.steps),
        batch_tokens=int(args.batch_tokens),
        eval_batch_tokens=int(args.eval_batch_tokens),
        eval_every=int(args.eval_every),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        seed=int(args.seed) + int(layer_idx),
        progress=progress,
    )
    result.update(
        {
            "backend": backend,
            "layer": int(layer_idx),
            "parameter_count": int(parameter_count),
            "train_tokens": int(train_inputs.shape[0]),
            "val_tokens": int(val_inputs.shape[0]),
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "activation_metadata": payload["metadata"],
            "experts": int(args.experts),
            "topk": int(args.topk),
            "global_expert": bool(args.global_expert),
            "computation_trace": trace_lines,
        }
    )
    _json_write(result_path, result)
    if args.save_best_states:
        state_path = output_dir / "states" / f"{backend}_layer{layer_idx:02d}.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, state_path)
    print(
        f"[done] backend={backend} layer={layer_idx} best_step={result['best_step']} "
        f"initial_rel={result['initial']['rel_l2']:.6f} best_rel={result['best']['rel_l2']:.6f} "
        f"best_cos={result['best']['cosine']:.6f} sec={result['elapsed_sec']:.1f}",
        flush=True,
    )
    del module, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _write_summary(output_dir: Path, payload: Dict[str, object], results: Iterable[Dict[str, object]]) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "backend": result["backend"],
                "layer": result["layer"],
                "parameter_count": result["parameter_count"],
                "initial_rel_l2": result["initial"]["rel_l2"],
                "best_rel_l2": result["best"]["rel_l2"],
                "initial_cosine": result["initial"]["cosine"],
                "best_cosine": result["best"]["cosine"],
                "best_step": result["best_step"],
                "elapsed_sec": result["elapsed_sec"],
                "peak_memory_mib": result["peak_memory_mib"],
            }
        )
    summary = {
        "format_version": 1,
        "experiment": "qwen35_last5_real_activation_teacher_init",
        "activation_metadata": payload["metadata"],
        "rows": sorted(rows, key=lambda row: (int(row["layer"]), str(row["backend"]))),
    }
    _json_write(output_dir / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description="Compare Sparse and Mixed MiXT teacher initialization on Qwen3.5-4B FFN activations")
    parser.add_argument("--model-path", default=r"D:\AI\Models\Qwen3.5-4B")
    parser.add_argument("--wiki-jsonl", default=r"D:\AI\Datas\WiKI\wikien_2023.jsonl")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "qwen35_teacher_init_last5"))
    parser.add_argument("--layers", default="27,28,29,30,31")
    parser.add_argument("--backends", default="sparse_mixt,mixed_mixt")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--train-sequences", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--capture-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--eval-batch-tokens", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--global-expert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-best-states", action="store_true")
    args = parser.parse_args()
    args.layers = _parse_csv(args.layers, int)
    args.backends = _parse_csv(args.backends, str)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(output_dir / "command_config.json", vars(args))
    cache_path = output_dir / "teacher_activations.pt"
    payload = _load_or_capture(args, cache_path)
    results = []
    for layer_idx in args.layers:
        for backend in args.backends:
            results.append(_run_one(args, payload, backend=backend, layer_idx=layer_idx, output_dir=output_dir))
            _write_summary(output_dir, payload, results)
    _write_summary(output_dir, payload, results)
    print(f"[result] completed {len(results)} runs; summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
