from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch as tc

from ..audit import to_jsonable
from ..diagnostics.hooks import capture_mlp_io, get_mlp_module
from ..diagnostics.jacobian import estimate_jvp_stats
from ..diagnostics.mote_trace import PROJ_NAMES, move_to_module_device_dtype, trace_qwen_mlp_projection
from ..diagnostics.prompt_sampling import diagnostic_prompt_metadata, load_diagnostic_prompts
from ..diagnostics.spectral import compare_matrices
from ..model import MOTNFFNLayer, unwrap_y


def parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Offline projection-level MOTE spectral diagnostics")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--mote_ckpt_dir", type=str, required=True)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--prompt_source", type=str, default=None, choices=["task", "json", "jsonl", "text"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sample_split", type=str, default=None)
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--prompt_template", type=str, default=None, choices=["config", "raw", "gsm8k_direct", "gsm8k_cot"])
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_block_tokens", type=int, default=1024)
    parser.add_argument("--max_blocks_trace", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"])
    parser.add_argument("--mode", type=str, default="both", choices=["natural", "shared", "both"])
    parser.add_argument("--top_k_svd", type=int, default=128)
    parser.add_argument("--jvp_samples", type=int, default=0)
    parser.add_argument("--jacobian_tokens", type=int, default=8)
    parser.add_argument("--save_raw_tensors", action="store_true")
    parser.add_argument("--save_router_tensors", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args(argv)


def _parse_layers(text: str) -> List[int]:
    layers = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    if not layers:
        raise ValueError("--layers must contain at least one layer index")
    return layers


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def _tokenize(tokenizer, prompts: List[str], device: tc.device, max_tokens: int) -> Dict[str, tc.Tensor]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
    return {k: v.to(device) for k, v in enc.items()}


def _forward_with_hooks(model, inputs, layers, *, cpu: bool = False):
    with capture_mlp_io(model, layers, cpu=cpu) as (captured, warnings):
        with tc.no_grad():
            model(**inputs)
    return captured, warnings


def _layer_summary(layer_result: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in list(PROJ_NAMES) + ["down_proj_shared_mid", "mlp_output"]:
        rec = layer_result.get("shared", {}).get(key) or layer_result.get(key)
        if not isinstance(rec, dict):
            continue
        item: Dict[str, Any] = {}
        if "compare" in rec:
            cmp = rec["compare"]
            item["cosine_mean"] = cmp.get("cosine_mean")
            item["pointwise_l2_relative"] = cmp.get("pointwise_l2_relative")
            item["cov_fro_diff"] = cmp.get("cov_fro_diff")
            item["effective_rank_ratio"] = cmp.get("effective_rank_ratio")
        router = rec.get("router")
        if isinstance(router, dict) and router.get("available", True):
            item["router_entropy"] = router.get("entropy_mean")
            item["usage_gini"] = router.get("usage_gini")
            item["active_expert_count"] = router.get("active_expert_count")
        blocks = rec.get("blocks")
        if isinstance(blocks, dict):
            item["block_output_diversity_summary"] = blocks.get("diversity_summary")
        glob = rec.get("global")
        if isinstance(glob, dict):
            item["global_output_routed_output_norm_ratio"] = glob.get("global_output_routed_output_norm_ratio")
        summary[key] = item
    return summary


def _run_jacobian(base_mlp, mote_mlp, x, args) -> Dict[str, Any]:
    if int(args.jvp_samples) <= 0:
        return {}
    n = min(int(args.jacobian_tokens), x.reshape(-1, x.shape[-1]).shape[0])
    xs = x.reshape(-1, x.shape[-1])[:n].detach()
    out: Dict[str, Any] = {}
    for name in PROJ_NAMES:
        base_proj = getattr(base_mlp, name, None)
        mote_proj = getattr(mote_mlp, name, None)
        if base_proj is None or mote_proj is None:
            continue
        if name == "down_proj":
            with tc.no_grad():
                bg = base_mlp.gate_proj(xs.to(next(base_mlp.parameters()).device, dtype=next(base_mlp.parameters()).dtype))
                bu = base_mlp.up_proj(xs.to(next(base_mlp.parameters()).device, dtype=next(base_mlp.parameters()).dtype))
                bmid = base_mlp.act_fn(bg) * bu
                mg = unwrap_y(mote_mlp.gate_proj(xs.to(next(mote_mlp.parameters()).device, dtype=next(mote_mlp.parameters()).dtype)))
                mu = unwrap_y(mote_mlp.up_proj(xs.to(next(mote_mlp.parameters()).device, dtype=next(mote_mlp.parameters()).dtype)))
                mmid = mote_mlp.act(mg) * mu
            out[name] = {
                "dense": estimate_jvp_stats(base_proj, bmid, int(args.jvp_samples)),
                "mote": estimate_jvp_stats(lambda z, p=mote_proj: unwrap_y(p(z)), mmid, int(args.jvp_samples)),
            }
        else:
            out[name] = {
                "dense": estimate_jvp_stats(base_proj, xs.to(next(base_mlp.parameters()).device, dtype=next(base_mlp.parameters()).dtype), int(args.jvp_samples)),
                "mote": estimate_jvp_stats(lambda z, p=mote_proj: unwrap_y(p(z)), xs.to(next(mote_mlp.parameters()).device, dtype=next(mote_mlp.parameters()).dtype), int(args.jvp_samples)),
            }
    return out


def main(argv: List[str] | None = None):
    args = parse_args(argv)
    from ..eval.restore import restore_fitmotn_model
    from ..runtime import load_causal_lm_and_tokenizer
    from ..config import load_config

    out_dir = Path(args.out_dir).expanduser().resolve()
    per_layer_dir = out_dir / "per_layer"
    spectra_dir = out_dir / "spectra"
    routing_dir = out_dir / "routing"
    raw_dir = out_dir / "raw_tensors"
    for path in (per_layer_dir, spectra_dir, routing_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_raw_tensors:
        raw_dir.mkdir(parents=True, exist_ok=True)

    layers = _parse_layers(args.layers)
    fit_cfg = load_config(args.config_json) if args.config_json else None
    prompts = load_diagnostic_prompts(args, fit_cfg=fit_cfg)
    prompt_meta = diagnostic_prompt_metadata(args, fit_cfg=fit_cfg)
    device = tc.device(args.device)
    warnings: List[str] = []

    base_model, tokenizer, base_dtype = load_causal_lm_and_tokenizer(
        args.base_model_path,
        device=device,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        use_cache=True,
    )
    mote_model, mote_tokenizer, metadata = restore_fitmotn_model(args.mote_ckpt_dir, device=args.device)
    base_model.eval()
    mote_model.eval()

    meta_base = str(metadata.get("base_model_path", ""))
    if meta_base and Path(meta_base).expanduser().resolve() != Path(args.base_model_path).expanduser().resolve():
        warnings.append(f"checkpoint base_model_path differs from --base_model_path: {meta_base}")

    inputs = _tokenize(tokenizer, prompts, device, int(args.max_tokens))
    base_nat, hook_warnings = _forward_with_hooks(base_model, inputs, layers, cpu=True)
    warnings.extend([f"base hook: {w}" for w in hook_warnings])

    mote_nat = {}
    if args.mode in {"natural", "both"}:
        mote_inputs = _tokenize(mote_tokenizer, prompts, device, int(args.max_tokens))
        mote_nat, hook_warnings = _forward_with_hooks(mote_model, mote_inputs, layers, cpu=True)
        warnings.extend([f"mote hook: {w}" for w in hook_warnings])

    per_layer_summary: Dict[str, Any] = {}
    failed_layers: List[int] = []
    num_tokens_used = 0
    for layer_idx in layers:
        layer_result: Dict[str, Any] = {"layer": int(layer_idx), "warnings": []}
        if args.mode in {"natural", "both"}:
            b = base_nat.get(layer_idx, {})
            m = mote_nat.get(layer_idx, {})
            natural: Dict[str, Any] = {}
            if "mlp_input" in b and "mlp_input" in m:
                natural["mlp_input_drift"] = compare_matrices(b["mlp_input"], m["mlp_input"], top_k_svd=int(args.top_k_svd), max_tokens=int(args.max_tokens))
            if "mlp_output" in b and "mlp_output" in m:
                natural["mlp_output_drift"] = compare_matrices(b["mlp_output"], m["mlp_output"], top_k_svd=int(args.top_k_svd), max_tokens=int(args.max_tokens))
            layer_result["natural"] = natural

        if args.mode in {"shared", "both"}:
            x = base_nat.get(layer_idx, {}).get("mlp_input")
            if x is None:
                layer_result["warnings"].append("missing base natural mlp_input; shared-input tracing skipped")
            else:
                num_tokens_used = max(num_tokens_used, min(int(args.max_tokens), int(x.reshape(-1, x.shape[-1]).shape[0])))
                try:
                    base_mlp = get_mlp_module(base_model, layer_idx)
                    mote_mlp = get_mlp_module(mote_model, layer_idx)
                    if not isinstance(mote_mlp, MOTNFFNLayer):
                        layer_result["warnings"].append(f"mote layer {layer_idx} mlp is {type(mote_mlp).__name__}, MOTE router/block tracing may be unavailable")
                    x_base_layer = move_to_module_device_dtype(x, base_mlp)
                    shared = trace_qwen_mlp_projection(
                        base_mlp,
                        mote_mlp,
                        x_base_layer,
                        max_tokens=int(args.max_tokens),
                        max_block_tokens=int(args.max_block_tokens),
                        max_blocks_trace=args.max_blocks_trace,
                        top_k_svd=int(args.top_k_svd),
                        save_raw_tensors=bool(args.save_raw_tensors),
                        save_router_tensors=bool(args.save_router_tensors),
                        out_paths={
                            "spectra_dir": spectra_dir,
                            "routing_dir": routing_dir,
                            "raw_dir": raw_dir,
                            "prefix": f"layer_{layer_idx}",
                        },
                    )
                    jac = _run_jacobian(base_mlp, mote_mlp, x, args)
                    if jac:
                        shared["jacobian"] = jac
                    layer_result["shared"] = shared
                except Exception as exc:
                    failed_layers.append(int(layer_idx))
                    layer_result["warnings"].append(f"shared tracing failed: {type(exc).__name__}: {exc}")
                    layer_result["shared_traceback"] = traceback.format_exc()

        per_layer_summary[str(layer_idx)] = _layer_summary(layer_result)
        _json_dump(per_layer_dir / f"layer_{layer_idx}.json", layer_result)
        if tc.cuda.is_available():
            tc.cuda.empty_cache()

    summary = {
        "base_model_path": str(Path(args.base_model_path).expanduser().resolve()),
        "mote_ckpt_dir": str(Path(args.mote_ckpt_dir).expanduser().resolve()),
        "layers": layers,
        "num_prompts": len(prompts),
        "num_tokens_used": num_tokens_used,
        "model_dtype": str(base_dtype),
        "mode": args.mode,
        **prompt_meta,
        "per_layer_summary": per_layer_summary,
        "failed_layers": failed_layers,
        "warnings": warnings,
    }
    _json_dump(out_dir / "diagnostics_summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
