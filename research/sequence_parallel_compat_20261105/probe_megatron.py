"""Record a real Megatron Core import / CP configuration probe without GPU work."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import traceback


def probe():
    import torch
    result = dict(torch=torch.__version__, nccl=torch.distributed.is_nccl_available(),
                  gpu_training_attempted=False)
    for package in ("megatron-core", "transformer-engine", "triton"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.dot_product_attention import DotProductAttention
        from megatron.core.transformer.enums import AttnMaskType
        config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4,
                                   context_parallel_size=2, tensor_model_parallel_size=1)
        result["config_constructed"] = True
        try:
            DotProductAttention(config, layer_number=1, attn_mask_type=AttnMaskType.causal,
                                attention_type="self")
        except AssertionError as exc:
            result["non_te_attention_cp_guard"] = str(exc)
        result["status"] = "requires_transformer_engine_and_gpu_validation"
    except Exception as exc:
        result.update(status="blocked_at_import", error=repr(exc), traceback=traceback.format_exc())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2))
