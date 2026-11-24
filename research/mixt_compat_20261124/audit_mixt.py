"""Read-only checkpoint audit and numerical interoperability probe.

This is an assessment, not a production converter. Source tensors are never saved
back. The proof folds operator bias only to test the existing contraction kernel;
production import should preserve its separate trained parametrization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fitmotn.ADTN import GatedADTNLayer
from fitmotn.gate import gate_config_from_mapping
from fitmotn.init.approx import collect_dense_ffn_targets
from fitmotn.model import SparseMiXTFFNLayer
from fitmotn.chat_formatting import make_standard_chat_supervised_example


def relative_error(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


class ImportedMainProof(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.config = dict(E=source.num_gates_to_apply, topk=source.num_gates_to_apply, d=source.d, k_in=source.k_in,
                           gate_type="softmax", pos_strategy="sliding", global_expert_enabled=False)
        self.core = GatedADTNLayer(
            dim_input=source.d ** source.q_number,
            dim_output=source.d ** (source.q_number - source.k_in + source.k_out),
            num_blocks=source.num_gates_to_apply, d=source.d, k_in=source.k_in,
            gate_config=gate_config_from_mapping(self.config, data_dim=source.d ** source.q_number),
            pos_strategy="sliding", dtype=source.bricks.dtype, device=source.bricks.device)
        assert self.core.positions == source.op_indices
        with torch.no_grad():
            for i, block in enumerate(self.core.blocks):
                block.U.copy_((source.bricks[i] + source.bias[i]).reshape_as(block.U))
            self.core.gate.router.linear.weight.zero_()

    def forward(self, x):
        return self.core(x).to(x.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261124)
    ckpt = args.checkpoint
    config = json.loads((ckpt / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("partner_mixt_projector", ckpt / "mixt_projector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (ckpt / "model.safetensors").open("rb") as f:
        header_bytes = f.read(struct.unpack("<Q", f.read(8))[0])
    header = json.loads(header_bytes)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    report = {
        "checkpoint": str(ckpt), "torch": torch.__version__,
        "safetensors_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ckpt / p).read_bytes()).hexdigest() for p in
                          ["config.json", "mixt_projector.py", "modeling_qwen3_5_mixt.py", "chat_template.jinja"]},
        "parameters": sum(math.prod(v["shape"]) for v in tensors.values()),
        "tensor_count": len(tensors), "dtypes": dict(Counter(v["dtype"] for v in tensors.values())),
        "index_keys_match": set(index["weight_map"]) == set(tensors),
        "index_files": sorted(set(index["weight_map"].values())),
        "mixT_parameters": sum(math.prod(v["shape"]) for k, v in tensors.items()
                               if ".mlp." in k and int(k.split(".")[2]) in config["mixt_compress_layers"]),
        "projection_probes": [], "failure_probes": {},
    }
    mlps = {}
    with safe_open(ckpt / "model.safetensors", framework="pt", device="cpu") as sf:
        for layer in config["mixt_compress_layers"]:
            projections = {}
            for kind in ("gate", "up", "down"):
                incoming, outgoing = ((2560, 9216) if kind != "down" else (9216, 2560))
                source = module.HybridMainMixTProjector(
                    incoming, outgoing, rank=112,
                    main_in_features=config["mixt_main_in_features"][kind],
                    main_out_features=config["mixt_main_out_features"][kind], dtype=torch.bfloat16)
                prefix = f"model.layers.{layer}.mlp.{kind}_proj."
                state = {k[len(prefix):]: sf.get_tensor(k) for k in tensors if k.startswith(prefix)}
                source.load_state_dict(state, strict=True)
                source = source.float().eval()
                candidate = copy.deepcopy(source)
                candidate.mixt00 = ImportedMainProof(source.mixt00)
                x = torch.randn(2, 3, incoming)
                with torch.no_grad():
                    expected = source(x)
                    actual = candidate(x)
                    row = {"layer": layer, "projection": kind,
                           "fp32_relative_l2": relative_error(actual, expected),
                           "fp32_max_abs": float((actual - expected).abs().max()),
                           "alphas": [float(getattr(source, k)) for k in ("alpha10", "alpha01", "alpha11")],
                           "operator_bias_max_abs": float(source.mixt00.bias.abs().max())}
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                if layer == 19:
                    # A training signal must reach both newly added routing and inherited blocks.
                    candidate(x).square().mean().backward()
                    row["router_gradient_norm"] = float(candidate.mixt00.core.gate.router.linear.weight.grad.norm())
                    row["expert_gradient_norm"] = float(candidate.mixt00.core.blocks[0].U.grad.norm())
                    assert row["router_gradient_norm"] > 0 and row["expert_gradient_norm"] > 0
                    # Original BF16 operator addition is kept in BF16 before importing.
                    bf_source = copy.deepcopy(source).bfloat16()
                    bf_candidate = copy.deepcopy(bf_source)
                    bf_candidate.mixt00 = ImportedMainProof(bf_source.mixt00)
                    with torch.no_grad():
                        row["bf16_relative_l2"] = relative_error(bf_candidate(x.bfloat16()), bf_source(x.bfloat16()))
                    singular = torch.linalg.svdvals(source.w11.weight.detach())
                    energy = singular.square()
                    row["corner_rank112_best_relative_frobenius"] = float((energy[112:].sum() / energy.sum()).sqrt())
                report["projection_probes"].append(row)
                projections[f"{kind}_proj"] = source
            if layer == 19:
                mlp = nn.Module()
                for name, value in projections.items():
                    setattr(mlp, name, value)
                mlp.act_fn = nn.SiLU()
                mlps[layer] = mlp
            print(f"verified layer {layer}", flush=True)

    fake_model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlps[19])]))
    try:
        collect_dense_ffn_targets(fake_model, [0])
    except Exception as e:
        report["failure_probes"]["training_collect_dense_targets"] = f"{type(e).__name__}: {e}"
    cfg = dict(E=4, topk=4, d=2, k_in=8, gate_type="softmax", pos_strategy="sliding",
               global_expert_enabled=False, sparse_mixt={"boundary_init": "randomized_svd"})
    try:
        SparseMiXTFFNLayer(mlps[19], cfg, device=torch.device("cpu"))
    except Exception as e:
        report["failure_probes"]["sparse_mixt_svd_import"] = f"{type(e).__name__}: {e}"

    from transformers import AutoConfig, AutoTokenizer
    import transformers
    report["transformers"] = transformers.__version__
    hfconfig = AutoConfig.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    report["config_class"] = type(hfconfig).__name__
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    report["tokenizer"] = {"class": type(tok).__name__, "length": len(tok), "pad": tok.pad_token_id,
                           "eos": tok.eos_token_id, "has_generation_tag": "{% generation" in tok.chat_template}
    examples = {
        "single_cot": [{"role": "user", "content": "1+1=?"},
                       {"role": "assistant", "content": "<think>Add one and one.</think>2"}],
        "multi_turn_tir": [{"role": "user", "content": "1+1=?"},
                           {"role": "assistant", "content": "<tool_call>print(1+1)</tool_call>"},
                           {"role": "tool", "content": "2"}, {"role": "assistant", "content": "2"}],
    }
    report["chat_probes"] = {}
    for name, messages in examples.items():
        try:
            encoded = make_standard_chat_supervised_example(tok, messages, max_len=1024, enable_thinking=True)
            report["chat_probes"][name] = {"tokens": encoded["input_ids"].numel(),
                                             "supervised_tokens": int((encoded["labels"] != -100).sum())}
        except Exception as e:
            report["chat_probes"][name] = {"error": f"{type(e).__name__}: {e}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "projection_probes"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
