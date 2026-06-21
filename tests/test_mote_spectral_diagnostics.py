from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg

from MOTE.cli.mote_spectral_diagnose import _layer_summary, parse_args
from MOTE.diagnostics.mote_trace import gini, move_to_module_device_dtype, trace_motn_projection, trace_qwen_mlp_projection
from MOTE.diagnostics.prompt_sampling import load_diagnostic_prompts, resolve_prompt_template
from MOTE.diagnostics.spectral import compare_matrices, spectral_stats
from MOTE.model import MOTNFFNLayer


class FakeQwenMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def motn_cfg():
    return {
        "gate_type": "topk",
        "E": 4,
        "topk": 2,
        "temperature": 1.0,
        "use_ste": True,
        "jitter_eps": 0.0,
        "capacity_factor": 10.0,
        "min_capacity": 1,
        "drop_tokens": False,
        "drop_policy": "probs",
        "aux_coeff": 0.0,
        "zloss_coeff": 0.0,
        "gate_arch": "linear",
        "gate_hidden_dim": 0,
        "gate_hidden_mult": 0.0625,
        "gate_hidden_min": 4,
        "gate_hidden_max": 16,
        "gate_activation": "silu",
        "gate_norm": "none",
        "gate_dropout": 0.0,
        "gate_mlp_bias": True,
        "gate_output_init_std": 1e-3,
        "gate_residual_delta_scale": 1.0,
        "d": 2,
        "k_in": 2,
        "pos_strategy": "sliding",
        "init_gamma": 0.6,
        "seed": 0,
        "warmup_ratio": 0.5,
        "warmup_stride": 1,
        "global_expert_enabled": False,
    }


class MoteSpectralDiagnosticsTests(unittest.TestCase):
    def test_gini_accepts_cpu_tensor(self):
        value = gini(torch.ones(4, device="cpu"))
        self.assertIsInstance(value, float)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gini_accepts_cuda_tensor(self):
        value = gini(torch.ones(4, device="cuda"))
        self.assertIsInstance(value, float)

    def test_spectral_stats_handles_regular_and_low_rank_matrices(self):
        x = torch.randn(6, 4)
        stats = spectral_stats(x, top_k_svd=3)
        self.assertEqual(stats["shape"], [6, 4])
        self.assertIn("effective_rank", stats)
        low = torch.ones(3, 8)
        low_stats = spectral_stats(low, top_k_svd=8)
        self.assertEqual(low_stats["shape"], [3, 8])
        self.assertIn("stable_rank", low_stats)
        cmp = compare_matrices(x, x + 0.01, top_k_svd=3)
        self.assertIn("cosine_mean", cmp)

    def test_trace_motn_projection_returns_router_and_blocks(self):
        dense = FakeQwenMLP()
        mote = MOTNFFNLayer(dense, motn_cfg(), device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(5, 8)
        result = trace_motn_projection(mote.gate_proj, x, max_tokens=5, top_k_svd=4)
        self.assertTrue(result["router"]["available"])
        self.assertEqual(len(result["router"]["expert_usage_counts"]), mote.gate_proj.core.num_blocks)
        self.assertIn("entropy_mean", result["router"])
        self.assertNotIn("top1_ids", result["router"])
        self.assertNotIn("topk_ids_shape", result["router"])
        self.assertIn("per_block", result["blocks"])

    def test_trace_motn_projection_limits_block_stats(self):
        dense = FakeQwenMLP()
        mote = MOTNFFNLayer(dense, motn_cfg(), device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(5, 8)
        result = trace_motn_projection(
            mote.gate_proj,
            x,
            max_tokens=5,
            max_block_tokens=3,
            max_blocks_trace=1,
            top_k_svd=4,
        )
        summary = result["blocks"]["diversity_summary"]
        self.assertEqual(summary["tokens_used"], 3)
        self.assertEqual(summary["blocks_traced"], 1)
        self.assertEqual(len(result["blocks"]["per_block"]), 1)
        self.assertEqual(summary["skipped_blocks"], list(range(1, mote.gate_proj.core.num_blocks)))

    def test_trace_qwen_mlp_projection_returns_projection_records(self):
        dense = FakeQwenMLP()
        mote = MOTNFFNLayer(FakeQwenMLP(), motn_cfg(), device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(2, 3, 8)
        result = trace_qwen_mlp_projection(dense, mote, x, max_tokens=6, top_k_svd=4)
        for key in ("gate_proj", "up_proj", "down_proj", "mlp_output"):
            self.assertIn(key, result)
            self.assertIn("compare", result[key])
        self.assertIn("down_proj_shared_mid", result)
        self.assertIn("compare", result["down_proj_shared_mid"])
        self.assertIn("router", result["gate_proj"])

    def test_trace_qwen_mlp_projection_moves_input_dtype_to_modules(self):
        dense = FakeQwenMLP().to(dtype=torch.float32)
        mote = MOTNFFNLayer(FakeQwenMLP(), motn_cfg(), device=torch.device("cpu"), dtype=torch.float32)
        x = torch.randn(2, 3, 8, dtype=torch.float64)
        moved = move_to_module_device_dtype(x, dense)
        self.assertEqual(moved.dtype, next(dense.parameters()).dtype)
        result = trace_qwen_mlp_projection(dense, mote, x, max_tokens=6, top_k_svd=4)
        self.assertIn("mlp_output", result)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_trace_qwen_mlp_projection_accepts_cpu_input_for_cuda_modules(self):
        dense = FakeQwenMLP().to(device="cuda", dtype=torch.float32)
        mote = MOTNFFNLayer(FakeQwenMLP(), motn_cfg(), device=torch.device("cuda"), dtype=torch.float32)
        x = torch.randn(2, 3, 8, device="cpu", dtype=torch.float32)
        result = trace_qwen_mlp_projection(dense, mote, x, max_tokens=6, top_k_svd=4)
        self.assertIn("mlp_output", result)

    def test_layer_summary_extracts_shared_projection_and_router_metrics(self):
        layer_result = {
            "shared": {
                "mlp_output": {"compare": {"cosine_mean": 0.9, "effective_rank_ratio": 1.1}},
                "gate_proj": {
                    "compare": {"effective_rank_ratio": 0.8},
                    "router": {"available": True, "entropy_mean": 0.5, "usage_gini": 0.2, "active_expert_count": 3},
                },
                "up_proj": {"compare": {"effective_rank_ratio": 0.7}},
                "down_proj": {"compare": {"effective_rank_ratio": 0.6}},
            }
        }
        summary = _layer_summary(layer_result)
        self.assertEqual(summary["mlp_output"]["cosine_mean"], 0.9)
        self.assertEqual(summary["mlp_output"]["effective_rank_ratio"], 1.1)
        self.assertEqual(summary["gate_proj"]["effective_rank_ratio"], 0.8)
        self.assertEqual(summary["gate_proj"]["router_entropy"], 0.5)
        self.assertEqual(summary["gate_proj"]["usage_gini"], 0.2)
        self.assertEqual(summary["gate_proj"]["active_expert_count"], 3)

    def test_cli_parse_args(self):
        args = parse_args(
            [
                "--base_model_path",
                "/tmp/base",
                "--mote_ckpt_dir",
                "/tmp/mote",
                "--prompt_text",
                "hello",
                "--tasks",
                "gsm8k",
                "--layers",
                "1,2",
                "--out_dir",
                "/tmp/out",
                "--max_block_tokens",
                "7",
                "--max_blocks_trace",
                "2",
            ]
        )
        self.assertEqual(args.layers, "1,2")
        self.assertEqual(args.tasks, ["gsm8k"])
        self.assertEqual(args.mode, "both")
        self.assertEqual(args.max_block_tokens, 7)
        self.assertEqual(args.max_blocks_trace, "2")

    def test_prompt_sampling_from_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"question": "What is 2+2?"}',
                        '{"prompt": "already formatted"}',
                        '{"text": "plain text"}',
                        '{"messages": [{"role": "user", "content": "hi"}]}',
                    ]
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--base_model_path",
                    "/tmp/base",
                    "--mote_ckpt_dir",
                    "/tmp/mote",
                    "--prompts_file",
                    str(path),
                    "--num_samples",
                    "4",
                    "--sample_seed",
                    "123",
                    "--prompt_template",
                    "gsm8k_direct",
                    "--layers",
                    "0",
                    "--out_dir",
                    "/tmp/out",
                ]
            )
            prompts = load_diagnostic_prompts(args, fit_cfg=None)
            self.assertEqual(len(prompts), 4)
            self.assertIn("Question:\nWhat is 2+2?\nAnswer:\n", prompts)
            self.assertIn("already formatted", prompts)
            self.assertIn("plain text", prompts)
            self.assertIn("user: hi", prompts)

    def test_prompt_template_modes(self):
        base_args = parse_args(
            [
                "--base_model_path",
                "/tmp/base",
                "--mote_ckpt_dir",
                "/tmp/mote",
                "--prompt_text",
                "x",
                "--prompt_template",
                "raw",
                "--layers",
                "0",
                "--out_dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(resolve_prompt_template(base_args), "{question}")

        direct_args = parse_args(
            [
                "--base_model_path",
                "/tmp/base",
                "--mote_ckpt_dir",
                "/tmp/mote",
                "--prompt_text",
                "x",
                "--prompt_template",
                "gsm8k_direct",
                "--layers",
                "0",
                "--out_dir",
                "/tmp/out",
            ]
        )
        self.assertIn("Answer:", resolve_prompt_template(direct_args))


if __name__ == "__main__":
    unittest.main()
