"""Standalone full-size HF export/reload/generation acceptance."""
import faulthandler
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fitmotn.export.hf_export import export_fitmotn_hf_roundtrip

torch.set_num_threads(8)
faulthandler.dump_traceback_later(120, repeat=True)
source, imported, exported = map(Path, sys.argv[1:4])
report = {}
start = time.monotonic()
try:
    result = export_fitmotn_hf_roundtrip(imported, exported, torch_dtype="bfloat16",
                                        base_trust_remote_code=True, roundtrip_device="cpu")
    report["export_missing"] = result.missing_keys
    report["export_unexpected"] = result.unexpected_keys
    print("export completed", flush=True)
    gc.collect()
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    inputs = tok("Explain why the sum of two even integers is even.", return_tensors="pt")
    base = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation="eager").eval()
    with torch.no_grad():
        expected = base(**inputs, use_cache=False).logits
        expected_generation = base.generate(**inputs, max_new_tokens=3, do_sample=False)
    del base
    gc.collect()
    loaded, info = AutoModelForCausalLM.from_pretrained(exported, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, output_loading_info=True)
    loaded.eval()
    assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    report["hf_reload_keys_exact"] = True
    report["router_dtype"] = str(loaded.model.layers[19].mlp.gate_proj.core.gate.router.linear.weight.dtype)
    assert report["router_dtype"] == "torch.float32"
    with torch.no_grad():
        actual = loaded(**inputs, use_cache=False).logits
        delta = actual.float() - expected.float()
        report["hf_reload_logits_max_abs"] = float(delta.abs().max())
        report["hf_reload_logits_relative_l2"] = float(delta.norm() / expected.float().norm())
        assert torch.equal(actual, expected)
        generated = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
        report["generated_token_count"] = generated.shape[-1] - inputs["input_ids"].shape[-1]
        report["greedy_tokens_identical_to_partner"] = torch.equal(generated, expected_generation)
        report["first_generated_token_matches_full_prompt"] = bool(generated[0, inputs["input_ids"].shape[-1]] == actual[0, -1].argmax())
    assert report["greedy_tokens_identical_to_partner"]
    assert report["first_generated_token_matches_full_prompt"]
    report["ok"] = True
except Exception as exc:
    import traceback
    report.update(ok=False, error=str(exc), traceback=traceback.format_exc())
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    report["elapsed_seconds"] = time.monotonic() - start
    Path(__file__).with_name("export_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
