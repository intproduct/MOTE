"""Real checkpoint acceptance: restore, two training steps, HF export/reload."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fitmotn.eval.restore import restore_fitmotn_model
from fitmotn.export.hf_export import export_fitmotn_hf_roundtrip
from fitmotn.patching import set_trainable_patch_only
from fitmotn.chat_formatting import make_standard_chat_supervised_example, supervision_chat_template
from fitmotn.data.adapters import normalize_chat_messages

torch.set_num_threads(8)
torch.manual_seed(20261125)
source, imported, exported = map(Path, sys.argv[1:4])
result_path = Path(__file__).with_name("acceptance.json")
report = {}
start = time.monotonic()

def save():
    report["elapsed_seconds"] = time.monotonic() - start
    result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

try:
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    inputs = tok("Explain why the sum of two even integers is even.", return_tensors="pt")
    base = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation="eager").eval()
    with torch.no_grad():
        expected = base(**inputs, use_cache=False).logits
    del base
    gc.collect()
    model, tok, meta = restore_fitmotn_model(imported, device="cpu", torch_dtype="bfloat16")
    with torch.no_grad():
        restored = model(**inputs, use_cache=False).logits
    report["raw_restore_max_abs"] = float((restored - expected).abs().max())
    assert torch.equal(restored, expected)
    report["eos_ids"] = model.generation_config.eos_token_id
    assert tok.eos_token_id in report["eos_ids"]
    print("raw checkpoint restore passed", flush=True)
    save()

    messages = [{"role": "user", "content": "Calculate 7 * 8 with Python."},
                {"role": "assistant", "content": None, "reasoning_content": "I will calculate it.",
                 "tool_calls": [{"function": {"name": "python", "arguments": {"code": "print(7 * 8)"}}}]},
                {"role": "tool", "content": "TOOL_ONLY_56"},
                {"role": "assistant", "content": "56"}]
    normalized = normalize_chat_messages({"messages": messages})
    original_text = tok.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
    marked_text = tok.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False,
                                         chat_template=supervision_chat_template(tok.chat_template))
    assert original_text == marked_text
    encoded = make_standard_chat_supervised_example(tok, normalized, 1024, enable_thinking=True)
    supervised = tok.decode(encoded["labels"][encoded["labels"] != -100])
    assert "TOOL_ONLY_56" not in supervised and "print(7 * 8)" in supervised and "56" in supervised
    assert encoded["input_ids"].tolist() == tok.encode(original_text, add_special_tokens=False)
    report["tir"] = {"rendered_bytes_identical": True, "supervised_tokens": int((encoded["labels"] != -100).sum()),
                     "tool_response_masked": True, "reasoning_preserved": "I will calculate it." in supervised,
                     "no_extra_eos": True}
    save()

    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_trainable_patch_only(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    training_ids = tok("1 + 1 = 2", return_tensors="pt")["input_ids"]
    report["training_losses"] = []
    for step in range(2):
        optimizer.zero_grad()
        loss = model(input_ids=training_ids, labels=training_ids, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        report["training_losses"].append(float(loss.detach()))
        optimizer.step()
        print(f"training step {step + 1} passed", flush=True)
    report["router_nonzero_gradient_tensors_after_step2"] = sum(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters)
    report["trainable_parameters"] = sum(p.numel() for p in parameters)
    assert report["router_nonzero_gradient_tensors_after_step2"] > 39
    del optimizer, parameters, model, loss
    gc.collect()
    save()

    # Export the untouched step-zero import, not the deliberately trained probe.
    exported_result = export_fitmotn_hf_roundtrip(imported, exported, torch_dtype="bfloat16",
                                                base_trust_remote_code=True, roundtrip_device="cpu")
    report["export_missing"] = exported_result.missing_keys
    report["export_unexpected"] = exported_result.unexpected_keys
    save()
    print("HF export saved", flush=True)
    loaded, info = AutoModelForCausalLM.from_pretrained(exported, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True)
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
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        generated = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
        report["generated_token_count"] = generated.shape[-1] - inputs["input_ids"].shape[-1]
    assert report["generated_token_count"] > 0
    report["ok"] = True
except Exception as exc:
    import traceback
    report["error"] = str(exc)
    report["traceback"] = traceback.format_exc()
    report["ok"] = False
    raise
finally:
    save()
