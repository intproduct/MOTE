"""Load the unmodified partner checkpoint and test short causal-LM forwards."""
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
checkpoint, output = Path(sys.argv[1]), Path(sys.argv[2])
result = {"checkpoint": str(checkpoint), "transformers": transformers.__version__,
          "torch": torch.__version__, "device": "cpu", "dtype": "bfloat16"}
start = time.monotonic()
try:
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, output_loading_info=True, attn_implementation="eager")
    result["load_seconds"] = time.monotonic() - start
    result["load_info"] = info
    result["class"] = type(model).__name__
    result["parameters"] = sum(p.numel() for p in model.parameters())
    result["tied_embeddings"] = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    model.eval()
    inputs = tok("1+1=", return_tensors="pt")
    with torch.no_grad():
        full = model(**inputs, use_cache=False)
        cached = model(**inputs, use_cache=True)
        result["logits_shape"] = list(full.logits.shape)
        result["finite_logits"] = bool(torch.isfinite(full.logits).all())
        result["prefill_cache_max_abs"] = float((full.logits - cached.logits).abs().max())
        token = full.logits[:, -1].argmax(-1, keepdim=True)
        incremental = model(input_ids=token, past_key_values=cached.past_key_values, use_cache=True)
        joined = torch.cat([inputs["input_ids"], token], -1)
        reference = model(input_ids=joined, use_cache=False).logits[:, -1:]
        result["incremental_relative_l2"] = float((incremental.logits.float()-reference.float()).norm()/reference.float().norm())
        result["incremental_top1_equal"] = bool(torch.equal(incremental.logits.argmax(-1), reference.argmax(-1)))
        # Full-model equivalence of the FP32 numerical import proof.
        from audit_mixt import ImportedMainProof
        model.float()
        before = model(**inputs, use_cache=False).logits
        replaced = 0
        for layer in model.config.mixt_compress_layers:
            for kind in model.config.mixt_projection_kinds:
                projection = getattr(model.model.layers[layer].mlp, kind + "_proj")
                projection.mixt00 = ImportedMainProof(projection.mixt00)
                replaced += 1
        after = model(**inputs, use_cache=False).logits
        result["fp32_imported_projections"] = replaced
        result["fp32_import_logits_max_abs"] = float((after - before).abs().max())
        result["fp32_import_logits_relative_l2"] = float((after-before).norm()/before.norm())
        torch.testing.assert_close(after, before, rtol=2e-5, atol=2e-5)
except Exception as e:
    import traceback
    result["error"] = f"{type(e).__name__}: {e}"
    result["traceback"] = traceback.format_exc()
result["total_seconds"] = time.monotonic() - start
serialized = json.dumps(result, indent=2, ensure_ascii=False, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
output.write_text(serialized, encoding="utf-8")
print(serialized)
