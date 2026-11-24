"""Compare partner, imported, and exported models on a real 32K GPU prefill."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fitmotn.patching import patch_qwen_ffn_layers, set_motn_usage_tracking

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--export", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--tokens", type=int, default=32768)
args = parser.parse_args()
torch.set_num_threads(8)
torch.manual_seed(20261125)
report = dict(torch=torch.__version__, tokens=args.tokens, dtype="bfloat16", attention="sdpa", stages={})
path = Path(args.output)
def save():
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True, local_files_only=True)
seed = torch.tensor(tok.encode("Read this context carefully. The blue key is 7319. Explain the arithmetic in full detail. ", add_special_tokens=False), device="cuda:0")
ids = seed.repeat((args.tokens + seed.numel() - 1) // seed.numel())[:args.tokens].unsqueeze(0)

def evaluate(model, stage):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        logits = model(ids, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
        generated = model.generate(ids, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)[0, args.tokens:].tolist()
    torch.cuda.synchronize()
    report["stages"][stage] = dict(seconds=time.monotonic()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                  finite=bool(torch.isfinite(logits).all()), generated_tokens=generated)
    save()
    return logits, generated

try:
    model = AutoModelForCausalLM.from_pretrained(args.source, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    reference, reference_tokens = evaluate(model, "partner")
    config = dict(patch_backend="pretrained_mixt", gate_type="softmax", gate_arch="linear", topk=4, E=4,
        d=2, k_in=8, dtype=torch.bfloat16, drop_tokens=False, global_expert_enabled=False,
        pretrained_mixt=dict(version=1, uniform_init=True, train_scope="gate_only"))
    patch_qwen_ffn_layers(model, model.config.mixt_compress_layers, config, torch.device("cuda:0"), dtype=torch.bfloat16)
    set_motn_usage_tracking(model, False)
    imported, imported_tokens = evaluate(model, "imported")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.export, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    set_motn_usage_tracking(model, False)
    projected = []
    hook = model.lm_head.register_forward_pre_hook(lambda module, inputs: projected.append(inputs[0].shape[1]))
    exported, exported_tokens = evaluate(model, "exported")
    hook.remove()
    report["export_projected_positions"] = projected
    report["import_max_abs"] = float((imported-reference).abs().max())
    report["export_max_abs"] = float((exported-reference).abs().max())
    report["import_bitwise_equal"] = torch.equal(imported, reference)
    report["export_bitwise_equal"] = torch.equal(exported, reference)
    report["generated_tokens_match"] = reference_tokens == imported_tokens == exported_tokens
    report["ok"] = report["import_bitwise_equal"] and report["export_bitwise_equal"] and report["generated_tokens_match"] and all(n == 1 for n in projected)
    save()
    assert report["ok"], report
except Exception as exc:
    report.update(ok=False, error=repr(exc))
    save()
    raise
