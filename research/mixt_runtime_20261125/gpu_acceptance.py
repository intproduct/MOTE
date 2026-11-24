"""Bounded GPU acceptance for the real partner model; writes after each probe."""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fitmotn.patching import patch_qwen_ffn_layers, set_trainable_patch_only, set_motn_usage_tracking

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--lengths", nargs="+", type=int, default=[512, 4096, 8192, 16384, 32768])
parser.add_argument("--device", default=None)
parser.add_argument("--train-length", type=int, default=None)
parser.add_argument("--parity-length", type=int, default=128)
parser.add_argument("--skip-training", action="store_true")
parser.add_argument("--gate-arch", choices=["linear", "quantum"], default="linear")
args = parser.parse_args()
device = args.device or next(f"cuda:{i}" for i in range(torch.cuda.device_count()) if "RTX PRO 4000" in torch.cuda.get_device_name(i))
torch.cuda.set_device(device)
torch.set_num_threads(8)
torch.manual_seed(20261125)
report = dict(torch=torch.__version__, device=device, gpu=torch.cuda.get_device_name(),
              dtype="bfloat16", attention="sdpa", gate_arch=args.gate_arch, inference=[])
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
start = time.monotonic()

def save():
    report["elapsed_seconds"] = time.monotonic() - start
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

def sample_ids(n):
    return seed_ids.repeat((n + seed_ids.numel() - 1) // seed_ids.numel())[:n].unsqueeze(0)

try:
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    seed_ids = torch.tensor(tok.encode("Read this context carefully. The blue key is 7319. Explain the arithmetic in full detail. ", add_special_tokens=False), device=device)
    initial = sample_ids(args.parity_length)
    report["parity_tokens"] = args.parity_length
    with torch.no_grad():
        before = model(initial, use_cache=False, logits_to_keep=1).logits.detach().float()
    config = dict(patch_backend="pretrained_mixt", gate_type="softmax", gate_arch=args.gate_arch, topk=4, E=4,
        d=2, k_in=8, dtype=torch.bfloat16, drop_tokens=False, global_expert_enabled=False,
        pretrained_mixt=dict(version=1, uniform_init=True, train_scope="gate_only"))
    patch_qwen_ffn_layers(model, model.config.mixt_compress_layers, config, torch.device(device), dtype=torch.bfloat16)
    set_motn_usage_tracking(model, False)
    with torch.no_grad():
        after = model(initial, use_cache=False, logits_to_keep=1).logits.detach().float()
    report["gpu_import_max_abs"] = float((after - before).abs().max())
    report["gpu_import_relative_l2"] = float((after-before).norm()/before.norm())
    report["gpu_import_bitwise_equal"] = torch.equal(after, before)
    save()
    del before, after
    for length in args.lengths:
        ids = sample_ids(length)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        try:
            with torch.no_grad():
                out = model(ids, use_cache=False, logits_to_keep=1).logits
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(out).all())
            row = dict(tokens=length, seconds=time.monotonic()-tick, finite=finite,
                       output_shape=list(out.shape), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            assert finite
            report["inference"].append(row)
            print(json.dumps(row), flush=True)
            save()
            del out
        except torch.cuda.OutOfMemoryError as exc:
            report["inference"].append(dict(tokens=length, ok=False, error="out_of_memory", detail=str(exc)))
            save()
            break
    longest = max(row["tokens"] for row in report["inference"] if row.get("finite"))
    ids = sample_ids(longest)
    with torch.no_grad():
        generated = model.generate(ids, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
    report["cached_generation"] = dict(prompt_tokens=longest, new_tokens=generated.shape[-1]-longest)
    save()
    del generated, ids
    torch.cuda.empty_cache()
    if args.skip_training:
        report["completed"] = True
        save()
        sys.exit(0)
    set_trainable_patch_only(model)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    report["training"] = []
    for length, supervision in [(512, "all_tokens"), (min(longest, args.train_length or longest), "last_32_tokens")]:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        ids = sample_ids(length)
        try:
            if supervision == "all_tokens":
                loss = model(ids, labels=ids, use_cache=False).loss
            else:
                logits = model(ids, use_cache=False, logits_to_keep=33).logits
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, -32:].reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            nonzero = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in params)
            assert nonzero > 0, "No trainable router received a nonzero gradient"
            previous = [p.detach().clone() for p in params]
            optimizer.step()
            updated = sum(not torch.equal(p.detach(), old) for p, old in zip(params, previous))
            assert updated > 0, "Optimizer did not update any trainable router"
            del previous
            torch.cuda.synchronize()
            row = dict(tokens=length, supervision=supervision, loss=float(loss.detach()),
                       nonzero_gradient_tensors=nonzero, seconds=time.monotonic()-tick,
                       updated_parameter_tensors=updated,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, ok=True)
            report["training"].append(row)
            print(json.dumps(row), flush=True)
            save()
            del loss
        except torch.cuda.OutOfMemoryError as exc:
            report["training"].append(dict(tokens=length, supervision=supervision, ok=False, error="out_of_memory", detail=str(exc)))
            save()
            break
    report["completed"] = True
except Exception as exc:
    report.update(completed=False, error=str(exc), traceback=traceback.format_exc())
    raise
finally:
    save()
