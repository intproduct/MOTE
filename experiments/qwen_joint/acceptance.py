"""CPU-created common initialization and isolated native GPU acceptance stages.

This command is deliberately unable to start the formal matrix.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import traceback

from .resources import GPULease, PRO4000, V100
from .state import digest, fresh_directory, write_json


def source_identity():
    root = Path(__file__).resolve().parents[2]
    import subprocess
    tracked = subprocess.check_output(["git", "ls-files", "*.py", "pyproject.toml"], cwd=root, text=True).splitlines()
    paths = {root / p for p in tracked}
    paths.update(Path(__file__).parent.glob("*.py"))
    paths.update(Path(__file__).parent.glob("*.json"))
    paths.update((Path(__file__).parent / "configs").glob("*.json"))
    return {str(p.relative_to(root)): digest(p) for p in paths}


def verify_model(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf8"))
    for entry in manifest["files"]:
        path = Path(manifest["path"]) / entry["name"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Frozen model changed: {path}")
    return manifest


def initialize(output, model_manifest, seed):
    import torch
    from transformers import AutoModelForCausalLM
    from .model import AdapterConfig, patch_qwen, tensor_hash
    output = fresh_directory(output)
    write_json(output / "STATUS.json", {"phase": "common_initialization", "pid": os.getpid(), "seed": seed})
    torch.set_num_threads(8)
    manifest = verify_model(model_manifest)
    model = AutoModelForCausalLM.from_pretrained(manifest["path"], local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="sdpa")
    cfg = AdapterConfig(seed=seed)
    report = patch_qwen(model, "mlp", cfg)
    common = dict(report)
    common["state"] = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad and ".router." not in n}
    common["state_hash"] = tensor_hash(common["state"])
    common["model_manifest_sha256"] = digest(model_manifest)
    common["code"] = source_identity()
    checkpoint = output / "common.pt"
    torch.save(common, checkpoint)
    write_json(output / "COMMON_INITIALIZATION.json", {**report, "state_hash": common["state_hash"],
        "checkpoint_sha256": digest(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "model_manifest_sha256": digest(model_manifest), "code": common["code"], "created_epoch": time.time()})
    write_json(output / "STATUS.json", {"phase": "complete", "seed": seed, "parameters": report["parameters"]})


def probe(output, root, device_uuid):
    output = fresh_directory(output)
    with GPULease(root, device_uuid, "native_runtime_precision_probe", output) as lease:
        import torch
        import transformers
        from .state import MasterAdamW
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Lease must expose exactly one CUDA device")
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        if device_uuid in V100 and capability != (7, 0):
            raise RuntimeError(f"Expected V100 sm70, found {capability}")
        dtype = torch.bfloat16 if device_uuid == PRO4000 else torch.float16
        model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).to("cuda", dtype)
        opt = MasterAdamW(model)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=1024)
        before = [p.detach().clone() for p in model.parameters()]
        x = torch.randn(32, 256, device="cuda", dtype=dtype)
        loss = model(x).float().square().mean()
        scaler.scale(loss).backward()
        update = opt.step(scaler)
        torch.cuda.synchronize()
        if not update["updated"] or not bool(torch.isfinite(loss)) or not any(not torch.equal(a, b) for a, b in zip(before, model.parameters())):
            raise RuntimeError("Native precision forward/backward/update failed")
        write_json(output / "RUNTIME_PROBE.json", {"passed": True, "uuid": device_uuid, "lease": lease.record,
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda_build": torch.version.cuda,
            "architectures": torch.cuda.get_arch_list(), "device_name": properties.name, "capability": list(capability),
            "precision": str(dtype), "loss": float(loss.detach()), "update": update,
            "optimizer_states": sorted({str(s["exp_avg"].dtype) for s in opt.optimizer.state.values()}),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def full_model(output, root, device_uuid, model_manifest, common_path):
    output = fresh_directory(output)
    code = source_identity()
    with GPULease(root, device_uuid, "all_method_full_qwen_acceptance", output) as lease:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .model import AdapterConfig, JointProjection, patch_qwen, set_model_precision, tensor_hash
        from .routers import METHODS
        from .state import MasterAdamW
        from .task import next_target_loss
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest = verify_model(model_manifest)
        common_path = Path(common_path)
        init_manifest = json.loads((common_path.parent / "COMMON_INITIALIZATION.json").read_text(encoding="utf8"))
        if digest(common_path) != init_manifest["checkpoint_sha256"]:
            raise ValueError("Common initialization file hash mismatch")
        common = torch.load(common_path, map_location="cpu", weights_only=False)
        if common["model_manifest_sha256"] != digest(model_manifest):
            raise ValueError("Common initialization belongs to a different model")
        cfg = AdapterConfig(**common["adapter"])
        dtype = torch.bfloat16 if device_uuid == PRO4000 else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(manifest["path"], local_files_only=True)
        fixture = "This is a deterministic software integration fixture. A small model predicts the next token in a sequence. " * 300
        ids = tokenizer.encode(fixture, add_special_tokens=False)[:1025]
        if len(ids) != 1025:
            raise RuntimeError("Fixture too short")
        x = torch.tensor([ids[:-1]], device="cuda")
        y = torch.tensor([ids[1:]], device="cuda")
        rows = []
        for method in (*METHODS, "dense"):
            write_json(output / "STATUS.json", {"phase": "full_model_acceptance", "method": method, "pid": os.getpid(), "uuid": device_uuid})
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
            start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(manifest["path"], local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="sdpa")
            report = patch_qwen(model, method, cfg, common=None if method == "dense" else common)
            if report["layers"] != [21, 22, 23, 24, 25, 26, 27]:
                raise ValueError("Unexpected Qwen layer selection")
            set_model_precision(model, dtype, "cuda")
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            for module in model.modules():
                if isinstance(module, JointProjection):
                    module.capture_metrics = True
            optimizer = MasterAdamW(model)
            scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=1024)
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            timed = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                ce, valid, aux = next_target_loss(model, x, y)
                loss = ce / valid + 0.01 * aux
            scaler.scale(loss).backward()
            gradient_groups = {}
            for group, fragment in (("router", ".router."), ("experts", ".experts."), ("residual_tail", ".residual.")):
                parameters = [p for n, p in model.named_parameters() if p.requires_grad and fragment in n]
                gradient_groups[group] = {"parameters_with_gradient": sum(p.grad is not None for p in parameters),
                    "finite": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters),
                    "nonzero": sum(p.grad is not None and bool(p.grad.float().norm() > 0) for p in parameters)}
            update = optimizer.step(scaler)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - timed
            after = dict(model.named_parameters())
            changed = {group: sum(not torch.equal(p, after[n]) for n, p in before.items() if fragment in n)
                for group, fragment in (("router", ".router."), ("experts", ".experts."), ("all_trainable", ""))}
            if not update["updated"] or not bool(torch.isfinite(loss)) or changed["all_trainable"] == 0:
                raise RuntimeError(f"No finite joint update: {method}")
            if method != "dense" and any(not gradient_groups[g]["finite"] or gradient_groups[g]["nonzero"] == 0 or changed[g] == 0 for g in ("router", "experts")):
                raise RuntimeError(f"Missing router/expert joint gradient or update: {method}")
            row = {"method": method, "passed": True, "precision": str(dtype), "sequence_length": 1024, "valid_targets": valid,
                "ce": float(ce.detach() / valid), "aux": float(aux.detach()), "gradient_groups": gradient_groups,
                "changed_parameters": changed, "update": update, "step_seconds": seconds,
                "total_seconds_with_load": time.perf_counter() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "parameters": report["parameters"],
                "router_metrics": {n: m.last_metrics for n, m in model.named_modules() if isinstance(m, JointProjection)}}
            write_json(output / (method + ".json"), row)
            rows.append(row)
            del model, optimizer, scaler, before, after, parameters, module, x, y, ce, aux, loss
            gc.collect()
            torch.cuda.empty_cache()
            x, y = torch.tensor([ids[:-1]], device="cuda"), torch.tensor([ids[1:]], device="cuda")
        write_json(output / "FULL_MODEL_ACCEPTANCE.json", {"passed": True, "rows": rows, "code": code, "lease": lease.record,
            "common_initialization_sha256": digest(common_path), "model_manifest_sha256": digest(model_manifest),
            "fixture": "Deterministic synthetic software test only; not smoke, pilot, or evaluation evidence",
            "formal_acceptance": False})
        write_json(output / "STATUS.json", {"phase": "complete", "methods": len(rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "probe", "full-model"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="E:/AI/MOTE-quantum")
    parser.add_argument("--gpu", choices=(PRO4000, *V100))
    parser.add_argument("--model-manifest", default="E:/AI/MOTE-quantum/queue-20261030/model-manifest.json")
    parser.add_argument("--common")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    existed_before = Path(args.output).exists()
    try:
        if args.stage == "initialize":
            initialize(args.output, args.model_manifest, args.seed)
        elif args.stage == "probe":
            if not args.gpu:
                parser.error("probe requires --gpu")
            probe(args.output, args.root, args.gpu)
        else:
            if not args.gpu or not args.common:
                parser.error("full-model requires --gpu and --common")
            full_model(args.output, args.root, args.gpu, args.model_manifest, args.common)
    except BaseException:
        if not existed_before and Path(args.output).is_dir():
            write_json(Path(args.output) / "FAILURE.json", {"error": traceback.format_exc(), "time": time.time(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
