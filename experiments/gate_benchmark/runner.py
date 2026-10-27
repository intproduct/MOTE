from __future__ import annotations
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .config import digest
from .data import TensorDictDataset, load_data, bert_path
from .models import (
    ImageClassifier,
    BertClassifier,
    auxiliary_loss,
    routing_snapshot,
    shared_state,
    load_shared,
    state_hash,
    parameter_report,
    warm_initialize,
    routed_modules,
    is_router_key,
)


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def environment():
    import transformers, datasets

    revision, dirty = None, None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        check = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            check=False,
        )
        dirty = check.returncode != 0
    except (subprocess.CalledProcessError, OSError):
        pass
    root = Path(__file__).resolve().parents[2]
    sources = ('gate.py', 'structured_routers.py', 'advanced_routers.py',
               'experiments/gate_benchmark/models.py', 'experiments/gate_benchmark/runner.py',
               'experiments/gate_benchmark/config.py')
    return {
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in sources if (root/name).exists()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_commit": revision,
        "git_tracked_dirty": dirty,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def metrics(logits, labels, task):
    preds = logits.argmax(-1)
    result = {
        "accuracy": float((preds == labels).float().mean()),
        "ce": float(F.cross_entropy(logits, labels)),
    }
    if task == "mrpc":
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        result["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        result["score"] = (result["accuracy"] + result["f1"]) / 2
    else:
        result["score"] = result["accuracy"]
    return result


@torch.no_grad()
def evaluate(model, data, task, batch_size, device):
    model.eval()
    logits, labels = [], []
    router_sums = {}
    for batch in DataLoader(
        TensorDictDataset(data), batch_size=batch_size, shuffle=False
    ):
        y = batch.pop("labels")
        pred = model(**{k: v.to(device) for k, v in batch.items()})
        logits.append(pred.float().cpu())
        labels.append(y)
        for name, m in routed_modules(model):
            a, n = m.last_aux, m.last_tokens
            if a is None:
                continue
            r = router_sums.setdefault(
                name,
                {
                    "tokens": 0,
                    "H_tok": 0.0,
                    "drop_rate": 0.0,
                    "importance": torch.zeros_like(a["importance"], device="cpu"),
                    "load": torch.zeros_like(a["load"], device="cpu"),
                },
            )
            r["tokens"] += n
            for key in ("H_tok", "drop_rate"):
                r[key] += float(a[key]) * n
            for key in ("importance", "load"):
                r[key] += a[key].detach().cpu() * n
    route = {}
    for name, r in router_sums.items():
        n = r["tokens"]
        imp, load = r["importance"] / n, r["load"] / n
        route[name] = {
            "tokens": n,
            "H_tok": r["H_tok"] / n,
            "H_dataset": float(-(imp * imp.clamp_min(1e-12).log()).sum()),
            "drop_rate": r["drop_rate"] / n,
            "importance": imp.tolist(),
            "load": load.tolist(),
            "load_cv": float(load.std(unbiased=False) / load.mean().clamp_min(1e-12)),
            "unused_experts": int((load == 0).sum()),
        }
    return metrics(torch.cat(logits), torch.cat(labels), task), route


def train_one(
    model, cfg, stage, task, arch, seed, data, metadata, out, common_hash, warm_report
):
    out = Path(out)
    if out.exists():
        raise FileExistsError(
            f"Run directory already exists: {out}. Completed runs are skipped by run; move incomplete runs before restarting."
        )
    out.mkdir(parents=True)
    device = torch.device(cfg["device"])
    model.to(device)
    if cfg["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("fp16 benchmark requires CUDA; CPU/Mac smoke uses fp32")
    settings = cfg[stage]
    manifest = {
        "config": cfg,
        "stage": stage,
        "task": task,
        "gate": arch,
        "seed": seed,
        "shared_state_sha256": common_hash,
        "data": metadata,
        "parameters": parameter_report(model),
        "warm_initialization": warm_report,
        "environment": environment(),
    }
    write_json(out / "manifest.json", manifest)
    initial_router = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                      if is_router_key(n) and p.numel()}
    seed_all(seed, cfg["deterministic"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDictDataset(data["train"]),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    initial, initial_route = evaluate(
        model, data["dev"], task, settings["batch_size"], device
    )
    write_json(out / "initial_dev.json", {"metrics": initial, "routing": initial_route})
    best, best_step = -math.inf, 0
    train_seconds, samples, skipped = 0.0, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    with (out / "curves.jsonl").open("w") as log:
        for step in range(1, settings["steps"] + 1):
            model.train()
            sync(device)
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            # Identical linear decay after a 5% warmup for all three routers.
            warm_steps = max(1, round(settings["steps"] * 0.05))
            factor = min(
                step / warm_steps,
                max(
                    0.0,
                    (settings["steps"] - step + 1)
                    / max(1, settings["steps"] - warm_steps),
                ),
            )
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"] * factor
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=cfg["precision"] == "fp16"
            ):
                pred = model(**batch)
                ce = F.cross_entropy(pred.float(), y)
                aux = auxiliary_loss(model)
                loss = ce + aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=cfg["precision"] == "fp32"
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            sync(device)
            train_seconds += time.perf_counter() - start
            samples += len(y)
            if step % cfg["log_every"] == 0 or step == 1:
                row = {
                    "step": step,
                    "train_ce": float(ce.detach()),
                    "aux": float(aux.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(norm) if torch.isfinite(norm) else None,
                    "training_seconds": train_seconds,
                    "examples_per_second": samples / train_seconds,
                    "amp_skipped_steps": skipped,
                    "routing_last_batch": routing_snapshot(model),
                }
                if step == 1:
                    row["router_gradient_audit"] = {
                        n: {"finite": bool(torch.isfinite(p.grad).all()),
                            "nonzero": bool(p.grad.abs().sum() > 0)}
                        if p.grad is not None else {"finite": False, "nonzero": False}
                        for n, p in model.named_parameters() if n in initial_router
                    }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    f"{stage}/{task}/{arch}/{seed} step {step}/{settings['steps']} ce={row['train_ce']:.4f}",
                    flush=True,
                )
            if step % cfg["eval_every"] == 0 or step == settings["steps"]:
                score, route = evaluate(
                    model, data["dev"], task, settings["batch_size"], device
                )
                log.write(
                    json.dumps({"step": step, "dev": score, "routing_dev": route})
                    + "\n"
                )
                log.flush()
                if score["score"] > best:
                    best, best_step = score["score"], step
                    tmp = out / "best.tmp"
                    torch.save(
                        {"model": model.state_dict(), "step": step, "dev": score}, tmp
                    )
                    tmp.replace(out / "best.pt")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final, final_route = evaluate(
        model, data["test"], task, settings["batch_size"], device
    )
    result = {
        "router_update_audit": {
            n: {"changed": not torch.equal(p.detach().cpu(), initial_router[n]),
                "l2_delta": float((p.detach().cpu()-initial_router[n]).double().norm())}
            for n, p in model.named_parameters() if n in initial_router
        },
        "stage": stage,
        "task": task,
        "gate": arch,
        "seed": seed,
        "best_step": best_step,
        "best_dev": checkpoint["dev"],
        "final": final,
        "final_routing": final_route,
        "training_seconds": train_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
        "examples_per_second": samples / train_seconds,
        "amp_skipped_steps": skipped,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "shared_state_sha256": common_hash,
        "data_sha256": metadata.get("tensor_file_sha256", "synthetic"),
        "router_parameters": manifest["parameters"]["router_total"],
        "total_parameters": manifest["parameters"]["total"],
    }
    write_json(out / "result.json", result)
    return result


def run(cfg, stage="all", tasks=None, seeds=None, gates=None):
    if gates is not None and not set(gates) <= set(cfg["gates"]):
        raise ValueError('requested gates are not in the configured experiment plan')
    if seeds is not None and not set(seeds) <= set(cfg["seeds"]):
        raise ValueError('requested seeds are not in the configured experiment plan')
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use doctor, or --device cpu for smoke only"
        )
    experiment = Path(cfg["output_dir"]) / digest(cfg)[:12]
    experiment.mkdir(parents=True, exist_ok=True)
    write_json(experiment / "config.json", cfg)
    for s in ("mlp", "bert"):
        if stage not in {"all", s}:
            continue
        model_path, model_info = bert_path(cfg) if s == "bert" else (None, None)
        for task in cfg[s]["tasks"]:
            if tasks is not None and task not in tasks:
                continue
            data, meta = load_data(cfg, task)
            meta["pretrained_model"] = model_info
            for seed in cfg["seeds"] if seeds is None else seeds:

                def build(arch):
                    seed_all(seed, cfg["deterministic"])
                    return (
                        ImageClassifier(
                            cfg, arch, seed, meta["input_dim"], meta["classes"]
                        )
                        if s == "mlp"
                        else BertClassifier(
                            cfg, arch, seed, meta["classes"], model_path=model_path
                        )
                    ).to(device)

                destinations = {
                    arch: experiment / f"{s}_{task}_{arch}_seed{seed}"
                    for arch in (cfg["gates"] if gates is None else gates)
                }
                pending = [
                    a
                    for a, p in destinations.items()
                    if not (p / "result.json").exists()
                ]
                if not pending:
                    continue
                for arch in pending:
                    if destinations[arch].exists():
                        raise FileExistsError(
                            f"Incomplete run: {destinations[arch]}; move it before restarting"
                        )
                template = build("mlp")
                warm_report = warm_initialize(
                    template, cfg[s]["warm_init_steps"], seed + 10000
                )
                common = shared_state(template)
                common_hash = state_hash(common)
                del template
                gc.collect()
                for arch in pending:
                    model = build(arch)
                    load_shared(model, common)
                    warm_initialize(model, 0, seed)  # release unused source weights
                    assert state_hash(shared_state(model)) == common_hash
                    train_one(
                        model,
                        cfg,
                        s,
                        task,
                        arch,
                        seed,
                        data,
                        meta,
                        destinations[arch],
                        common_hash,
                        warm_report,
                    )
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    print(f"results: {experiment}", flush=True)
    return experiment
