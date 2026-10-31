"""Independent equal-token smoke/pilot training; formal runs require a signed-off release file."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .acceptance import source_identity, verify_model
from .data import TargetWindows
from .resources import GPULease, PRO4000, V100
# Model imports transitively reach legacy losses.py, which queries CUDA at import.
# Delay them until the owned UUID lease is acquired and physically verified.
METHODS = ("mlp", "affine_born", "reencode_cz", "reencode_product")
from .state import MasterAdamW, TokenSchedule, digest, fresh_directory, rng_state, restore_rng, write_json
from .checkpoint import load_checkpoint, save_checkpoint
from .retention import Retention
from .reproducibility import configure, PROFILE, state_hashes, tree_digest
from .thermal import Pacer, POLICY
from .validation import formal_budget


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_common(path, model_manifest):
    path = Path(path)
    metadata = json.loads((path.parent / "COMMON_INITIALIZATION.json").read_text(encoding="utf8"))
    if digest(path) != metadata["checkpoint_sha256"]:
        raise ValueError("Common initialization file hash mismatch")
    common = torch.load(path, map_location="cpu", weights_only=False)
    if common["model_manifest_sha256"] != digest(model_manifest):
        raise ValueError("Common initialization model mismatch")
    return common


def verify_data(path):
    root = Path(path)
    release = json.loads((root / "DATA_RELEASE.json").read_text(encoding="utf8"))
    if release["status"] != "frozen" or release["sequence_length"] != 1024:
        raise ValueError("A frozen compatible data release is required")
    for name, entry in release["files"].items():
        if (root / name).stat().st_size != entry["bytes"] or digest(root / name) != entry["sha256"]:
            raise ValueError(f"Data release changed: {name}")
    return release


@torch.no_grad()
def evaluate(model, data, seed, dtype, target=1_048_576, guard=None, device='cuda'):
    """Fixed-dev evaluation preserves all training RNG and mode, even on error."""
    from .task import next_target_loss
    saved_rng, was_training = rng_state(), model.training
    model.eval()
    try:
        windows = TargetWindows(data, 'internal_dev', 17)
        total, count, window_count = 0.0, 0, 0
        sample_hash = hashlib.sha256()
        while count < target:
            if guard is not None and window_count % 16 == 0:
                guard.check('evaluation')
            ids, labels, sample = windows.next(target-count)
            window_count += 1
            if sample['valid_targets'] == 0:
                continue
            with torch.autocast(device, dtype=dtype, enabled=device == 'cuda'):
                ce, valid, _ = next_target_loss(model, torch.from_numpy(ids)[None].to(device),
                                               torch.from_numpy(labels)[None].to(device))
            if not bool(torch.isfinite(ce)):
                raise FloatingPointError('Nonfinite dev cross entropy')
            total += float(ce)
            count += valid
            sample_hash.update(json.dumps(sample, sort_keys=True).encode())
        return dict(ce=total/count, nll_sum=total, valid_targets=count, sample_hash=sample_hash.hexdigest(),
                    evaluation_order_seed=17)
    finally:
        restore_rng(saved_rng)
        model.train(was_training)


def make_guard(args, output):
    from .resources import command
    policy = dict(POLICY)
    if args.gpu != PRO4000:
        policy.update(profile='v100-thermal-pacing-v1',minimum_update_start_interval_seconds=0)
    def stop():
        return any(p.exists() for p in (Path(args.root)/'STOP',Path(args.root)/'queue-20261030/STOP',
                                      output/'STOP',output.parent/'STOP'))
    def query():
        return command(['nvidia-smi','-i',args.gpu,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'])
    def record(value):
        with (output/'thermal-pacing.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(value)+'\n')
    write_json(output/'THERMAL_PROFILE.json',policy)
    return Pacer(query,stop,record,policy=policy)


def run(args):
    output = fresh_directory(args.output)
    configure()  # cuBLAS configuration precedes the lease's first CUDA call.
    if args.stage in ('smoke','pilot','replay') and args.seed != 17:
        raise ValueError('Acceptance uses seed17 for every method')
    if args.stage != 'replay' and args.run_updates is not None:
        raise ValueError('Bounded partial updates are confined to replay acceptance')
    code = source_identity()
    write_json(output/'STATUS.json',dict(phase='validating_release',stage=args.stage,method=args.method,pid=os.getpid(),time=time.time()))
    data_release = verify_data(args.data)
    manifest = verify_model(args.model_manifest)
    common = None if args.method == 'dense' else load_common(args.common,args.model_manifest)
    identity = dict(code=code,model_manifest_sha256=digest(args.model_manifest),
        data_sha256=digest(Path(args.data)/'DATA_RELEASE.json'),common_sha256=None if common is None else digest(args.common),
        method=args.method,seed=args.seed,gpu=args.gpu,stage=args.stage,
        config_sha256=digest(Path(__file__).parent/'configs/integration.json'),reproducibility=PROFILE)
    if args.stage == 'formal':
        if not args.acceptance:raise ValueError('Formal matrix requires INTEGRATION_ACCEPTANCE.json')
        acceptance=json.loads(Path(args.acceptance).read_text(encoding='utf8'))
        target=formal_budget(acceptance,code,identity['data_sha256'],identity['model_manifest_sha256'],
            identity['config_sha256'],args.method,args.seed,args.gpu,identity['common_sha256'],args.root)
        identity['acceptance_sha256']=digest(args.acceptance)
    else:
        target=655360 if args.stage=='smoke' else 1048576 if args.stage in ('pilot','replay') else 0
    identity['target_tokens']=target
    with GPULease(args.root, args.gpu, args.stage + "_" + args.method, output) as lease:
        from .model import AdapterConfig, JointProjection, patch_qwen, set_model_precision, tensor_hash
        from .task import next_target_loss
        cfg = AdapterConfig(seed=args.seed)
        if common is not None and common["adapter"] != asdict(cfg):
            raise ValueError("Seed/common initialization configuration mismatch")
        identity["adapter"] = asdict(cfg)
        identity["cuda_device"] = lease.record["cuda_device"]
        write_json(output / "RUN_CONFIG.json", identity)
        write_json(output / "STATUS.json", {"phase": "loading_model", "stage": args.stage, "pid": os.getpid(), "cuda_device": lease.record["cuda_device"]})
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        seed_everything(args.seed)
        dtype = torch.bfloat16 if args.gpu == PRO4000 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(manifest["path"], local_files_only=True,
            torch_dtype=torch.float32, trust_remote_code=False, attn_implementation="sdpa")
        report = patch_qwen(model, args.method, cfg, common=common)
        router_before=tensor_hash({n:p for n,p in model.named_parameters() if '.router.' in n})
        set_model_precision(model, dtype, "cuda")
        router_after=tensor_hash({n:p for n,p in model.named_parameters() if '.router.' in n})
        if router_before != router_after:raise RuntimeError('Router placement changed initial FP32 values')
        write_json(output/'ROUTER_PLACEMENT.json',dict(before=router_before,after=router_after,exact=True))
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        write_json(output / "PARAMETERS.json", report)
        guard=make_guard(args,output)
        stage_times=dict(training_seconds=0.0,checkpoint_seconds=0.0,evaluation_seconds=0.0)
        def measured_evaluation(tokens=1048576):
            torch.cuda.synchronize();begin=time.perf_counter()
            result=evaluate(model,args.data,args.seed,dtype,target=tokens,guard=guard)
            torch.cuda.synchronize();stage_times['evaluation_seconds']+=time.perf_counter()-begin
            return result
        def measured_save(path,cursor,extra=None):
            torch.cuda.synchronize();begin=time.perf_counter()
            save_checkpoint(path,model,optimizer,schedule,scaler,cursor,identity,extra=extra,manifest_path=args.model_manifest)
            torch.cuda.synchronize();stage_times['checkpoint_seconds']+=time.perf_counter()-begin
        if args.stage == "dense-eval":
            if args.method != "dense" or args.resume:
                raise ValueError("Untouched Dense evaluation requires dense without resume")
            write_json(output / "STATUS.json", {"phase": "evaluating_dense", "stage": args.stage, "pid": os.getpid(), "cuda_device": lease.record["cuda_device"], "time": time.time()})
            result = measured_evaluation()
            write_json(output / "DENSE_EVALUATION.json", {**result, "identity": identity, "precision": str(dtype), "updates": 0})
            write_json(output / "STATUS.json", {"phase": "complete", "stage": args.stage})
            return
        optimizer = MasterAdamW(model)
        projections = {n: m for n, m in model.named_modules() if isinstance(m, JointProjection)}
        schedule = TokenSchedule(optimizer, target)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=1024)
        cursor = None
        if args.resume:
            cursor, _ = load_checkpoint(args.resume, model, optimizer, schedule, scaler, identity)
        stream = TargetWindows(args.data, "train", args.seed, cursor)
        # Formal rotation owns a fresh checkpoint subdirectory; it never adopts
        # an old run. Resumed prefix assets remain external and untouched.
        retention=Retention.create(output,output/'checkpoints',identity,target) if args.stage=='formal' else None
        checkpoint_root=retention.path if retention is not None else output
        start_tokens=schedule.tokens
        if start_tokens >= target:
            raise ValueError('Checkpoint already reached this token budget; refusing duplicate completed work')
        run_limit=min(target,start_tokens+32768*(args.run_updates or 2)) if args.stage=='replay' else target
        if args.stage=='replay':
            write_json(output/'START_STATE_HASHES.json',state_hashes(model,optimizer,schedule,scaler,stream))
        metrics = (output / "updates.jsonl").open("x", encoding="utf8")
        sample_log = (output / "samples.jsonl").open("x", encoding="utf8")
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        phase = "running"
        update_in_progress = False
        before_cursor, before_rng = stream.state_dict(), rng_state()
        try:
            while schedule.tokens < run_limit:
                temperature=int(guard.check('update'))
                update_tokens = min(32768, target - schedule.tokens)
                before_cursor, before_rng = stream.state_dict(), rng_state()
                update_in_progress = True
                schedule.set_next_update(update_tokens)
                # GradScaler-overflow retries repeat the same target stream and RNG.
                torch.cuda.synchronize()
                timed = time.perf_counter()
                for attempt in range(9):
                    optimizer.zero_grad()
                    total, nll, balance, samples = 0, 0.0, 0.0, []
                    while total < update_tokens:
                        if guard.stop():raise RuntimeError('STOP marker during update; preserve incomplete state')
                        ids, labels, sample = stream.next(update_tokens - total)
                        if not sample["valid_targets"]:
                            continue
                        # Capture one explicitly identified first window per update.
                        # Recomputed activation-checkpoint forwards have the same sample.
                        for module in projections.values():
                            module.capture_metrics = total == 0
                        with torch.autocast("cuda", dtype=dtype):
                            ce, valid, aux = next_target_loss(model, torch.from_numpy(ids)[None].to("cuda"), torch.from_numpy(labels)[None].to("cuda"))
                            loss = ce / update_tokens + 0.01 * aux * valid / update_tokens
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Nonfinite task loss")
                        scaler.scale(loss).backward()
                        total += valid
                        nll += float(ce.detach())
                        balance += float(aux.detach()) * valid
                        samples.append(sample)
                    update = optimizer.step(scaler, capture_stats=True)
                    torch.cuda.synchronize()
                    if update["updated"]:
                        break
                    write_json(output / f"overflow-{schedule.tokens}-{attempt}.json", {"tokens": schedule.tokens, "attempt": attempt, "scaler": scaler.state_dict()})
                    stream = TargetWindows(args.data, "train", args.seed, before_cursor)
                    restore_rng(before_rng)
                else:
                    raise FloatingPointError("GradScaler failed nine identical-stream attempts")
                schedule.advance(total)
                update_in_progress = False
                seconds = time.perf_counter() - timed
                stage_times['training_seconds']+=seconds
                row = {"tokens": schedule.tokens, "update_tokens": total, "ce": nll / total, "load_balance": balance / total,
                       "seconds": seconds, "valid_tokens_per_second": total / seconds, "grad_norm": update["grad_norm"],
                       "overflow_retries": attempt, "temperature_c": temperature, "cursor": stream.state_dict(),
                       "optimizer_update": update, "routing_sample": samples[0],
                       "routing_first_window": {n: m.last_metrics for n, m in projections.items()},
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                sample_log.write(json.dumps({"tokens": schedule.tokens, "samples": samples}) + "\n")
                sample_log.flush()
                write_json(output / "STATUS.json", {"phase": phase, "tokens": schedule.tokens, "target": target, "pid": os.getpid(), "uuid": args.gpu, "cursor": stream.state_dict(), "time": time.time()})
                cadence=target//10 if args.stage=='formal' else target
                if args.stage=='replay':
                    # Only the first reference checkpoint is required to prove
                    # interruption recovery; terminal hashes retain every state.
                    if not args.resume and schedule.tokens==start_tokens+32768:
                        measured_save(output/f'checkpoint-{schedule.tokens:010d}',stream.state_dict())
                    evaluation=measured_evaluation(4096)
                    write_json(output/f'dev-{schedule.tokens:010d}.json',evaluation)
                    fingerprints=state_hashes(model,optimizer,schedule,scaler,stream)
                    fingerprints['samples']=tree_digest(samples)
                    write_json(output/f'state-{schedule.tokens:010d}.json',fingerprints)
                elif schedule.tokens%cadence==0 or schedule.tokens==target:
                    checkpoint=checkpoint_root/f'checkpoint-{schedule.tokens:010d}'
                    measured_save(checkpoint,stream.state_dict())
                    evaluation=measured_evaluation()
                    write_json(output/f'dev-{schedule.tokens:010d}.json',evaluation)
                    if retention is not None:
                        if guard.stop():raise RuntimeError('STOP before retention; preserve all checkpoints')
                        retention.register(checkpoint.name,evaluation['ce'])
                        write_json(output/f'retention-{schedule.tokens:010d}.json',retention.prune())
            completed=schedule.tokens==target
            result=dict(phase='replay_complete' if args.stage=='replay' else 'complete' if completed else phase,
                tokens=schedule.tokens,target=target,elapsed_seconds=time.perf_counter()-start,
                stage_seconds=stage_times,identity=identity,lease=lease.record,precision=str(dtype),
                samples_sha256=digest(output/'samples.jsonl'),start_tokens=start_tokens,
                internal_dev=evaluation)
            if args.stage=='replay':
                result['component_hashes']=fingerprints
                result['replay_updates']=(schedule.tokens-start_tokens)//32768
                result['next_ce']=row['ce']
            else:
                result['checkpoint']=str(checkpoint)
            write_json(output/'RESULT.json',result)
            write_json(output/'STATUS.json',result)
        except BaseException:
            # Preserve the actual failed state separately; never label it complete.
            failure = {"error": traceback.format_exc(), "tokens": schedule.tokens,
                       "incomplete_update": update_in_progress, "observed_cursor": stream.state_dict(),
                       "last_update_start_cursor": before_cursor, "time": time.time()}
            try:
                failed_rng = rng_state()
                if update_in_progress:
                    restore_rng(before_rng)
                measured_save(output/'failure-checkpoint',before_cursor if update_in_progress else stream.state_dict(),
                    extra=dict(failure=failure,failed_rng=failed_rng,automatic_resume_permitted=False))
            except BaseException:
                failure["failure_checkpoint_error"] = traceback.format_exc()
            write_json(output / "TRAINING_FAILURE_STATE.json", failure)
            raise
        finally:
            metrics.close()
            sample_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dense-eval", "smoke", "pilot", "replay", "formal"))
    parser.add_argument("--method", required=True, choices=(*METHODS, "dense"))
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--gpu", required=True, choices=(PRO4000, *V100))
    parser.add_argument("--data", required=True)
    parser.add_argument("--common")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="E:/AI/MOTE-quantum")
    parser.add_argument("--model-manifest", default="E:/AI/MOTE-quantum/queue-20261030/model-manifest.json")
    parser.add_argument("--resume")
    parser.add_argument("--acceptance")
    parser.add_argument("--run-updates",type=int,choices=(1,2))
    args = parser.parse_args()
    if args.method != "dense" and not args.common:
        parser.error("MoTE methods require --common")
    existed_before = Path(args.output).exists()
    try:
        run(args)
    except BaseException:
        if not existed_before and Path(args.output).is_dir():
            write_json(Path(args.output) / "FAILURE.json", {"error": traceback.format_exc(), "time": time.time(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
