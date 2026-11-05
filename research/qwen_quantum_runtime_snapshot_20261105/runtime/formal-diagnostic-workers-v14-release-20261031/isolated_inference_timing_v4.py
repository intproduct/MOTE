"""Evaluation-only serial native-precision timing from a verified full checkpoint."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

ROOT = Path('E:/AI/MOTE-quantum')
REF = ROOT / 'diagnostics/mechanism-v1-20261031'


def run(args):
    # No model import or CUDA initialization before masking and acquiring a lease.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from fitmotn.experiments.qwen_joint.acceptance import source_identity
    from fitmotn.experiments.qwen_joint.checkpoint import decode
    from fitmotn.experiments.qwen_joint.resources import GPULease, PRO4000, V100, gpu_snapshot
    from fitmotn.experiments.qwen_joint.state import write_json
    from checkpoint_mechanism_v4 import read, digest, load_checkpoint
    from production_resource_preflight_v3 import check as resource_check
    from inference_timing_core_v4 import (serial_measure, summarize, positive_integer,
        assert_plain_evaluation, assert_owned_observation)

    positive_integer(args.warmup, 'warmup'); positive_integer(args.repeats, 'repeats')
    out = Path(args.output)
    if not out.resolve().is_relative_to((ROOT / 'diagnostics').resolve()):
        raise ValueError('Output outside diagnostic root')
    if args.gpu not in (PRO4000, *V100): raise ValueError('Unknown physical GPU')
    out.mkdir(parents=True, exist_ok=False)
    def status(phase, **details):
        write_json(out / 'STATUS.json', dict(phase=phase, pid=os.getpid(), gpu=args.gpu, time=time.time(), **details))
    started = time.perf_counter()
    try:
        status('verifying_checkpoint')
        release = read(ROOT / 'releases/integration-v3.json')
        assert source_identity() == release['code']
        marker, packed, source_mode = load_checkpoint(args.checkpoint, release)
        identity = marker['identity']
        native = torch.bfloat16 if args.gpu == PRO4000 else torch.float16
        cohort = torch.bfloat16 if identity['gpu'] == PRO4000 else torch.float16
        if native != cohort: raise ValueError('Checkpoint precision cohort differs from diagnostic device')
        assert {v.dtype for k,v in packed['model'].items() if '.router.' not in k} == {native}
        budget = resource_check(); reserve = 10_000_000
        assert budget['acceptance_launch_fits'] and budget['processing_actual_plus_reserved_bytes'] + reserve < 20_000_000_000
        write_json(out / 'RESERVATION.json', dict(**budget, additional_diagnostic_reservation_bytes=reserve))
        protocol = read(REF / 'MECHANISM_PROTOCOL.json')
        assert digest(REF / 'fixed_windows.npz') == protocol['arrays_sha256']
        assert protocol['model_manifest_sha256'] == identity['model_manifest_sha256']
        assert protocol['data_release_sha256'] == identity['data_sha256']
        with np.load(REF / 'fixed_windows.npz', allow_pickle=False) as arrays:
            windows = [torch.from_numpy(arrays[f'ids_{i}'].copy()).reshape(1, -1) for i in range(2)]
            assert all(tuple(ids.shape) == (1,1024) for ids in windows)
            valid_targets = sum(int((arrays[f'labels_{i}'] != -100).sum()) for i in range(2))
        assert valid_targets == 2046
        seeds = read(ROOT / 'initialization/seed-release-v2-20261031/SEED_RELEASE.json')
        initial = next(row for row in seeds['initializations'] if row['seed'] == identity['seed'])
        assert digest(initial['common_path']) == initial['checkpoint_sha256']
        assert identity['common_sha256'] == (None if identity['method'] == 'dense' else initial['checkpoint_sha256'])
        common = torch.load(initial['common_path'], map_location='cpu', weights_only=False, mmap=True)
        restored = decode(packed); del packed
        with GPULease(ROOT, args.gpu, 'isolated_inference_timing_v4', out) as lease:
            from transformers import AutoModelForCausalLM
            from fitmotn.experiments.qwen_joint.model import AdapterConfig, patch_qwen, set_model_precision, tensor_hash
            from fitmotn.experiments.qwen_joint.reproducibility import configure
            from fitmotn.experiments.qwen_joint.train import make_guard
            profile = configure()
            guard = make_guard(SimpleNamespace(root=ROOT, gpu=args.gpu), out)
            def check(): guard.check('diagnostic')
            def owned():
                return assert_owned_observation(gpu_snapshot(args.gpu), lease.record,
                    read(lease.path), os.getpid())
            check(); status('loading_model')
            model_manifest = read(ROOT / 'queue-20261030/model-manifest.json')
            model = AutoModelForCausalLM.from_pretrained(model_manifest['path'], local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='sdpa')
            patch_qwen(model, identity['method'], AdapterConfig(**identity['adapter']),
                common=common if identity['method'] != 'dense' else None)
            model.load_state_dict(restored['model'], strict=True)
            model.requires_grad_(False).eval(); model.config.use_cache = False
            model.gradient_checkpointing_disable()
            set_model_precision(model, native, 'cuda')
            before_hash = tensor_hash(model.state_dict())
            assert before_hash == tensor_hash(restored['model']), 'Native model restore changed stored weights'
            del restored, common; gc.collect()
            windows = [ids.to('cuda') for ids in windows]
            assert_plain_evaluation(model)
            # Global hooks also run on every module and would contaminate measurements.
            global_hooks = torch.nn.modules.module
            for name in ('_global_forward_hooks','_global_forward_pre_hooks','_global_backward_hooks','_global_backward_pre_hooks'):
                if getattr(global_hooks, name, {}): raise RuntimeError('Global model hooks are active')
            all_devices_before = [gpu_snapshot(gpu) for gpu in (PRO4000,*V100)]
            owned()
            # Validate complete logits on fixed inputs outside timing and without diagnostic hooks.
            validation = []
            with torch.inference_mode(), torch.autocast('cuda',dtype=native):
                for ids in windows:
                    check()
                    logits = model(input_ids=ids, use_cache=False).logits
                    assert tuple(logits.shape) == (1,1024,model.config.vocab_size)
                    assert torch.isfinite(logits).all()
                    validation.append(dict(shape=list(logits.shape),dtype=str(logits.dtype),
                        final_position_argmax=int(logits[0,-1].argmax())))
                    del logits
            torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
            def forward():
                for ids in windows:
                    output = model(input_ids=ids, use_cache=False)
                    del output
            def before(phase, index):
                check(); assert_plain_evaluation(model)
                status('timing_' + phase, repetition=index, method=identity['method'], checkpoint_tokens=marker['tokens'])
                observation = owned()
                torch.cuda.reset_peak_memory_stats()
                return observation
            def after(row):
                result = dict(**owned(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                write_json(out / (row['phase'] + '-' + str(row['index']) + '.json'), dict(**row, observation_after=result))
                return result
            with torch.inference_mode(), torch.autocast('cuda',dtype=native):
                rows = serial_measure(forward, torch.cuda.synchronize, time.perf_counter, before, after,
                    warmup=args.warmup, repeats=args.repeats)
            assert before_hash == tensor_hash(model.state_dict()), 'Timing changed checkpoint values'
            owned(); assert source_identity() == release['code']
            summary = summarize([row['seconds'] for row in rows if row['phase'] == 'measured'], 2048)
            payload = dict(passed=True, phase='complete', method=identity['method'], seed=identity['seed'],
                checkpoint_tokens=marker['tokens'], checkpoint=str(Path(args.checkpoint)),
                checkpoint_payload_sha256=marker['files']['training.pt']['sha256'],
                checkpoint_marker_sha256=digest(Path(args.checkpoint) / 'complete.json'),
                stored_model_hash=before_hash, native_precision=str(native), router_precision='torch.float32',
                physical_device=lease.record['cuda_device'], lease=lease.record, profile=profile,
                warmup_repeats=args.warmup, measured_repeats=args.repeats, batch_size=1,
                sequence_length=1024, windows_per_repeat=2, valid_targets_per_repeat=valid_targets,
                task='full vocabulary forward throughput on fixed corpus windows; not generation or training throughput',
                kv_cache_policy='disabled; full windows only, no autoregressive generation',
                timed_scope=['backbone','native sparse dispatch','all selected FFN projections','full vocabulary output',
                    'Python dispatch overhead','CUDA synchronization after complete forward'],
                excluded_scope=['tokenization','disk reads','CPU to GPU input transfer','checkpoint/model loading',
                    'warmup','occupancy observations','temperature/STOP checks and cooling','logit validation','weight hashing'],
                selected_device_no_foreign_compute_at_observations=True,
                host_exclusive=False, desktop_sharing_recorded=True,
                isolation_limit='Owned UUID lease with pre/post repeat process snapshots; other GPU jobs may share CPU, host memory and PCIe. No continuous foreign-process proof.',
                all_devices_before=all_devices_before,
                all_devices_after=[gpu_snapshot(gpu) for gpu in (PRO4000,*V100)],
                rows=rows, summary=summary, logit_validation=validation,
                protocol_sha256=digest(REF / 'MECHANISM_PROTOCOL.json'), arrays_sha256=protocol['arrays_sha256'],
                source_identity_mode=source_mode, source_code=release['code'],
                release_sha256=digest(ROOT / 'releases/integration-v3.json'),
                helper_sha256={name:digest(Path(__file__).parent / name) for name in
                    ('isolated_inference_timing_v4.py','inference_timing_core_v4.py','checkpoint_mechanism_v4.py','production_resource_preflight_v3.py')},
                elapsed_seconds=time.perf_counter()-started, formal_approved=False)
            write_json(out / 'RESULT.json', payload)
            actual = sum(p.stat().st_size for p in out.iterdir() if p.is_file()); assert actual < reserve
            status('complete', result=str(out / 'RESULT.json'), bytes=actual)
    except BaseException:
        write_json(out / 'FAILURE.json', dict(error=traceback.format_exc(), time=time.time(), pid=os.getpid()))
        status('failed_preserved'); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--gpu',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=10)
    run(parser.parse_args())
