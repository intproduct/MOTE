"""Read-only projection precision probes on frozen Dense reference inputs.

Reference arithmetic uses the actual stored native checkpoint values restored
to FP32. Raw records retain sufficient statistics, routes and per-anchor errors;
high-dimensional activations remain reproducible from checkpoint plus inputs.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

ROOT = Path('E:/AI/MOTE-quantum')
REF = ROOT / 'diagnostics/mechanism-v1-20261031'


def compact_error(reference, variant):
    import numpy as np
    import torch
    if reference.shape != variant.shape or reference.ndim not in (2, 3) or not reference.numel():
        raise ValueError('Aligned nonempty [anchor,feature] or [expert,anchor,feature] arrays required')
    a = reference.detach().double().cpu(); b = variant.detach().double().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite function output')
    delta = b - a
    raw = dict(reference_energy=a.square().sum(-1).numpy(), variant_energy=b.square().sum(-1).numpy(),
        error_energy=delta.square().sum(-1).numpy(), error_max_abs=delta.abs().amax(-1).numpy())
    ref_norm = np.sqrt(raw['reference_energy'].sum()); error_norm = np.sqrt(raw['error_energy'].sum())
    summary = dict(shape=list(a.shape), output_features=a.shape[-1], relative_l2=float(error_norm / max(ref_norm, 1e-12)),
        error_rms=float(error_norm / np.sqrt(a.numel())), reference_rms=float(ref_norm / np.sqrt(a.numel())),
        max_abs_error=float(raw['error_max_abs'].max()))
    return raw, summary


def capture_projection(module, inputs, native_dtype=None):
    import torch
    native = native_dtype is not None
    if native and inputs.device.type != 'cuda':
        raise ValueError('Native GPU arithmetic requires CUDA; CPU quantization is separately labeled')
    if native_dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        raise ValueError('Native BF16 unsupported on this device')
    if native_dtype not in (None, torch.float16, torch.bfloat16):
        raise ValueError('Unsupported native precision')
    captured = []
    handle = None
    routed = hasattr(module, 'router')
    if routed:
        handle = module.router.register_forward_hook(lambda _mod, _args, result: captured.append(result))
    try:
        with torch.no_grad(), torch.autocast(device_type=inputs.device.type, dtype=native_dtype, enabled=native):
            experts = torch.stack([expert(inputs[:, :module.main_in]) for expert in module.experts]) if routed else None
            output = module(inputs)
        value = dict(projection=output.float().cpu())
        if routed:
            if len(captured) != 1: raise RuntimeError('Expected exactly one actual sparse router call')
            route = captured[0]
            value.update(experts=experts.float().cpu(), indices=route.indices.cpu(), weights=route.weights.cpu(),
                probabilities=route.probabilities.cpu())
        if not all(torch.isfinite(v).all() for v in value.values()): raise ValueError('Nonfinite projection')
        return value
    finally:
        if handle is not None: handle.remove()


def comparisons(conditions, native_dtype_name=None):
    """Compare both input roundtrips to FP32; native also to aligned FP32 input."""
    names = ['stored_reference_fp32', 'input_float16_roundtrip', 'input_bfloat16_roundtrip']
    if native_dtype_name: names.append('native_' + native_dtype_name)
    if set(conditions) != set(names): raise ValueError('Incomplete or unknown condition set')
    pairs = [(name, names[0]) for name in names[1:]]
    if native_dtype_name: pairs.append((names[-1], 'input_' + native_dtype_name + '_roundtrip'))
    raw = {}; rows = []
    routed = 'indices' in conditions[names[0]]
    for index, name in enumerate(names):
        condition = conditions[name]
        if ('indices' in condition) != routed: raise ValueError('Routed and Dense conditions mixed')
        if routed:
            for key in ('indices', 'weights', 'probabilities'):
                raw[f'condition_{index}_{key}'] = condition[key].numpy()
    for index, (variant, reference) in enumerate(pairs):
        left, right = conditions[reference], conditions[variant]
        row = dict(variant=variant, reference=reference, statistics={})
        for key in ('projection', 'experts') if routed else ('projection',):
            records, summary = compact_error(left[key], right[key])
            raw.update({f'comparison_{index}_{key}_{name}': value for name, value in records.items()})
            row['statistics'][key] = summary
        if routed:
            import torch
            switched = (left['indices'].sort(-1).values != right['indices'].sort(-1).values).any(-1)
            row.update(switched=int(switched.sum()), anchors=len(switched),
                route_indices_exact=torch.equal(left['indices'], right['indices']),
                route_weights_exact=torch.equal(left['weights'], right['weights']),
                route_probabilities_exact=torch.equal(left['probabilities'], right['probabilities']))
        rows.append(row)
    if native_dtype_name and routed:
        if not all(rows[-1][name] for name in ('route_indices_exact', 'route_weights_exact', 'route_probabilities_exact')):
            raise ValueError('FP32 routers differ for identical weights and aligned native inputs')
    return raw, rows, names


def probe_projection(module, inputs, native_dtype, include_native):
    import torch
    from fitmotn.experiments.qwen_joint.model import set_model_precision, tensor_hash
    device = next(module.parameters()).device
    baseline = tensor_hash(module.state_dict())
    router_hash = tensor_hash(module.router.state_dict()) if hasattr(module, 'router') else None
    conditions = {}
    names = ['stored_reference_fp32', 'input_float16_roundtrip', 'input_bfloat16_roundtrip']
    if include_native: names.append('native_' + str(native_dtype).split('.')[-1])
    try:
        for name in names:
            native = name.startswith('native_')
            cast = native_dtype if native else torch.float16 if 'float16_roundtrip' in name and 'bfloat16' not in name else torch.bfloat16 if 'bfloat16_roundtrip' in name else None
            # Both roundtrips happen on CPU, including BF16 on V100.
            x = inputs.cpu().to(cast).float() if cast else inputs.cpu().float()
            if not torch.isfinite(x).all(): raise ValueError('Input quantization overflow')
            if native: set_model_precision(module, native_dtype, device)
            x = x.to(device=device, dtype=native_dtype if native else torch.float32)
            conditions[name] = capture_projection(module, x, native_dtype if native else None)
            if router_hash is not None and tensor_hash(module.router.state_dict()) != router_hash:
                raise ValueError('Diagnostic changed FP32 router values')
        records, rows, names = comparisons(conditions, str(native_dtype).split('.')[-1] if include_native else None)
        return records, rows, names
    finally:
        set_model_precision(module, torch.float32, device)
        if tensor_hash(module.state_dict()) != baseline:
            raise ValueError('Diagnostic changed stored checkpoint values')


def run(args):
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu or ''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from contextlib import nullcontext
    from types import SimpleNamespace
    from fitmotn.experiments.qwen_joint.acceptance import source_identity
    from fitmotn.experiments.qwen_joint.resources import GPULease, PRO4000, V100
    from fitmotn.experiments.qwen_joint.state import write_json
    from checkpoint_mechanism_v4 import read, digest, load_checkpoint
    from production_resource_preflight_v3 import check as resource_check
    out = Path(args.output)
    if Path.cwd().resolve() != Path('D:/AI/MOTE').resolve(): raise ValueError('Wrong experiment workspace')
    if not out.resolve().is_relative_to((ROOT / 'diagnostics').resolve()): raise ValueError('Output outside diagnostic root')
    if gpu and gpu not in (PRO4000, *V100): raise ValueError('Unknown physical GPU')
    out.mkdir(parents=True, exist_ok=False)
    def status(phase, **extra):
        write_json(out / 'STATUS.json', dict(phase=phase, pid=os.getpid(), gpu=gpu, gpu_used=bool(gpu), time=time.time(), **extra))
    def stop_check():
        if any(p.exists() for p in (ROOT / 'STOP', ROOT / 'queue-20261030/STOP', out / 'STOP')):
            raise RuntimeError('STOP marker')
        if not gpu and torch.cuda.is_initialized(): raise RuntimeError('Unexpected CUDA initialization in CPU mode')
    started = time.perf_counter()
    try:
        stop_check(); status('verifying_inputs')
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        release = read(ROOT / 'releases/integration-v3.json'); assert source_identity() == release['code']
        budget = resource_check(); reserve = 30_000_000
        assert budget['acceptance_launch_fits'] and budget['processing_actual_plus_reserved_bytes'] + reserve <= 20_000_000_000
        budget['additional_diagnostic_reservation_bytes'] = reserve; write_json(out / 'RESERVATION.json', budget)
        seed_path = ROOT / 'initialization/seed-release-v2-20261031/SEED_RELEASE.json'
        assert digest(seed_path) == '2f88b8735ec4b4303af3578caf0c228a33a3ab697acf8b3466312996b4648f71'
        seeds = read(seed_path); packed = None
        if args.checkpoint:
            marker, packed, mode = load_checkpoint(args.checkpoint, release); identity = marker['identity']
            native = torch.bfloat16 if identity['gpu'] == PRO4000 else torch.float16
            assert {v.dtype for name, v in packed['model'].items() if '.router.' not in name} == {native}
            binding = dict(checkpoint=str(args.checkpoint), marker_sha256=digest(Path(args.checkpoint) / 'complete.json'),
                payload_sha256=marker['files']['training.pt']['sha256'], tokens=marker['tokens'], source_mode=mode)
        else:
            if args.method not in ('mlp', 'affine_born', 'reencode_cz', 'reencode_product', 'dense') or args.seed not in (17,29,43):
                raise ValueError('Initial diagnostics require explicit approved method and seed')
            from fitmotn.experiments.qwen_joint.checkpoint import verified_base
            verified_base(ROOT / 'queue-20261030/model-manifest.json', release['model_manifest_sha256'])
            native = torch.bfloat16 if gpu == PRO4000 or not gpu and args.native_precision == 'bfloat16' else torch.float16
            identity = dict(method=args.method, seed=args.seed)
            binding = dict(tokens=0, source_mode='initial_common_experts_with_native_weight_rounding')
        if gpu and native != (torch.bfloat16 if gpu == PRO4000 else torch.float16):
            raise ValueError('Checkpoint precision cohort differs from diagnostic GPU')
        initial = next(row for row in seeds['initializations'] if row['seed'] == identity['seed'])
        assert digest(initial['common_path']) == initial['checkpoint_sha256']
        if packed is not None: assert identity['common_sha256'] == (None if identity['method'] == 'dense' else initial['checkpoint_sha256'])
        common = torch.load(initial['common_path'], map_location='cpu', weights_only=False, mmap=True)
        if packed is None: identity['adapter'] = common['adapter']
        reference = read(REF / 'REFERENCE_ACTIVATIONS.json'); protocol = read(REF / 'MECHANISM_PROTOCOL.json')
        assert digest(REF / 'dense_reference.pt') == reference['sha256']
        assert digest(REF / 'MECHANISM_PROTOCOL.json') == reference['protocol_sha256']
        assert protocol['model_manifest_sha256'] == release['model_manifest_sha256'] and protocol['data_release_sha256'] == release['data_sha256']
        anchors = torch.load(REF / 'dense_reference.pt', map_location='cpu', weights_only=False)['inputs']
        lease_context = GPULease(ROOT, gpu, 'local_checkpoint_precision_v4', out) if gpu else nullcontext()
        with lease_context as lease:
            from fitmotn.experiments.qwen_joint.model import tensor_hash
            from checkpoint_mechanism_v4 import projection, initial_values
            if gpu:
                from fitmotn.experiments.qwen_joint.reproducibility import configure
                from fitmotn.experiments.qwen_joint.train import make_guard
                configure(); guard = make_guard(SimpleNamespace(root=ROOT, gpu=gpu), out)
            else: guard = None
            assert tensor_hash(common['state']) == initial['state_hash']
            rows = []
            for name in protocol['reference_activations']['projections']:
                stop_check()
                if guard: guard.check('diagnostic')
                status('projection_precision', projection=name, method=identity['method'], tokens=binding['tokens'])
                x = anchors[name]; assert tensor_hash({'input': x}) == reference['rows'][name]['sha256']
                values = packed['model'] if packed is not None else initial_values(name, identity, common, native)
                module = projection(name, identity, common, values).to('cuda' if gpu else 'cpu').eval()
                raw, comparisons_out, variants = probe_projection(module, x, native, include_native=bool(gpu))
                assert all(np.isfinite(value).all() for value in raw.values())
                path = out / (name + '.npz')
                with path.open('xb') as file: np.savez(file, **raw)
                row = dict(projection=name, input_sha256=reference['rows'][name]['sha256'],
                    routing_applicable=identity['method'] != 'dense', variants=variants, comparisons=comparisons_out,
                    raw_path=str(path), raw_sha256=digest(path), raw_bytes=path.stat().st_size)
                rows.append(row); write_json(out / (name + '.json'), row); del module
            stop_check(); assert source_identity() == release['code']
            result = dict(passed=True, phase='complete', method=identity['method'], seed=identity['seed'], binding=binding,
                native_checkpoint_precision=str(native), rows=rows, gpu_used=bool(gpu),
                native_arithmetic_measured=bool(gpu), physical_device=lease.record['cuda_device'] if lease else None,
                lease=lease.record if lease else None, code=release['code'], release_sha256=digest(ROOT / 'releases/integration-v3.json'),
                seed_release_sha256=digest(seed_path), common_sha256=initial['checkpoint_sha256'],
                protocol_sha256=reference['protocol_sha256'], reference_sha256=reference['sha256'],
                reference='Actual stored native expert/residual/tail and FP32 router values restored to FP32; not recovered master parameters',
                inputs='Same 64 frozen Dense anchors; FP16/BF16 input roundtrips performed on CPU',
                raw_scope='Per-expert/per-anchor squared norms and differences, maximum absolute errors, all route indices/weights/probabilities; full activations reproducible from retained checkpoints and anchors',
                isolated_timing=False, formal_approved=False, elapsed_seconds=time.perf_counter()-started,
                helper_sha256={name:digest(Path(__file__).parent / name) for name in ('local_checkpoint_precision_v4.py', 'checkpoint_mechanism_v4.py', 'production_resource_preflight_v3.py')})
            write_json(out / 'RESULT.json', result)
            actual = sum(p.stat().st_size for p in out.iterdir() if p.is_file()); assert actual < reserve
            status('complete', bytes=actual, native_arithmetic_measured=bool(gpu))
            print(dict(output=str(out), method=identity['method'], tokens=binding['tokens'], seconds=result['elapsed_seconds'], bytes=actual))
    except BaseException:
        write_json(out / 'FAILURE.json', dict(error=traceback.format_exc(), pid=os.getpid(), time=time.time()))
        status('failed_preserved'); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint'); source.add_argument('--initial', action='store_true')
    parser.add_argument('--method'); parser.add_argument('--seed', type=int)
    parser.add_argument('--native-precision', choices=('float16','bfloat16'), default='float16')
    parser.add_argument('--gpu'); parser.add_argument('--output', required=True)
    run(parser.parse_args())
