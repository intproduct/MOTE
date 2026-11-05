"""Read-only checkpoint-local diagnostics on frozen Dense anchors; CPU only.

Keep raw distances, per-anchor energies, routes and geometry. Full activation
tensors are recomputable from the retained checkpoint and are not duplicated.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path('E:/AI/MOTE-quantum')
REF = ROOT / 'diagnostics/mechanism-v1-20261031'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def canonical_identity(identity, release, checkpoint):
    """Current identities are exact; staged aliases need an audited payload."""
    if identity['code'] == release['code']:
        return 'promoted_checkout'
    staged_path = Path(release['staged_release'])
    if digest(staged_path) != release['staged_release_sha256']:
        raise ValueError('Staged release changed')
    staged = read(staged_path)
    if identity['code'] != staged['code']:
        raise ValueError('Unknown checkpoint source identity')
    # The original alias inventory is accepted only for already independently
    # audited recovery artifacts, never for a new unbound checkpoint.
    payload = str(Path(checkpoint).resolve() / 'training.pt')
    for cohort in ('pro4000', 'v100b'):
        audit = read(ROOT / 'runtime' / f'production-replay-artifact-audit-v3-{cohort}.json')
        if not audit['passed']:
            continue
        for row in audit['rows']:
            for item in row['checkpoint_files']:
                if str(Path(item['path']).resolve()) == payload and digest(payload) == item['sha256']:
                    return 'audited_staged_bytes_promoted_unchanged'
    raise ValueError('Staged checkpoint lacks an audited payload binding')


def validate_metadata(marker, packed, release):
    identity = marker['identity']
    if marker.get('format') != 'qwen-immutable-base-full-state-v1':
        raise ValueError('Expected a complete immutable-base checkpoint')
    if packed['identity'] != identity or marker['tokens'] != packed['schedule']['tokens'] or marker['cursor'] != packed['cursor']:
        raise ValueError('Checkpoint metadata mismatch')
    if packed.get('extra', {}).get('automatic_resume_permitted') is False:
        raise ValueError('Failure state requires separate explicit review')
    for key in ('model_manifest_sha256', 'data_sha256', 'config_sha256'):
        if identity[key] != release[key]:
            raise ValueError('Checkpoint release mismatch: ' + key)
    if identity['method'] not in ('mlp', 'affine_born', 'reencode_cz', 'reencode_product', 'dense'):
        raise ValueError('Unknown method')
    if identity['seed'] not in (17, 29, 43):
        raise ValueError('Unknown seed')
    if set(packed['model']) != set(packed['optimizer']['names']):
        raise ValueError('Incomplete trainable parameter inventory')
    if set(packed['model']) & set(packed['compact_reference']['frozen']):
        raise ValueError('Trainable/frozen overlap')
    return identity


def load_checkpoint(path, release):
    import torch
    from fitmotn.experiments.qwen_joint.checkpoint import verified_base
    path = Path(path)
    marker = read(path / 'complete.json')
    spec = marker['files']['training.pt']
    payload = path / 'training.pt'
    if payload.stat().st_size != spec['bytes'] or digest(payload) != spec['sha256']:
        raise ValueError('Checkpoint payload changed')
    packed = torch.load(payload, map_location='cpu', weights_only=False, mmap=True)
    identity = validate_metadata(marker, packed, release)
    mode = canonical_identity(identity, release, path)
    base, expected = verified_base(ROOT / 'queue-20261030/model-manifest.json', identity['model_manifest_sha256'])
    reference = packed['compact_reference']
    if str(base) != reference['base_path'] or expected['sha256'] != reference['base_sha256']:
        raise ValueError('Immutable model binding mismatch')
    return marker, packed, mode


def projection(name, identity, common, values):
    import torch
    from fitmotn.experiments.qwen_joint.model import AdapterConfig, JointProjection
    layer = int(name.split('.')[2]); kind = name.split('.')[-1]
    offset = ('gate_proj', 'up_proj', 'down_proj').index(kind)
    inputs, outputs = (3072, 1024) if kind == 'down_proj' else (1024, 3072)
    if identity['method'] == 'dense':
        module = torch.nn.Linear(inputs, outputs, bias=False)
    else:
        module = JointProjection.from_spec(inputs, outputs, identity['method'], AdapterConfig(**identity['adapter']),
            layer * 100 + offset * 10, common['projections'][f'{layer}.{kind}'])
    subset = {key[len(name) + 1:]: value.float() for key, value in values.items() if key.startswith(name + '.')}
    module.load_state_dict(subset, strict=True)
    module.eval()
    return module


def initial_values(name, identity, common, native_dtype):
    import torch
    from safetensors import safe_open
    from fitmotn.experiments.qwen_joint.model import AdapterConfig, JointProjection
    layer = int(name.split('.')[2]); kind = name.split('.')[-1]
    offset = ('gate_proj', 'up_proj', 'down_proj').index(kind)
    inputs, outputs = (3072, 1024) if kind == 'down_proj' else (1024, 3072)
    if identity['method'] == 'dense':
        manifest = read(ROOT / 'queue-20261030/model-manifest.json')
        with safe_open(str(Path(manifest['path']) / 'model.safetensors'), framework='pt', device='cpu') as source:
            return {name + '.weight': source.get_tensor(name + '.weight').to(native_dtype).float()}
    module = JointProjection.from_spec(inputs, outputs, identity['method'], AdapterConfig(**identity['adapter']),
        layer * 100 + offset * 10, common['projections'][f'{layer}.{kind}'])
    values = {name + '.' + key: value for key, value in module.state_dict().items() if key.startswith('router.')}
    values.update({key: value.to(native_dtype).float() for key, value in common['state'].items() if key.startswith(name + '.')})
    return values


def local_measurements(module, initial, x, method):
    import torch
    from fitmotn.experiments.qwen_joint.routers import reencoded_state
    from mechanism_metrics_candidate_v1 import functional_distances, match_functions, real_state_qfi, sparse_feature_jacobian
    raw = {}
    with torch.no_grad():
        actual = module(x); baseline = initial(x)
        raw.update(projection_energy=actual.double().square().sum(-1).numpy(),
                   initial_projection_energy=baseline.double().square().sum(-1).numpy(),
                   projection_delta_energy=(actual.double() - baseline.double()).square().sum(-1).numpy(),
                   projection_delta_max_abs=(actual - baseline).abs().amax(-1).numpy())
        if method == 'dense':
            return raw, dict(routing_applicable=False, output_features=actual.shape[-1])
        outputs = torch.stack([expert(x[:, :module.main_in]) for expert in module.experts])
        first = torch.stack([expert(x[:, :initial.main_in]) for expert in initial.experts])
        distance = functional_distances(outputs); matched = match_functions(first, outputs)
        route = module.router(x); initial_route = initial.router(x)
        raw.update(function_distance=distance.numpy(), initial_current_distance=functional_distances(first, outputs).numpy(),
                   expert_energy=outputs.double().square().sum(-1).numpy(),
                   initial_expert_energy=first.double().square().sum(-1).numpy(),
                   indices=route.indices.numpy(), weights=route.weights.numpy(), probabilities=route.probabilities.numpy(),
                   initial_indices=initial_route.indices.numpy(), initial_weights=initial_route.weights.numpy())
        features = torch.tanh(module.router.input_proj(module.router.norm(x[:4])))
        logits = module.router.logits(x[:4])
    geometry = []
    for row, feature in enumerate(features):
        jac = sparse_feature_jacobian(module.router, feature)
        matrix = jac.pop('jacobian'); raw[f'jacobian_{row}'] = matrix.numpy()
        item = dict(anchor_row=row, feature=feature.tolist(), logits=logits[row].tolist(), **jac)
        if method.startswith('reencode'):
            with torch.no_grad():
                angles = feature.new_tensor([math.pi / 2] * 4 + [.35] * 4) + math.pi / 2 * module.router.output(feature).tanh()
            qfi = real_state_qfi(lambda a: reencoded_state(a, method == 'reencode_cz'), angles)
            raw[f'angle_qfi_{row}'] = qfi.numpy()
            item['angle_qfi'] = dict(angles=angles.tolist(), eigenvalues=torch.linalg.eigvalsh(qfi).tolist(), trace=float(qfi.trace()))
        geometry.append(item)
    off = distance[~torch.eye(len(distance), dtype=torch.bool)]
    summary = dict(routing_applicable=True, expert_output_features=outputs.shape[-1], projection_output_features=actual.shape[-1],
        off_diagonal=dict(mean=float(off.mean()), median=float(torch.quantile(off, .5)), min=float(off.min()), max=float(off.max())),
        initial_current_function_matching=matched, router_geometry=geometry,
        initial_current_topk_switches=int((route.indices.sort(-1).values != initial_route.indices.sort(-1).values).any(-1).sum()),
        anchor_count=len(x), coordinates='angle-state QFI, feature-weight Jacobian, function distances are separate')
    return raw, summary


def run(args):
    import numpy as np
    import torch
    from fitmotn.experiments.qwen_joint.acceptance import source_identity
    from fitmotn.experiments.qwen_joint.model import tensor_hash
    from fitmotn.experiments.qwen_joint.state import write_json
    from production_resource_preflight_v3 import check as resource_check
    out = Path(args.output)
    if not out.resolve().is_relative_to((ROOT / 'diagnostics').resolve()):
        raise ValueError('Diagnostics must remain under the experiment data root')
    out.mkdir(parents=True, exist_ok=False)
    def status(phase, **extra):
        write_json(out / 'STATUS.json', dict(phase=phase, pid=os.getpid(), time=time.time(), gpu_used=False, **extra))
    def check():
        if torch.cuda.is_initialized() or any(p.exists() for p in (ROOT / 'STOP', ROOT / 'queue-20261030/STOP', out / 'STOP')):
            raise RuntimeError('STOP marker or unexpected GPU initialization')
    started = time.perf_counter()
    try:
        check(); status('verifying_checkpoint')
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        release = read(ROOT / 'releases/integration-v3.json')
        assert source_identity() == release['code']
        budget = resource_check(); reserve = 20_000_000
        assert budget['acceptance_launch_fits'] and budget['processing_actual_plus_reserved_bytes'] + reserve < 20_000_000_000
        budget.update(additional_diagnostic_reservation_bytes=reserve); write_json(out / 'RESERVATION.json', budget)
        marker, packed, mode = load_checkpoint(args.checkpoint, release); identity = marker['identity']
        seeds = read(ROOT / 'initialization/seed-release-v2-20261031/SEED_RELEASE.json')
        initial = next(item for item in seeds['initializations'] if item['seed'] == identity['seed'])
        assert digest(initial['common_path']) == initial['checkpoint_sha256']
        assert identity['common_sha256'] == (None if identity['method'] == 'dense' else initial['checkpoint_sha256'])
        common = torch.load(initial['common_path'], map_location='cpu', weights_only=False, mmap=True)
        assert tensor_hash(common['state']) == initial['state_hash']
        reference = read(REF / 'REFERENCE_ACTIVATIONS.json'); protocol = read(REF / 'MECHANISM_PROTOCOL.json')
        assert digest(REF / 'dense_reference.pt') == reference['sha256']
        assert digest(REF / 'MECHANISM_PROTOCOL.json') == reference['protocol_sha256']
        anchors = torch.load(REF / 'dense_reference.pt', map_location='cpu', weights_only=False)['inputs']
        rows = []
        for name in protocol['reference_activations']['projections']:
            check(); status('projection_diagnostics', projection=name, method=identity['method'], tokens=marker['tokens'])
            x = anchors[name]; assert tensor_hash({'input': x}) == reference['rows'][name]['sha256']
            native_dtype = next(value.dtype for key, value in packed['model'].items() if key.startswith(name + '.') and '.router.' not in key)
            assert native_dtype in (torch.float16, torch.bfloat16)
            module = projection(name, identity, common, packed['model'])
            baseline = projection(name, identity, common, initial_values(name, identity, common, native_dtype))
            raw, summary = local_measurements(module, baseline, x, identity['method'])
            assert all(np.isfinite(value).all() for value in raw.values())
            path = out / (name + '.npz')
            with path.open('xb') as file: np.savez(file, **raw)
            row = dict(projection=name, input_sha256=reference['rows'][name]['sha256'], native_checkpoint_dtype=str(native_dtype),
                raw_path=str(path), raw_sha256=digest(path), raw_bytes=path.stat().st_size, **summary)
            rows.append(row); write_json(out / (name + '.json'), row)
            del module, baseline, raw
        assert len(rows) == 21 and source_identity() == release['code']
        payload = dict(passed=True, phase='complete', checkpoint=str(Path(args.checkpoint)), checkpoint_payload_sha256=marker['files']['training.pt']['sha256'],
            checkpoint_marker_sha256=digest(Path(args.checkpoint) / 'complete.json'), method=identity['method'], seed=identity['seed'],
            tokens=marker['tokens'], training_gpu=identity['gpu'], source_identity_mode=mode, source_code=release['code'], rows=rows,
            native_training_precision=str(native_dtype), evaluation_precision='float32_cpu_from_stored_native_checkpoint_values',
            initial_reference='same seed initialization, experts rounded to checkpoint native dtype then FP32; FP32 routers remain exact',
            protocol_sha256=reference['protocol_sha256'], reference_activations_sha256=reference['sha256'],
            runner_sha256=digest(__file__), metrics_sha256=digest(ROOT / 'runtime/mechanism_metrics_candidate_v1.py'),
            raw_scope='All 16x16 distance matrices, 64-anchor expert energies and routes, four per-projection Jacobians/QFI. Full activations are recomputable from the retained checkpoint and anchors, not duplicated.',
            elapsed_seconds=time.perf_counter()-started, gpu_used=False, isolated_gpu_timing=False, formal_approved=False)
        write_json(out / 'RESULT.json', payload)
        actual = sum(p.stat().st_size for p in out.iterdir() if p.is_file()); assert actual < reserve
        status('complete', result=str(out / 'RESULT.json'), bytes=actual)
        print(json.dumps(dict(output=str(out), method=identity['method'], tokens=marker['tokens'], bytes=actual, elapsed_seconds=payload['elapsed_seconds'])))
    except BaseException:
        write_json(out / 'FAILURE.json', dict(error=traceback.format_exc(), time=time.time(), pid=os.getpid()))
        status('failed_preserved'); raise


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--output', required=True)
    run(parser.parse_args())
