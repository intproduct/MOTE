"""Candidate compact full-state codec. No production trainer changes or retention."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from .state import digest, restore_rng, rng_state, write_json

FORMAT = 'qwen-immutable-base-full-state-v1'


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verified_base(manifest_path, manifest_sha=None):
    manifest_path = Path(manifest_path)
    if manifest_sha is not None and digest(manifest_path) != manifest_sha:
        raise ValueError('Base model manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    base = Path(manifest['path']) / 'model.safetensors'
    expected = next(x for x in manifest['files'] if x['name'] == 'model.safetensors')
    if base.stat().st_size != expected['bytes'] or digest(base) != expected['sha256']:
        raise ValueError('Base model file changed')
    return base, expected


def encode(state, manifest_path):
    base, expected = verified_base(manifest_path)
    trainable = set(state['optimizer']['names'])
    if len(trainable) != len(state['optimizer']['names']) or not trainable <= set(state['model']):
        raise ValueError('Optimizer parameter inventory mismatch')
    frozen = {}
    with safe_open(str(base), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        for name, value in state['model'].items():
            if name in trainable:
                continue
            source_key = name if name in keys else 'model.embed_tokens.weight' if name == 'lm_head.weight' else None
            if source_key not in keys:
                raise ValueError('Unknown frozen tensor: ' + name)
            base_value = source.get_tensor(source_key).float().to(value.dtype)
            if tensor_digest(base_value) != tensor_digest(value):
                raise ValueError('Frozen tensor differs from immutable base: ' + name)
            frozen[name] = dict(source_key=source_key, dtype=str(value.dtype), shape=list(value.shape), sha256=tensor_digest(value))
    trainable_model = OrderedDict((k,v) for k,v in state['model'].items() if k in trainable)
    packed = {k:trainable_model if k == 'model' else v for k,v in state.items()}
    packed['compact_reference'] = dict(format=FORMAT, model_manifest_path=str(manifest_path),
        model_manifest_sha256=digest(manifest_path), base_path=str(base), base_sha256=expected['sha256'],
        frozen=frozen, model_order=list(state['model']), model_metadata=getattr(state['model'], '_metadata', None))
    return packed


def decode(packed):
    reference = packed['compact_reference']
    if reference['format'] != FORMAT:
        raise ValueError('Unknown compact checkpoint format')
    base, expected = verified_base(reference['model_manifest_path'], reference['model_manifest_sha256'])
    if str(base) != reference['base_path'] or expected['sha256'] != reference['base_sha256']:
        raise ValueError('Compact immutable base binding mismatch')
    order = reference['model_order']
    if len(order) != len(set(order)):
        raise ValueError('Repeated model tensor name')
    if set(packed['model']) & set(reference['frozen']) or set(order) != set(packed['model']) | set(reference['frozen']):
        raise ValueError('Incomplete or overlapping compact model inventory')
    restored = OrderedDict()
    with safe_open(str(base), framework='pt', device='cpu') as source:
        for name in order:
            if name in packed['model']:
                restored[name] = packed['model'][name]
            else:
                spec = reference['frozen'][name]
                value = source.get_tensor(spec['source_key']).float().to(getattr(torch, spec['dtype'].removeprefix('torch.')))
                if list(value.shape) != spec['shape'] or tensor_digest(value) != spec['sha256']:
                    raise ValueError('Reconstructed frozen tensor mismatch: ' + name)
                restored[name] = value
    if reference['model_metadata'] is not None:
        restored._metadata = reference['model_metadata']
    state = {k:v for k,v in packed.items() if k != 'compact_reference'}
    state['model'] = restored
    return state


def save_state(path, state, manifest_path, fault=None):
    """Keep incomplete directory as evidence; publish marker last, never overwrite."""
    path = Path(path)
    packed = encode(state, manifest_path)
    path.mkdir(parents=True, exist_ok=False)
    temporary = path / 'training.pt.partial'
    with temporary.open('xb') as f:
        torch.save(packed, f)
        f.flush()
        os.fsync(f.fileno())
    if fault == 'after_payload':
        raise OSError('Injected interruption after payload write')
    target = path / 'training.pt'
    os.replace(temporary, target)
    if fault == 'before_marker':
        raise OSError('Injected interruption before complete marker')
    write_json(path / 'complete.json', dict(schema=1, format=FORMAT, created_epoch=time.time(),
        identity=state['identity'], tokens=state['schedule']['tokens'], cursor=state['cursor'],
        files={'training.pt':dict(sha256=digest(target), bytes=target.stat().st_size)}))


def read_state(path, identity):
    path = Path(path)
    manifest = json.loads((path / 'complete.json').read_text(encoding='utf8'))
    target = path / 'training.pt'
    expected = manifest['files']['training.pt']
    if manifest.get('format') != FORMAT or manifest['identity'] != identity:
        raise ValueError('Compact checkpoint identity or format mismatch')
    if target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
        raise ValueError('Compact checkpoint payload hash mismatch')
    packed = torch.load(target, map_location='cpu', weights_only=False, mmap=True)
    if packed['identity'] != identity or packed['schedule']['tokens'] != manifest['tokens'] or packed['cursor'] != manifest['cursor']:
        raise ValueError('Compact checkpoint embedded metadata mismatch')
    return decode(packed)


def load_checkpoint(path, model, optimizer, schedule, scaler, identity):
    state = read_state(path, identity)
    if state.get('extra', {}).get('automatic_resume_permitted') is False:
        raise ValueError('Failure checkpoint requires explicit recovery review; automatic resume refused')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['schedule'])
    if state['scaler'] is not None:
        if scaler is None:
            raise ValueError('Compact checkpoint requires GradScaler')
        scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])
    return state['cursor'], state['extra']


def save_checkpoint(path, model, optimizer, schedule, scaler, cursor, identity, extra=None, *, manifest_path):
    state = dict(schema=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        schedule=schedule.state_dict(), scaler=scaler.state_dict() if scaler is not None else None,
        rng=rng_state(), cursor=cursor, identity=identity, extra=extra or {})
    save_state(path, state, manifest_path)
