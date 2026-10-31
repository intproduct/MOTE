import copy
from collections import OrderedDict
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytest
import torch
from safetensors.torch import save_file
from fitmotn.experiments.qwen_joint import checkpoint as codec


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    frozen = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    save_file({'model.embed_tokens.weight': frozen}, str(base / 'model.safetensors'))
    manifest = tmp_path / 'model-manifest.json'
    codec.write_json(manifest, dict(path=str(base), files=[dict(name='model.safetensors',
        sha256=codec.digest(base / 'model.safetensors'), bytes=(base / 'model.safetensors').stat().st_size)]))
    model = OrderedDict([('model.embed_tokens.weight', frozen.float().half()),
        ('adapter.weight', torch.ones(2, 4)), ('lm_head.weight', frozen.float().half())])
    model._metadata = {'': {'version': 1}, 'adapter': {'version': 1}}
    state = dict(schema=1, model=model, optimizer={'names':['adapter.weight'], 'masters':[torch.ones(2,4)]},
        schedule={'tokens':32768}, scaler={'scale':1024.0}, rng={'torch':torch.get_rng_state()},
        cursor={'position':32,'offset':17}, identity={'method':'fixture'}, extra={'retained':True})
    return tmp_path, manifest, state


def test_roundtrip_and_metadata(fixture):
    root, manifest, state = fixture
    codec.save_state(root / 'good', state, manifest)
    actual = codec.read_state(root / 'good', state['identity'])
    assert list(actual) == list(state)
    assert list(actual['model']) == list(state['model'])
    assert actual['model']._metadata == state['model']._metadata
    assert all(torch.equal(actual['model'][n],v) and actual['model'][n].dtype == v.dtype for n,v in state['model'].items())
    assert torch.equal(actual['optimizer']['masters'][0], state['optimizer']['masters'][0])
    assert torch.equal(actual['rng']['torch'], state['rng']['torch'])
    assert actual['cursor'] == state['cursor'] and actual['scaler'] == state['scaler']
    assert actual['extra'] == state['extra']


@pytest.mark.parametrize('point', ['after_payload', 'before_marker'])
def test_interrupted_write_has_no_commit_marker(fixture, point):
    root, manifest, state = fixture
    with pytest.raises(OSError, match='Injected'):
        codec.save_state(root / point, state, manifest, fault=point)
    assert not (root / point / 'complete.json').exists()
    assert list((root / point).iterdir())
    with pytest.raises(FileNotFoundError):
        codec.read_state(root / point, state['identity'])


def test_no_overwrite(fixture):
    root, manifest, state = fixture
    codec.save_state(root / 'good', state, manifest)
    before = codec.digest(root / 'good/training.pt')
    with pytest.raises(FileExistsError):
        codec.save_state(root / 'good', state, manifest)
    assert codec.digest(root / 'good/training.pt') == before


def test_identity_and_payload_corruption(fixture):
    root, manifest, state = fixture
    codec.save_state(root / 'good', state, manifest)
    with pytest.raises(ValueError, match='identity'):
        codec.read_state(root / 'good', {'method':'wrong'})
    target = root / 'good/training.pt'
    with target.open('r+b') as f:
        f.seek(100)
        value = f.read(1)
        f.seek(100)
        f.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match='payload hash'):
        codec.read_state(root / 'good', state['identity'])


def test_base_change_detected(fixture):
    root, manifest, state = fixture
    packed = codec.encode(state, manifest)
    with (root / 'base/model.safetensors').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Base model file changed'):
        codec.decode(packed)


def test_manifest_change_detected(fixture):
    root, manifest, state = fixture
    packed = codec.encode(state, manifest)
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest changed'):
        codec.decode(packed)


def test_modified_frozen_tensor_rejected(fixture):
    root, manifest, state = fixture
    state['model']['model.embed_tokens.weight'][0,0] = 123
    with pytest.raises(ValueError, match='Frozen tensor differs'):
        codec.encode(state, manifest)


def test_incomplete_inventory_rejected(fixture):
    root, manifest, state = fixture
    packed = codec.encode(state, manifest)
    del packed['compact_reference']['frozen']['lm_head.weight']
    with pytest.raises(ValueError, match='inventory'):
        codec.decode(packed)


def test_wrong_frozen_tensor_hash_rejected(fixture):
    root, manifest, state = fixture
    packed = codec.encode(state, manifest)
    packed['compact_reference']['frozen']['lm_head.weight']['sha256'] = 'bad'
    with pytest.raises(ValueError, match='Reconstructed frozen tensor mismatch'):
        codec.decode(packed)


def test_failed_state_is_readable_but_not_automatically_resumable(fixture):
    root, manifest, state = fixture
    state['extra']['automatic_resume_permitted'] = False
    codec.save_state(root/'failed',state,manifest)
    assert codec.read_state(root/'failed',state['identity'])['extra']['retained']
    with pytest.raises(ValueError,match='recovery review'):
        codec.load_checkpoint(root/'failed',None,None,None,None,state['identity'])
