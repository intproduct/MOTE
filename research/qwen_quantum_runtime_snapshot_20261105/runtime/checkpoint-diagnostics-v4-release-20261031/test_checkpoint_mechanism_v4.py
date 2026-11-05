import copy
import numpy as np
import pytest
import torch
from checkpoint_mechanism_v4 import local_measurements, validate_metadata
from fitmotn.experiments.qwen_joint.model import AdapterConfig, JointProjection


@pytest.mark.parametrize('method', ['mlp', 'affine_born', 'reencode_cz', 'reencode_product'])
def test_local_records_match_independent_direct_functions_and_leave_weights_unchanged(method):
    torch.manual_seed(17)
    baseline = JointProjection(torch.nn.Linear(128, 192, bias=False), method,
        AdapterConfig(residual_rank=4, tail_rank=4), 3)
    module = copy.deepcopy(baseline)
    with torch.no_grad():
        module.experts[0].u.mul_(1.1)
        module.router.log_scale.add_(.02)
    before = {name: value.clone() for name, value in module.state_dict().items()}
    x = torch.randn(8, 128)
    raw, result = local_measurements(module, baseline, x, method)
    with torch.no_grad():
        functions = torch.stack([expert(x) for expert in module.experts]).double().numpy()
        a, b = functions[0], functions[1]
        expected = np.sqrt(np.mean((a-b)**2)) / np.sqrt((np.mean(a*a)+np.mean(b*b))/2)
        assert raw['function_distance'][0, 1] == pytest.approx(expected, abs=1e-12)
        np.testing.assert_allclose(raw['expert_energy'], (functions*functions).sum(-1), rtol=1e-12)
        output = module(x).double().numpy(); initial = baseline(x).double().numpy()
        np.testing.assert_allclose(raw['projection_delta_energy'], ((output-initial)**2).sum(-1), rtol=1e-12)
    assert len(result['router_geometry']) == 4
    assert all(torch.equal(value, module.state_dict()[name]) for name, value in before.items())
    assert all(np.isfinite(value).all() for value in raw.values())
    for row in range(4):
        np.testing.assert_allclose(raw[f'jacobian_{row}'].sum(0), 0, atol=1e-7)
        if method.startswith('reencode'):
            assert np.linalg.eigvalsh(raw[f'angle_qfi_{row}']).min() >= -1e-10


def test_dense_records_have_no_fabricated_experts():
    module = torch.nn.Linear(8, 5, bias=False)
    baseline = copy.deepcopy(module)
    raw, result = local_measurements(module, baseline, torch.randn(7, 8), 'dense')
    assert not result['routing_applicable'] and 'indices' not in raw
    assert np.array_equal(raw['projection_delta_energy'], np.zeros(7))


@pytest.mark.parametrize('field', ['identity', 'tokens', 'cursor', 'format', 'failure', 'trainable_inventory', 'base_overlap'])
def test_checkpoint_mismatches_and_failure_states_refused(field):
    release = {'config_sha256':'c', 'data_sha256':'d', 'model_manifest_sha256':'m'}
    identity = dict(release, method='mlp', seed=17)
    marker = dict(identity=identity, tokens=7, cursor={'next':3}, format='qwen-immutable-base-full-state-v1')
    packed = dict(identity=copy.deepcopy(identity), schedule={'tokens':7}, cursor={'next':3},
        model={'trainable':1}, optimizer={'names':['trainable']}, compact_reference={'frozen':{'fixed':{}}})
    assert validate_metadata(marker, packed, release) == identity
    if field == 'identity':packed['identity']['seed'] = 29
    elif field == 'tokens':marker['tokens'] += 1
    elif field == 'cursor':packed['cursor']['next'] += 1
    elif field == 'format':marker['format'] = 'other'
    elif field == 'failure':packed['extra'] = {'automatic_resume_permitted':False}
    elif field == 'trainable_inventory':packed['model']['another'] = 2
    else:packed['compact_reference']['frozen']['trainable'] = {}
    with pytest.raises(ValueError):validate_metadata(marker, packed, release)
