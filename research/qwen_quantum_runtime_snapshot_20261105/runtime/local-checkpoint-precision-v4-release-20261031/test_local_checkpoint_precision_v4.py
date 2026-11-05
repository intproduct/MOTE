import copy
import numpy as np
import pytest
import torch
from local_checkpoint_precision_v4 import compact_error, capture_projection, comparisons, probe_projection
from fitmotn.experiments.qwen_joint.model import JointProjection, AdapterConfig, set_model_precision, tensor_hash


@pytest.mark.parametrize('shape', [(5,7), (3,5,7)])
def test_compact_statistics_reconstruct_independent_full_array_errors(shape):
    torch.manual_seed(29)
    a = torch.randn(shape); b = a + torch.randn(shape) * .03
    raw, row = compact_error(a, b)
    aa, bb = a.double().numpy(), b.double().numpy(); difference = bb - aa
    np.testing.assert_allclose(raw['error_energy'], np.square(difference).sum(-1), atol=1e-14)
    np.testing.assert_allclose(raw['reference_energy'], np.square(aa).sum(-1), atol=1e-14)
    np.testing.assert_allclose(raw['variant_energy'], np.square(bb).sum(-1), atol=1e-14)
    np.testing.assert_allclose(raw['error_max_abs'], np.abs(difference).max(-1), atol=1e-14)
    assert row['relative_l2'] == pytest.approx(np.linalg.norm(difference) / np.linalg.norm(aa), rel=1e-12)
    assert row['error_rms'] == pytest.approx(np.sqrt(np.square(difference).mean()), rel=1e-12)
    assert row['reference_rms'] == pytest.approx(np.sqrt(np.square(aa).mean()), rel=1e-12)
    assert row['max_abs_error'] == pytest.approx(np.abs(difference).max(), rel=1e-12)


@pytest.mark.parametrize('bad', ['shape','nonfinite','empty','dimensions'])
def test_invalid_records_refused(bad):
    a = torch.ones(2,4); b = a.clone()
    if bad == 'shape': b = b[:1]
    elif bad == 'nonfinite': b[0,0] = float('inf')
    elif bad == 'empty': a = a[:0]; b = b[:0]
    else: a = a[None,None]; b = b[None,None]
    with pytest.raises(ValueError): compact_error(a,b)


@pytest.mark.parametrize('method', ['mlp','affine_born','reencode_cz','reencode_product'])
def test_real_sparse_projection_cpu_roundtrips_match_direct_reference_and_do_not_mutate(method):
    torch.manual_seed(43)
    module = JointProjection(torch.nn.Linear(128,192,bias=False),method,AdapterConfig(residual_rank=4,tail_rank=4),3).eval()
    set_model_precision(module,torch.float16,'cpu'); set_model_precision(module,torch.float32,'cpu')
    before = tensor_hash(module.state_dict()); inputs = torch.randn(8,128)
    raw, rows, variants = probe_projection(module,inputs,torch.float16,False)
    assert tensor_hash(module.state_dict()) == before and len(rows) == 2
    assert variants == ['stored_reference_fp32','input_float16_roundtrip','input_bfloat16_roundtrip']
    with torch.no_grad():
        actual = module(inputs.half().float()).double().numpy(); reference = module(inputs).double().numpy()
    np.testing.assert_allclose(raw['comparison_0_projection_error_energy'], ((actual-reference)**2).sum(-1), atol=1e-12)
    np.testing.assert_allclose(raw['comparison_0_projection_error_max_abs'], np.abs(actual-reference).max(-1), atol=1e-12)
    assert raw['condition_0_indices'].shape == (8,4) and raw['comparison_0_experts_error_energy'].shape == (16,8)
    assert not module.router._forward_hooks


def test_dense_has_no_fabricated_routes_or_experts_and_native_cpu_refused():
    layer = torch.nn.Linear(8,6,bias=False).eval(); inputs = torch.randn(5,8)
    raw, rows, _ = probe_projection(layer,inputs,torch.float16,False)
    assert not any('condition_' in name or 'experts' in name for name in raw)
    assert all('switched' not in row for row in rows)
    with pytest.raises(ValueError, match='requires CUDA'): capture_projection(layer,inputs,torch.float16)


def test_hook_removed_when_sparse_forward_fails():
    module = JointProjection(torch.nn.Linear(128,128,bias=False),'mlp',AdapterConfig(residual_rank=4,tail_rank=4),3)
    def fail_after_router(x):
        module.router(x)
        raise RuntimeError('injected projection failure')
    module.forward = fail_after_router
    with pytest.raises(RuntimeError, match='injected'): capture_projection(module,torch.randn(3,128))
    assert not module.router._forward_hooks


def test_aligned_native_router_mismatch_rejected_and_missing_conditions_refused():
    values = dict(projection=torch.ones(3,7),experts=torch.ones(16,3,7),indices=torch.tensor([[0,1,2,3]]*3),
        weights=torch.full((3,4),.25),probabilities=torch.full((3,16),1/16))
    conditions = {name:copy.deepcopy(values) for name in ('stored_reference_fp32','input_float16_roundtrip','input_bfloat16_roundtrip','native_float16')}
    _, rows, _ = comparisons(conditions,'float16'); assert rows[-1]['route_weights_exact']
    conditions['native_float16']['indices'][0,0] = 5
    with pytest.raises(ValueError, match='routers differ'): comparisons(conditions,'float16')
    del conditions['native_float16']
    with pytest.raises(ValueError, match='condition set'): comparisons(conditions,'float16')
