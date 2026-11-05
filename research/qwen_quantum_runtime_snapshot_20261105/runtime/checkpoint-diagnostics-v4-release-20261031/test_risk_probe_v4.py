from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fitmotn.experiments.qwen_joint.model import AdapterConfig, JointProjection
from risk_probe_v4 import aligned_probe, native_precision


def test_native_precision_never_substitutes_bfloat_roundtrip():
    assert native_precision('native_bfloat16', 'cuda', (12, 0)) == torch.bfloat16
    assert native_precision('native_float16', 'cuda', (7, 0)) == torch.float16
    with pytest.raises(ValueError):native_precision('native_bfloat16', 'cuda', (7, 0))
    with pytest.raises(ValueError):native_precision('native_float16', 'cpu')
    with pytest.raises(ValueError):native_precision('native_bfloat16', 'cpu')


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(31, 128)
        self.proj = JointProjection(torch.nn.Linear(128, 192, bias=False), 'mlp',
            AdapterConfig(residual_rank=4, tail_rank=4), 3)
    def forward(self, input_ids, use_cache):
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=Backbone(); self.lm_head=torch.nn.Linear(192,31,bias=False)


def fixture():
    torch.manual_seed(17); model=Model().eval()
    arrays={}
    for i in range(2):
        arrays[f'ids_{i}']=torch.randint(31,(1024,)).numpy()
        arrays[f'labels_{i}']=torch.randint(31,(1024,)).numpy(); arrays[f'labels_{i}'][0]=-100
        arrays[f'positions_{i}']=np.linspace(1,1023,32,dtype=np.int64)
    return model, arrays


def test_probe_fixed_window_alignment_batch_shape_and_hook_cleanup():
    model, arrays=fixture()
    a, _=aligned_probe(model, arrays, ['model.proj'], 'fp32_reference', lambda:None)
    b, _=aligned_probe(model, arrays, ['model.proj'], 'fp32_repeat', lambda:None)
    assert np.array_equal(a['logprobs'],b['logprobs']) and np.array_equal(a['topk'],b['topk'])
    c, _=aligned_probe(model, arrays, ['model.proj'], 'batch_two', lambda:None)
    np.testing.assert_allclose(a['logprobs'],c['logprobs'],atol=1e-6,rtol=1e-6)
    assert a['topk'].shape==(1,2048,4) and a['logprobs'][0]==a['logprobs'][1024]==0
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    def fail():raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):aligned_probe(model, arrays, ['model.proj'], 'input_bfloat16_roundtrip', fail)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
