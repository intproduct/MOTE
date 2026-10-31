import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch import nn
import pytest
from fitmotn.experiments.qwen_joint.routers import JointRouter, METHODS
from fitmotn.experiments.qwen_joint.model import set_model_precision
from fitmotn.experiments.qwen_joint.model import set_model_precision as set_precision_preserving_routers


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_preserves_router_values_outputs_and_gradients(method, dtype):
    router = JointRouter(16, method, 12017)
    nested = nn.ModuleDict({'backbone': nn.Linear(16,16), 'adapted': nn.ModuleList([router])})
    nested.register_buffer('counter', torch.tensor(7, dtype=torch.int64))
    x = torch.linspace(-1, 1, 96).reshape(6,16)
    logits = router.logits(x)
    logits.square().sum().backward()
    before = {name: value.clone() for name,value in router.state_dict().items()}
    grads = {name: p.grad.clone() if p.grad is not None else None for name,p in router.named_parameters()}
    for target in (dtype, torch.float32, dtype):
        set_precision_preserving_routers(nested, target, 'cpu')
        assert nested['backbone'].weight.dtype == target
        assert nested.counter.dtype == torch.int64 and nested.counter.item() == 7
        assert all(value.dtype == torch.float32 and torch.equal(value,before[name]) for name,value in router.state_dict().items())
        assert torch.equal(router.logits(x),logits)
        for name,p in router.named_parameters():
            assert (p.grad is None and grads[name] is None) or torch.equal(p.grad,grads[name])


