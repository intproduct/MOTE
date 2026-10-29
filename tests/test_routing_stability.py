import json
from pathlib import Path
import pytest
import torch
from fitmotn.gate import TopKGate
from fitmotn.experiments.routing_stability.routers import (
    SharedBornRouter, ProbabilityTopKGate, boundary_loss, standardized_scores,
)
from fitmotn.experiments.routing_stability.model import initialized_model, full_state_hash
from fitmotn.experiments.gate_benchmark.models import (
    shared_state, state_hash, parameter_report, auxiliary_loss, is_router_key,
)


def configuration():
    cfg = json.loads((Path(__file__).parents[1]/'experiments/routing_stability/configs/pilot.json').read_text())
    cfg['device'] = 'cpu'
    cfg['model'].update(dim=128, ff_dim=256, layers=1, context=12, vocab_size=37)
    return cfg


@pytest.mark.parametrize('experts', [4, 16, 64])
def test_orthogonality_orientation_autograd(experts):
    r = SharedBornRouter(13, experts, 5).double()
    with torch.no_grad():
        r.rotation.normal_(std=.3)
        r.coupling.normal_(std=.3)
    x = torch.randn(2, 3, experts, dtype=torch.float64, requires_grad=True)
    matrix = r.matrix(torch.float64)
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(experts, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    direct, dense = r.evolve(x), x @ matrix
    torch.testing.assert_close(dense, direct, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(dense)
    a = torch.autograd.grad((dense*probe).sum(), (x, r.rotation, r.coupling), retain_graph=True)
    b = torch.autograd.grad((direct*probe).sum(), (x, r.rotation, r.coupling))
    for one, two in zip(a,b):
        torch.testing.assert_close(one, two, atol=1e-11, rtol=1e-11)


def test_cached_operator_tracks_mutation_load_and_dtype():
    r = SharedBornRouter(13,16,5).eval()
    x = torch.randn(3,13)
    with torch.no_grad():
        a = r(x)
        first = r._matrix_cache
        torch.testing.assert_close(r(x), a, atol=0, rtol=0)
        assert first is r._matrix_cache
        r.rotation.add_(.2)
        b = r(x)
        assert first is not r._matrix_cache and not torch.allclose(a,b)
        state = {k:v.clone() for k,v in r.state_dict().items()}
        state['rotation'].zero_()
        r.load_state_dict(state)
        c = r(x)
        assert not torch.allclose(b,c)
        r.double()
        d = r(x.double())
        assert d.dtype == torch.float64
        assert r._matrix_cache.dtype == torch.float64
    # Eval with gradients must build a differentiable, fresh operator.
    r(x.double()).square().sum().backward()
    assert r.rotation.grad is not None and torch.isfinite(r.rotation.grad).all()


def test_direct_probabilities_match_log_softmax_weights_and_ste_gradient():
    class Prob(torch.nn.Module):
        data_dim, num_experts, gate_arch = 7, 16, 'shared_born'
        def forward(self,x):
            return x.softmax(-1)
    class LogProb(torch.nn.Module):
        def forward(self,x):
            return x.log_softmax(-1)
    a = ProbabilityTopKGate(Prob(),4,.01)
    b = TopKGate(7,16,k=4,drop_tokens=False)
    b.router = LogProb()
    x = torch.randn(23,16,requires_grad=True)
    pa,ma,aa = a(x)
    pb,mb,ab = b(x)
    assert torch.equal(ma,mb)
    torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-6)
    weight = torch.randn_like(pa)
    ga = torch.autograd.grad((pa*weight).sum()+aa['l_aux'],x,retain_graph=True)[0]
    gb = torch.autograd.grad((pb*weight).sum()+ab['l_aux'],x)[0]
    torch.testing.assert_close(ga,gb,atol=2e-7,rtol=1e-5)


def test_margin_scale_shift_and_boundary_gradient():
    x = torch.tensor([[2.,1.,.99,-2.]],dtype=torch.float64,requires_grad=True)
    l,g = boundary_loss(x,2,.1)
    assert l > 0
    for scale,shift in [(10.,3.),(.1,-.3)]:
        value,_ = boundary_loss(x*scale+shift,2,.1)
        torch.testing.assert_close(value,l,atol=1e-10,rtol=1e-8)
    grad = torch.autograd.grad(l,x)[0]
    assert grad[0,1] < 0 and grad[0,2] > 0
    assert abs((grad*x).sum()) < 1e-10  # catches detached normalization scales
    zero,_ = boundary_loss(torch.tensor([[3.,2.,-1.,-2.]]),2,.1)
    assert zero == 0
    tied = torch.zeros(2,16,requires_grad=True)
    loss,_ = boundary_loss(tied,4,.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(tied.grad).all()


@pytest.mark.parametrize('base,margin', [('mlp_half','mlp_margin'),('amp_half','amp_margin'),('born_lite','born_lite_margin')])
def test_pairs_identical_start_eval_no_margin_graph_and_real_updates(base,margin):
    torch.set_num_threads(1)
    cfg = configuration()
    a,sa = initialized_model(cfg,base,17)
    b,sb = initialized_model(cfg,margin,17)
    assert full_state_hash(a.state_dict()) == full_state_hash(b.state_dict())
    assert state_hash(sa) == state_hash(sb) == state_hash(shared_state(b))
    x=torch.randint(2,37,(2,12)); y=torch.randint(2,37,(2,12))
    a.eval(); b.eval()
    assert not b._margin_handles
    torch.testing.assert_close(a(x),b(x),atol=0,rtol=0)
    assert b.regularizer() == 0 and not b._margin_terms
    before={n:p.clone() for n,p in b.named_parameters() if is_router_key(n)}
    b.training_step=cfg['train']['steps']; b.train()
    opt=torch.optim.AdamW(b.parameters(),lr=.001)
    for _ in range(3):
        opt.zero_grad()
        logits=b(x)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())+auxiliary_loss(b)+b.regularizer()
        loss.backward()
        assert b._margin_terms
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
        opt.step()
    assert all((p-before[n]).abs().max()>0 for n,p in b.named_parameters() if n in before)
    b.eval()
    assert not b._margin_handles and not b._margin_terms
    b(x); assert not b._margin_terms
    # Future tokens cannot affect earlier positions for either new design.
    with torch.no_grad():
        original=b(x); altered=x.clone();altered[:,6:]=(altered[:,6:]+1)%37
        torch.testing.assert_close(original[:,:6],b(altered)[:,:6],atol=1e-6,rtol=1e-5)


def test_full_size_budgets_and_shared_anchor():
    cfg=json.loads((Path(__file__).parents[1]/'experiments/routing_stability/configs/pilot.json').read_text())
    counts={};hashes=set()
    for variant in ['mlp_half','amp_half','born_lite']:
        model,state=initialized_model(cfg,variant,17)
        counts[variant]=parameter_report(model)['router_total']
        hashes.add(state_hash(state))
    assert len(hashes)==1
    assert abs(counts['born_lite']/counts['mlp_half']-1)<.03


@pytest.mark.parametrize('variant,old_variant',[('mlp_half','mlp_half'),('amp_half','amp_d1_half')])
def test_unregularized_baselines_match_original_initialization(variant,old_variant):
    from fitmotn.experiments.autoregressive_gate.model import initialized_model as original
    cfg=configuration()
    a,_=initialized_model(cfg,variant,17)
    b,_=original(cfg,old_variant,17)
    assert full_state_hash(a.state_dict())==full_state_hash(b.state_dict())
    a.eval();b.eval();x=torch.randint(2,37,(2,12))
    torch.testing.assert_close(a(x),b(x),atol=0,rtol=0)

