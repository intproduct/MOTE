import pytest
import numpy as np
import torch
from torch import nn
from fitmotn.experiments.operator_geometry.metrics import (
    substitution_indices, fixed_mass_delta, head, head_jvp, compare_outputs,
    spearman, bootstrap_window_mean,
)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(7, eps=1e-5).double()
        self.embedding = nn.Embedding(11,7).double()


def test_analytic_layernorm_jvp_matches_autograd():
    torch.manual_seed(17)
    m = Head()
    h,d = torch.randn(5,7,dtype=torch.double),torch.randn(5,7,dtype=torch.double)
    _, expected = torch.autograd.functional.jvp(lambda h:head(m,h),h,d)
    torch.testing.assert_close(head_jvp(m,h,d),expected,atol=1e-12,rtol=1e-12)


def test_fisher_predicts_local_kl_not_signed_nll():
    torch.manual_seed(4)
    m = Head()
    h,d = torch.randn(5,7,dtype=torch.double),torch.randn(5,7,dtype=torch.double)*1e-4
    r = compare_outputs(head(m,h),head(m,h+d),head_jvp(m,h,d),torch.arange(5))
    torch.testing.assert_close(r['fisher_kl'],r['kl'],atol=1e-10,rtol=.003)
    torch.testing.assert_close(r['first_order_nll'],r['delta_nll'],atol=1e-6,rtol=.01)
    assert (r['kl'] >= 0).all()


def test_every_replacement_preserves_topk_cardinality_and_mass():
    torch.manual_seed(29)
    w=torch.rand(6,16); mask=torch.zeros_like(w).scatter(1,w.topk(4).indices,1)
    w=w*mask; w=w/w.sum(-1,keepdim=True)
    a,b=substitution_indices(w,4)
    assert a.shape == (6,48)
    outputs=torch.randn(6,16,7)
    for j in range(48):
        n=torch.arange(6); changed=w.clone()
        changed[n,b[:,j]]=w[n,a[:,j]];changed[n,a[:,j]]=0
        assert ((changed>0).sum(-1)==4).all()
        torch.testing.assert_close(changed.sum(-1),w.sum(-1))
        actual=(changed[:,:,None]*outputs).sum(1)-(w[:,:,None]*outputs).sum(1)
        torch.testing.assert_close(actual,fixed_mass_delta(outputs,w,a[:,j],b[:,j]),atol=1e-6,rtol=1e-5)


def test_rank_ties_and_window_bootstrap():
    assert spearman([1,1,2,3],[3,3,2,1]) == pytest.approx(-1)
    assert spearman([1,1],[2,3]) is None
    values=np.array([1,1,1,1,2,2,2,2.]); windows=np.repeat([0,1],4)
    assert bootstrap_window_mean(values,windows,17)==[1.,2.]


def test_zero_intervention_is_exact_identity():
    m=Head(); h=torch.randn(5,7,dtype=torch.double)
    r=compare_outputs(head(m,h),head(m,h),head_jvp(m,h,h*0),torch.arange(5))
    for value in r.values():
        assert not value.any()


@pytest.mark.parametrize('method',['mlp','spectral_functional','spectral_random','shared_born',
    'angle_natural','angle_random','angle_functional','product_functional','exp_adapter'])
def test_candidate_probabilities_and_gradients(method):
    from fitmotn.experiments.operator_geometry.routers import ControlledRouter,spectral_codes
    torch.manual_seed(43)
    vectors=np.random.default_rng(17).normal(size=(16,6))
    d=((vectors[:,None]-vectors[None,:])**2).sum(-1)
    r=ControlledRouter(13,16,8,method,np.arange(15,-1,-1),spectral_codes(d),np.roll(np.arange(16),3)).double()
    x=torch.randn(7,13,dtype=torch.double,requires_grad=True);p=r(x)
    torch.testing.assert_close(p.sum(-1),torch.ones(7,dtype=torch.double),atol=1e-12,rtol=1e-12)
    assert (p>0).all() and torch.isfinite(p).all()
    loss=-(p[:,3].log()).mean();loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert sum(float(t.grad.abs().sum()) for t in r.parameters() if t.grad is not None)>0


def test_product_born_equals_classical_product_bernoulli():
    from fitmotn.experiments.operator_geometry.routers import ControlledRouter
    r=ControlledRouter(7,16,4,'product_functional',np.arange(16),np.ones((16,4)),np.arange(16)).double()
    with torch.no_grad():r.rotation.normal_(std=.2)
    angles=torch.randn(5,4,dtype=torch.double)
    q=torch.sin((angles+r.rotation.sum(0))/2).square()
    bits=((torch.arange(16)[:,None]>>torch.arange(3,-1,-1))&1).bool()
    classical=torch.where(bits[None],q[:,None,:],1-q[:,None,:]).prod(-1)
    torch.testing.assert_close(r.state(angles).square(),classical,atol=1e-12,rtol=1e-12)


def test_square_orthogonal_cosine_score_is_basis_invariant():
    torch.manual_seed(29)
    x,c=torch.randn(3,7,dtype=torch.double),torch.randn(3,7,dtype=torch.double)
    u,_=torch.linalg.qr(torch.randn(7,7,dtype=torch.double))
    torch.testing.assert_close(torch.nn.functional.cosine_similarity(x@u,c@u),
        torch.nn.functional.cosine_similarity(x,c),atol=1e-12,rtol=1e-12)


def test_functional_assignment_is_a_permutation_and_does_not_increase_cost():
    from fitmotn.experiments.operator_geometry.routers import optimize_assignment
    vectors=np.random.default_rng(29).normal(size=(16,6));d=((vectors[:,None]-vectors[None,:])**2).sum(-1)
    p,result=optimize_assignment(d,17,restarts=2,steps=100)
    assert sorted(p)==list(range(16))
    assert result['functional_cost'] <= result['identity_cost']
