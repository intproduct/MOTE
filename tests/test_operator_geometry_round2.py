import torch
from torch import nn
from fitmotn.experiments.operator_geometry.metrics import head
from fitmotn.experiments.operator_geometry.round2_data import directional_cost


def test_directional_expert_cost_predicts_small_fixed_mass_replacement():
    torch.manual_seed(37)
    model=nn.Module();model.norm=nn.LayerNorm(7).double();model.embedding=nn.Embedding(13,7).double()
    model.requires_grad_(False)
    h=torch.randn(6,7,dtype=torch.double);outputs=torch.randn(6,16,7,dtype=torch.double);y=torch.arange(6)
    c=directional_cost(model,h,outputs,y)
    torch.testing.assert_close(c.mean(-1),torch.zeros(6,dtype=torch.double),atol=1e-12,rtol=1e-12)
    eps=1e-5
    baseline=torch.nn.functional.cross_entropy(head(model,h),y,reduction='none')
    changed=torch.nn.functional.cross_entropy(head(model,h+eps*(outputs[:,9]-outputs[:,2])),y,reduction='none')
    torch.testing.assert_close((changed-baseline)/eps,c[:,9]-c[:,2],atol=2e-3,rtol=2e-3)
    assert all(p.grad is None for p in model.parameters())


def test_directional_cost_is_invariant_to_common_expert_output_shift():
    torch.manual_seed(41)
    model=nn.Module();model.norm=nn.LayerNorm(7).double();model.embedding=nn.Embedding(13,7).double()
    model.requires_grad_(False)
    h=torch.randn(6,7,dtype=torch.double);outputs=torch.randn(6,16,7,dtype=torch.double);y=torch.arange(6)
    a=directional_cost(model,h,outputs,y)
    b=directional_cost(model,h,outputs+torch.randn(6,1,7,dtype=torch.double),y)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_replacement_threshold_never_uses_validation_and_can_choose_no_change():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_train import action_arrays,calibrate_threshold
    cost=np.array([[2.,1.,0.],[2.,1.,0.]])
    cache={'weights':torch.tensor([[.5,.5,0.],[.5,.5,0.]])}
    exact={'source':np.array([[0,1],[0,1]]),'target':np.array([[2,2],[2,2]]),'delta_nll':np.ones((2,2))}
    predicted,actual,chosen=action_arrays(cost,cache,exact)
    np.testing.assert_allclose(predicted,[-1.,-1.]);np.testing.assert_allclose(actual,[1.,1.])
    threshold,rows=calibrate_threshold(cost,cache,exact,[0.,.5,None])
    assert threshold is None and rows[-1]['changed']==0


def test_utility_prediction_does_not_accept_labels_or_expert_outputs():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_train import UtilityPredictor
    torch.manual_seed(73)
    model=UtilityPredictor(7,4,'angle_functional',np.arange(16),np.zeros((16,4)),np.arange(16))
    x=torch.randn(5,7,requires_grad=True);cost=model(x)
    assert cost.shape==(5,16) and torch.isfinite(cost).all()
    cost[:,0].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_reduced_rank_ridge_solves_full_rank_normal_equation():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_ridge import reduced_rank_solution
    rng=np.random.default_rng(83);x=rng.normal(size=(100,7));y=rng.normal(size=(100,5))
    gram=x.T@x/len(x);cross=x.T@y/len(x);values,vectors=np.linalg.eigh(gram)
    w=reduced_rank_solution(values,vectors,cross,.1,5)
    np.testing.assert_allclose((gram+.1*np.eye(7))@w,cross,atol=1e-12)
    low=reduced_rank_solution(values,vectors,cross,.1,2)
    assert np.linalg.matrix_rank(low,tol=1e-9)==2
    def obj(a):return np.square(x@a-y).sum()/len(x)+.1*np.square(a).sum()
    assert obj(w)<=obj(low)<=obj(np.zeros_like(w))


def test_product_top4_boundary_and_real_readout_fisher():
    from fitmotn.experiments.operator_geometry.round2_theory import product_boundary,real_fisher
    assert product_boundary(samples=1000)['hamming_one']==0
    assert real_fisher(samples=2)['passed']


def test_vectorized_spearman_agrees_with_average_ties_reference():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_numeric import row_spearman
    from fitmotn.experiments.operator_geometry.metrics import spearman
    rng=np.random.default_rng(91);a=rng.integers(0,5,(50,12));b=rng.integers(0,5,(50,12));a[0]=1
    fast=row_spearman(a,b)
    for i in range(len(a)):
        slow=spearman(a[i],b[i])
        if slow is None:assert np.isnan(fast[i])
        else:np.testing.assert_allclose(fast[i],slow,atol=1e-14)


def test_ray_geometry_loses_sign_and_anchor_recovers_a_sign_difference():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_geometry import pair_scores
    f=np.array([[[1.,0.],[-1.,0.]]]);s=np.array([[0]]);t=np.array([[1]])
    assert pair_scores(f,s,t,None)[0,0]==4
    assert pair_scores(f,s,t,0)[0,0]==0
    assert pair_scores(f,s,t,1)[0,0]==1


def test_energy_decomposition_recovers_function_distance():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_energy import proxies
    rng=np.random.default_rng(103);out=rng.normal(size=(5,16,7));w=np.ones((5,16))/16
    s=np.tile(np.arange(4)[:,None],(1,12)).reshape(1,48).repeat(5,axis=0)
    t=np.tile(np.arange(4,16),(5,4))
    energy=np.square(out).sum(-1).mean(0)
    d=np.square(out[:,:,None]-out[:,None,:]).sum(-1).mean(0)
    values,_,_=proxies(out,w,{'source':s,'target':t},d,energy)
    ix=np.arange(5)[:,None]
    np.testing.assert_allclose(values['local_full_distance'],np.square(out[ix,s]-out[ix,t]).sum(-1),atol=1e-12)


def test_regular_basis_graph_assignment_cannot_change_energy_sum():
    import numpy as np
    from fitmotn.experiments.operator_geometry.routers import cube_edges
    rng=np.random.default_rng(107);energy=rng.uniform(0,2,16);edges=cube_edges(16)
    for _ in range(20):
        mapping=rng.permutation(16)
        actual=(energy[mapping[edges[:,0]]]+energy[mapping[edges[:,1]]]).sum()
        np.testing.assert_allclose(actual,4*energy.sum(),atol=1e-12)


def test_positive_control_logprob_teacher_preserves_centered_scores_and_order():
    import numpy as np
    from fitmotn.experiments.operator_geometry.round2_positive import teacher_cost,measures
    torch.manual_seed(109)
    logits=torch.randn(20,16,dtype=torch.double)
    target=teacher_cost(logits.softmax(-1))
    torch.testing.assert_close(target,-logits+logits.mean(-1,keepdim=True),atol=1e-12,rtol=1e-12)
    a=target.numpy();result=measures(a,a,np.zeros(16))
    assert result['r2_vs_constant']==1 and result['top4_set_agreement']==1
    np.testing.assert_allclose(result['within_token_rho'],1,atol=1e-12)


def test_fixed_boundary_derivative_matches_directional_finite_difference():
    import numpy as np
    from fitmotn.experiments.operator_geometry.routers import ControlledRouter
    from fitmotn.experiments.operator_geometry.round2_boundary_gradient import local_boundary
    torch.manual_seed(113)
    router=ControlledRouter(7,16,4,'angle_functional',np.arange(16),np.zeros((16,4)),np.arange(16)).double().eval()
    x=torch.randn(8,7,dtype=torch.double);direction=torch.randn_like(x)
    _,pair,margin,gradient=local_boundary(router,x)
    def difference(y):
        logp=router(y).log()
        return logp.gather(1,pair[:,0:1])[:,0]-logp.gather(1,pair[:,1:2])[:,0]
    eps=1e-5;numeric=(difference(x+eps*direction)-difference(x-eps*direction))/(2*eps)
    torch.testing.assert_close(numeric,(gradient*direction).sum(-1),atol=1e-9,rtol=1e-5)


def test_state_boundary_bounds_and_expert_function_pullback():
    from fitmotn.experiments.operator_geometry.round2_metric_bridge import state_checks,functional_checks
    assert state_checks(samples=2)['passed']
    assert functional_checks()['passed']
