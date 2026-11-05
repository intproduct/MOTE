import math
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import pytest
import torch
from mechanism_metrics_candidate_v1 import functional_distances,match_functions,real_state_qfi,feature_logits,sparse_feature_jacobian,replacement_experts,switch_risk
from fitmotn.experiments.qwen_joint.routers import JointRouter,reencoded_state

def test_function_distance_invariances():
    a=torch.tensor([[[1.,2.],[3.,4.]], [[-1.,-2.],[-3.,-4.]]])
    d=functional_distances(a)
    assert torch.allclose(d,torch.tensor([[0.,2.],[2.,0.]],dtype=torch.float64))
    assert torch.allclose(d,functional_distances(a*13))
    assert torch.isfinite(functional_distances(torch.zeros_like(a))).all()

def test_permutation_matching():
    a=torch.randn(5,7,9,generator=torch.Generator().manual_seed(4))
    perm=torch.tensor([3,0,4,1,2]);r=match_functions(a,a[perm])
    assert r['mean_matched_distance']<1e-7
    assert all(int(perm[j])==i for i,j in r['assignment'])

def test_analytic_single_ry_qfi():
    f=lambda x:torch.stack([torch.cos(x[0]/2),torch.sin(x[0]/2)])
    assert torch.allclose(real_state_qfi(f,torch.tensor([.7])),torch.ones(1,1,dtype=torch.float64),atol=1e-12)

def test_product_reencoding_qfi_analytic():
    q=real_state_qfi(lambda x:reencoded_state(x[None],False)[0],torch.arange(8).double()/7)
    expected=torch.eye(4,dtype=torch.float64).repeat(2,2)
    assert torch.allclose(q,expected,atol=1e-12)

def test_qfi_rejects_unnormalized():
    with pytest.raises(ValueError):real_state_qfi(lambda x:torch.stack([x[0]*0+2,x[0]*0]),torch.tensor([0.]))

@pytest.mark.parametrize('method',['mlp','affine_born','reencode_cz','reencode_product'])
def test_feature_adapter_agrees_with_frozen_router(method):
    router=JointRouter(16,method,17)
    x=torch.randn(3,16,generator=torch.Generator().manual_seed(7))
    feature=torch.tanh(router.input_proj(router.norm(x)))
    assert torch.equal(feature_logits(router,feature),router.logits(x))
    j=sparse_feature_jacobian(router,feature[0])['jacobian']
    assert torch.isfinite(j).all() and torch.allclose(j.sum(0),torch.zeros(8),atol=1e-7)
    assert int((j.norm(dim=1)>0).sum())==4

def test_mlp_jacobian_matches_closed_form():
    router=JointRouter(16,'mlp',19);x=torch.arange(8).float()/7
    logits=feature_logits(router,x[None])[0]
    ids=torch.argsort(logits,descending=True,stable=True)[:4]
    w=logits[ids].softmax(0);softmax_jac=torch.diag(w)-torch.outer(w,w)
    expected=torch.zeros(16,8);expected[ids]=softmax_jac@router.output.weight[ids]
    assert torch.allclose(sparse_feature_jacobian(router,x)['jacobian'],expected,atol=1e-8)

def test_replacement_excludes_selected_and_breaks_ties_by_index():
    outputs=torch.tensor([0.,1.,-1.,4.,8.])[:,None,None].expand(-1,2,-1)
    selected=torch.tensor([[0,4],[3,4]])
    assert replacement_experts(outputs,selected).tolist()==[1,1]
    assert replacement_experts(outputs,selected,True).tolist()==[3,2]

def test_harmful_switch_denominators_mask_and_set_order():
    top=np.tile(np.array([0,1,2,3]),(2,4,1));other=top.copy()
    other[0,0]=[3,2,1,0];other[1,1,3]=4;other[0,2,3]=5
    r=switch_risk([-2]*4,[-2,-2.02,-1.9,-100],top,other,[1,1,1,0])
    assert r['valid_targets']==3 and r['switched_targets']==2
    assert r['harmful'][0]['count']==1 and r['harmful'][0]['probability_given_switch']==.5
    assert abs(r['switch_associated_positive_nll_per_valid_target']-.02/3)<1e-12

def test_zero_switch_and_empty_targets():
    top=np.zeros((1,2,4),dtype=int)
    r=switch_risk([-1,-2],[-1,-2],top,top,[1,1])
    assert r['harmful'][0]['probability_given_switch'] is None
    with pytest.raises(ValueError):switch_risk([-1,-2],[-1,-2],top,top,[0,0])
