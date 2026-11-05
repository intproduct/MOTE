"""Candidate mechanism mathematics; distinct coordinate systems stay separate."""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from fitmotn.experiments.qwen_joint.routers import hard_topk,reencoded_state


def functional_distances(a,b=None):
    """[experts, aligned samples, output features] -> normalized RMS matrix."""
    b=a if b is None else b
    if a.ndim!=3 or b.ndim!=3 or a.shape[1:]!=b.shape[1:]:raise ValueError('Aligned functional samples required')
    a=a.double().flatten(1);b=b.double().flatten(1)
    na=a.square().mean(1);nb=b.square().mean(1)
    mse=(na[:,None]+nb[None,:]-2*(a@b.T)/a.shape[1]).clamp_min(0)
    return mse.sqrt()/((na[:,None]+nb[None,:])/2).sqrt().clamp_min(1e-12)


def match_functions(a,b):
    distances=functional_distances(a,b)
    left,right=linear_sum_assignment(distances.cpu().numpy())
    return dict(assignment=list(zip(left.tolist(),right.tolist())),
        mean_matched_distance=float(distances[left,right].mean()),distance_matrix=distances.tolist())


def real_state_qfi(state_function,angles):
    x=angles.detach().double().requires_grad_(True)
    psi=state_function(x).reshape(-1)
    if not torch.allclose(psi.square().sum(),psi.new_tensor(1),atol=1e-10,rtol=1e-10):
        raise ValueError('QFI requires a normalized real pure state')
    jac=torch.autograd.functional.jacobian(lambda z:state_function(z).reshape(-1),x).reshape(len(psi),-1)
    projected=jac.T@psi
    matrix=4*(jac.T@jac-torch.outer(projected,projected))
    return matrix.detach()


def feature_logits(router,feature):
    v=router.output(feature)
    if router.method=='mlp':logits=v
    else:
        if router.method=='affine_born':p=(1+v).square()
        else:
            offset=v.new_tensor([math.pi/2]*4+[0.35]*4)
            p=reencoded_state(offset+(math.pi/2)*v.tanh(),router.method=='reencode_cz').square()
        p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
        p=(p+1e-6)/(1+16e-6)
        logits=p.log()
    return (logits-logits.mean(-1,keepdim=True))*router.log_scale.clamp(-3,3).exp()


def sparse_feature_jacobian(router,feature):
    x=feature.detach().float().reshape(-1).requires_grad_(True)
    def weights(z):
        route=hard_topk(feature_logits(router,z[None]),router.top_k)
        return torch.zeros_like(route.probabilities).scatter(1,route.indices,route.weights)[0]
    logits=feature_logits(router,x[None])[0]
    ranked=logits.sort(descending=True).values
    jac=torch.autograd.functional.jacobian(weights,x).detach()
    return dict(jacobian=jac, frobenius_norm=float(jac.norm()),spectral_norm=float(torch.linalg.matrix_norm(jac,2)),
        fourth_fifth_margin=float((ranked[3]-ranked[4]).detach()))


def replacement_experts(outputs,selected,farthest=False):
    """Select nearest/farthest currently unselected expert per actual input."""
    if outputs.ndim!=3 or selected.shape[0]!=outputs.shape[1]:raise ValueError('Misaligned expert outputs')
    per_token=outputs.transpose(0,1)
    target=per_token[torch.arange(len(selected)),selected[:,0]]
    distance=(per_token-target[:,None]).square().sum(-1)
    distance.scatter_(1,selected,-torch.inf if farthest else torch.inf)
    return distance.argmax(-1) if farthest else distance.argmin(-1)


def switch_risk(reference_logprob,variant_logprob,reference_topk,variant_topk,valid,thresholds=(0.01,0.001,0.1)):
    ref=np.asarray(reference_logprob,dtype=np.float64);var=np.asarray(variant_logprob,dtype=np.float64)
    left=np.asarray(reference_topk);right=np.asarray(variant_topk);valid=np.asarray(valid,dtype=bool)
    if ref.ndim!=1 or ref.shape!=var.shape or valid.shape!=ref.shape or left.shape!=right.shape or left.ndim!=3 or left.shape[1]!=len(ref):
        raise ValueError('Aligned target and router arrays required')
    if not (np.isfinite(ref[valid]).all() and np.isfinite(var[valid]).all()):raise ValueError('Nonfinite target logprob')
    changed=(np.sort(left,axis=-1)!=np.sort(right,axis=-1)).any(-1)
    union=changed.any(0)[valid];error=(var-ref)[valid];increase=-error
    def tails(x):
        if not len(x):return None
        return {name:float(np.quantile(np.abs(x),q,method='linear')) for name,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]}
    count=int(valid.sum());switched=int(union.sum())
    if count==0:raise ValueError('At least one valid target required')
    return dict(valid_targets=count,switched_targets=switched,any_router_switch_rate=switched/count,
        per_router_switch_rate=changed[:,valid].mean(1).tolist(),target_logprob_abs_error=tails(error),
        nonswitch_logprob_abs_error=tails(error[~union]),
        switch_associated_positive_nll_per_valid_target=float(np.maximum(increase,0)[union].sum()/count),
        mean_positive_nll_increase_given_switch=float(np.maximum(increase,0)[union].mean()) if switched else None,
        harmful=[dict(threshold_nats=t,count=int((union&(increase>t)).sum()),
            probability_given_switch=float((union&(increase>t)).sum()/switched) if switched else None) for t in thresholds])
