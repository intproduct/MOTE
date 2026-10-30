"""Numerical checks bridging state, readout-boundary, and output metrics.

All equations are own derivations from normalized states and the softmax head.
These are toy mathematical checks, not empirical evidence of quantum utility.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .routers import ControlledRouter
from .metrics import head,head_jvp
from ..autoregressive_gate.data import digest,write_json


def state_checks(samples=64,seed=20261034):
    torch.manual_seed(seed);rows=[]
    for method in ['angle_natural','product_functional']:
        router=ControlledRouter(7,16,4,method,np.arange(16),np.zeros((16,4)),np.arange(16)).double()
        with torch.no_grad():router.rotation.normal_(std=.7)
        for i in range(samples):
            theta=.2+torch.rand(4,dtype=torch.double)*2.6
            psi=router.state(theta[None])[0];jac=torch.autograd.functional.jacobian(lambda a:router.state(a[None])[0],theta)
            p=psi.square();order=p.argsort(descending=True);a,b=order[3:5]
            grad=2*jac[a]/psi[a]-2*jac[b]/psi[b]
            bound=(1/p[a]+1/p[b]).sqrt()
            assert grad.norm()<=bound+1e-9
            # A fixed unitary leaves the overlap of the product encodings intact.
            scale=[1e-4,1e-3,1e-2,.1][i%4];delta=torch.randn(4,dtype=torch.double)*scale
            phi=router.state((theta+delta)[None])[0];q=phi.square()
            expected=torch.cos(delta/2).prod();overlap=psi@phi
            assert abs(overlap-expected)<1e-12
            trace_distance=(1-overlap.square()).clamp_min(0).sqrt()
            max_pair_drift=(q-p).max()-(q-p).min()
            assert max_pair_drift<=2*trace_distance+1e-12
            certificate=bool(p[a]-p[b]>2*trace_distance)
            changed=not torch.equal(order[:4].sort().values,q.topk(4).indices.sort().values)
            assert not (certificate and changed)
            rows.append({'method':method,'sample':i,'perturbation_scale':scale,
                'logmargin_gradient_bound_ratio':float(grad.norm()/bound),'overlap_formula_abs_error':float(abs(overlap-expected)),
                'trace_distance':float(trace_distance),'probability_margin':float(p[a]-p[b]),
                'max_pair_drift':float(max_pair_drift),'certified_top4_unchanged':certificate,'top4_changed':changed})
    return {'passed':True,'rows':rows,
        'differential_bound':'|d log(p_i/p_j)| <= sqrt(1/p_i+1/p_j) sqrt(dtheta^T F_Q dtheta); real full-support Born measurement, unfloored probabilities',
        'finite_bound':'Every pair gap changes by at most2 sqrt(1-|<psi|phi>|^2); p_(4)-p_(5) greater than this implies unchanged top4',
        'product_overlap':'<psi(theta)|psi(theta+delta)>=product cos(delta_j/2), for the same fixed real circuit after the product RY encoding',
        'scope':'A sufficient finite bound and a local differential bound, both independent of expert function costs. A probability floor rescales the gap and drift equally; arbitrary learned input encoders also contribute their Jacobian.',
        'novelty':'Own specialization/check of standard state-overlap and Cauchy-Schwarz inequalities; no novelty or advantage claim for the generic inequalities.'}


def functional_checks(seed=20261035):
    torch.manual_seed(seed);model=nn.Module();model.norm=nn.LayerNorm(7).double();model.embedding=nn.Embedding(13,7).double()
    model.requires_grad_(False);h=torch.randn(7,dtype=torch.double);outputs=torch.randn(16,7,dtype=torch.double)
    logits=head(model,h[None])[0];prob=logits.softmax(-1)
    jac=torch.autograd.functional.jacobian(lambda z:head(model,z[None])[0],h)
    softmax_fisher=torch.diag(prob)-prob[:,None]*prob[None,:]
    functional=outputs@jac.T@softmax_fisher@jac@outputs.T
    weight_direction=torch.randn(16,dtype=torch.double);weight_direction-=weight_direction.mean()
    hidden_direction=weight_direction@outputs;v=head_jvp(model,h[None],hidden_direction[None])[0]
    direct=(prob*v.square()).sum()-(prob*v).sum().square()
    pullback=weight_direction@functional@weight_direction
    assert abs(direct-pullback)<1e-10
    assert torch.linalg.eigvalsh(functional).min()>-1e-10
    lp=logits.log_softmax(-1);rows=[]
    for epsilon in [.01,.001,.0001]:
        lq=head(model,(h+epsilon*hidden_direction)[None])[0].log_softmax(-1)
        kl=(prob*(lp-lq)).sum();approx=.5*epsilon**2*pullback
        rows.append({'epsilon':epsilon,'kl':float(kl),'fisher_kl':float(approx),'relative_error':float(abs(kl-approx)/kl)})
    assert rows[-1]['relative_error']<.002
    shifted=outputs+torch.randn(1,7,dtype=torch.double)
    same=weight_direction@(shifted@jac.T@softmax_fisher@jac@shifted.T)@weight_direction
    assert abs(same-pullback)<1e-10
    return {'passed':True,'rows':rows,'pullback_jvp_abs_error':float(abs(direct-pullback)),
        'common_expert_shift_quadratic_abs_error':float(abs(same-pullback)),
        'equation':'G_function(x)=V(x)^T J_head(x)^T [diag(p_LM)-p_LM p_LM^T] J_head(x) V(x); KL approximately(1/2) Delta_w^T G_function Delta_w',
        'notation':'V has expert outputs as columns; code stores its transpose. Head Jacobian includes final LayerNorm. Sum(Delta_w)=0 gives common-output-shift invariance.',
        'distinction':'Gate categorical Fisher diag(p_gate)-p_gate p_gate^T in logits and state QFI describe other maps. Equality to the functional metric requires additional expert/head structure.',
        'hard_selection':'Within a fixed active set one can pull G_function through the weight Jacobian. Across a top-k exchange, directly measure the finite displacement or its labeled KL approximation; an ordinary derivative of hard indices omits the jump.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'checks.json',{'state_boundary':state_checks(),'functional':functional_checks(),'source_sha256':digest(Path(__file__))})
    print('METRIC_BRIDGE_PASSED',out,flush=True)


if __name__=='__main__':main()
