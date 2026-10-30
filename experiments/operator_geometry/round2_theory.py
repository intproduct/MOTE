"""CPU checks of product-state top-k boundaries and real-state Fisher geometry.

These are mathematical diagnostics, not training or evidence of task advantage.
No labels, development samples, validation samples, or GPU are used.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .routers import ControlledRouter


def product_boundary(samples=100000,seed=20261030):
    rng=np.random.default_rng(seed)
    # Positive costs for flipping each Bernoulli bit away from its modal value.
    costs=np.sort(rng.uniform(.001,10,(samples,4)),axis=1)
    bits=((np.arange(16)[:,None]>>np.arange(3,-1,-1))&1)
    energies=costs@bits.T
    order=np.argsort(energies,axis=1,kind='stable')
    fourth,fifth=order[:,3],order[:,4]
    hamming=bits[fourth].__ne__(bits[fifth]).sum(-1)
    # The five cheapest subsets are empty,a,b and the cheapest two of c,ab,d.
    # With a <= b <= c <= d, if ab<c then the fourth/fifth are ab,c;
    # otherwise they are c,min(ab,d). Their Hamming distance is 3 or 2.
    ab=costs[:,0]+costs[:,1];c=costs[:,2];d=costs[:,3]
    expected=np.where(ab<c,3,np.where(ab<d,3,2))
    assert np.array_equal(hamming,expected)
    assert (hamming>=2).all()
    return {'samples':samples,'seed':seed,'hamming_one':int((hamming==1).sum()),
            'hamming_two':int((hamming==2).sum()),'hamming_three':int((hamming==3).sum()),
            'exact_case_formula_agrees':True,
            'conditions':'four independent nondegenerate Bernoulli bits; distinct subset costs; top4 selection; monotone probability floor preserves ranks',
            'reason':'sort modal-bit flip costs a<b<c<d; top3 are empty,a,b; fourth/fifth are ab,c or c,min(ab,d)',
            'implication':'Hypercube Hamming-one edges exclude every generic fourth/fifth boundary of a four-bit product router.'}


def fisher_point(router,angles,epsilon=1e-6):
    state=lambda a:router.state(a[None]).squeeze(0)
    psi=state(angles)
    jac=torch.autograd.functional.jacobian(state,angles)
    p=psi.square();dp=2*psi[:,None]*jac
    qfi=4*(jac.T@jac-(jac.T@psi)[:,None]*(psi@jac)[None,:])
    cfi=dp.T@(dp/p[:,None])
    smooth=(p+epsilon)/(1+len(p)*epsilon)
    dsmooth=dp/(1+len(p)*epsilon)
    smooth_cfi=dsmooth.T@(dsmooth/smooth[:,None])
    return {'minimum_probability':float(p.min()),
            'qfi_identity_max_abs':float((qfi-torch.eye(4,dtype=angles.dtype)).abs().max()),
            'cfi_qfi_max_abs':float((cfi-qfi).abs().max()),
            'smooth_cfi_min_eigenvalue':float(torch.linalg.eigvalsh(smooth_cfi).min()),
            'smooth_cfi_max_eigenvalue':float(torch.linalg.eigvalsh(smooth_cfi).max()),
            'probability_floor':epsilon}


def real_fisher(samples=64,seed=20261031):
    torch.manual_seed(seed);rows=[]
    for method in ['angle_natural','product_functional']:
        router=ControlledRouter(7,16,4,method,np.arange(16),np.zeros((16,4)),np.arange(16)).double()
        with torch.no_grad():router.rotation.normal_(std=.7)
        for i in range(samples):
            a=torch.rand(4,dtype=torch.double)*2.6+.2
            rows.append({'method':method,'sample':i,**fisher_point(router,a)})
    assert max(r['qfi_identity_max_abs'] for r in rows)<1e-10
    assert max(r['cfi_qfi_max_abs'] for r in rows)<1e-10
    assert max(r['smooth_cfi_max_eigenvalue'] for r in rows)<=1+1e-10
    return {'rows':rows,'passed':True,
            'equation':'real normalized psi: dp=2 psi dpsi, hence CFI=sum(dp dp^T/p)=4 Jpsi^T Jpsi=QFI away from zero amplitudes; a fixed real orthogonal U preserves the input-angle QFI=I',
            'coordinate':'four encoding angles before the fixed trained real circuit, not encoder weights and not all trainable circuit parameters jointly',
            'scope':'RY/CZ real-state ansatz only; generic complex-state measurements need not saturate QFI; hard top-k and sparse renormalization are outside this smooth metric',
            'implication':'A fixed entangling readout changes top-k boundaries without changing the unsmoothed local full-probability Fisher metric in these angle coordinates.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result={'product_boundary':product_boundary(),'real_fisher':real_fisher(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'passed':True,'product_samples':100000,'fisher_points':128,'output':str(output)}))


if __name__=='__main__':main()
