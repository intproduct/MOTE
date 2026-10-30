"""Check the analytic eight-angle QFI of RY(b) CZ RY(a)."""
from pathlib import Path
import json
import numpy as np
import torch
from .round3_capacity import ReencodingPredictor

def analytic_metric(a,entangle=True):
    n=len(a);out=torch.eye(2*n,dtype=a.dtype,device=a.device)
    if not entangle:
        out[:n,n:]=torch.eye(n,dtype=a.dtype);out[n:,:n]=torch.eye(n,dtype=a.dtype);return out
    def neighbors(i):return [j for j in [i-1,i+1] if 0<=j<n]
    for i in range(n):
        cross=torch.prod(torch.cos(a[neighbors(i)]));out[i,n+i]=cross;out[n+i,i]=cross
        for j in neighbors(i):
            outer=[k for k in neighbors(i)+neighbors(j) if k not in [i,j]]
            v=torch.sin(a[i])*torch.sin(a[j])*torch.prod(torch.cos(a[outer]));out[n+i,n+j]=v
    return out

def run():
    torch.set_num_threads(1);torch.manual_seed(2026103012)
    out=Path('outputs/operator_geometry_round3/theory/interleaved_qfi_v1');out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for method in ['reencode_cz','reencode_product']:
        model=ReencodingPredictor(12,8,method,np.arange(16)).double()
        for index in range(128):
            angles=torch.rand(8,dtype=torch.float64)*2.6+.2
            fn=lambda z:model.state(z[None])[0]
            psi=fn(angles);jac=torch.autograd.functional.jacobian(fn,angles)
            qfi=4*(jac.T@jac-(jac.T@psi[:,None])@(psi[None,:]@jac))
            expected=analytic_metric(angles[:4],method=='reencode_cz')
            err=float(abs(qfi-expected).max());assert err<1e-12
            jp=2*psi[:,None]*jac;measured=jp.T@(jp/psi.square()[:,None])
            assert float(abs(measured-qfi).max())<1e-12
            floored=jp.T@(jp/(psi.square()[:,None]+1e-6))/(1+16e-6)
            eig=torch.linalg.eigvalsh(qfi);gap=torch.linalg.eigvalsh(qfi-floored)
            assert float(gap.min())>-1e-11
            rank=int(torch.linalg.matrix_rank(qfi,atol=1e-8));assert rank==(8 if method=='reencode_cz' else 4)
            rows.append({'method':method,'point':index,'formula_max_abs':err,'rank':rank,'qfi_eigenvalues':eig.tolist(),
                'floor_information_loss_trace':float(torch.trace(qfi-floored)), 'minimum_floor_loss_eigenvalue':float(gap.min())})
    center=torch.full((4,),torch.pi/2,dtype=torch.float64)
    torch.testing.assert_close(analytic_metric(center),torch.eye(8,dtype=torch.float64),atol=1e-12,rtol=0)
    result={'passed':True,'cases':len(rows),'rows':rows,'pi_over_2_encoding_metric_identity':True,
        'formula':'For chain neighbors N(i): Faa=I; Fab_ij=delta_ij product_{k in N(i)}cos(a_k); Fbb_ii=1; Fbb_ij=sin(a_i)sin(a_j) product_{k in (N(i) union N(j)) excluding{i,j}}cos(a_k) if adjacent, otherwise0. Product/noCZ: [[I,I],[I,I]].',
        'scope':'Coordinates are8 independent data angles before input-network pullback, fixedCZ, full real unfloored Born probabilities. Nativefloor reduces measured Fisher. Finite256 numerical checks support the analytic derivation; genericrank conclusions require nondegenerate coordinates.'}
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('METRIC_PASS',len(rows),'max_error',max(r['formula_max_abs'] for r in rows),flush=True)

if __name__=='__main__':run()
