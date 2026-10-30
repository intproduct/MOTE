"""CPU-only state-angle to selected-mixture tangent checks."""
from pathlib import Path
import json,math
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp
from .round3_capacity import ReencodingPredictor
from .round3_functional import ROOT,read
from .round3_replay_review import csvwrite

def tangent(model,angles,beta,floor):
    fn=lambda a:model.state(a[None])[0]
    psi=fn(angles);js=torch.autograd.functional.jacobian(fn,angles)
    p=psi.square();jp=2*psi[:,None]*js;idx=p.topk(4).indices
    w=(beta*(p[idx]+floor).log()).softmax(0)
    j=(torch.diag(w)-w[:,None]*w[None,:])@(beta*jp[idx]/(p[idx,None]+floor))
    singular=torch.linalg.svdvals(j)
    return p,idx,w,j,singular

def checks():
    torch.manual_seed(17683);rows=[];bits=((torch.arange(16)[:,None]>>torch.arange(3,-1,-1))&1).double()
    for method in ['reencode_cz','reencode_product']:
        model=ReencodingPredictor(12,8,method,np.arange(16)).double()
        for n in range(64):
            a=torch.rand(8,dtype=torch.float64)*2+.2;beta=1.4
            p,idx,w,j,sv=tangent(model,a,beta,0.)
            fn=lambda z:(beta*(model.state(z[None])[0].square()[idx]).log()).softmax(0)
            exact=torch.autograd.functional.jacobian(fn,a);err=float(abs(exact-j).max());assert err<1e-11
            direction=torch.randn(8,dtype=torch.float64);h=1e-6;fd=(fn(a+h*direction)-fn(a-h*direction))/(2*h)
            assert float(abs(fd-j@direction).max())<1e-7
            rank=int((sv>1e-8).sum());bitrank=int(torch.linalg.matrix_rank(bits[idx]-bits[idx].mean(0),atol=1e-10))
            if method=='reencode_product':assert rank==bitrank and rank in [2,3]
            else:assert rank==3
            rows.append({'method':method,'point':n,'rank':rank,'bit_affine_rank':bitrank,'chain_autodiff_max_error':err})
    return rows

def run():
    setup();toy=checks();root=ROOT/'outputs/operator_geometry_round3';out=root/'theory/sparse_tangent_v1';out.mkdir(parents=True,exist_ok=False)
    functional=Path(read(root/'functional/STATUS.json')['group']);cfg=read(functional/'config.json');rows=[];summaries=[];hashes={}
    bits=((torch.arange(16)[:,None]>>torch.arange(3,-1,-1))&1).double()
    for seed in cfg['seeds']:
        geo=read(ROOT/cfg['geometry_group']/f's{seed}/assignment.json');assignment=np.asarray(geo['expert_for_basis'])
        for method in ['reencode_cz','reencode_product']:
            for objective in ['initial','coefficient','output','fisher']:
                path=(ROOT/cfg['capacity_group']/f'runs/{method}_s{seed}/best_dev.pt' if objective=='initial' else functional/'runs'/f'{method}_{objective}_s{seed}'/'best_dev.pt')
                ck=torch.load(path,map_location='cpu',weights_only=False);hashes[str(path.relative_to(ROOT))]=digest(path)
                model=ReencodingPredictor(1024,8,method,assignment).double().eval();model.load_state_dict(ck['model'])
                beta=float(model.log_scale.clamp(-3,3).exp().detach())*ck['scale'] if objective=='initial' else ck['beta']
                for split in ['dev','validation']:
                    x=torch.load(ROOT/cfg['data_group']/f's{seed}/{split}.pt',map_location='cpu',weights_only=True)['x'][::128].double()
                    with torch.no_grad():v=model.output(torch.tanh(model.input_proj(model.norm(x))));angles=v.new_tensor([math.pi/2]*4+[.35]*4)+(math.pi/2)*v.tanh()
                    block=[]
                    for i,a in enumerate(angles):
                        for floor in [0.,1e-6]:
                            p,idx,w,j,sv=tangent(model,a,beta,floor)
                            br=int(torch.linalg.matrix_rank(bits[idx]-bits[idx].mean(0),atol=1e-10));rank=int((sv>1e-8).sum())
                            assert rank<=3 and torch.isfinite(sv).all()
                            if method=='reencode_product' and floor==0:assert rank==br
                            r={'seed':seed,'method':method,'objective':objective,'split':split,'position_index':i*128,'floor':floor,'rank_abs_1e8':rank,'bit_affine_rank':br,
                               's1':float(sv[0]),'s2':float(sv[1]),'s3':float(sv[2]),'s3_over_s1':float(sv[2]/sv[0]),'selected_basis':idx.tolist()}
                            rows.append(r);block.append(r)
                    for floor in [0.,1e-6]:
                        z=[r for r in block if r['floor']==floor]
                        summaries.append({'seed':seed,'method':method,'objective':objective,'split':split,'floor':floor,'positions':len(z),'affine_rank2_count':sum(r['bit_affine_rank']==2 for r in z),
                                          'weight_rank2_count':sum(r['rank_abs_1e8']==2 for r in z),'median_s3_over_s1':float(np.median([r['s3_over_s1'] for r in z]))})
    assert len(rows)==3072;write_json(out/'raw.json',rows);csvwrite(out/'summary.csv',summaries);write_json(out/'SUMMARY.json',summaries);write_json(out/'CHECKPOINT_HASHES.json',hashes)
    write_json(out/'AUDIT.json',{'passed':True,'toy_checks':toy,'checkpoint_count':len(hashes),'positions':len(rows)//2,'jacobians':len(rows),'script_sha256':digest(Path(__file__)),'plan_sha256':digest(root/'SPARSE_TANGENT_PLAN.json'),'completed':timestamp()})
    print(json.dumps({'passed':True,'checkpoints':len(hashes),'jacobians':len(rows)}),flush=True)

if __name__=='__main__':run()
