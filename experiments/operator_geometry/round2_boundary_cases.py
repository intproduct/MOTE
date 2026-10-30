"""Read-only diagnosis of exceptions to the fixed fourth/fifth approximation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .routers import ControlledRouter
from .round2_boundary_gradient import local_boundary
from ..autoregressive_gate.data import digest,write_json


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);repo=Path(__file__).resolve().parents[2];group=Path(args.group);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=read(group/'config.json');source=repo/cfg['source_group'];data=repo/cfg['data_group'];pcfg=read(source/'config.json');rows=[]
    for row in read(group/'raw_summary.json'):
        file=group/f"{row['method']}_s{row['seed']}_{row['split']}.npz";assert digest(file)==row['arrays_sha256'];arrays=np.load(file)
        for condition in ['bf16_input_roundtrip','fp16_input_roundtrip']:
            actual=arrays[condition+'__pair_crossed'];pred=arrays[condition+'__predicted_pair_crossed'];flips=arrays[condition+'__set_flip']
            ix=np.flatnonzero((actual!=pred)|(flips&~actual))
            if not len(ix):continue
            seed,method,split=row['seed'],row['method'],row['split'];path=data/f's{seed}/{split}.pt'
            assert digest(path)==read(data/f's{seed}/result.json')['files'][path.name]
            cache=torch.load(path,map_location='cpu',weights_only=True);x=cache['x'][ix];del cache
            path=source/f'runs/{method}_s{seed}/final.pt';assert digest(path)==row['checkpoint_sha256']
            state=torch.load(path,map_location='cpu',weights_only=False)['model'];geo=read(source/f'data/s{seed}/geometry.json')
            codes=np.load(source/f'data/s{seed}/codes.npy');checks={}
            perturbed=x.to(torch.bfloat16 if condition.startswith('bf16') else torch.float16).float()
            for precision,dtype in [('fp32',torch.float32),('fp64',torch.float64)]:
                router=ControlledRouter(x.shape[-1],16,pcfg['rank'],method,np.asarray(geo['expert_for_basis']),codes,np.asarray(geo['random_expert_for_basis'])).to(dtype)
                router.load_state_dict(state,strict=True);router.requires_grad_(False);router.eval()
                a,pair,margin,gradient=local_boundary(router,x.to(dtype))
                with torch.no_grad():b=router(perturbed.to(dtype))
                order=a.argsort(descending=True,dim=-1);orderq=b.argsort(descending=True,dim=-1)
                for j,index in enumerate(ix):
                    old,new=set(order[j,:4].tolist()),set(orderq[j,:4].tolist())
                    removed=list(old-new);added=list(new-old);logp=a[j].log();logq=b[j].log()
                    detail={'old_top6':order[j,:6].tolist(),'new_top6':orderq[j,:6].tolist(),'removed':removed,'added':added,
                        'old_probability':a[j].tolist(),'new_probability':b[j].tolist(),'fixed_pair':pair[j].tolist(),
                        'fixed_pair_initial_margin':float(margin[j]),'fixed_pair_final_margin':float(logq[pair[j,0]]-logq[pair[j,1]]),
                        'fixed_pair_predicted_final_margin':float(margin[j]+gradient[j]@(perturbed[j].to(dtype)-x[j].to(dtype)))}
                    if removed:
                        detail['removed_original_rank']=int((order[j]==removed[0]).nonzero()[0])+1
                        detail['added_original_rank']=int((order[j]==added[0]).nonzero()[0])+1
                    checks.setdefault(int(index),{})[precision]=detail
            for index in ix:rows.append({'seed':seed,'method':method,'split':split,'condition':condition,'token_row':int(index),'precision_checks':checks[int(index)]})
    write_json(out/'cases.json',{'cases':rows,'source_sha256':digest(Path(__file__)),
        'scope':'Only exceptions to the existing fixed4/5 approximation, identified before FP64 replay. This selected-case analysis is not a full FP64 robustness rate.'})
    print('BOUNDARY_EXCEPTION_CASES',len(rows),flush=True)


if __name__=='__main__':main()
