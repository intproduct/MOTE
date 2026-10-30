"""Separate pre-softmax selection, probability ties and batch arithmetic."""
from pathlib import Path
import json,os
import numpy as np
import torch
from ..routing_stability.runner import setup
from ..routing_stability.model import CausalLM
from .round3_replay import reweight

def main():
    setup();torch.cuda.set_per_process_memory_fraction(.35)
    base=Path('outputs/operator_geometry_round3/diagnostics/selector_v1');base.mkdir(parents=True,exist_ok=False)
    lock=Path('outputs/operator_geometry_round2/PIPELINE_LOCK.json')
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump({'stage':'selector_diagnostic','pid':os.getpid()},f)
    try:
        cfg=json.loads(Path('experiments/routing_stability/configs/titan_v.json').read_text());rows=[]
        for seed in [17,29,43]:
            model=CausalLM(cfg,'mlp_half',seed).cuda().eval();model.load_state_dict(torch.load(f'outputs/routing_stability_round1/formal/756b04397b2d/runs/mlp_half_s{seed}/best.pt',map_location='cpu',weights_only=False)['model']);model.requires_grad_(False)
            gate=model.get_submodule('blocks.1.down').mote.core.gate
            for split in ['dev','validation']:
                c=torch.load(f'outputs/operator_geometry_round2/data/86a2ba9f5aa2/s{seed}/{split}.pt',map_location='cpu',weights_only=True)
                logits=c['scores'].cuda();p=c['teacher'].cuda();ref=c['weights'].cuda();idx=logits.topk(4,1).indices
                w=torch.zeros_like(p).scatter(1,idx,p.gather(1,idx));w/=w.sum(1,keepdim=True)
                probw=reweight(p);cached_logits_set_change=((w>0)!=(ref>0)).any(1)
                cached_prob_set_change=((probw>0)!=(ref>0)).any(1)
                pred=torch.cat([gate.router(c['x'][i:i+256].cuda()).cpu() for i in range(0,len(logits),256)]).cuda()
                pp=gate._softmax_temp(pred);idx2=pred.topk(4,1).indices
                exact=torch.zeros_like(pp).scatter(1,idx2,pp.gather(1,idx2));exact/=exact.sum(1,keepdim=True)
                probnew=reweight(pp);exactchange=((exact>0)!=(ref>0)).any(1);probchange=((probnew>0)!=(ref>0)).any(1)
                cases=[]
                for index in torch.where(cached_prob_set_change|exactchange|probchange)[0].tolist():
                    one=gate.router(c['x'][index:index+1].cuda());direct=gate(c['x'][index:index+1].cuda())[0]
                    order=logits[index].argsort(descending=True);chosen=order[2:6]
                    cases.append({'index':index,'window':int(c['window'][index]),'position':int(c['position'][index]),'rank3to6':chosen.tolist(),
                        'cached_logits':logits[index,chosen].tolist(),'cached_p':p[index,chosen].tolist(),
                        'batch256_logits':pred[index,chosen].tolist(),'batch256_p':pp[index,chosen].tolist(),'batch1_logits':one[0,chosen].tolist(),
                        'cached_selected':torch.where(ref[index]>0)[0].tolist(),'batch256_logits_selected':torch.where(exact[index]>0)[0].tolist(),
                        'batch256_p_selected':torch.where(probnew[index]>0)[0].tolist(),'batch1_actual_gate_selected':torch.where(direct[0]>0)[0].tolist()})
                rows.append({'seed':seed,'split':split,'cached_logits_selection_changes':int(cached_logits_set_change.sum()),'cached_prob_selection_changes':int(cached_prob_set_change.sum()),
                    'regenerated_logits_selection_changes':int(exactchange.sum()),'regenerated_prob_selection_changes':int(probchange.sum()),
                    'cached_logits_weight_max_abs':float(abs(w-ref).max()),'regenerated_logits_max_abs':float(abs(pred-logits).max()),'cases':cases})
                print('DIAGNOSED',seed,split,rows[-1],flush=True)
            del model
        (base/'summary.json').write_text(json.dumps({'rows':rows,'scope':'Originalgate selects pre-softmax logits. Probability-based surrogate is a different finite-precision selector. Compare native batch1/256 separately.'},indent=2)+'\n',encoding='utf8')
    finally:lock.unlink()

if __name__=='__main__':main()
