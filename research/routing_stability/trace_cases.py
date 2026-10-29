"""Inspect two observed batch-shape route changes; evaluation only."""
from pathlib import Path
import json
import numpy as np
import torch
from fitmotn.experiments.autoregressive_gate.model import CausalLM
from fitmotn.experiments.autoregressive_gate.runner import setup
from fitmotn.experiments.gate_benchmark.models import routed_modules

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/routing_stability/20261028'
OLD=ROOT/'outputs/autoregressive_gate'
cfg=json.loads((OLD/'formal/9b3af7494afc/config.json').read_text())
protocol=json.loads((OUT/'native_topk/PROTOCOL.json').read_text())
tokens=np.load(OLD/'data/validation.npy')
cases=[]
setup()
with torch.inference_mode():
 for variant,seed in [('mpo_c1_half',17),('mlp_half',29)]:
    model=CausalLM(cfg,variant,seed).cuda().eval()
    model.load_state_dict(torch.load(OLD/f'formal/9b3af7494afc/runs/{variant}_s{seed}/best.pt',map_location='cpu',weights_only=False)['model'])
    saved={}
    def hook(m,inp,out):
        saved['input']=inp[0].clone();saved['score']=out.clone()
    router=dict(routed_modules(model))['blocks.1.down'].mote.core.gate.router
    h=router.register_forward_hook(hook)
    ids=torch.tensor(np.stack([tokens[s:s+128] for s in protocol['window_starts']]),device='cuda',dtype=torch.long)
    full=model(ids);base=saved.copy()
    output=[];inputs=[];scores=[]
    for row in ids:
        output.append(model(row.unsqueeze(0)));inputs.append(saved['input']);scores.append(saved['score'])
    output=torch.cat(output);input2=torch.cat(inputs);score2=torch.cat(scores)
    idx1=base['score'].topk(4,dim=-1).indices.sort(-1).values
    idx2=score2.topk(4,dim=-1).indices.sort(-1).values
    changed=(idx1!=idx2).any(-1).nonzero().flatten()
    for n in changed.tolist():
        b,t=divmod(n,128)
        z1,z2=base['score'][n],score2[n]
        values,indices=z1.topk(5)
        target=int(tokens[protocol['window_starts'][b]+t+1])
        lp1=full[b,t].log_softmax(-1);lp2=output[b,t].log_softmax(-1)
        cases.append({'variant':variant,'seed':seed,'module':'blocks.1.down','window':b,'position':t,
            'experts_batch16':idx1[n].tolist(),'experts_batch1':idx2[n].tolist(),
            'top5_ids_batch16':indices.tolist(),'top5_scores_batch16':values.tolist(),
            'same_five_scores_batch1':z2[indices].tolist(),
            'margin_batch16':float(values[3]-values[4]),
            'input_max_abs_difference':float((base['input'][n]-input2[n]).abs().max()),
            'router_score_max_abs_difference':float((z1-z2).abs().max()),
            'vocab_logits_max_abs_difference':float((full[b,t]-output[b,t]).abs().max()),
            'target_token_logprob_delta':float(lp2[target]-lp1[target]),
            'predicted_token_batch16':int(full[b,t].argmax()),'predicted_token_batch1':int(output[b,t].argmax())})
    h.remove()
    del model
    torch.cuda.empty_cache()
(OUT/'CASE_STUDIES.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
print(json.dumps(cases,indent=2))
