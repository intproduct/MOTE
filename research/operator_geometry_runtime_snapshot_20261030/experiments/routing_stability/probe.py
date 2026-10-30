"""Validation-only numerical stress tests; input rounding is not native BF16."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .runner import read_json, canonical_hash, setup, timestamp, digest, write_json, REPO
from .model import CausalLM
from .routers import SharedBornRouter, standardized_scores
from ..gate_benchmark.models import routed_modules


def top_mask(z, k):
    return torch.zeros_like(z,dtype=torch.bool).scatter_(-1,z.topk(k,-1).indices,True)


def compare(z, other, k):
    a,b=top_mask(z,k),top_mask(other,k)
    z,other=z.double(),other.double()
    if not torch.isfinite(z).all() or not torch.isfinite(other).all():
        raise FloatingPointError('non-finite router stress test')
    error=((z-z.mean(-1,keepdim=True))-(other-other.mean(-1,keepdim=True))).abs().amax(-1)
    top=z.topk(k+1,-1).values;gap=top[...,k-1]-top[...,k]
    flip=(a!=b).any(-1); safe=gap>2*error
    assert not (safe & flip).any(), 'measured sufficient margin bound violated'
    return {'n':len(z),'changed':int(flip.sum()),'replaced':int((a & ~b).sum()),
            'measured_bound_safe':int(safe.sum()),'max_score_difference':float(error.max())}


def capture(model, ids):
    values={}; handles=[]
    for name,module in routed_modules(model):
        def hook(router,inp,out,name=name):
            values[name]=(inp[0].detach().clone(),out.detach().clone())
        handles.append(module.mote.core.gate.router.register_forward_hook(hook))
    try:
        logits=model(ids)
    finally:
        for h in handles:h.remove()
    return logits,values


def output_error(ref,other,targets):
    a=F.log_softmax(ref,-1).gather(-1,targets[...,None]).squeeze(-1)
    b=F.log_softmax(other,-1).gather(-1,targets[...,None]).squeeze(-1)
    delta=(a-b).abs().flatten().cpu()
    return {'n':delta.numel(),'target_logprob_abs_sum':float(delta.sum()),
            'target_logprob_abs_max':float(delta.max()),
            'target_logprob_abs_p99':float(torch.quantile(delta,.99)),
            'target_logprob_abs_gt_01':int((delta>.1).sum()),
            'next_token_changed':int((ref.argmax(-1)!=other.argmax(-1)).sum()),
            'max_vocab_logit_difference':float((ref-other).abs().max())}


@torch.no_grad()
def probe_run(path,cfg,data_root):
    out=path/'stability.json'
    if out.exists():
        assert read_json(out)['checkpoint_sha256']==digest(path/'best.pt')
        return
    checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
    model=CausalLM(cfg,checkpoint['variant'],checkpoint['seed']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    tokens=np.load(data_root/'validation.npy')
    length=cfg['model']['context']; windows=cfg['probe']['windows']; batch=cfg['probe']['batch']
    starts=np.linspace(0,len(tokens)-length-1,windows,dtype=np.int64)
    rows=[]; distributions=[]; end_to_end=[]
    modules=[(n,m.mote.core.gate.router) for n,m in routed_modules(model)]
    for offset in range(0,windows,batch):
        chosen=starts[offset:offset+batch]
        ids=torch.tensor(np.stack([tokens[s:s+length] for s in chosen]),device='cuda',dtype=torch.long)
        targets=torch.tensor(np.stack([tokens[s+1:s+length+1] for s in chosen]),device='cuda',dtype=torch.long)
        reference,saved=capture(model,ids)
        singles=[]; single_scores={n:[] for n,_ in modules}
        for i in range(len(ids)):
            output,scores=capture(model,ids[i:i+1]);singles.append(output)
            for n in single_scores:single_scores[n].append(scores[n][1])
        candidate=torch.cat(singles)
        end_to_end.append({'batch_offset':offset,'condition':'batch16_vs1',**output_error(reference,candidate,targets)})
        for index,(name,router) in enumerate(modules):
            x,z=saved[name]
            key={'module':name,'batch_offset':offset}
            scale_scores=z.log() if isinstance(router,SharedBornRouter) else z
            top=standardized_scores(scale_scores).topk(cfg['router']['topk']+1,-1).values
            gap=top[...,-2]-top[...,-1]
            distributions.append({**key,'n':len(gap),'gap_mean':float(gap.mean()),
                'gap_p01':float(torch.quantile(gap,.01)),'gap_p05':float(torch.quantile(gap,.05)),
                'gap_median':float(gap.median()),'ties':int((gap==0).sum()),
                'below_threshold':int((gap<cfg['margin']['threshold']).sum())})
            candidates={'repeat':router(x),'batch16_vs1':torch.cat(single_scores[name]),
                        'input_fp16_roundtrip':router(x.half().float()),
                        'input_bf16_roundtrip':router(x.bfloat16().float())}
            generator=torch.Generator(device='cuda').manual_seed(cfg['probe']['seed']+index+offset)
            direction=torch.rand(x.shape,device=x.device,generator=generator)*2-1
            rms=x.square().mean(-1,keepdim=True).sqrt()
            for level in cfg['probe']['rms_levels']:
                candidates[f'input_uniform_rms_{level:g}']=router(x+level*rms*direction)
            for condition,value in candidates.items():
                rows.append({**key,'condition':condition,**compare(z,value,cfg['router']['topk'])})
        # All routers receive rounded hidden inputs in the same forward, allowing
        # propagation through subsequent layers. Expert input itself is not cast.
        for dtype,label in [(torch.float16,'input_fp16_roundtrip'),(torch.bfloat16,'input_bf16_roundtrip')]:
            handles=[]
            for _,router in modules:
                def rounding(module,inp,dtype=dtype):return (inp[0].to(dtype).float(),)
                handles.append(router.register_forward_pre_hook(rounding))
            try:
                candidate,candidate_scores=capture(model,ids)
            finally:
                for h in handles:h.remove()
            end_to_end.append({'batch_offset':offset,'condition':label,**output_error(reference,candidate,targets)})
            for name,_ in modules:
                rows.append({'module':name,'batch_offset':offset,'condition':'propagated_'+label,
                             **compare(saved[name][1],candidate_scores[name][1],cfg['router']['topk'])})
        del reference,saved,candidates,candidate,singles
    result={'variant':checkpoint['variant'],'seed':checkpoint['seed'],
            'checkpoint_sha256':digest(path/'best.pt'),'time':timestamp(),
            'protocol':{'split':'validation','window_starts':starts.tolist(),'context':length,
                'precision':'FP32; FP16/BF16 roundtrip inputs only, not native low-precision kernels',
                'scope':'same fixed weights; no training or test-based selection; no cross-engine/RL claim',
                'mean_of_batch_quantiles':'p01/p05/p99 summaries are per-batch, not pooled quantiles'},
            'rows':rows,'gap_distributions':distributions,'end_to_end':end_to_end}
    write_json(out,result)
    print('STABILITY',path.name,len(rows),'comparisons',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--root',default='outputs/routing_stability_round1');args=parser.parse_args()
    setup();cfg=read_json(args.config);group=Path(args.root)/cfg['phase']/canonical_hash(cfg)[:12]
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            probe_run(group/'runs'/f'{variant}_s{seed}',cfg,(REPO/cfg['data_root']).resolve())
            torch.cuda.empty_cache()


if __name__=='__main__':main()
