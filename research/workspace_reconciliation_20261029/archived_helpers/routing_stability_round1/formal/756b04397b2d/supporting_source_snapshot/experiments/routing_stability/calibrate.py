"""Training-split calibration only; never uses test for threshold selection."""
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.model import CausalLM
from .runner import setup, read_json, digest, write_json, REPO, timestamp
from .probe import capture
from .routers import standardized_scores
from ..gate_benchmark.models import routed_modules


@torch.no_grad()
def main():
    setup()
    old=REPO/'outputs/autoregressive_gate/formal/9b3af7494afc'
    cfg=read_json(old/'config.json')
    tokens=np.load(REPO/'outputs/autoregressive_gate/data/train.npy')
    starts=np.linspace(0,len(tokens)-129,32,dtype=np.int64)
    ids=torch.tensor(np.stack([tokens[s:s+128] for s in starts]),device='cuda',dtype=torch.long)
    rows=[]
    for variant in ['mlp_half','amp_d1_half']:
        file=old/'runs'/f'{variant}_s17'/'best.pt'
        cp=torch.load(file,map_location='cpu',weights_only=False)
        model=CausalLM(cfg,variant,17).cuda().eval();model.load_state_dict(cp['model'])
        _,saved=capture(model,ids)
        for name,module in routed_modules(model):
            router=module.mote.core.gate.router
            x,s=saved[name];z=standardized_scores(s)
            for dtype,label in [(torch.float16,'fp16'),(torch.bfloat16,'bf16')]:
                other=standardized_scores(router(x.to(dtype).float()))
                two_error=2*(z-other).abs().amax(-1)
                rows.append({'variant':variant,'module':name,'input_roundtrip':label,
                    'twice_error_quantiles':{str(q):float(torch.quantile(two_error,q)) for q in [.5,.95,.99,.999,1.]},
                    'fraction_below_threshold_01':float((two_error < .1).float().mean()),
                    'checkpoint_sha256':digest(file)})
        del model,saved
    write_json(REPO/'outputs/routing_stability_round1/CALIBRATION.json',{
        'time':timestamp(),'split':'train','window_starts':starts.tolist(),'rows':rows,
        'threshold':.1,'weight':1.,'schedule':'zero first10%, linear ramp to30%, then full',
        'selection_rule':'Preselected tau=.1 in centered RMS units; report observed error coverage, not a worst-case guarantee. Same tau/lambda for all arms. Pilot acceptance is numerical/procedural, not a gate ranking.',
        'limitation':'Calibration uses two previously trained seed17 baselines and train inputs. No final test access. Smooth scale sqrt(RMS^2+1e-12) participates in gradients.'})
    print('TRAIN CALIBRATION COMPLETE',flush=True)


if __name__=='__main__':main()
