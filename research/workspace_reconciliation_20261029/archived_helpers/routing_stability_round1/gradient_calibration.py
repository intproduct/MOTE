import json
from pathlib import Path
import torch
from torch.nn import functional as F
from fitmotn.experiments.routing_stability.runner import setup, load_tokens, training_batch, write_json
from fitmotn.experiments.routing_stability.model import CausalLM
from fitmotn.experiments.routing_stability.routers import boundary_loss,SharedBornRouter
from fitmotn.experiments.gate_benchmark.models import auxiliary_loss,routed_modules,is_router_key

setup();root=Path('outputs/routing_stability_round1');group=root/'pilot/5b7d1b6a5189'
tokens=load_tokens('outputs/autoregressive_gate/data','train')
rows=[]
for variant in ['mlp_half','amp_half','born_lite']:
    cp=torch.load(group/'runs'/f'{variant}_s17'/'best.pt',map_location='cpu',weights_only=False)
    cfg=cp['config'];model=CausalLM(cfg,variant,17).cuda().train();model.load_state_dict(cp['model'])
    terms=[];handles=[]
    for _,module in routed_modules(model):
        def hook(router,inp,out):
            s=out.log() if isinstance(router,SharedBornRouter) else out
            terms.append(boundary_loss(s,4,.05)[0])
        handles.append(module.mote.core.gate.router.register_forward_hook(hook))
    x,y,_=training_batch(tokens,32,128,torch.Generator().manual_seed(735921),'cuda')
    logits=model(x);task=F.cross_entropy(logits.flatten(0,1),y.flatten())+auxiliary_loss(model)
    margin=torch.stack(terms).mean()
    params=[p for n,p in model.named_parameters() if is_router_key(n)]
    task_grad=torch.autograd.grad(task,params,retain_graph=True,allow_unused=True)
    margin_grad=torch.autograd.grad(margin,params,allow_unused=True)
    norm=lambda gs:float(torch.stack([g.square().sum() for g in gs if g is not None]).sum().sqrt())
    a,b=norm(task_grad),norm(margin_grad)
    rows.append({'variant':variant,'train_task_loss':float(task),'raw_margin_loss':float(margin),
                 'task_router_grad_norm':a,'margin_router_grad_norm':b,
                 'weighted_ratio_lambda001':.01*b/a,'weighted_ratio_lambda01':.1*b/a})
    for h in handles:h.remove()
    print(rows[-1],flush=True)
write_json(root/'GRADIENT_CALIBRATION.json',{'split':'train','threshold':.05,'pilot_checkpoint_step':120,'rows':rows})
