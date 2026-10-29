"""Read-only method checks; no training, checkpoints or repository source changes."""
import json
from pathlib import Path
import statistics as st
import torch
from torch import nn
from fitmotn.gate import TopKGate, RouterLogits
from fitmotn.experiments.gate_benchmark.config import load_config
from fitmotn.experiments.gate_benchmark.models import router_config

root=Path('outputs/gate_benchmark')
c=load_config('experiments/gate_benchmark/configs/titan_v.json')
torch.set_num_threads(1)

# Use the actual TopKGate and a transparent logits input. Only selected expert
# coefficients contribute, matching the sparse dispatch's incoming gradient.
g=TopKGate(data_dim=16,num_experts=16,k=4,use_ste=True,drop_tokens=False,aux_coeff=0.,zloss_coeff=0.)
g.router=nn.Identity()
x=torch.linspace(-1.,1.,16).reshape(1,16).requires_grad_()
out,mask,_=g(x)
coeff=torch.tensor([[.1,-.2,.3,-.4,.5,-.6,.7,-.8,.9,-1.,1.1,-1.2,1.3,-1.4,1.5,-1.6]])*mask
actual=torch.autograd.grad((out*coeff).sum(),x,retain_graph=True)[0]
p=torch.softmax(x,dim=-1)
h=p*mask
h=h/h.sum(-1,keepdim=True)
hard_grad=torch.autograd.grad((h*coeff).sum(),x,retain_graph=True)[0]
soft_grad=torch.autograd.grad((p*coeff).sum(),x)[0]
assert torch.allclose(actual,hard_grad+soft_grad,atol=1e-7,rtol=1e-6)
assert not torch.allclose(actual,soft_grad,atol=1e-6,rtol=1e-5)
assert not torch.allclose(actual,hard_grad,atol=1e-6,rtol=1e-5)
ste={'forward_matches_topk_normalized':bool(torch.allclose(out,h)),
     'actual_gradient_equals_hard_plus_soft':True,
     'max_error_vs_hard_plus_soft':float((actual-hard_grad-soft_grad).abs().max()),
     'max_difference_vs_soft_only_STE':float((actual-soft_grad).abs().max()),
     'max_difference_vs_hard_only':float((actual-hard_grad).abs().max()),
     'actual':actual.tolist(),'hard':hard_grad.tolist(),'soft':soft_grad.tolist()}

budget=[]
for d in (256,1024,2048):
 for arch in c['gates']:
  rc=router_config(c,arch,d,17)
  model=RouterLogits(d,16,gate_arch=arch,gate_hidden_dim=rc['gate_hidden_dim'],
                     gate_norm=rc['gate_norm'],gate_learnable_gain=rc['gate_learnable_gain'],
                     gate_quantum=rc['gate_quantum'],gate_mps=rc['gate_mps'])
  budget.append({'input_dim':d,'gate':arch,'hidden_or_rank_or_sites':model.resolved_hidden_dim,
                 'parameters_including_norm_and_gain':sum(p.numel() for p in model.parameters())})

initial=[]
for task in c['mlp']['tasks']+c['bert']['tasks']:
 for gate in c['gates']:
  per_seed=[]
  stage='mlp' if task in c['mlp']['tasks'] else 'bert'
  paths=sorted((root/'45c21049b433').glob(f'{stage}_{task}_{gate}_seed*/initial_dev.json'))
  assert len(paths)==3,(task,gate,len(paths))
  for pth in paths:
   routes=json.loads(pth.read_text())['routing'].values()
   per_seed.append({k:st.mean(r[k] for r in routes) for k in ('H_tok','load_cv','unused_experts')})
  initial.append({'task':task,'gate':gate,**{k+'_mean':st.mean(r[k] for r in per_seed) for k in per_seed[0]}})

out={'ste_check':ste,'actual_router_budgets':budget,'initial_dev_routing':initial,
     'note':'These are mathematical/metadata checks of the unchanged implementation; not new training runs.'}
(root/'method_review_checks.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out,indent=2))
