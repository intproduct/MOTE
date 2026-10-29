import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import torch
from fitmotn.experiments.gate_benchmark.config import load_config
from fitmotn.experiments.gate_benchmark.data import load_data
from fitmotn.experiments.gate_benchmark.models import ImageClassifier, auxiliary_loss
from fitmotn.experiments.gate_benchmark.runner import seed_all
c = load_config('experiments/gate_benchmark/configs/pilot.json')
seed_all(17)
data, meta = load_data(c, 'fashion_mnist')
m = ImageClassifier(c, 'mlp', 17, 784, 10).cuda()
x, y = data['train']['features'][:128].cuda(), data['train']['labels'][:128].cuda()
opt = torch.optim.AdamW(m.parameters(), lr=.001)
def step():
    opt.zero_grad(set_to_none=True)
    with torch.profiler.record_function('FORWARD'):
        out = m(features=x)
        loss = torch.nn.functional.cross_entropy(out, y) + auxiliary_loss(m)
    with torch.profiler.record_function('BACKWARD'):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
    with torch.profiler.record_function('OPTIMIZER'):
        opt.step()
    torch.cuda.synchronize()
for _ in range(2):
    step()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
    for _ in range(2):
        step()
print(p.key_averages().table(sort_by='self_cpu_time_total', row_limit=35))
print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
p.export_chrome_trace('outputs/gate_benchmark/logs/step_profile_trace.json')
