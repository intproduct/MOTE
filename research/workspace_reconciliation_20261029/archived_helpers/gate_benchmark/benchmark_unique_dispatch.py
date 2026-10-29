"""Measure an equivalent grouped index lookup without changing repository code."""
import copy
import inspect
import json
import os
from pathlib import Path
import statistics
import textwrap
import time
import types
os.environ['HF_HOME'] = str(Path('outputs/hf_cache').resolve())
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
from fitmotn.experiments.gate_benchmark.config import load_config
from fitmotn.experiments.gate_benchmark.models import ImageClassifier, BertClassifier, routed_modules, auxiliary_loss
from fitmotn.experiments.gate_benchmark.data import load_data, bert_path
from fitmotn.experiments.gate_benchmark.runner import seed_all

cfg = load_config('experiments/gate_benchmark/configs/pilot.json')
report = []
for stage, task in [('mlp', 'fashion_mnist'), ('bert', 'sst2')]:
    data, meta = load_data(cfg, task)
    batch = {k: v[:cfg[stage]['batch_size']].cuda() for k, v in data['train'].items()}
    labels = batch.pop('labels')
    for gate in cfg['gates']:
        seed_all(17)
        ref = (ImageClassifier(cfg, gate, 17, meta['input_dim'], meta['classes']) if stage == 'mlp' else
               BertClassifier(cfg, gate, 17, meta['classes'], model_path=bert_path(cfg)[0])).cuda()
        new = copy.deepcopy(ref)
        for _, m in routed_modules(new):
            core = m.mote.core
            source = textwrap.dedent(inspect.getsource(type(core).forward))
            from unique_dispatch_ops import unique_select, unique_add
            source = source.replace('x_sub = x_sites.index_select(0, idx)', 'x_sub = unique_select(x_sites, idx)')
            source = source.replace('w_sub = probs_use.index_select(0, idx)', 'w_sub = unique_select(probs_use, idx)')
            source = source.replace('out_sites.index_add_(0, idx, y_sub)', 'out_sites = unique_add(out_sites, idx, y_sub)')
            namespace = {}
            exec(source, dict(type(core).forward.__globals__, unique_select=unique_select, unique_add=unique_add), namespace)
            core.forward = types.MethodType(namespace['forward'], core)
        opts = [torch.optim.AdamW(m.parameters(), lr=cfg[stage]['lr'], weight_decay=cfg[stage]['weight_decay']) for m in (ref, new)]
        timings = {'reference': [], 'unique': []}
        max_output_diff = 0.0
        max_grad_diff = 0.0
        for step in range(6):
            outputs, gradients = [], []
            for label, model, opt in zip(('reference', 'unique'), (ref, new), opts):
                seed_all(100 + step)
                model.train()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                output = model(**batch)
                loss = torch.nn.functional.cross_entropy(output, labels) + auxiliary_loss(model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                torch.cuda.synchronize()
                timings[label].append(time.perf_counter() - begin)
                outputs.append(output.detach())
                gradients.append({n: p.grad.detach().clone() if p.grad is not None else None for n, p in model.named_parameters()})
                opt.step()
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            max_output_diff = max(max_output_diff, float((outputs[0] - outputs[1]).abs().max()))
            for n, g in gradients[0].items():
                h = gradients[1][n]
                assert (g is None) == (h is None), n
                if g is not None:
                    torch.testing.assert_close(g, h, atol=0, rtol=0)
                    max_grad_diff = max(max_grad_diff, float((g-h).abs().max()))
            for (n, p), (_, q) in zip(ref.named_parameters(), new.named_parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
        item = {'stage': stage, 'gate': gate, 'steps_compared': 6, 'bitwise_equal_outputs_gradients_and_adamw_updates': True,
                'max_output_diff': max_output_diff, 'max_grad_diff': max_grad_diff,
                'forward_backward_clip_seconds': {k: statistics.median(v[1:]) for k, v in timings.items()}, 'timings': timings}
        report.append(item)
        print(json.dumps(item), flush=True)
        del ref, new, opts, gradients
        torch.cuda.empty_cache()
Path('outputs/gate_benchmark/unique_dispatch_benchmark.json').write_text(json.dumps(report, indent=2))
