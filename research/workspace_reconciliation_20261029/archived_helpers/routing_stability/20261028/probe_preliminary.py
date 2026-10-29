"""Post-hoc, no-training routing sensitivity probe on frozen causal-LM checkpoints.

Only validation inputs are used. Input fp16/bf16 casts simulate input rounding;
they are NOT native mixed-precision model/kernel benchmarks on TITAN V.
"""
from pathlib import Path
import copy
import csv
import datetime as dt
import hashlib
import json
import time
import numpy as np
import torch
from fitmotn.experiments.autoregressive_gate.model import CausalLM
from fitmotn.experiments.autoregressive_gate.runner import setup
from fitmotn.experiments.gate_benchmark.models import routed_modules

REPO = Path(__file__).resolve().parents[2]
OLD = REPO/'outputs/autoregressive_gate'
GROUP = OLD/'formal/9b3af7494afc'
OUT = REPO/'outputs/routing_stability/20261028'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, x):
    (OUT/name).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def mask(z, k=4):
    return torch.zeros_like(z, dtype=torch.bool).scatter_(1, z.topk(k, dim=-1).indices, True)

def compare(z, other):
    z, other = z.double(), other.double()
    assert torch.isfinite(z).all() and torch.isfinite(other).all()
    a, b = mask(z), mask(other)
    diff = (z-z.mean(-1, keepdim=True))-(other-other.mean(-1, keepdim=True))
    err = diff.abs().amax(-1)
    top = z.topk(5, dim=-1).values
    margin = top[:, 3]-top[:, 4]
    flip = (a != b).any(-1)
    certified = margin > 2*err
    assert not (flip & certified).any(), 'Top-k margin bound violated'
    return {'n': z.shape[0], 'changed': int(flip.sum()), 'flip_rate': float(flip.double().mean()),
            'mean_replaced_fraction': float((a & ~b).sum(-1).double().mean()/4),
            'max_centered_score_difference': float(err.max()),
            'p99_centered_score_difference': float(torch.quantile(err, .99)),
            'certified_unchanged_fraction': float(certified.double().mean())}

@torch.inference_mode()
def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'RESULTS.json').exists(), 'Refuse to overwrite completed probe.'
    cfg = read(GROUP/'config.json')
    frozen = read(GROUP/'frozen_sources.json')
    assert all(sha(REPO/p) == h for p, h in frozen.items())
    tokens = np.load(OLD/'data/validation.npy')
    # Sixteen fixed disjoint windows spread across validation. Same for all runs.
    starts = np.linspace(0, len(tokens)-129, 16, dtype=np.int64)
    ids = torch.tensor(np.stack([tokens[s:s+128] for s in starts]), dtype=torch.long, device='cuda')
    protocol = {'created': dt.datetime.now().astimezone().isoformat(), 'split': 'validation',
        'windows': 16, 'context': 128, 'tokens_per_module_per_run': 2048,
        'window_starts': starts.tolist(), 'k': 4, 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'precision': 'FP32; TF32 disabled',
        'input_rounding': 'fp16/bf16 roundtrip -> fp32, weights and arithmetic stay fp32',
        'noise': 'one common fixed uniform[-1,1] direction per seed/module, scaled by per-token input RMS',
        'scope': 'diagnostic only, no optimization, no held-out test tuning, no native serving-engine comparison',
        'frozen_sources': frozen, 'script_sha256': sha(__file__)}
    write('PROTOCOL.json', protocol)
    rows, distributions, end_to_end, checkpoints = [], [], [], []
    begin = time.perf_counter()
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            path = GROUP/'runs'/f'{variant}_s{seed}'/'best.pt'
            checksum = sha(path)
            cp = torch.load(path, map_location='cpu', weights_only=False)
            model = CausalLM(cfg, variant, seed).cuda().eval()
            model.load_state_dict(cp['model'])
            modules = [(name, m.mote.core.gate.router) for name, m in routed_modules(model)]

            def capture(x):
                saved, hooks = {}, []
                for name, router in modules:
                    def hook(mod, inp, out, name=name):
                        saved[name] = (inp[0].detach().clone(), out.detach().clone())
                    hooks.append(router.register_forward_hook(hook))
                try:
                    logits = model(x)
                finally:
                    for h in hooks:
                        h.remove()
                return logits, saved

            full, base = capture(ids)
            single_scores = {n: [] for n, _ in modules}
            single_logits = []
            for i in range(len(ids)):
                output, got = capture(ids[i:i+1])
                single_logits.append(output)
                for name in single_scores:
                    single_scores[name].append(got[name][1])
            single = torch.cat(single_logits)
            short, short_scores = capture(ids[:, :64])
            for condition, ref, out in [('batch16_vs_batch1', full, single), ('causal_length128_vs64', full[:, :64], short)]:
                end_to_end.append({'variant': variant, 'seed': seed, 'condition': condition,
                    'max_vocab_logit_difference': float((ref-out).abs().max()),
                    'next_token_argmax_changed': int((ref.argmax(-1)!=out.argmax(-1)).sum()),
                    'n_predictions': ref.shape[0]*ref.shape[1]})

            for mi, (name, router) in enumerate(modules):
                x, z = base[name]
                identity = {'variant': variant, 'seed': seed, 'module': name}
                top = z.double().topk(5, dim=-1).values
                margin = top[:, 3]-top[:, 4]
                scale = (z.double()-z.double().mean(-1, keepdim=True)).square().mean(-1).sqrt().clamp_min(1e-30)
                normalized = margin/scale
                distributions.append({**identity, 'n': len(x), 'exact_boundary_ties': int((margin==0).sum()),
                    'margin_quantiles': {str(q): float(torch.quantile(margin, q)) for q in [0., .01, .05, .5, .95]},
                    'relative_margin_quantiles': {str(q): float(torch.quantile(normalized, q)) for q in [0., .01, .05, .5, .95]}})
                conditions = {'identical_repeat': router(x),
                    'router_chunk32': torch.cat([router(t) for t in x.split(32)]),
                    'input_roundtrip_fp16': router(x.half().float()),
                    'input_roundtrip_bf16': router(x.bfloat16().float()),
                    'model_batch16_vs_batch1': torch.cat(single_scores[name])}
                r64 = copy.deepcopy(router).double()
                conditions['router_fp64_reference'] = r64(x.double())
                del r64
                gen = torch.Generator(device='cuda').manual_seed(seed*100+mi+20261028)
                direction = torch.rand(x.shape, generator=gen, device=x.device)*2-1
                rms = x.square().mean(-1, keepdim=True).sqrt()
                for level in [1e-6, 1e-4, 1e-3, 1e-2]:
                    conditions[f'input_uniform_rms_{level:g}'] = router(x + level*rms*direction)
                for condition, candidate in conditions.items():
                    rows.append({**identity, 'condition': condition, **compare(z, candidate)})
                sliced = z.reshape(16, 128, -1)[:, :64].reshape(-1, 16)
                rows.append({**identity, 'condition': 'model_causal_length128_vs64',
                             **compare(sliced, short_scores[name][1])})
            assert sha(path) == checksum
            checkpoints.append({'variant': variant, 'seed': seed, 'sha256': checksum})
            write('PARTIAL.json', {'rows': rows, 'distributions': distributions, 'end_to_end': end_to_end})
            print(f'DONE {variant}_s{seed}; elapsed={time.perf_counter()-begin:.1f}s', flush=True)
            del model, cp, base, conditions, full, single, short
            torch.cuda.empty_cache()
    assert all(sha(REPO/p) == h for p, h in frozen.items())
    result = {'protocol': protocol, 'rows': rows, 'distributions': distributions,
              'end_to_end': end_to_end, 'checkpoints': checkpoints, 'elapsed_seconds': time.perf_counter()-begin,
              'frozen_sources_unchanged': True, 'checkpoints_unchanged': True}
    write('RESULTS.json', result)
    with (OUT/'sensitivity.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'COMPLETE: {len(checkpoints)} checkpoints; {len(rows)} module comparisons', flush=True)

if __name__ == '__main__':
    main()
