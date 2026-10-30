"""Audit completed pilots, replay saved routers, and calculate paired contrasts.

Writes only a new review directory. Training artifacts remain immutable.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.data import digest, write_json
from ..routing_stability.runner import setup, read_json, canonical_hash, timestamp
from ..routing_stability.model import CausalLM, full_state_hash
from .runner import REPO
from .pilot import evaluate, rounding, pilot_hashes
from .routers import ControlledRouter
from .metrics import bootstrap_window_mean


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('group', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--review-name', default='review_v1'); args = parser.parse_args()
    group = args.group.resolve(); cfg = read_json(group/'config.json')
    if not (group/'raw_summary.json').exists():
        raise RuntimeError('All training must finish before checkpoint replay')
    root = group.parent.parent
    if (root/'PIPELINE_LOCK.json').exists():
        raise RuntimeError('Training lock exists; do not compete for the GPU')
    dest = group/args.review_name; dest.mkdir(exist_ok=False)
    setup()
    if args.device == 'cuda': torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    issues = []; rows = []; pairs = []; results = {}; curves = {}; nlls = {}; baseline = {}
    missing = [f'{m}_s{s}' for s in cfg['seeds'] for m in cfg['methods']
               if not (group/'runs'/f'{m}_s{s}'/'result.json').exists()]
    if missing: raise RuntimeError(f'Missing runs: {missing}')
    original_hashes = read_json(group/'source_hashes.json')
    if pilot_hashes() != original_hashes: issues.append('Frozen source changed')
    for path, sha in original_hashes.items():
        if digest(group/'source_snapshot'/path) != sha: issues.append(f'Snapshot mismatch: {path}')
    source_cfg = read_json(REPO/cfg['source_config'])
    replay = []; geometry_rows = []
    for seed in cfg['seeds']:
        data = group/'data'/f's{seed}'; geo = read_json(data/'geometry.json')
        checkpoint = REPO/cfg['source_group']/'runs'/f'mlp_half_s{seed}'/'best.pt'
        if digest(checkpoint) != geo['checkpoint_sha256']: issues.append(f'Original checkpoint changed: {seed}')
        model = CausalLM(source_cfg, 'mlp_half', seed).to(args.device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
        model.requires_grad_(False)
        if full_state_hash(model.state_dict()) != geo['frozen_state_hash']: issues.append(f'Backbone state hash: {seed}')
        cache = torch.load(data/'validation.pt', map_location='cpu', weights_only=True)
        window_ids = cache['window'].numpy().copy()
        cache = {k:v.to(args.device) for k,v in cache.items()}
        codes = np.load(data/'codes.npy'); assignment = np.array(geo['expert_for_basis'])
        random_perm = np.array(geo['random_expert_for_basis'])
        baseline[seed] = np.load(data/'baseline_validation_nll.npy').astype(float)
        geometry_rows.append({'seed':seed, 'functional_cost':geo['functional_cost'],
                              'identity_cost':geo['identity_cost'], 'random_mean_cost':geo['random_mean_cost'],
                              'relative_reduction_vs_random':1-geo['functional_cost']/geo['random_mean_cost']})
        samples = []; angle_initial = []
        for method in cfg['methods']:
            run = group/'runs'/f'{method}_s{seed}'; result = read_json(run/'result.json')
            results[method,seed] = result; manifest = read_json(run/'manifest.json')
            if manifest['config_hash'] != canonical_hash(cfg): issues.append(f'Config hash: {run.name}')
            if manifest['expert_pool_sha256'] != geo['checkpoint_sha256']: issues.append(f'Expert pool: {run.name}')
            for name, sha in result['files'].items():
                if digest(run/name) != sha: issues.append(f'File checksum: {run.name}/{name}')
            samples.append(result['files']['sample_indices.npy'])
            indices = np.load(run/'sample_indices.npy')
            if len(indices) != cfg['steps']*cfg['batch_size']: issues.append(f'Sampling count: {run.name}')
            if indices.min() < 0 or indices.max() >= cfg['train_windows']*len(cfg['positions']): issues.append(f'Sampling bounds: {run.name}')
            curve = [json.loads(line) for line in (run/'curves.jsonl').read_text().splitlines()]
            curves[method,seed] = curve
            if [r['step'] for r in curve] != list(range(0,cfg['steps']+1,cfg['eval_every'])): issues.append(f'Curve steps: {run.name}')
            if not all(np.isfinite(r['ce']) for r in curve): issues.append(f'Nonfinite curve: {run.name}')
            if not all(np.isfinite(r['gradient_norm']) and r['gradient_norm'] > 0 for r in curve[1:]): issues.append(f'Gradient: {run.name}')
            if result['finite_steps'] != cfg['steps'] or not result['router_changed']: issues.append(f'Training status: {run.name}')
            saved = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
            if saved['step'] != cfg['steps'] or saved['config'] != cfg: issues.append(f'Checkpoint selection: {run.name}')
            router = ControlledRouter(cache['x'].shape[-1],16,cfg['rank'],method,assignment,codes,random_perm).to(args.device)
            router.load_state_dict(saved['model'], strict=True)
            if method in ['angle_natural','angle_random','angle_functional']:
                # Match trainable initialization, independently of label buffers.
                torch.manual_seed(seed+900000)
                initial = ControlledRouter(cache['x'].shape[-1],16,cfg['rank'],method,assignment,codes,random_perm)
                angle_initial.append(full_state_hash(dict(initial.named_parameters())))
            metric, values = evaluate(model,router,cache); rounded = rounding(model,router,cache)
            stored = np.load(run/'validation_nll.npy'); err = float(np.max(abs(values-stored)))
            if err > 5e-5: issues.append(f'Checkpoint replay NLL exceeds 5e-5: {run.name}: {err}')
            if rounded['flips'] != result['rounding']['flips'] or rounded['harm'] != result['rounding']['harm']:
                issues.append(f'Roundtrip event replay mismatch: {run.name}')
            replay.append({'method':method,'seed':seed,'max_abs_nll_error':err,'ce':metric['ce'],
                           'rounding_flips':rounded['flips'],'rounding_harm':rounded['harm']})
            nlls[method,seed] = stored.astype(float)
            row = {'method':method,'seed':seed,'router_parameters':result['router_parameters'],
                   'baseline_ce':float(baseline[seed].mean()),'validation_ce':metric['ce'],
                   'validation_ppl':metric['ppl'],'delta_ce_vs_original':float((stored-baseline[seed]).mean()),
                   'imitation_end_ce':next(r['ce'] for r in curve if r['step']==cfg['imitation_steps']),
                   'train_ce':result['train']['ce'],'load_cv':metric['load_cv'], 'unused_experts':metric['unused_experts'],
                   'roundtrip_tokens':rounded['tokens'],'roundtrip_flips':rounded['flips'],'roundtrip_harm':rounded['harm'],
                   'roundtrip_abs_gt_01':rounded['abs_gt_01'],'roundtrip_abs_p99':rounded['abs_delta_p99'],
                   'wall_seconds_shared_gpu':result['seconds']}
            rows.append(row)
            print('REPLAY', method, seed, 'max NLL error', err, flush=True)
        if len(set(samples)) != 1: issues.append(f'Sampling differs within seed {seed}')
        if len(set(angle_initial)) != 1: issues.append(f'Angle initialization differs within seed {seed}')
        for candidate,reference in [('angle_functional','angle_random'),('angle_functional','angle_natural'),
                                    ('angle_functional','product_functional'),('angle_functional','spectral_functional'),
                                    ('spectral_functional','spectral_random')]:
            d = nlls[candidate,seed]-nlls[reference,seed]
            ci = bootstrap_window_mean(d,window_ids,20261030+seed,repeats=2000)
            pairs.append({'candidate':candidate,'reference':reference,'seed':seed,'delta_ce':float(d.mean()),
                          'window_bootstrap_ci95_low':ci[0],'window_bootstrap_ci95_high':ci[1],
                          'harm_count_difference':results[candidate,seed]['rounding']['harm']-results[reference,seed]['rounding']['harm']})
        del model,cache,router
        if args.device == 'cuda': torch.cuda.empty_cache()
    aggregate = []
    for method in cfg['methods']:
        subset = [r for r in rows if r['method']==method]
        row = {'method':method,'seeds':len(subset),'router_parameters':subset[0]['router_parameters']}
        for key in ['validation_ce','validation_ppl','delta_ce_vs_original','imitation_end_ce','train_ce','load_cv']:
            values=np.array([r[key] for r in subset]); row[key+'_mean']=float(values.mean());row[key+'_sd']=float(values.std(ddof=1))
        for key in ['roundtrip_tokens','roundtrip_flips','roundtrip_harm','roundtrip_abs_gt_01']:
            row[key+'_total']=sum(r[key] for r in subset)
        row['max_unused_experts']=max(r['unused_experts'] for r in subset)
        aggregate.append(row)
    functional = [r for r in aggregate if r['method']=='angle_functional'][0]
    random = [r for r in aggregate if r['method']=='angle_random'][0]
    criteria = {
        'ce_improves_vs_random_all_three_seeds':all(r['delta_ce']<0 for r in pairs if r['reference']=='angle_random'),
        'ce_within_001_of_classical_each_seed':all(r['delta_ce']<=.01 for r in pairs if r['reference']=='spectral_functional'),
        'harmful_events_lower_than_random_pooled':functional['roundtrip_harm_total']<random['roundtrip_harm_total'],
    }
    audit={'finished':timestamp(),'expected_runs':len(cfg['methods'])*len(cfg['seeds']),'runs':len(rows),
           'checkpoint_replays':len(replay),'missing':missing,'issues':issues,'passed':not issues,
           'device':args.device,'config_hash':canonical_hash(cfg),'criteria':criteria,
           'large_training_eligible':not issues and all(criteria.values()),
           'selection_rule':cfg['continue_criterion'],'geometry':geometry_rows,
           'scope':'Fixed final checkpoint; sampled validation CE and BPE PPL, not full validation. Window CIs are within checkpoint, 3-seed SD across checkpoints. No test access. Shared-GPU wall seconds are not latency comparisons.',
           'review_source_sha256':digest(Path(__file__))}
    csv_write(dest/'runs.csv',rows);csv_write(dest/'aggregate.csv',aggregate);csv_write(dest/'paired.csv',pairs)
    write_json(dest/'checkpoint_replay.json',replay);write_json(dest/'AUDIT.json',audit)
    write_json(dest/'curves.json',[{'method':m,'seed':s,'values':v} for (m,s),v in curves.items()])
    print(json.dumps({'audit':audit,'aggregate':aggregate,'paired':pairs},ensure_ascii=False,indent=2))
    if issues:raise RuntimeError(issues)


if __name__=='__main__':main()
