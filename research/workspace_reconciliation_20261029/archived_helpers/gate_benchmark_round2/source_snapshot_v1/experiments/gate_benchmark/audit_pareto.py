"""Read-only validation of the parameter-efficiency matrix and saved checkpoints."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import traceback
import torch
from .config import load_config, digest
from .data import load_data, bert_path
from .models import ImageClassifier, BertClassifier, is_router_key, state_hash
from .runner import seed_all, write_json


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def audit(cfg, check_models=True):
    root = Path(cfg['output_dir']) / digest(cfg)[:12]
    issues, rows, paired = [], [], defaultdict(list)
    data_records = {}
    for stage in ['mlp', 'bert']:
        for task in cfg[stage]['tasks']:
            data, meta = load_data(cfg, task)
            ix = meta['indices']
            assert not set(ix['train']) & set(ix['dev']), task
            assert all(len(v) == len(set(v)) for v in ix.values()), task
            assert all(len(data[s]['labels']) == len(ix[s]) == meta['sizes'][s] for s in ix), task
            assert meta['final_source_split'] == ('test' if stage == 'mlp' else 'validation'), task
            data_records[task] = dict(sizes=meta['sizes'], source_split=meta['final_source_split'],
                                      sha256=meta['tensor_file_sha256'], train_dev_disjoint=True)
            for seed in cfg['seeds']:
                for gate in cfg['gates']:
                    dest = root/f'{stage}_{task}_{gate}_seed{seed}'
                    try:
                        for name in ['manifest.json', 'initial_dev.json', 'curves.jsonl', 'best.pt', 'result.json']:
                            assert (dest/name).is_file() and (dest/name).stat().st_size > 0, name+' missing'
                        m, r, initial = [json.loads((dest/name).read_text()) for name in
                                         ['manifest.json', 'result.json', 'initial_dev.json']]
                        curves = [json.loads(line) for line in (dest/'curves.jsonl').read_text().splitlines()]
                        assert all(finite(v) for v in [m, r, initial, curves]), 'nonfinite saved value'
                        assert m['config'] == cfg
                        for key, value in [('stage',stage), ('task',task), ('gate',gate), ('seed',seed)]:
                            assert m[key] == r[key] == value
                        train = [v for v in curves if 'train_ce' in v]
                        dev = [v for v in curves if 'dev' in v]
                        steps = cfg[stage]['steps']
                        assert [v['step'] for v in train] == [s for s in range(1,steps+1) if s == 1 or s % cfg['log_every'] == 0]
                        assert [v['step'] for v in dev] == [s for s in range(1,steps+1) if s == steps or s % cfg['eval_every'] == 0]
                        assert all(v['grad_norm'] is not None and v['grad_norm'] > 0 for v in train)
                        assert all(v['finite'] for v in train[0]['router_gradient_audit'].values()), 'router gradient absent/nonfinite'
                        assert all(v['nonzero'] for v in train[0]['router_gradient_audit'].values()), 'zero first-step router gradient'
                        assert all(v['changed'] and v['l2_delta'] > 0 for v in r['router_update_audit'].values()), 'unchanged router parameter'
                        best = max(dev, key=lambda v:v['dev']['score'])
                        assert best['step'] == r['best_step'] and best['dev'] == r['best_dev']
                        assert r['amp_skipped_steps'] == 0
                        assert m['shared_state_sha256'] == r['shared_state_sha256']
                        assert r['data_sha256'] == meta['tensor_file_sha256']
                        assert m['environment']['gpu'] == 'NVIDIA TITAN V'
                        assert m['environment']['torch'] == '2.6.0+cu118'
                        assert len(m['environment']['source_sha256']) == 6
                        assert all(len(pos) == cfg['experts'] for pos in m['parameters']['positions'].values())
                        assert m['parameters']['router_total'] == r['router_parameters']
                        for route in r['final_routing'].values():
                            assert len(route['load']) == cfg['experts']
                            assert abs(route['drop_rate']) < 3e-7
                            assert abs(sum(route['load'])-cfg['topk']) < 1e-5
                            assert abs(sum(route['importance'])-1) < 1e-4
                            assert route['unused_experts'] == sum(v == 0 for v in route['load'])
                        checkpoint = torch.load(dest/'best.pt', map_location='cpu', weights_only=True)
                        assert checkpoint['step'] == r['best_step'] and checkpoint['dev'] == r['best_dev']
                        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
                        trained_shared = {k:v for k,v in checkpoint['model'].items() if not is_router_key(k)}
                        assert state_hash(trained_shared) != r['shared_state_sha256']
                        if check_models:
                            seed_all(seed, cfg['deterministic'])
                            model = (ImageClassifier(cfg,gate,seed,meta['input_dim'],meta['classes']) if stage == 'mlp' else
                                     BertClassifier(cfg,gate,seed,meta['classes'],model_path=bert_path(cfg)[0]))
                            before = {n:p for n,p in model.named_parameters() if is_router_key(n) and p.numel()}
                            assert all(not torch.equal(p,checkpoint['model'][n]) for n,p in before.items())
                            model.load_state_dict(checkpoint['model'], strict=True)
                            assert sum(p.numel() for p in model.parameters()) == r['total_parameters']
                            del model, before
                        rows.append(dict(run=dest.name, stage=stage,task=task,gate=gate,seed=seed,passed=True,
                                         first_ce=train[0]['train_ce'],last_ce=train[-1]['train_ce'],
                                         router_parameters=r['router_parameters'],best_dev=r['best_dev'],
                                         seconds_per_step=r['training_seconds']/steps,peak_cuda_bytes=r['peak_cuda_bytes']))
                        paired[stage,task,seed].append(m)
                        del checkpoint, trained_shared
                    except Exception as exc:
                        issues.append(dict(run=str(dest),error=repr(exc),traceback=traceback.format_exc()))
    for key, manifests in paired.items():
        try:
            assert len(manifests) == len(cfg['gates'])
            assert len({m['shared_state_sha256'] for m in manifests}) == 1
            assert len({m['data']['tensor_file_sha256'] for m in manifests}) == 1
            assert len({json.dumps(m['environment']['source_sha256'],sort_keys=True) for m in manifests}) == 1
            assert all(m['parameters']['positions'] == manifests[0]['parameters']['positions'] for m in manifests)
            baseline = next(m for m in manifests if m['gate'] == 'mlp')
            for m in manifests:
                fraction = cfg.get('variants',{}).get(m['gate'],{}).get('fraction',1.0)
                assert abs(m['parameters']['router_total']/baseline['parameters']['router_total']-fraction) < .065
        except Exception as exc:
            issues.append(dict(group=key,error=repr(exc),traceback=traceback.format_exc()))
    expected = sum(len(cfg[s]['tasks']) for s in ['mlp','bert'])*len(cfg['gates'])*len(cfg['seeds'])
    result = dict(expected=expected,completed_audited=len(rows),passed=not issues and len(rows)==expected,
                  issues=issues,data=data_records,runs=rows)
    write_json(root/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','data']},indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--skip-model-load',action='store_true')
    args = parser.parse_args()
    result = audit(load_config(args.config), not args.skip_model_load)
    raise SystemExit(0 if result['passed'] else 1)
