"""Independent audit of completed benchmark artifacts; does not alter results."""
import argparse
import collections
import hashlib
import json
import math
import traceback
from pathlib import Path
import torch
from fitmotn.experiments.gate_benchmark.config import load_config, digest
from fitmotn.experiments.gate_benchmark.models import (
    ImageClassifier, BertClassifier, shared_state, state_hash, is_router_key,
)
from fitmotn.experiments.gate_benchmark.runner import seed_all
from fitmotn.experiments.gate_benchmark.data import bert_path, load_data


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def audit(config_path, check_initial=False):
    cfg = load_config(config_path)
    root = Path(cfg['output_dir']) / digest(cfg)[:12]
    issues, rows, paired = [], [], collections.defaultdict(list)
    data_audit = {}
    for stage in ('mlp', 'bert'):
        for task in cfg[stage]['tasks']:
            data, meta = load_data(cfg, task)
            ix = meta['indices']
            assert not set(ix['train']) & set(ix['dev']), task
            assert all(len(x) == len(set(x)) for x in ix.values()), task
            assert all(len(data[s]['labels']) == len(ix[s]) == meta['sizes'][s] for s in ix), task
            assert meta['final_source_split'] == ('test' if stage == 'mlp' else 'validation')
            for split in data.values():
                assert split['labels'].min() >= 0 and split['labels'].max() < meta['classes']
                if stage == 'mlp':
                    assert torch.isfinite(split['features']).all()
                    assert split['features'].min() >= -1 and split['features'].max() <= 1
                else:
                    assert split['input_ids'].shape[1] == cfg['max_length']
                    assert ((split['attention_mask'] == 0) | (split['attention_mask'] == 1)).all()
            data_audit[task] = {
                'sizes': meta['sizes'], 'source_split': meta['final_source_split'],
                'revision': meta['revision'], 'sha256': meta['tensor_file_sha256'],
                'train_dev_disjoint': True,
                'label_counts': {s: torch.bincount(d['labels']).tolist() for s, d in data.items()},
            }
            for seed in cfg['seeds']:
                for gate in cfg['gates']:
                    dest = root / f'{stage}_{task}_{gate}_seed{seed}'
                    if not (dest / 'result.json').exists():
                        issues.append({'run': str(dest), 'error': 'incomplete' if dest.exists() else 'missing'})
                        continue
                    try:
                        required = ('manifest.json', 'initial_dev.json', 'curves.jsonl', 'best.pt', 'result.json')
                        assert all((dest / x).is_file() and (dest / x).stat().st_size > 0 for x in required)
                        m, r, initial = [json.loads((dest / p).read_text()) for p in ('manifest.json', 'result.json', 'initial_dev.json')]
                        for key, expected_identity in (('stage', stage), ('task', task), ('gate', gate), ('seed', seed)):
                            assert m[key] == r[key] == expected_identity, (dest.name, key, 'matrix identity mismatch')
                        curves = [json.loads(line) for line in (dest / 'curves.jsonl').read_text().splitlines()]
                        assert all(finite(v) for v in (m, r, initial, curves))
                        tr = [x for x in curves if 'train_ce' in x]
                        dev = [x for x in curves if 'dev' in x]
                        expected = [i for i in range(1, cfg[stage]['steps'] + 1) if i == 1 or i % cfg['log_every'] == 0]
                        assert [x['step'] for x in tr] == expected
                        expected_dev = [i for i in range(1, cfg[stage]['steps'] + 1) if i % cfg['eval_every'] == 0 or i == cfg[stage]['steps']]
                        assert [x['step'] for x in dev] == expected_dev, 'missing or duplicate dev evaluation'
                        assert dev[-1]['step'] == cfg[stage]['steps']
                        assert all(x['grad_norm'] is not None and x['grad_norm'] > 0 for x in tr)
                        best = max(dev, key=lambda x: x['dev']['score'])
                        assert best['step'] == r['best_step'] and best['dev'] == r['best_dev']
                        assert r['amp_skipped_steps'] == 0
                        assert r['data_sha256'] == meta['tensor_file_sha256']
                        assert m['config'] == cfg
                        assert m['environment']['git_commit'] == '807366cb8afc58b1fc37086f998ae791e8bb16a6'
                        assert m['environment']['git_tracked_dirty'] is False
                        assert m['environment']['gpu'] == 'NVIDIA TITAN V'
                        assert m['environment']['torch'] == '2.6.0+cu118'
                        assert m['shared_state_sha256'] == r['shared_state_sha256']
                        assert all(len(pos) == cfg['experts'] for pos in m['parameters']['positions'].values())
                        assert m['parameters']['router_total'] == r['router_parameters']
                        ckpt = torch.load(dest / 'best.pt', map_location='cpu', weights_only=True)
                        assert ckpt['step'] == r['best_step'] and ckpt['dev'] == r['best_dev']
                        assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
                        shared_trained = {k: v for k, v in ckpt['model'].items() if not is_router_key(k)}
                        assert state_hash(shared_trained) != r['shared_state_sha256'], 'shared parameters did not update'
                        for route in r['final_routing'].values():
                            assert len(route['load']) == cfg['experts']
                            # CUDA float32 mean may round 4.0 to 3.99999976 for
                            # irregular token counts. This metric is not an integer count.
                            assert abs(route['drop_rate']) <= 2 * torch.finfo(torch.float32).eps
                            assert abs(sum(route['load']) - cfg['topk']) < 1e-5
                            assert route['unused_experts'] == sum(x == 0 for x in route['load'])
                            assert abs(sum(route['importance']) - 1) < 1e-4
                        row = {
                            'run': dest.name, 'passed': True, 'first_train_ce': tr[0]['train_ce'],
                            'last_train_ce': tr[-1]['train_ce'], 'initial_dev': initial['metrics'],
                            'best_dev': r['best_dev'], 'final': r['final'], 'best_step': r['best_step'],
                            'shared_parameters_updated': True, 'logged_gradient_norm_range': [min(x['grad_norm'] for x in tr), max(x['grad_norm'] for x in tr)],
                            'max_abs_reported_drop_rate': max(abs(v['drop_rate']) for v in r['final_routing'].values()),
                            'training_seconds': r['training_seconds'], 'seconds_per_step': r['training_seconds'] / cfg[stage]['steps'],
                            'wall_seconds': r['wall_seconds'], 'peak_cuda_mib': r['peak_cuda_bytes'] / 2**20,
                            'examples_per_second': r['examples_per_second'],
                            'mean_route': {key: sum(v[key] for v in r['final_routing'].values()) / len(r['final_routing']) for key in ('H_tok', 'H_dataset', 'load_cv', 'unused_experts')},
                        }
                        if check_initial:
                            seed_all(seed, cfg['deterministic'])
                            if stage == 'mlp':
                                model = ImageClassifier(cfg, gate, seed, meta['input_dim'], meta['classes'])
                            else:
                                model = BertClassifier(cfg, gate, seed, meta['classes'], model_path=bert_path(cfg)[0])
                            before = {k: v for k, v in model.state_dict().items() if is_router_key(k)}
                            changed = [k for k, v in before.items() if not torch.equal(v, ckpt['model'][k])]
                            assert changed, 'router parameters did not update'
                            row['router_tensors_changed'] = len(changed)
                            row['router_tensors_total'] = len(before)
                            model.load_state_dict(ckpt['model'], strict=True)
                            assert sum(p.numel() for p in model.parameters()) == r['total_parameters']
                            row['checkpoint_strict_load'] = True
                            del model
                        rows.append(row)
                        paired[(stage, task, seed)].append(m)
                    except Exception as exc:
                        issues.append({'run': str(dest), 'error': repr(exc), 'traceback': traceback.format_exc()})
    for key, manifests in paired.items():
        assert len({m['shared_state_sha256'] for m in manifests}) == 1, key
        assert len({m['data']['tensor_file_sha256'] for m in manifests}) == 1, key
        assert all(m['parameters']['positions'] == manifests[0]['parameters']['positions'] for m in manifests), key
        counts = [m['parameters']['router_total'] for m in manifests]
        assert max(counts) / min(counts) < 1.06, key
    expected = sum(len(cfg[s]['tasks']) for s in ('mlp', 'bert')) * len(cfg['gates']) * len(cfg['seeds'])
    result = {'experiment': root.name, 'expected': expected, 'completed_audited': len(rows), 'passed': not issues and len(rows) == expected,
              'issues': issues, 'data': data_audit, 'runs': rows,
              'note': 'Loss and gradient finiteness enforced every optimization step by runner; independent audit verifies logged values and saved state. Timing excludes initial model loading and initial dev.'}
    (root / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--check-initial', action='store_true')
    a = p.parse_args()
    audit(a.config, a.check_initial)
