"""CPU-only integrity monitoring. Does not modify experiments or GPU processes."""
from pathlib import Path
import contextlib
import datetime
import hashlib
import io
import json
import math
import runpy

root = Path(__file__).resolve().parent
repo = root.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def lines(path):
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    parts = text.splitlines()
    if text and not text.endswith('\n'):
        parts = parts[:-1]  # a writer may still be appending the final record
    return [json.loads(line) for line in parts if line.strip()]


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(root/'monitor.py'), run_name='__main__')
snapshot = read(root/'MONITOR_STATUS.json')
group = Path(snapshot['active']['group'])
cfg = read(group/'config.json')
assert cfg == read(Path(snapshot['active']['config']))
assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == snapshot['active']['config_hash']
assert not snapshot['finite_problems'] and not snapshot['changed_frozen_sources']
completed, initial, shared, orders, data_hashes, curves = [], {}, {}, {}, set(), {}
for path in (group/'runs').glob('*'):
    if (path/'manifest.json').exists():
        manifest = read(path/'manifest.json')
        key = (manifest['seed'], cfg['designs'][manifest['variant']]['base'])
        initial.setdefault(key, set()).add(manifest['initial_model_hash'])
        shared.setdefault(manifest['seed'], set()).add(manifest['shared_init_hash'])
        data_hashes.add(manifest['data_manifest_sha256'])
    if (path/'result.json').exists():
        r = read(path/'result.json')
        assert finite(r)
        assert r['config_hash'] == snapshot['active']['config_hash']
        assert r['steps'] == r['finite_steps'] == r['nonzero_router_grad_steps'] == cfg['train']['steps']
        assert r['shared_parameters_updated']
        assert all(v > 0 for v in r['router_parameter_max_changes'].values())
        assert r['checkpoint_sha256'] == sha(path/'best.pt')
        assert r['batch_order_sha256'] == sha(path/'batch_starts.npy')
        orders.setdefault(r['seed'], set()).add(r['batch_order_sha256'])
        for name in ['last.pt', 'curves.jsonl', 'routing.jsonl', 'manifest.json']:
            assert (path/name).is_file()
        completed.append(path.name)
    records = lines(path/'curves.jsonl')
    curves[path.name] = {r['step']: r for r in records if r['kind'] == 'train'}
    assert all(finite(r) for r in records)
    assert all(r['router_grad_l1_after_clip'] > 0 for r in curves[path.name].values())
    assert all(finite(r) for r in lines(path/'routing.jsonl'))
assert all(len(values) == 1 for values in [*initial.values(), *shared.values(), *orders.values()])
assert len(data_hashes) <= 1
assert not data_hashes or data_hashes == {sha(repo/cfg['data_root']/'manifest.json')}
pair_checks = []
for seed in cfg['seeds']:
    for baseline, margin in [('mlp_half', 'mlp_margin'), ('amp_half', 'amp_margin'), ('born_lite', 'born_lite_margin')]:
        a, b = curves.get(f'{baseline}_s{seed}', {}), curves.get(f'{margin}_s{seed}', {})
        common = [s for s in set(a) & set(b) if s <= cfg['margin']['start_fraction'] * cfg['train']['steps']]
        if common:
            pair_checks.append({'seed': seed, 'baseline': baseline, 'margin': margin,
                                'common_logged_steps_before_margin': len(common),
                                'max_ce_difference_before_margin': max(abs(a[s]['ce']-b[s]['ce']) for s in common)})
routing = []
current = snapshot['status'].get('run')
if current:
    records = lines(group/'runs'/current/'routing.jsonl')
    if records:
        for name, value in records[-1]['routing'].items():
            load = value['load']
            mean = sum(load)/len(load)
            cv = math.sqrt(sum((x-mean)**2 for x in load)/len(load))/mean
            routing.append({'module': name, 'step': records[-1]['step'], 'cv': cv,
                            'unused_experts': sum(x == 0 for x in load)})
result = {'time': datetime.datetime.now().astimezone().isoformat(),
          'complete': len(completed), 'expected': len(cfg['seeds'])*len(cfg['gates']),
          'completed_audited': completed, 'latest': snapshot['latest'], 'problems': [],
          'config_and_frozen_sources_unchanged': True,
          'shared_and_pair_initialization_and_sampling_consistent': True,
          'before_margin_pair_checks': pair_checks, 'current_routing': routing,
          'evaluation_count': snapshot['evaluations'], 'stability_count': snapshot['stability_probes']}
(root/'LATEST_HEALTH_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
