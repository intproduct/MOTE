"""CPU-only heartbeat checks; never launches training or edits frozen files."""
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def lines(path):
    if not path.exists():
        return []
    # A writer may be midway through the final append; only complete records count.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


active = read(ROOT/'ACTIVE_PROCESS.json')
group = Path(active['group'])
cfg = read(group/'config.json')
frozen = read(group/'frozen_sources.json')
problems, runs = [], []
initializations, orders, data_hashes = {}, {}, set()
for seed in cfg['seeds']:
    for variant in cfg['gates']:
        name = f'{variant}_s{seed}'
        folder = group/'runs'/name
        if not (folder/'manifest.json').exists():
            continue
        manifest = read(folder/'manifest.json')
        initializations.setdefault(seed, set()).add(manifest['shared_init_hash'])
        data_hashes.add(manifest['data_manifest_sha256'])
        curves, routes = lines(folder/'curves.jsonl'), lines(folder/'routing.jsonl')
        validations = [row for row in curves if row['kind'] == 'validation']
        row = {'run': name, 'completed': (folder/'result.json').exists(),
               'logged_step': curves[-1]['step'] if curves else 0,
               'logged_values_finite': finite(curves) and finite(routes),
               'latest_validation': validations[-1] if validations else None}
        if not row['logged_values_finite']:
            problems.append(name+': nonfinite log')
        if routes:
            row['latest_unused_experts'] = {k: sum(x == 0 for x in r['load'])
                                           for k,r in routes[-1]['routing'].items()}
        if row['completed']:
            result = read(folder/'result.json')
            orders.setdefault(seed, set()).add(result['batch_order_sha256'])
            checks = {
                'finite_steps': result['finite_steps'] == cfg['train']['steps'],
                'nonzero_router_grad_steps': result['nonzero_router_grad_steps'] == cfg['train']['steps'],
                'all_router_tensors_updated': all(v > 0 for v in result['router_parameter_max_changes'].values()),
                'shared_parameters_updated': result['shared_parameters_updated'],
                'best_checkpoint_hash_matches': sha(folder/'best.pt') == result['checkpoint_sha256'],
                'batch_order_file_matches': sha(folder/'batch_starts.npy') == result['batch_order_sha256'],
                'result_finite': finite(result),
            }
            row['checks'] = checks
            row['best_step'] = result['best_step']
            problems.extend(name+': '+key for key,value in checks.items() if not value)
        runs.append(row)
changed = [p for p,h in frozen.items() if sha(REPO/p) != h]
shared_ok = all(len(v) == 1 for v in initializations.values())
orders_ok = all(len(v) == 1 for v in orders.values())
data_ok = data_hashes == {sha(ROOT/'data/manifest.json')}
if changed or not (shared_ok and orders_ok and data_ok):
    problems.append('frozen source, shared init, batch order, or data manifest inconsistency')
record = {'checked_at': datetime.datetime.now().astimezone().isoformat(),
          'completed': sum(r['completed'] for r in runs),
          'expected': len(cfg['seeds'])*len(cfg['gates']),
          'changed_frozen_sources': changed, 'shared_initialization_consistent': shared_ok,
          'completed_batch_orders_consistent': orders_ok, 'data_manifest_consistent': data_ok,
          'problems': problems, 'runs': runs,
          'stderr_bytes': (ROOT/'logs/formal_pipeline.err').stat().st_size}
tmp = ROOT/'MONITOR_STATUS.json.tmp'
tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
tmp.replace(ROOT/'MONITOR_STATUS.json')
print(json.dumps(record, ensure_ascii=False, indent=2))
