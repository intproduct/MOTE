"""Read-only health snapshot of the in-progress formal matrix (no GPU use)."""
import collections
import datetime
import json
import math
from pathlib import Path

root = Path('outputs/gate_benchmark/45c21049b433')
issues, records = [], []
hashes = collections.defaultdict(set)
for run in sorted(root.iterdir()):
    if not run.is_dir() or not (run / 'manifest.json').exists():
        continue
    manifest = json.loads((run / 'manifest.json').read_text())
    hashes[(manifest['task'], manifest['seed'])].add(manifest['shared_state_sha256'])
    path = run / 'curves.jsonl'
    lines = path.read_text().splitlines() if path.exists() else []
    curve = []
    for i, line in enumerate(lines):
        try:
            curve.append(json.loads(line))
        except json.JSONDecodeError:
            # The running writer can be between write and newline. Only the
            # final line may be pending; finished runs get a strict full audit.
            if i != len(lines)-1 or (run / 'result.json').exists():
                issues.append({'run': run.name, 'error': 'invalid JSON line', 'line': i+1})
    train = [x for x in curve if 'train_ce' in x]
    for x in train:
        for key in ('train_ce', 'train_loss', 'grad_norm'):
            value = x.get(key)
            if value is not None and not math.isfinite(value):
                issues.append({'run': run.name, 'step': x['step'], 'error': 'nonfinite '+key})
    records.append({'run': run.name, 'complete': (run / 'result.json').exists(),
        'last_logged_step': train[-1]['step'] if train else 0,
        'logged_points': len(train),
        'gradient_min': min((x['grad_norm'] for x in train), default=None),
        'gradient_max': max((x['grad_norm'] for x in train), default=None)})
for (task, seed), values in hashes.items():
    if len(values) != 1:
        issues.append({'task': task, 'seed': seed, 'error': 'shared hash mismatch'})
snapshot = {'time': datetime.datetime.now().astimezone().isoformat(),
    'started': len(records), 'complete': sum(r['complete'] for r in records),
    'logged_points': sum(r['logged_points'] for r in records), 'issues': issues,
    'runs': records, 'scope': 'in-progress JSON health only; does not replace final audit'}
Path('outputs/gate_benchmark/live_health.json').write_text(json.dumps(snapshot, indent=2))
print(json.dumps({k: v for k, v in snapshot.items() if k != 'runs'}))
raise SystemExit(bool(issues))
