import os
from pathlib import Path
os.environ.update(HF_HOME=str(Path('outputs/hf_cache').resolve()), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import torch
from fitmotn.experiments.gate_benchmark.config import load_config, digest
from audit_results import audit

c = load_config('experiments/gate_benchmark/configs/pilot.json')
serial = Path(c['output_dir']) / digest(c)[:12]
c['output_dir'] = 'outputs/gate_parallel_pilot'
parallel = Path(c['output_dir']) / digest(c)[:12]
config_path = Path(c['output_dir']) / 'config_used.json'
config_path.write_text(json.dumps(c, indent=2))
audit(config_path, check_initial=True)
rows = []
for p in sorted(serial.glob('*/result.json')):
    other = parallel / p.parent.name
    a, b = json.loads(p.read_text()), json.loads((other / 'result.json').read_text())
    for key in ('final', 'final_routing', 'best_step', 'best_dev', 'shared_state_sha256', 'data_sha256', 'router_parameters', 'total_parameters', 'amp_skipped_steps'):
        assert a[key] == b[key], (p.parent.name, key)
    ca = [json.loads(x) for x in (p.parent / 'curves.jsonl').read_text().splitlines()]
    cb = [json.loads(x) for x in (other / 'curves.jsonl').read_text().splitlines()]
    assert len(ca) == len(cb)
    for x, y in zip(ca, cb):
        for key in set(x) - {'training_seconds', 'examples_per_second'}:
            assert x[key] == y[key], (p.parent.name, x['step'], key)
    sa = torch.load(p.parent / 'best.pt', map_location='cpu', weights_only=True)['model']
    sb = torch.load(other / 'best.pt', map_location='cpu', weights_only=True)['model']
    assert sa.keys() == sb.keys()
    assert all(torch.equal(v, sb[k]) for k, v in sa.items()), p.parent.name
    rows.append({'run': p.parent.name, 'bitwise_equal_checkpoint_and_training_curves': True,
                 'serial_training_seconds': a['training_seconds'], 'concurrent_training_seconds': b['training_seconds']})
assert len(rows) == 6
report = {'passed': True, 'runs': rows, 'interpretation': 'Concurrency preserves measured learning trajectory exactly on all six pilots; timing contains contention and is reported separately.'}
(parallel / 'serial_equivalence.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
