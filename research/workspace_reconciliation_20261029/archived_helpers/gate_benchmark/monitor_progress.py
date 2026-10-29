import datetime
import json
from pathlib import Path

root = Path('outputs/gate_benchmark/45c21049b433')
c = json.loads((root / 'config.json').read_text())
done, active, missing = [], [], []
steps_done = 0
stage_counts = {}
for stage in ('mlp', 'bert'):
    stage_done = 0
    for task in c[stage]['tasks']:
        for seed in c['seeds']:
            for gate in c['gates']:
                name = f'{stage}_{task}_{gate}_seed{seed}'
                p = root / name
                if (p / 'result.json').exists():
                    done.append(name)
                    steps_done += c[stage]['steps']
                    stage_done += 1
                elif p.exists():
                    tr = []
                    if (p / 'curves.jsonl').exists():
                        for line in (p / 'curves.jsonl').read_text().splitlines():
                            try:
                                record = json.loads(line)
                                if 'training_seconds' in record:
                                    tr.append(record)
                            except json.JSONDecodeError:
                                pass
                    last = tr[-1] if tr else {}
                    steps_done += last.get('step', 0)
                    active.append({'run': name, 'step': last.get('step', 0), 'steps': c[stage]['steps'],
                                   'training_seconds': round(last.get('training_seconds', 0), 1),
                                   'ce': round(last.get('train_ce', 0), 4),
                                   'examples_per_second': round(last.get('examples_per_second', 0), 2)})
                else:
                    missing.append(name)
    stage_counts[stage] = {'complete': stage_done, 'expected': len(c[stage]['tasks']) * len(c['seeds']) * len(c['gates'])}
total_steps = sum(len(c[s]['tasks']) * len(c['seeds']) * len(c['gates']) * c[s]['steps'] for s in ('mlp', 'bert'))
r = {'time': datetime.datetime.now().astimezone().isoformat(), 'completed_runs': len(done), 'expected_runs': len(done)+len(active)+len(missing),
     'logged_steps': steps_done, 'planned_steps': total_steps, 'progress_percent_by_steps': round(100*steps_done/total_steps, 2),
     'stages': stage_counts, 'active': active, 'pending_runs': len(missing)}
Path('outputs/gate_benchmark/STATUS.json').write_text(json.dumps(r, indent=2))
phase = 'formal_mlp' if stage_counts['mlp']['complete'] < 27 else ('formal_bert' if stage_counts['bert']['complete'] < 27 else 'postprocess')
Path('outputs/gate_benchmark/logs/active_phase.json').write_text(json.dumps({'phase': phase, 'updated_at': r['time'],
    'driver_pid': 12116 if len(done) < 54 else None, 'completed_runs': len(done), 'expected_runs': 54,
    'log': 'outputs/gate_benchmark/logs/formal_concurrent_driver.log', 'status_file': 'outputs/gate_benchmark/STATUS.json'}, indent=2))
print(json.dumps(r, ensure_ascii=False))
