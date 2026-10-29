"""Run each three-gate triplet concurrently; one owner writes group metadata.

Accuracy protocol, RNGs and defaults are unchanged. Concurrent training timings
must not be interpreted as isolated throughput. See the serial efficiency audit.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.update(HF_HOME=str(ROOT / 'outputs/hf_cache'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  PYTHONUTF8='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
os.environ['PATH'] = str(ROOT / '.venv-titan/Library/bin') + os.pathsep + os.environ['PATH']


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--stage', choices=['mlp', 'bert'])
    p.add_argument('--task')
    p.add_argument('--seed', type=int)
    p.add_argument('--gate')
    p.add_argument('--max-workers', type=int, default=3)
    a = p.parse_args()
    from fitmotn.experiments.gate_benchmark.config import load_config, digest
    c = load_config(a.config)
    if a.output:
        c['output_dir'] = a.output
    group = Path(c['output_dir']) / digest(c)[:12]
    if a.worker:
        from fitmotn.experiments.gate_benchmark import runner
        original = runner.write_json
        def owned_write(path, value):
            if Path(path) == group / 'config.json':
                assert json.loads(Path(path).read_text()) == value
            else:
                original(path, value)
        runner.write_json = owned_write
        runner.run(c, stage=a.stage, tasks=[a.task], seeds=[a.seed], gates=[a.gate])
        return
    group.mkdir(parents=True, exist_ok=True)
    config_file = group / 'config.json'
    if config_file.exists():
        assert json.loads(config_file.read_text()) == c
    else:
        config_file.write_text(json.dumps(c, indent=2))
    execution = group / 'execution.jsonl'
    logroot = Path(c['output_dir']) / 'logs'
    logroot.mkdir(parents=True, exist_ok=True)
    assert 1 <= a.max_workers <= 6
    failed = []
    for stage in ('mlp', 'bert'):
        if a.stage and a.stage != stage:
            continue
        pending = []
        for task in c[stage]['tasks']:
            for seed in c['seeds']:
                for gate in c['gates']:
                    name = f'{stage}_{task}_{gate}_seed{seed}'
                    dest = group / name
                    if (dest / 'result.json').exists():
                        print('SKIP ' + name, flush=True)
                    elif dest.exists():
                        raise FileExistsError(f'Incomplete run retained: {dest}')
                    else:
                        pending.append((task, seed, gate, name))
        children = []
        while pending or children:
            while pending and not failed and len(children) < a.max_workers:
                task, seed, gate, name = pending.pop(0)
                log = logroot / f'{group.name}_{name}_{int(time.time())}.log'
                handle = log.open('w', encoding='utf-8')
                cmd = [sys.executable, '-u', __file__, '--worker', '--config', a.config,
                       '--stage', stage, '--task', task, '--seed', str(seed), '--gate', gate]
                if a.output:
                    cmd += ['--output', a.output]
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
                record = {'event': 'start', 'run': name, 'pid': proc.pid, 'driver_pid': os.getpid(),
                          'time': now(), 'log': str(log), 'max_concurrent_workers': a.max_workers,
                          'active_workers_after_start': len(children)+1}
                with execution.open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                children.append((name, proc, handle))
            active = []
            for name, proc, handle in children:
                code = proc.poll()
                if code is None:
                    active.append((name, proc, handle))
                    continue
                handle.write(f'\nEXIT_CODE={code}\n')
                handle.close()
                record = {'event': 'end', 'run': name, 'pid': proc.pid, 'exit_code': code, 'time': now()}
                with execution.open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                if code:
                    failed.append(record)
            children = active
            if failed and not children:
                raise RuntimeError(f'Failed runs retained: {failed}')
            if children:
                time.sleep(2)
    (group / 'execution_complete.json').write_text(json.dumps({'completed_at': now(), 'max_concurrent_workers': a.max_workers, 'stages': a.stage or 'mlp_then_bert'}))


if __name__ == '__main__':
    main()
