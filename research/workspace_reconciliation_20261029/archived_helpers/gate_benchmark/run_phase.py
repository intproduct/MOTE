"""Logged, single-process stage driver. Never overwrites logs or run directories."""
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
LOG = ROOT / 'outputs/gate_benchmark/logs'
os.environ.update(HTTP_PROXY='http://127.0.0.1:7897', HTTPS_PROXY='http://127.0.0.1:7897',
                  HF_HOME=str(ROOT / 'outputs/hf_cache'), HF_HUB_DISABLE_XET='1',
                  HF_HUB_ETAG_TIMEOUT='60', HF_HUB_DOWNLOAD_TIMEOUT='120',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONUTF8='1')
os.environ['PATH'] = str(ROOT / '.venv-titan/Library/bin') + os.pathsep + os.environ['PATH']


def command(name, args):
    stamp = datetime.datetime.now().astimezone().isoformat()
    log = LOG / f'{name}.log'
    if log.exists():
        log = LOG / f'{name}_{int(time.time())}.log'
    with log.open('w', encoding='utf-8') as f:
        f.write(json.dumps({'started': stamp, 'argv': [sys.executable, *args]}, ensure_ascii=False) + '\n')
        f.flush()
        proc = subprocess.Popen([sys.executable, '-u', *args], stdout=f, stderr=subprocess.STDOUT)
        (LOG / 'active_phase.json').write_text(json.dumps({'phase': name, 'pid': proc.pid, 'driver_pid': os.getpid(), 'started': stamp, 'log': str(log)}, indent=2))
        print(f'START {name} PID={proc.pid} {log}', flush=True)
        code = proc.wait()
        f.write(f'\nEXIT_CODE={code}\n')
    with (LOG / 'phase_history.jsonl').open('a') as f:
        f.write(json.dumps({'phase': name, 'exit_code': code, 'started': stamp, 'ended': datetime.datetime.now().astimezone().isoformat(), 'log': str(log)}) + '\n')
    if code:
        raise SystemExit(code)
    print(f'PASS {name}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['pilot', 'formal'])
    a = p.parse_args()
    module = ['-m', 'fitmotn.experiments.gate_benchmark']
    cfg = f'experiments/gate_benchmark/configs/{"pilot" if a.phase == "pilot" else "titan_v"}.json'
    command(a.phase + '_prepare', module + ['prepare', '--config', cfg])
    if a.phase == 'pilot':
        command('pilot_run', module + ['run', '--config', cfg])
    else:
        command('formal_mlp', module + ['run', '--config', cfg, '--stage', 'mlp'])
        command('formal_bert', module + ['run', '--config', cfg, '--stage', 'bert'])
    command(a.phase + '_audit', ['outputs/gate_benchmark/audit_results.py', cfg, '--check-initial'])
    command(a.phase + '_summarize', module + ['summarize'])
    (LOG / f'{a.phase}_complete.txt').write_text(datetime.datetime.now().astimezone().isoformat())
