import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
while True:
    p = subprocess.run([sys.executable, 'outputs/gate_benchmark/monitor_progress.py'], text=True, capture_output=True, check=True)
    r = json.loads(p.stdout)
    with Path('outputs/gate_benchmark/logs/progress_history.jsonl').open('a') as f:
        f.write(json.dumps(r) + '\n')
    active = ', '.join(f"{v['run']}={v['step']}/{v['steps']}" for v in r['active'])
    print(f"{r['time']} completed={r['completed_runs']}/54 steps={r['logged_steps']}/43200 ({r['progress_percent_by_steps']}%) | {active}", flush=True)
    if r['completed_runs'] == 54:
        break
    if not psutil.pid_exists(12116):
        print('DRIVER STOPPED before matrix completion; inspect retained logs.', flush=True)
        raise SystemExit(2)
    time.sleep(55)
