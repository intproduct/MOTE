"""Single-worker orchestration; never overwrites an incomplete training run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback
from .runner import read_json, canonical_hash, timestamp, SOURCE_PATHS, REPO
from .data import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', default='outputs/autoregressive_gate')
    args = parser.parse_args()
    root, config_path = Path(args.root).resolve(), Path(args.config).resolve()
    cfg = read_json(config_path)
    group = root/cfg['phase']/canonical_hash(cfg)[:12]
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'PIPELINE_LOCK.json'
    record = {'pid': os.getpid(), 'started': timestamp(), 'config': str(config_path),
              'config_hash': canonical_hash(cfg), 'group': str(group), 'state': 'running'}
    with lock.open('x', encoding='utf-8') as f:
        import json
        json.dump(record, f, indent=2)
    write_json(root/'ACTIVE_PROCESS.json', record)
    frozen = {p: digest(REPO/p) for p in SOURCE_PATHS}
    group.mkdir(parents=True, exist_ok=True)
    frozen_path = group/'frozen_sources.json'
    if frozen_path.exists():
        assert read_json(frozen_path) == frozen
    else:
        write_json(frozen_path, frozen)
    try:
        commands = ['run', 'audit'] + (['evaluate', 'summarize'] if cfg['phase'] == 'formal' else [])
        for command in commands:
            assert read_json(config_path) == cfg, 'configuration changed during experiment'
            assert all(digest(REPO/p) == sha for p, sha in frozen.items()), 'training code changed'
            write_json(root/'STATUS.json', {**record, 'stage': command, 'updated': timestamp()})
            module = 'fitmotn.experiments.autoregressive_gate.' + ('analysis' if command == 'summarize' else 'runner')
            cmd = [sys.executable, '-u', '-m', module]
            if command != 'summarize':
                cmd.append(command)
            cmd += ['--config', str(config_path), '--root', str(root)]
            print(timestamp(), 'START', command, flush=True)
            subprocess.run(cmd, cwd=REPO, check=True)
            print(timestamp(), 'DONE', command, flush=True)
        record.update(state='ready_for_review' if cfg['phase'] == 'formal' else 'pilot_passed', finished=timestamp())
    except Exception:
        record.update(state='failed', traceback=traceback.format_exc(), finished=timestamp())
        raise
    finally:
        write_json(root/'STATUS.json', record)
        write_json(root/'ACTIVE_PROCESS.json', record)
        lock.unlink()


if __name__ == '__main__':
    main()
