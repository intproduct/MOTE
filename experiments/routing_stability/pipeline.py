"""Single-worker orchestration; never overwrites an incomplete training run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
import traceback
from .runner import read_json, canonical_hash, timestamp, SOURCE_PATHS, REPO
from ..autoregressive_gate.data import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', default='outputs/routing_stability_round1')
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
    # Keep actual source bytes as well as hashes, including rejected pilots.
    for relative, checksum in frozen.items():
        destination = group/'source_snapshot'/relative
        if destination.exists():
            assert digest(destination) == checksum
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/relative, destination)
    try:
        commands = ['run', 'audit'] + (['evaluate', 'probe', 'summarize'] if cfg['phase'] == 'formal' else [])
        for command in commands:
            assert read_json(config_path) == cfg, 'configuration changed during experiment'
            assert all(digest(REPO/p) == sha for p, sha in frozen.items()), 'training code changed'
            write_json(root/'STATUS.json', {**record, 'stage': command, 'updated': timestamp()})
            module = 'fitmotn.experiments.routing_stability.' + ({'summarize': 'analysis', 'probe': 'probe'}.get(command, 'runner'))
            cmd = [sys.executable, '-u', '-m', module]
            if command not in ['summarize', 'probe']:
                cmd.append(command)
            cmd += ['--config', str(config_path), '--root', str(root)]
            print(timestamp(), 'START', command, flush=True)
            subprocess.run(cmd, cwd=REPO, check=True)
            print(timestamp(), 'DONE', command, flush=True)
        if cfg['phase'] == 'pilot':
            pilot_review(group, cfg)
        record.update(state='ready_for_review' if cfg['phase'] == 'formal' else 'pilot_passed', finished=timestamp())
    except Exception:
        record.update(state='failed', traceback=traceback.format_exc(), finished=timestamp())
        raise
    finally:
        write_json(root/'STATUS.json', record)
        write_json(root/'ACTIVE_PROCESS.json', record)
        lock.unlink()


def pilot_review(group, cfg):
    """Finite training is insufficient: reject near-constant sparse routing."""
    warnings, rows = [], []
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            path=group/'runs'/f'{variant}_s{seed}'
            r=read_json(path/'result.json')
            for module, values in r['best_validation']['routing'].items():
                row={'run':path.name,'module':module,'load_cv':values['load_cv'],
                     'unused_experts':values['unused_experts']}
                rows.append(row)
                if values['load_cv'] > 1.5 and values['unused_experts'] >= 6:
                    warnings.append({**row,'problem':'near-constant top-k expert collapse'})
            if cfg['designs'][variant]['margin']:
                import json
                curves=[json.loads(line) for line in (path/'curves.jsonl').read_text().splitlines()]
                if not any(v.get('margin_loss',0)>0 for v in curves):
                    warnings.append({'run':path.name,'problem':'margin branch never active'})
    review={'accepted':not warnings,'warnings':warnings,'routing':rows,'time':timestamp()}
    write_json(group/'pilot_review.json',review)
    if warnings:
        raise RuntimeError(f'pilot routing review rejected: {warnings}')


if __name__ == '__main__':
    main()
