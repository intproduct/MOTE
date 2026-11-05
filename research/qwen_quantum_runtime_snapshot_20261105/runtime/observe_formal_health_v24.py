import argparse, hashlib, json, math, os, time
from pathlib import Path
import psutil
from formal_health_boundaries_v18 import finished_training, diagnostic_jobs
from formal_health_freshness_v22 import post_evaluation_evidence
from formal_health_freshness_v23 import active_evaluation_evidence
from formal_health_processes_v24 import process_identity, held_exits_match

ROOT = Path(r'E:\AI\MOTE-quantum')
QUEUE = ROOT / 'queue-20261030'
ACCEPTANCE = '443cffa69bc28e743d6cd4d5552a9b751559315b1ec57087260ec9d013ac77e3'
DEV_SAMPLE = 'a9c4ad23d441a2c6d41f08e58bd4ddb2d2be8035c4c848e017062a406f31e548'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(command):
    return [str(x).replace('\\', '/').lower() for x in command]

def observe(label):
    # Read-only process/GPU observation. No process control, launches or CUDA imports.
    if not label or any(c not in '0123456789TZ' for c in label):
        raise ValueError('Use a UTC timestamp label')
    output = QUEUE / f'formal-health-{label}.json'
    if output.exists():
        raise FileExistsError(output)
    snapshot_path = QUEUE / f'heartbeat-observation-{label}.json'
    snapshot = read(snapshot_path)
    request_path = QUEUE / 'execution-request.json'
    previous = read(Path(read(request_path)['formal_health_observation']))
    now = time.time()
    errors, controllers, runs, diagnostics = [], {}, [], []
    def check(ok, message):
        if not ok:
            errors.append(message)
    check(digest(ROOT / 'releases/INTEGRATION_ACCEPTANCE.json') == ACCEPTANCE, 'acceptance changed')
    check(now - snapshot_path.stat().st_mtime < 300, 'snapshot older than five minutes')
    for kind, summary in snapshot['formal_controllers'].items():
        path = Path(summary['path'])
        controller = read(path)
        exit_dir = ROOT/'runtime/formal-controller-exits-v15-20261031'
        attached = read(exit_dir/(kind+'.ATTACHED.json'))
        expected = next(x for x in attached['workers'] if x['pid'] == controller['pid'])
        identity = process_identity(expected)
        exit_path = exit_dir/(kind+'.RESULT.json')
        receipt = read(exit_path) if exit_path.exists() else None
        held = receipt is not None and held_exits_match(attached, receipt)
        check(identity['matches'] or (not identity['present'] and held), f'{kind} controller identity or bound held exits')
        age = time.time() - path.stat().st_mtime
        check(age < 60 and controller['phase'] == 'running' and not controller.get('failed') and not controller.get('error'), f'{kind} controller health')
        check(not (path.parent / 'STOP').exists() and not (path.parent / 'FAILURE.json').exists(), f'{kind} stop/failure')
        controllers[kind] = {'pid':controller['pid'], 'command':expected['command'], 'create_time':expected['create_time'],
            'actual_identity':identity, 'phase':controller['phase'], 'status_age_seconds':age,
            'held_exits_confirmed':held, 'exit_receipt':str(exit_path) if receipt else None,
            'exit_receipt_sha256':digest(exit_path) if receipt else None,
            'successful_experiment_completion_implied_by_exit':False}
        if kind != 'training':
            diagnostics = diagnostic_jobs(controller, check)
            continue
        for name, job in controller['jobs'].items():
            if job['phase'] in ('waiting_for_formal_diagnostics', 'complete_audited'):
                runs.append(finished_training(name, job, check))
                continue
            if job['phase'] != 'running':
                errors.append(f'{name}: phase changed to {job["phase"]}; inspect boundary before further work')
                continue
            directory = Path(job['output'])
            status, config = read(directory / 'STATUS.json'), read(directory / 'RUN_CONFIG.json')
            lines = (directory / 'updates.jsonl').read_bytes().splitlines(keepends=True)
            updates = [json.loads(x) for x in lines if x.endswith(b'\n') and x.strip()]
            last = updates[-1]
            thermal_path = directory / 'thermal-pacing.jsonl'
            thermal_rows = [json.loads(x) for x in thermal_path.read_bytes().splitlines(keepends=True) if x.endswith(b'\n') and x.strip()]
            eval_rows = [x for x in thermal_rows if x.get('stage') == 'evaluation' and x.get('time', 0) > status['time']]
            checkpoint_boundary = status['tokens'] > 0 and status['tokens'] % (status['target']//10) == 0
            cp_current = directory / 'checkpoints' / f'checkpoint-{status["tokens"]:010d}' / 'complete.json'
            dev_current = directory / f'dev-{status["tokens"]:010d}.json'
            # Finish reading each worker's evidence before sampling its clock.
            # The pass-wide clock can precede a concurrent thermal-log append.
            checkpoint = read(cp_current) if cp_current.exists() else None
            dev = read(dev_current) if dev_current.exists() else None
            dev_mtime = dev_current.stat().st_mtime if dev is not None else None
            run_now = time.time()
            evaluation = active_evaluation_evidence(status, checkpoint is not None,
                dev is not None, thermal_rows, run_now)
            evaluation_active = evaluation is not None
            handoff = None
            if checkpoint_boundary and checkpoint is not None and dev is not None:
                handoff = post_evaluation_evidence(status, config, checkpoint, dev, dev_mtime, thermal_rows, run_now, ACCEPTANCE, DEV_SAMPLE)
                if handoff:
                    handoff.update({'checkpoint_manifest':str(cp_current), 'checkpoint_manifest_sha256':digest(cp_current), 'dev_file':str(dev_current), 'dev_sha256':digest(dev_current)})
            fresh_update = 0 <= run_now-status['time'] < 180
            check(status['phase'] == 'running' and (fresh_update or evaluation_active or handoff), f'{name} fresh training or active evaluation evidence')
            check(config['acceptance_sha256'] == ACCEPTANCE and config['stage'] == 'formal' and config['target_tokens'] == 32768000, f'{name} acceptance identity')
            prior = next((x for x in previous['runs'] if x['name'] == name), None)
            if prior:
                advanced = status['tokens'] > prior['observed_tokens']
                same = status['tokens'] == prior['observed_tokens']
                short_recheck = run_now-previous['observed_epoch'] < 600
                check(advanced or (same and (evaluation_active or handoff or (short_recheck and fresh_update))), f'{name} progressing or verified evaluation')
            check(all(math.isfinite(x['ce']) and math.isfinite(x['grad_norm']) and x['optimizer_update']['updated'] for x in updates), f'{name} finite successful updates')
            check(all(x['tokens'] == (i+1)*32768 and x['update_tokens'] == 32768 for i,x in enumerate(updates)), f'{name} effective token sequence')
            actual = []
            for recorded in job['native_processes']:
                identity = process_identity(recorded)
                check(identity['matches'], f'{name} native identity {recorded["pid"]}')
                actual.append(identity)
            lock_path = ROOT / 'locks' / (status['uuid']+'.lock')
            lock = read(lock_path) if lock_path.exists() else None
            owned = lock is not None and lock['pid'] == status['pid'] and Path(lock['run_directory']).resolve() == directory.resolve()
            check(owned, f'{name} GPU lock owner')
            initialization = job['initialization']
            ip = Path(initialization['path'])
            check(digest(ip) == initialization['sha256'] and read(ip)['passed'], f'{name} initialization audit')
            check(not (directory/'STOP').exists() and not (directory/'FAILURE.json').exists(), f'{name} stop/failure')
            checkpoints = []
            for cp in sorted(directory.glob('checkpoints/checkpoint-*/complete.json')):
                meta = read(cp)
                tokens = int(cp.parent.name.split('-')[-1])
                dev_path = directory / f'dev-{tokens:010d}.json'
                check(meta['identity']['acceptance_sha256'] == ACCEPTANCE, f'{name} checkpoint acceptance binding')
                dev = read(dev_path) if dev_path.exists() else None
                if dev:
                    check(dev['valid_targets'] == 1048576 and dev['sample_hash'] == DEV_SAMPLE and math.isfinite(dev['ce']), f'{name} fixed internal dev evaluation')
                checkpoints.append({'manifest':str(cp), 'manifest_sha256':digest(cp), 'tokens':tokens, 'internal_dev':dev, 'payload_hashes_recomputed_this_observation':False})
            runs.append({'name':name, 'run_directory':str(directory), 'gpu_uuid':status['uuid'], 'pid':status['pid'], 'actual_processes':actual, 'observed_tokens':status['tokens'], 'target_tokens':status['target'], 'updates':len(updates), 'total_overflow_retries':sum(x['overflow_retries'] for x in updates), 'last_ce':last['ce'], 'last_temperature_c':last['temperature_c'], 'status_age_seconds':run_now-status['time'], 'run_observed_epoch':run_now, 'thermal_tail_evidence':thermal_rows[-2:], 'observed_subphase':'internal_dev_evaluation' if evaluation_active else ('post_evaluation_handoff' if handoff else 'training'), 'evaluation_guard_evidence':eval_rows[-2:] if evaluation_active else [], 'post_evaluation_evidence':handoff, 'lock_owner_matches':owned, 'initialization_audit':initialization, 'checkpoint_observations':checkpoints})
    watcher = previous['controller_exit_observer']
    watcher_evidence = dict(watcher)
    try:
        process = psutil.Process(watcher['pid'])
        command = process.cmdline()
        check(norm(command) == norm(watcher['command']), 'exit watcher command')
        if watcher.get('create_time') is not None:
            check(abs(process.create_time()-watcher['create_time']) < .01, 'exit watcher create time')
        watcher_evidence.update(command=command, create_time=process.create_time(), present=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        watcher_evidence.update(present=False, observation_error=type(exc).__name__)
        check(isinstance(exc, psutil.NoSuchProcess) and all(x['held_exits_confirmed'] for x in controllers.values()), 'exit watcher absent before both bound exit receipts')
    check(not (ROOT/'runtime/formal-controller-exits-v15-20261031/FAILURE.json').exists(), 'exit watcher failure')
    check(not (ROOT/'STOP').exists() and not (QUEUE/'STOP').exists(), 'user STOP')
    result = {'passed':not errors, 'phase':snapshot['phase'], 'observed_epoch':now, 'snapshot':str(snapshot_path), 'acceptance_sha256':ACCEPTANCE, 'controllers':controllers, 'runs':runs, 'diagnostics':diagnostics, 'freshness_helper_sha256':digest(ROOT/'runtime/formal_health_freshness_v22.py'), 'active_evaluation_helper_sha256':digest(ROOT/'runtime/formal_health_freshness_v23.py'), 'boundary_helper_sha256':digest(ROOT/'runtime/formal_health_boundaries_v18.py'), 'controller_exit_observer':watcher_evidence, 'process_helper_sha256':digest(ROOT/'runtime/formal_health_processes_v24.py'), 'errors':errors, 'needs_review':bool(errors), 'new_stage_or_failure':bool(errors), 'observer_source_sha256':digest(Path(__file__)), 'next_action':'Preserve any failed controller and cooperative STOP. Observe existing workers to actual exits; never infer completion from exit code alone. Review resource-failure-v24 evidence and recheck the frozen shared budget before preparing a separately tested continuation. No duplicate launches or automatic retries.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    # Only observer-owned pointers change; frozen evidence and controller state stay read-only.
    request = read(request_path)
    request['formal_health_observation'] = str(output)
    request['formal_latest_tokens'] = {x['name']:x['observed_tokens'] for x in runs}
    temporary = request_path.with_name('execution-request.'+label+'.v24.tmp')
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, request_path)
    print(json.dumps({'health':str(output), 'passed':result['passed'], 'errors':errors, 'runs':[{**{k:x[k] for k in ('name','observed_tokens','updates','total_overflow_retries','last_temperature_c')}, 'checkpoint_manifests':len(x['checkpoint_observations'])} for x in runs]}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Observe running formal jobs without launching or controlling processes')
    parser.add_argument('--label', required=True)
    observe(parser.parse_args().label)
