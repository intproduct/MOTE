"""Read-only observation of audited training exits and formal diagnostics."""
import hashlib
import json
import math
import time
from pathlib import Path
import psutil

ROOT = Path('E:/AI/MOTE-quantum')
ACCEPTANCE = '443cffa69bc28e743d6cd4d5552a9b751559315b1ec57087260ec9d013ac77e3'
DEV = 'a9c4ad23d441a2c6d41f08e58bd4ddb2d2be8035c4c848e017062a406f31e548'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def same_process(record):
    try:
        p = psutil.Process(record['pid'])
        return abs(p.create_time()-record['create_time']) < .01
    except psutil.NoSuchProcess:
        return False

def barrier_observation(name, ref, training_ref, check):
    p = Path(ref['path'])
    value = read(p)
    check(sha(p) == ref['sha256'], name+': diagnostic barrier SHA256')
    check(value.get('passed') is True and value.get('schema') == 'qwen-formal-diagnostics-v12', name+': diagnostic barrier schema')
    check(value['name'] == name and value['training_audit_sha256'] == training_ref['sha256'], name+': diagnostic barrier training binding')
    expected = {(0,'full_model'),(0,'local_precision')} | {(t,k) for t in (3276800,16384000,32768000) for k in ('timing','full_model','local_precision')}
    cells = value['measurements']
    check(len(cells)==11 and {(x['tokens'],x['kind']) for x in cells}==expected, name+': eleven diagnostic measurements')
    for cell in cells:
        for key in ('result','audit'):
            check(sha(cell[key]) == cell[key+'_sha256'], name+': barrier cell '+key+' binding')
        check(read(cell['audit'])['passed'],name+': barrier cell audit')
    return ref

def terminal_contract(name, job, result, audit, config):
    errors = []
    def check(ok, text):
        if not ok: errors.append(name+': '+text)
    check(result['phase'] == 'complete' and result['tokens'] == result['target'] == 32768000, 'complete token budget')
    check(config['acceptance_sha256'] == ACCEPTANCE and config['stage'] == 'formal', 'acceptance binding')
    check(audit['passed'] and audit['phase'] == 'training_complete_audited' and audit['name'] == name, 'training audit')
    check(audit['counts']['optimizer_updates'] == 1000 and audit['primary_tokens'] == 32768000, 'audited updates and primary checkpoint')
    check(not audit['reserved_test_accessed'], 'reserved test must remain locked')
    check(result['samples_sha256'] == audit['samples_sha256'], 'sample hash binding')
    check(len(audit['internal_dev']) == 10 and {x['tokens'] for x in audit['internal_dev']} == {3276800*i for i in range(1,11)}, 'all decile evaluations')
    check(all(x['valid_targets'] == 1048576 and x['sample_hash'] == DEV and math.isfinite(x['ce']) for x in audit['internal_dev']), 'fixed dev results')
    check(result['internal_dev']['ce'] == audit['internal_dev'][-1]['ce'], 'final dev result')
    check({3276800,16384000,32768000}.issubset({x['tokens'] for x in audit['retained']}), 'required retained checkpoints')
    exits = job.get('exit_codes', [])
    check(len(exits) >= 2 and all(x['exit_code'] == 0 for x in exits), 'held native exit codes')
    check({(x['pid'],x['create_time']) for x in exits} == {(x['pid'],x['create_time']) for x in job['native_processes']}, 'exit process identities')
    return errors

def finished_training(name, job, check):
    d = Path(job['output'])
    ref = job['training_audit']
    audit = read(ref['path'])
    result = read(d/'RESULT.json')
    config = read(d/'RUN_CONFIG.json')
    for error in terminal_contract(name, job, result, audit, config): check(False,error)
    check(sha(ref['path']) == ref['sha256'], name+': audit receipt SHA256')
    refs = [audit['result'],audit['source_execution'],job['initialization']]
    refs += [x['evidence'] for x in audit['internal_dev']]
    refs += [x['marker'] for x in audit['retained']]
    for item in refs:
        p = Path(item['path'])
        check(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name+': bound file '+str(p))
    for item in audit['retained']:
        # Payload SHA256 was computed by the frozen training auditor at completion.
        # This lightweight observation verifies that receipt, marker and current size.
        check(Path(item['payload']['path']).stat().st_size == item['payload']['bytes'], name+': retained payload size')
    for process in job['exit_codes']:
        check(not same_process(process), name+': training process still alive')
    status = read(d/'STATUS.json')
    check(status['phase'] == 'complete' and status['tokens'] == 32768000, name+': terminal status')
    lock_path = ROOT/'locks'/(job['gpu']+'.lock')
    if lock_path.exists():
        lock = read(lock_path)
        check(Path(lock['run_directory']).resolve() != d.resolve(), name+': training GPU lock remains')
    check(not (d/'STOP').exists() and not (d/'FAILURE.json').exists(), name+': stop/failure')
    if job['phase'] == 'complete_audited':
        barrier_observation(name,job['diagnostics'],ref,check)
    lines = (d/'updates.jsonl').read_bytes().splitlines()
    last = json.loads(lines[-1])
    return dict(name=name,run_directory=str(d),gpu_uuid=job['gpu'],pid=result['lease']['pid'],
        actual_processes=[],observed_tokens=result['tokens'],target_tokens=result['target'],updates=audit['counts']['optimizer_updates'],
        total_overflow_retries=audit['counts']['overflow_retries'],last_ce=last['ce'],last_temperature_c=last['temperature_c'],
        observed_subphase=job['phase'],training_audit=ref,final_internal_dev=result['internal_dev'],elapsed_seconds=result['elapsed_seconds'],
        checkpoint_observations=[dict(tokens=x['tokens'],manifest=x['marker']['path'],manifest_sha256=x['marker']['sha256'],
            payload_hashes_recomputed_this_observation=False,payload_hash_scope='frozen completion audit; current size checked') for x in audit['retained']],
        training_native_exits=job['exit_codes'],training_gpu_lease_released=True)

def diagnostic_jobs(controller, check):
    rows=[]
    for name,j in controller['jobs'].items():
        phase=j['phase']
        row={'name':name,'phase':phase}
        if phase in ('waiting_for_training_audit','waiting_for_device_release'):
            check(j.get('no_gpu_lease_while_waiting') is True, name+': waiting without GPU lease')
        elif phase == 'complete_audited':
            cell=j.get('cell')
            if cell:
                for key in ('result','audit'):
                    check(sha(cell[key]) == cell[key+'_sha256'], name+': diagnostic '+key+' binding')
                check(read(cell['audit'])['passed'], name+': diagnostic audit passed')
                check(j.get('exits') and all(x['exit_code']==0 for x in j['exits']), name+': diagnostic held exits')
                row['cell']=cell
            else:
                row['barrier']=barrier_observation(name,j['diagnostics'],j['training_audit'],check)
        elif phase == 'running':
            d=Path(j['output'])
            check(not (d/'FAILURE.json').exists() and not (d/'STOP').exists(),name+': diagnostic stop/failure')
            actual=[]
            for rec in j.get('native_processes',[]):
                if same_process(rec):
                    p=psutil.Process(rec['pid'])
                    cmd=p.cmdline()
                    normalize=lambda c:[x.replace('\\','/').lower() for x in c]
                    check(normalize(cmd)==normalize(rec['command']),name+': diagnostic actual command')
                    actual.append({'pid':p.pid,'create_time':p.create_time(),'command':cmd})
            row['actual_processes']=actual
            age=time.time()-j['started_epoch']
            if not actual:
                # Controller status may lag a just-finished worker by a few seconds.
                recent_result=(d/'RESULT.json').exists() and time.time()-(d/'RESULT.json').stat().st_mtime < 60
                check(recent_result or age<30,name+': diagnostic process missing without recent result')
                row['observed_subphase']='awaiting_controller_exit_audit'
            lp=ROOT/'locks'/(j['gpu']+'.lock')
            if lp.exists():
                lock=read(lp)
                check(Path(lock['run_directory']).resolve()==d.resolve() and lock['pid'] in [p['pid'] for p in actual],name+': diagnostic GPU lock')
                row['lock']=lock
            else:
                check(age<120 or (d/'RESULT.json').exists(),name+': diagnostic lock absent beyond startup')
                row['observed_subphase']='startup_or_result_handoff'
            # A real process alone is not sufficient evidence of indefinite health.
            latest=max([j['started_epoch']]+[p.stat().st_mtime for p in d.glob('*') if p.is_file()]) if d.exists() else j['started_epoch']
            check(time.time()-latest<600 and age<3600,name+': diagnostic activity freshness')
        else:
            check(False,name+': unreviewed diagnostic phase '+phase)
        rows.append(row)
    return rows
