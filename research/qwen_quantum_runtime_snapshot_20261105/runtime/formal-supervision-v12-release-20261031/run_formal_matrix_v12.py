"""Three independent native queues with one global reservation and exact exit observation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import uuid

os.environ['CUDA_VISIBLE_DEVICES']=''
import formal_runtime_v12 as core
from watch_pro_product_exit_v9 import HeldProcess,atomic_json

OUT=core.ROOT/'runtime/formal-supervision-v12-20261031'
LOCK=core.ROOT/'locks/formal-matrix-v12.lock'


def stopped():
    return any(p.exists() for p in (core.ROOT/'STOP',core.ROOT/'queue-20261030/STOP',OUT/'STOP'))


def verify_worker(words,row):
    expected=row['launch']['argv']
    core.require(len(words)==len(expected),'Worker argument count differs')
    core.require(Path(words[0]).name.lower()=='python.exe','Unexpected native interpreter')
    # A venv redirector launches the base interpreter; every remaining argument
    # must be exactly the frozen formal command, including seed, UUID and paths.
    core.require(words[1:]==expected[1:],'Worker command differs from frozen matrix')


def verify_exits(observed):
    core.require(len(observed)>=2,'Both venv wrapper and native worker must have held handles')
    core.require(len({r['pid'] for r in observed})==len(observed),'Repeated process identity')
    core.require(all(r['exit_code']==0 for r in observed),'A real worker/wrapper did not exit successfully')
    return True


def matching_run_processes(row):
    import psutil
    found=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        words=process.info['cmdline'] or []
        if (process.info['name'] or '').lower() in ('python.exe','pythonw.exe') and row['output'] in words:
            found.append(dict(pid=process.pid,command=words))
    return found


def verify_diagnostic_barrier(row,training_audit_path):
    path=core.DIAGNOSTICS/row['name']/'DIAGNOSTICS_COMPLETE.json'
    if not path.exists():return None
    value=core.read(path)
    core.require(value.get('schema')=='qwen-formal-diagnostics-v12' and value.get('passed') is True,'Diagnostic completion schema differs')
    core.require(value['name']==row['name'] and value['training_audit_sha256']==core.sha(training_audit_path),'Diagnostics not bound to this trained run')
    expected={(0,'full_model'),(0,'local_precision')}
    expected|={(t,k) for t in (3276800,16384000,32768000) for k in ('timing','full_model','local_precision')}
    cells=value['measurements'];core.require(len(cells)==len(expected) and {(x['tokens'],x['kind']) for x in cells}==expected,'Incomplete formal diagnostic states')
    retained={r['tokens']:r for r in core.read(training_audit_path)['retained']}
    for cell in cells:
        result=core.read(cell['result']);audit=core.read(cell['audit'])
        core.bound(cell['result'],cell['result_sha256']);core.bound(cell['audit'],cell['audit_sha256'])
        core.require(result['passed'] and result['phase']=='complete' and audit['passed'] and audit['result_sha256']==cell['result_sha256'],'Diagnostic audit failed or unbound')
        core.require(result['protocol_sha256']==core.PROTOCOL_HASH,'Diagnostic protocol differs')
        core.require(result['method']==row['method'] and result['seed']==row['seed'] and result['release_sha256']==core.RELEASE_HASH,'Diagnostic model/release differs')
        core.require(result.get('checkpoint_tokens',result.get('binding',{}).get('tokens'))==cell['tokens'],'Diagnostic target budget differs')
        core.require(result.get('native_precision',result.get('native_checkpoint_precision'))==('torch.bfloat16' if row['cohort']=='bf16' else 'torch.float16'),'Diagnostic native precision differs')
        physical=result['physical_device']['uuid'].removeprefix('GPU-')
        allowed=(core.GPUS[:1] if row['cohort']=='bf16' else core.GPUS[1:])
        core.require(physical in [g.removeprefix('GPU-') for g in allowed],'Diagnostic hardware cohort differs')
        if cell['tokens']:
            checkpoint=retained[cell['tokens']]
            payload_hash=result.get('checkpoint_payload_sha256',result.get('binding',{}).get('payload_sha256'))
            actual_path=result.get('checkpoint',result.get('binding',{}).get('checkpoint'))
            core.require(payload_hash==checkpoint['payload']['sha256'] and Path(actual_path).resolve()==Path(checkpoint['checkpoint']).resolve(),'Diagnostic is not the retained formal checkpoint')
        else:
            core.require('checkpoint' not in result and 'checkpoint' not in result.get('binding',{}),'Initial diagnostic used a trained checkpoint')
    return core.bound(path)


class Supervisor:
    def __init__(self,rows,manifest,approval,acceptance_binding):
        self.rows=rows;self.manifest=manifest;self.approval=approval;self.acceptance_binding=acceptance_binding
        self.mutex=threading.RLock();self.launch_mutex=threading.Lock();self.drain=threading.Event()
        self.state=dict(phase='starting',pid=os.getpid(),created_epoch=time.time(),jobs={},matrix_sha256=core.MATRIX_HASH,
                        acceptance=acceptance_binding,formal_approved=True,training_scope='30 native independent runs; diagnostic barrier before each subsequent job',
                        gpu_uuids=list(core.GPUS),failed=False)

    def update(self,name,**fields):
        with self.mutex:self.state['jobs'].setdefault(name,{}).update(fields)

    def publish(self):
        with self.mutex:
            self.state['time']=time.time();atomic_json(OUT/'CONTROLLER.json',self.state)

    def wait_idle(self,row):
        from fitmotn.experiments.qwen_joint.resources import gpu_snapshot
        first=None
        while True:
            if stopped() or self.drain.is_set():return None
            lock=core.ROOT/'locks'/(row['gpu_uuid']+'.lock')
            snapshot=gpu_snapshot(row['gpu_uuid'])
            if snapshot['idle'] and not lock.exists():
                first=first or time.monotonic()
                if time.monotonic()-first>=60:return snapshot
            else:first=None
            self.update(row['name'],phase='waiting_for_device_release',gpu_observation=snapshot,no_gpu_lease_while_waiting=True)
            time.sleep(5)

    def launch_training(self,row,snapshot):
        """Serialize the final shared budget check and launch across the three cards."""
        import psutil
        with self.launch_mutex:
            if stopped() or self.drain.is_set():return None
            core.bound(core.ACCEPTANCE,self.acceptance_binding['sha256'])
            core.require(not Path(row['output']).exists(),'Existing formal output; preserve without re-launch')
            core.require(not (core.ROOT/'locks'/(row['gpu_uuid']+'.lock')).exists(),'Device lease acquired by another job')
            for process in psutil.process_iter(['pid','name','cmdline']):
                words=process.info['cmdline'] or []
                core.require(not (process.info['name'] or '').lower().startswith('python') or row['output'] not in words,'Matching formal worker already exists')
            capacity=core.capacity_snapshot(self.rows,self.approval)
            core.require(capacity['fits'],'Fresh global formal reservation no longer fits')
            preflight=OUT/(row['name']+'.PREFLIGHT.json')
            core.require(not preflight.exists(),'Previous launch attempt exists; no automatic retry')
            atomic_json(preflight,dict(capacity=capacity,gpu=snapshot,acceptance=self.acceptance_binding,configuration=row))
            env=dict(os.environ,**row['launch']['environment'])
            stdout=(OUT/(row['name']+'.stdout.log')).open('x',encoding='utf8')
            stderr=(OUT/(row['name']+'.stderr.log')).open('x',encoding='utf8')
            try:
                process=subprocess.Popen(row['launch']['argv'],cwd=row['working_directory'],env=env,stdout=stdout,stderr=stderr,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            finally:stdout.close();stderr.close()
            self.update(row['name'],phase='running',launcher_pid=process.pid,command=row['launch']['argv'],output=row['output'],
                        gpu=row['gpu_uuid'],started_epoch=time.time(),preflight=core.bound(preflight))
            return process

    def observe_training(self,row,process):
        import psutil
        held=[];seen=set();start_binding=None;observer_error=None
        try:
            first=HeldProcess(process.pid);held.append(first);seen.add(process.pid);verify_worker(first.record['command'],row)
            # Never infer native-worker exit from the redirector PID alone.
            while True:
                if first.poll() is None:
                    try:children=psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:children=[]
                    for child in children:
                        if child.pid in seen:continue
                        try:
                            if child.name().lower()!='python.exe':continue
                            observed=HeldProcess(child.pid)
                        except psutil.NoSuchProcess:continue
                        held.append(observed);seen.add(child.pid)
                        verify_worker(observed.record['command'],row)
                codes=[p.poll() for p in held]
                records=[dict(**p.record,exit_code=code) for p,code in zip(held,codes)]
                self.update(row['name'],native_processes=records)
                if observer_error is None and start_binding is None and all((Path(row['output'])/n).exists() for n in ('RUN_CONFIG.json','PARAMETERS.json','ROUTER_PLACEMENT.json')):
                    try:
                        start_binding=core.audit_initialization(row,self.acceptance_binding['sha256'])
                        start_path=OUT/(row['name']+'.INITIALIZATION.json')
                        atomic_json(start_path,start_binding)
                        self.update(row['name'],initialization=core.bound(start_path))
                    except BaseException:
                        observer_error=traceback.format_exc();self.drain.set()
                        self.update(row['name'],phase='initialization_audit_failed_observing_existing_workers',error=observer_error)
                if len(held)>=2 and all(c is not None for c in codes):
                    process.wait();verify_exits(records)
                    if observer_error is not None:raise RuntimeError(observer_error)
                    return records
                if first.poll() is not None and len(held)<2:
                    process.wait();raise RuntimeError('Wrapper exited before native process handles were captured; preserve for inspection')
                time.sleep(2)
        except BaseException:
            failure=traceback.format_exc();self.drain.set()
            # Keep the owned Popen and every successfully captured kernel handle
            # until actual exit, even when command or initialization audit fails.
            # No signal or automatic retry is sent to an experiment process.
            while True:
                try:
                    exits=[dict(**h.record,exit_code=h.poll()) for h in held]
                    live=matching_run_processes(row)
                    self.update(row['name'],phase='observer_failure_waiting_for_existing_workers',error=failure,
                                native_processes=exits,matching_commands=live,no_process_signals_sent=True)
                    if process.poll() is not None and all(p['exit_code'] is not None for p in exits) and not live:break
                except BaseException:
                    self.update(row['name'],phase='observer_failure_exit_unverified',error=failure,exit_observation_error=traceback.format_exc())
                time.sleep(2)
            self.update(row['name'],phase='observer_failure_workers_exited',wrapper_exit_code=process.returncode)
            raise
        finally:
            for p in held:p.close()

    def worker(self,gpu):
        try:
            for row in [r for r in self.rows if r['gpu_uuid']==gpu]:
                if stopped() or self.drain.is_set():return
                snapshot=self.wait_idle(row)
                if snapshot is None:return
                process=self.launch_training(row,snapshot)
                if process is None:return
                exits=self.observe_training(row,process)
                gpu_lock=core.ROOT/'locks'/(gpu+'.lock')
                if gpu_lock.exists():
                    core.require(core.read(gpu_lock)['pid'] not in {p['pid'] for p in exits},'Exited worker did not release its own GPU lease')
                self.update(row['name'],phase='auditing_training',exit_codes=exits,finished_epoch=time.time())
                audit=core.audit_training(row,self.acceptance_binding['sha256'])
                audit_path=OUT/(row['name']+'.TRAINING_AUDIT.json');atomic_json(audit_path,audit)
                self.update(row['name'],phase='waiting_for_formal_diagnostics',training_audit=core.bound(audit_path),no_gpu_lease_while_waiting=True)
                while not stopped() and not self.drain.is_set():
                    barrier=verify_diagnostic_barrier(row,audit_path)
                    if barrier:
                        self.update(row['name'],phase='complete_audited',diagnostics=barrier);break
                    time.sleep(10)
                if stopped() or self.drain.is_set():return
        except BaseException:
            self.drain.set()
            with self.mutex:
                self.state['failed']=True;self.state['phase']='failed_preserved_draining_existing_workers'
            error=dict(gpu=gpu,error=traceback.format_exc(),time=time.time(),no_process_signals_sent=True)
            atomic_json(OUT/(gpu+'.FAILURE.json'),error)
            self.update('worker-'+gpu,phase='failed_preserved',**error)

    def run(self):
        nonce=str(uuid.uuid4());created=False
        core.require(not OUT.exists() and not LOCK.exists(),'Existing supervisor/lock; inspect instead of re-launching')
        core.require(not stopped(),'STOP marker before formal launch')
        for row in self.rows:core.require(not Path(row['output']).exists(),'Existing formal run; fresh launch refuses adoption')
        budget=core.capacity_snapshot(self.rows,self.approval);core.require(budget['fits'],'Global formal reservation does not fit')
        try:
            with LOCK.open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),nonce=nonce,output=str(OUT)),f)
            OUT.mkdir(exist_ok=False);created=True
            atomic_json(OUT/'RESERVATION.json',budget);self.state['phase']='running';self.publish()
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures=[executor.submit(self.worker,gpu) for gpu in core.GPUS]
                while not all(f.done() for f in futures):
                    try:self.publish()
                    except BaseException:
                        self.drain.set()
                        with self.mutex:self.state.update(failed=True,phase='status_failure_draining_existing_workers',status_error=traceback.format_exc())
                        try:atomic_json(OUT/('STATUS_FAILURE-'+uuid.uuid4().hex+'.json'),self.state)
                        except BaseException:pass
                    time.sleep(5)
                for f in futures:f.result()
            if self.state['failed']:self.state['phase']='failed_preserved'
            elif stopped():self.state['phase']='stopped_preserved'
            else:
                core.require(all(self.state['jobs'].get(row['name'],{}).get('phase')=='complete_audited' for row in self.rows),'Not all formal cells completed')
                self.state['phase']='complete_audited'
            self.publish()
        finally:
            if LOCK.exists() and core.read(LOCK).get('nonce')==nonce:LOCK.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('validate','launch'));args=parser.parse_args()
    rows,manifest,approval,binding=core.validate_acceptance()
    bundle=Path(__file__).parent/'MANIFEST.json'
    core.bound(bundle,approval['formal_controller_manifest_sha256'])
    for name,expected in core.read(bundle)['files'].items():core.bound(bundle.parent/name,expected)
    if args.action=='validate':
        capacity=core.capacity_snapshot(rows,approval);core.require(capacity['fits'],'Global formal reservation does not fit')
        print(json.dumps(dict(passed=True,configurations=len(rows),capacity=capacity,gpu_used=False)));return
    Supervisor(rows,manifest,approval,binding).run()


if __name__=='__main__':main()
