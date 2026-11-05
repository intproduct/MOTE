"""Missing initial FP16 probes and checkpoint-bound formal diagnostic queues.

No numerical worker modifications, no automatic retries and no process signals.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import uuid

os.environ['CUDA_VISIBLE_DEVICES']=''
import diagnostic_plan_v14 as plan
from diagnostic_plan_v14 import core,atomic_json
from run_formal_matrix_v12 import Supervisor as TrainingSupervisor,matching_run_processes


def initial_budget():
    # This is a conservative reservation, not permission for formal training.
    return core.read(plan.V12/'formal-supervision-v12-capacity-candidate-20261031.json')


def legacy_capacity():
    path=plan.OLD/'production_resource_preflight_v3.py'
    spec=importlib.util.spec_from_file_location('initial_legacy_capacity',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=module.check()
    core.require(value['acceptance_launch_fits'],'Initial diagnostic legacy resource envelope does not fit')
    core.require(value['processing_actual_plus_reserved_bytes']+450_000_000<=20_000_000_000,'Initial local processing allowance does not fit')
    return value


def validate(mode):
    core.require(Path.cwd().resolve()==core.WORK.resolve(),'Wrong experiment workspace')
    dependencies=plan.dependencies();rows,matrix=core.matrix_rows()
    bundle=Path(__file__).parent/'MANIFEST.json'
    own=plan.verify_bundle(bundle,None)
    workers=plan.verify_bundle(plan.WORKERS/'MANIFEST.json',own['worker_manifest_sha256'])
    core.require(workers['numerical_worker_sources_unchanged'] is True,'Numerical worker provenance missing')
    from fitmotn.experiments.qwen_joint.acceptance import source_identity
    core.bound(core.RELEASE,core.RELEASE_HASH)
    core.require(source_identity()==core.read(core.RELEASE)['code'],'Training source changed')
    if mode=='initial':
        core.require(not core.ACCEPTANCE.exists(),'Initial missing probes must precede final acceptance')
        rows=[r for r in rows if r['cohort']=='fp16'];approval=initial_budget();binding=None
        core.require(len(rows)==15,'Incomplete initial cohort')
        legacy_capacity()
    else:
        rows,matrix,approval,binding=core.validate_acceptance()
        core.bound(bundle,approval['formal_diagnostic_controller_manifest_sha256'])
        core.bound(plan.WORKERS/'MANIFEST.json',approval['formal_diagnostic_worker_manifest_sha256'])
        core.bound(plan.INITIAL/'INITIAL_LOCAL_INDEX.json',approval['initial_local_fp16_index_sha256'])
        for row in rows:plan.initial_cells(row)
    # Capacity always reserves the entire formal matrix, including initial mode.
    all_rows,_=core.matrix_rows();capacity=core.capacity_snapshot(all_rows,approval)
    core.require(capacity['fits'],'Shared global capacity does not fit')
    return rows,matrix,approval,binding,capacity


def training_ready(row,state,acceptance_hash,now):
    core.require(0<=now-state['time']<300,'Formal training status is stale')
    core.require(not state.get('failed') and not state.get('error'),'Training controller failed')
    core.require(state['phase'] in ('starting','running','complete_audited'),'Training controller is not healthy')
    core.require(state['acceptance']['sha256']==acceptance_hash,'Training acceptance differs')
    job=state['jobs'].get(row['name'])
    if not job or job.get('phase')!='waiting_for_formal_diagnostics':return None
    audit=job['training_audit'];expected=plan.TRAINING/(row['name']+'.TRAINING_AUDIT.json')
    core.require(Path(audit['path']).resolve()==expected.resolve(),'Unexpected training audit path')
    core.bound(audit['path'],audit['sha256'])
    records=job['exit_codes']
    core.require(len(records)>=2 and all(r['exit_code']==0 for r in records),'Actual training exits are unconfirmed')
    core.require(all(r['command'][1:]==row['launch']['argv'][1:] for r in records),'Exited training command differs')
    core.require(not matching_run_processes(row),'Training command remains live')
    return audit


class Diagnostics:
    def __init__(self,mode,rows,matrix,approval,binding):
        self.mode=mode;self.rows=rows;self.matrix=matrix;self.approval=approval;self.acceptance_binding=binding
        self.all_rows,_=core.matrix_rows()
        self.out=plan.INITIAL if mode=='initial' else plan.FORMAL
        self.lock=core.ROOT/'locks'/('initial-local-fp16-v14.lock' if mode=='initial' else 'formal-diagnostics-v14.lock')
        self.mutex=threading.RLock();self.launch_mutex=threading.Lock();self.drain=threading.Event()
        self.state=dict(phase='starting',mode=mode,pid=os.getpid(),created_epoch=time.time(),jobs={},failed=False,
                        acceptance=binding,manifest=core.bound(Path(__file__).parent/'MANIFEST.json'))

    def stopped(self):
        paths=[core.ROOT/'STOP',core.ROOT/'queue-20261030/STOP',self.out/'STOP']
        if self.mode=='formal':paths.append(plan.TRAINING/'STOP')
        return any(p.exists() for p in paths)

    def update(self,name,**fields):
        with self.mutex:self.state['jobs'].setdefault(name,{}).update(fields)

    def publish(self):
        with self.mutex:
            self.state['time']=time.time();atomic_json(self.out/'CONTROLLER.json',self.state)

    def stop_future_training(self,error):
        if self.mode=='formal':
            # Only the paired, owned training queue's cooperative stop marker.
            # Existing workers continue to finish; no signal is ever sent.
            plan.TRAINING.mkdir(exist_ok=True)
            try:
                with (plan.TRAINING/'STOP').open('x',encoding='utf8') as f:json.dump(dict(reason=error,diagnostic_controller_pid=os.getpid()),f)
            except FileExistsError:pass

    def wait_idle(self,row,name):
        from fitmotn.experiments.qwen_joint.resources import gpu_snapshot
        first=None;first_snapshot=None
        while not self.stopped() and not self.drain.is_set():
            snapshot=gpu_snapshot(row['gpu_uuid'])
            if snapshot['idle'] and not (core.ROOT/'locks'/(row['gpu_uuid']+'.lock')).exists():
                if first is None:first=time.monotonic();first_snapshot=snapshot
                if time.monotonic()-first>=60:return dict(last=snapshot,first=first_snapshot,separation_seconds=time.monotonic()-first)
            else:first=None;first_snapshot=None
            self.update(name,phase='waiting_for_device_release',gpu_observation=snapshot,no_gpu_lease_while_waiting=True)
            time.sleep(5)
        return None

    def launch(self,row,name,kind,output,checkpoint,idle):
        with self.launch_mutex:
            if self.stopped() or self.drain.is_set():return None
            if self.mode=='formal':core.bound(core.ACCEPTANCE,self.acceptance_binding['sha256'])
            else:legacy_capacity()
            budget=core.capacity_snapshot(self.all_rows,self.approval)
            core.require(budget['fits'],'Fresh shared reservation no longer fits')
            core.require(not Path(output).exists(),'Existing diagnostic output: no automatic retry')
            core.require(not (core.ROOT/'locks'/(row['gpu_uuid']+'.lock')).exists(),'Another worker acquired this device')
            core.require(not matching_run_processes(dict(output=str(output))),'Matching diagnostic already runs')
            command=plan.worker_command(row,kind,output,checkpoint)
            preflight=self.out/(name+'.PREFLIGHT.json');core.require(not preflight.exists(),'Previous attempt exists')
            atomic_json(preflight,dict(capacity=budget,idle=idle,command=command,worker_bundle=core.bound(plan.WORKERS/'MANIFEST.json') if checkpoint else None))
            env=dict(os.environ,**row['launch']['environment'])
            if checkpoint:
                item=next(x for x in self.matrix['configurations'] if Path(x['path']).stem==row['name'])
                env['QWEN_DIAGNOSTIC_FORMAL_CONFIG']=item['path']
            with (self.out/(name+'.stdout.log')).open('x') as stdout,(self.out/(name+'.stderr.log')).open('x') as stderr:
                process=subprocess.Popen(command,cwd=str(core.WORK),env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            observed=dict(name=name,output=str(output),launch=dict(argv=command))
            self.update(name,phase='running',method=row['method'],seed=row['seed'],gpu=row['gpu_uuid'],kind=kind,output=str(output),
                        command=command,launcher_pid=process.pid,started_epoch=time.time(),preflight=core.bound(preflight))
            return process,observed

    def cell(self,row,kind,output,checkpoint=None):
        tokens=0 if checkpoint is None else checkpoint['tokens']
        name=row['name'] if checkpoint is None else row['name']+'-'+f'{tokens:010d}-{kind}'
        idle=self.wait_idle(row,name)
        if idle is None:return None
        launched=self.launch(row,name,kind,output,checkpoint,idle)
        if launched is None:return None
        process,observed=launched
        # Diagnostic workers do not create the three training initialization
        # files. This frozen observer holds wrapper/native handles and drains
        # on every command/observation failure without killing either process.
        exits=TrainingSupervisor.observe_training(self,observed,process)
        lock=core.ROOT/'locks'/(row['gpu_uuid']+'.lock')
        if lock.exists():core.require(core.read(lock)['pid'] not in {v['pid'] for v in exits},'Exited worker retained GPU lease')
        output=Path(output);result=core.read(output/'RESULT.json')
        bundle=plan.WORKERS if checkpoint else Path(core.read(plan.OLD/'MANIFEST.json')['diagnostic_bundles'][kind]['manifest']).parent
        for file,h in result['helper_sha256'].items():core.bound(bundle/file,h)
        if checkpoint:
            reservation=core.read(output/'RESERVATION.json')
            core.require(reservation['policy']=='formal-v14-global-reservation' and reservation['formal_configuration']==row['name'],'Diagnostic resource policy differs')
            core.require(reservation['acceptance']['sha256']==self.acceptance_binding['sha256'],'Diagnostic acceptance binding differs')
            core.bound(bundle/'production_resource_preflight_v3.py',reservation['policy_sha256'])
        audited=plan.audit_measurement(output,kind,initial=checkpoint is None)
        core.require(not (output/'AUDIT.json').exists(),'Existing audit')
        atomic_json(output/'AUDIT.json',audited)
        cell=plan.measurement_cell(output,tokens,kind);plan.validate_cell(row,cell,checkpoint)
        self.update(name,phase='complete_audited',exits=exits,finished_epoch=time.time(),cell=cell)
        return cell

    def wait_training(self,row):
        started=time.monotonic()
        while not self.stopped() and not self.drain.is_set():
            if (plan.TRAINING/'CONTROLLER.json').exists():
                audit=training_ready(row,core.read(plan.TRAINING/'CONTROLLER.json'),self.acceptance_binding['sha256'],time.time())
                if audit:return audit
            core.require(time.monotonic()-started<72*3600,'No trained result within 72h; preserve for review')
            self.update(row['name'],phase='waiting_for_training_audit',no_gpu_lease_while_waiting=True);time.sleep(10)
        return None

    def worker(self,gpu):
        try:
            for row in [r for r in self.rows if r['gpu_uuid']==gpu]:
                if self.stopped() or self.drain.is_set():return
                if self.mode=='initial':
                    if self.cell(row,'local_precision',plan.INITIAL/row['name']) is None:return
                else:
                    bound=self.wait_training(row)
                    if bound is None:return
                    audit=core.read(bound['path']);planned=plan.checkpoint_plan(row,audit);cells=plan.initial_cells(row)
                    for item in planned:
                        cell=self.cell(row,item['kind'],item['output'],item['checkpoint'])
                        if cell is None:return
                        cells.append(cell)
                    core.bound(bound['path'],bound['sha256'])
                    barrier=plan.publish_barrier(row,bound['path'],cells)
                    self.update(row['name'],phase='complete_audited',training_audit=bound,diagnostics=barrier)
        except BaseException:
            error=traceback.format_exc();self.drain.set()
            with self.mutex:self.state.update(failed=True,phase='failed_preserved_draining_existing_workers')
            self.stop_future_training(error)
            atomic_json(self.out/(gpu+'.FAILURE.json'),dict(error=error,time=time.time(),no_process_signals_sent=True))
            self.update('worker-'+gpu,phase='failed_preserved',error=error)

    def run(self):
        core.require(not self.out.exists() and not self.lock.exists(),'Existing controller; inspect instead of relaunching')
        core.require(not self.stopped(),'STOP exists')
        nonce=str(uuid.uuid4())
        with self.lock.open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),nonce=nonce,output=str(self.out)),f)
        try:
            self.out.mkdir(exist_ok=False);self.state['phase']='running';self.publish()
            gpus=core.GPUS[1:] if self.mode=='initial' else core.GPUS
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures=[pool.submit(self.worker,gpu) for gpu in gpus]
                while not all(f.done() for f in futures):
                    try:self.publish()
                    except BaseException:
                        self.drain.set();self.state.update(failed=True,phase='status_failure_draining_existing_workers')
                        self.stop_future_training(traceback.format_exc())
                    time.sleep(5)
                for future in futures:future.result()
            if self.state['failed']:self.state['phase']='failed_preserved'
            elif self.stopped():self.state['phase']='stopped_preserved'
            else:
                core.require(all(self.state['jobs'].get(r['name'],{}).get('phase')=='complete_audited' for r in self.rows),'Incomplete diagnostics')
                if self.mode=='initial':
                    cells=[dict(name=r['name'],cell=self.state['jobs'][r['name']]['cell']) for r in self.rows]
                    for r,c in zip(self.rows,cells):plan.validate_cell(r,c['cell'])
                    atomic_json(self.out/'INITIAL_LOCAL_INDEX.json',dict(phase='complete_audited',measurements=cells,manifest=self.state['manifest']))
                self.state['phase']='complete_audited'
            self.publish()
        finally:
            if self.lock.exists() and core.read(self.lock).get('nonce')==nonce:self.lock.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('initial','formal'));parser.add_argument('action',choices=('validate','launch'))
    args=parser.parse_args();rows,matrix,approval,binding,budget=validate(args.mode)
    if args.action=='validate':print(json.dumps(dict(passed=True,mode=args.mode,rows=len(rows),capacity=budget,gpu_used=False)));return
    Diagnostics(args.mode,rows,matrix,approval,binding).run()


if __name__=='__main__':main()
