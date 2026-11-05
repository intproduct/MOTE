"""Local read-only experiment dashboard. No training imports or process control."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import psutil

ROOT = Path('E:/AI/MOTE-quantum')
HERE = Path(__file__).resolve().parent
QUEUE = ROOT / 'queue-20261030'
TRAIN = ROOT / 'runtime/formal-supervision-v12-20261031/CONTROLLER.json'
DIAG = ROOT / 'runtime/formal-diagnostics-v14-20261031/CONTROLLER.json'
GPUS = {
    'GPU-67f0c332-3f01-a733-4651-5532e94fc051': 'PRO 4000',
    'GPU-f145a48f-ccc7-cccf-94ad-acf680ff0891': 'V100 · A',
    'GPU-3753e6c9-3189-164f-a329-507d9a4a16f6': 'V100 · B',
}
METHODS = {'mlp':'MLP', 'affine_born':'Affine Born', 'reencode_cz':'Re-encode CZ', 'reencode_product':'Re-encode Product', 'dense':'Dense FFN'}
CACHE = {}
GPU_CACHE = {'time':0, 'values':{}, 'error':None}
BUILD_LOCK = threading.Lock()
STATUS_CACHE = {'time':0, 'value':None}

def read(path, default=None):
    path = Path(path)
    try:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        item = CACHE.get(str(path))
        if item and item[0] == key:
            return item[1]
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        CACHE[str(path)] = (key, value)
        return value
    except (OSError, ValueError):
        return default

def age(path):
    try:return max(0, time.time()-Path(path).stat().st_mtime)
    except OSError:return None

def tail(path, count=40):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2); size=f.tell(); start=max(0,size-1024*1024)
            f.seek(start); lines=f.read().splitlines(keepends=True)
        if start:lines=lines[1:]
        return [json.loads(x) for x in lines if x.endswith(b'\n') and x.strip()][-count:]
    except (OSError, ValueError):return []

def norm(command):return [str(x).replace('\\','/').lower() for x in command]

def identity(record):
    try:
        process=psutil.Process(record['pid'])
        return abs(process.create_time()-record['create_time']) < .01 and norm(process.cmdline()) == norm(record['command'])
    except (KeyError,psutil.Error):return False

def gpu_status():
    if time.time()-GPU_CACHE['time'] < 30:return GPU_CACHE
    values={};error=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:raise RuntimeError('GPU status unavailable')
        for row in csv.reader(io.StringIO(result.stdout)):
            if row[0].strip() in GPUS:
                values[row[0].strip()]=dict(zip(('temperature','utilization','memory_used','memory_total'),[float(x.strip()) for x in row[1:]]))
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired):error='显卡即时读数暂不可用'
    GPU_CACHE.update(time=time.time(),values=values,error=error)
    return GPU_CACHE

def build_status():
    now=time.time();request=read(QUEUE/'execution-request.json',{})
    training=read(TRAIN);diagnostics=read(DIAG)
    if training is None or diagnostics is None:raise RuntimeError('训练记录暂不可读，请稍后刷新')
    manifest=read(ROOT/'configs/formal-matrix-v11-20261031/MATRIX.json')
    alerts=[];controllers=[]
    for label,controller,path,entry in [('训练调度',training,TRAIN,'run_formal_matrix_v12.py'),('评测调度',diagnostics,DIAG,'run_diagnostics_v14.py')]:
        try:alive=any(Path(x).name==entry for x in psutil.Process(controller['pid']).cmdline())
        except (KeyError,psutil.Error):alive=False
        fresh=(age(path) or 0)<60
        finished=controller.get('phase') in ('complete','complete_audited')
        ok=finished or (alive and fresh and not controller.get('failed') and not controller.get('error'))
        controllers.append({'label':label,'alive':alive,'ok':ok,'age':age(path),'phase':controller.get('phase')})
        if not ok:alerts.append(label+'需要检查：进程、状态新鲜度或失败标记异常')
    for p in [ROOT/'STOP',QUEUE/'STOP',TRAIN.parent/'STOP',DIAG.parent/'STOP']:
        if p.exists():alerts.append('检测到停止标记：'+p.parent.name)
    rows=[]
    for item in manifest['configurations']:
        cfg=read(item['path']);name=cfg['name'];job=training.get('jobs',{}).get(name,{})
        directory=Path(cfg['output']);status=read(directory/'STATUS.json',{})
        phase=job.get('phase','queued');completed=phase=='complete_audited'
        tokens=status.get('tokens',0);target=cfg['training']['target_supervised_tokens']
        active=phase=='running';thermal=tail(directory/'thermal-pacing.jsonl',2) if active else []
        last_guard=thermal[-1] if thermal else {}
        logs=tail(directory/'updates.jsonl',40) if active else []
        devs=[read(p) for p in sorted(directory.glob('dev-*.json'))] if job else []
        devs=[d for d in devs if d]
        state='complete' if completed else ('diagnostics' if phase=='waiting_for_formal_diagnostics' else ('running' if active else ('queued' if not job else 'waiting')))
        if phase in ('failed','error','stopped') or (directory/'FAILURE.json').exists() or (directory/'STOP').exists():state='issue'
        processes_ok=bool(job.get('native_processes')) and all(identity(x) for x in job.get('native_processes',[])) if active else None
        elapsed=now-job.get('started_epoch',now)
        if active and not processes_ok and elapsed>120 and not (directory/'RESULT.json').exists():
            state='issue';alerts.append(name+'：训练进程身份待检查')
        activity='训练'
        if active and last_guard.get('stage')=='evaluation':activity='开发集评估'
        if active and (last_guard.get('wait_seconds',0)>0 or last_guard.get('event')):activity='温控等待 / 检查'
        activity_age=age(directory/'thermal-pacing.jsonl') if active else None
        if active and activity_age is not None and activity_age>600:
            state='issue';alerts.append(name+'：活动记录超过 10 分钟未更新')
        cp=read(job.get('diagnostics',{}).get('path','')) if completed else None
        diagnostic_cells=len(cp.get('measurements',[])) if cp else 0
        if not completed:
            diagnostic_cells=sum(1 for k,j in diagnostics.get('jobs',{}).items() if k.startswith(name+'-') and j.get('phase')=='complete_audited' and j.get('cell'))
        rows.append({'name':name,'method':cfg['method'],'label':METHODS[cfg['method']],'seed':cfg['seed'],'cohort':cfg['cohort'].upper(),'gpu':cfg['gpu_uuid'],'gpu_label':GPUS[cfg['gpu_uuid']],'state':state,'phase':phase,'activity':activity,'tokens':tokens,'target':target,'percent':min(100,100*tokens/target),'updates':tokens//32768,'diagnostic_cells':diagnostic_cells,'processes_ok':processes_ok,'activity_age':activity_age,'dev_ce':devs[-1]['ce'] if devs else None,'dev_evaluations':len(devs),'loss_history':[[x['tokens']//32768,x['ce']] for x in logs],'last_ce':logs[-1]['ce'] if logs else None,'overflow_recent':sum(x['overflow_retries'] for x in logs),'elapsed_seconds':(job.get('finished_epoch',now)-job.get('started_epoch',now)) if job else 0})
    g=gpu_status();gpu_rows=[]
    for uuid,label in GPUS.items():
        lock=read(ROOT/'locks'/(uuid+'.lock'),{})
        tasks=[x for x in rows if x['gpu']==uuid and x['state'] in ('running','diagnostics','issue','waiting')]
        diagnostic=[j for j in diagnostics.get('jobs',{}).values() if j.get('gpu')==uuid and j.get('phase')=='running']
        owned=bool(lock) and psutil.pid_exists(lock.get('pid',-1))
        gpu_rows.append({'uuid':uuid,'label':label,'metrics':g['values'].get(uuid),'metrics_age':max(0,time.time()-g['time']),'lock_present':bool(lock),'lock_owner_alive':owned,'task':tasks[-1]['name'] if tasks else None,'diagnostic_running':bool(diagnostic),'purpose':lock.get('purpose')})
    hp=request.get('formal_health_observation');health=read(hp,{}) if hp else {}
    hc={'passed':health.get('passed'),'observed_epoch':health.get('observed_epoch'),'errors':health.get('errors',[])}
    completed=sum(x['state']=='complete' for x in rows)
    new_diags=sum(1 for j in diagnostics.get('jobs',{}).values() if j.get('phase')=='complete_audited' and j.get('cell'))
    return {'observed_epoch':time.time(),'phase':request.get('phase'),'summary':{'completed':completed,'total':len(rows),'running':sum(x['state']=='running' for x in rows),'diagnostics':sum(x['state']=='diagnostics' for x in rows),'queued':sum(x['state']=='queued' for x in rows),'tokens':sum(x['tokens'] for x in rows),'target_tokens':sum(x['target'] for x in rows),'new_diagnostics':new_diags,'target_diagnostics':270},'controllers':controllers,'health':hc,'alerts':alerts,'gpu_warning':g['error'],'gpus':gpu_rows,'rows':rows,'disk_free_bytes':shutil.disk_usage(ROOT).free,'latest_report':bool(request.get('latest_formal_stage_report'))}

def status():
    with BUILD_LOCK:
        if time.time()-STATUS_CACHE['time']>3:
            value=build_status();STATUS_CACHE.update(time=time.time(),value=value)
        return STATUS_CACHE['value']

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        host=self.headers.get('Host','').split(':')[0]
        if host not in ('127.0.0.1','localhost'):
            self.send_error(403);return
        path=urlparse(self.path).path
        try:
            if path=='/api/status':data=json.dumps(status(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
            elif path in ('/','/index.html','/style.css','/app.js'):
                file=HERE/('index.html' if path=='/' else path[1:]);data=file.read_bytes()
                mime={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8'}[file.suffix]
            elif path=='/report/latest':
                request=read(QUEUE/'execution-request.json',{});file=(Path(request['latest_formal_stage_report'])/'stage-report.pdf').resolve()
                if not file.is_relative_to((ROOT/'reports').resolve()):raise ValueError('Invalid report path')
                data=file.read_bytes();mime='application/pdf'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            data=json.dumps({'error':str(exc)},ensure_ascii=False).encode()
            self.send_response(503);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    process=psutil.Process()
    (HERE/'SERVER.json').write_text(json.dumps({'url':f'http://127.0.0.1:{args.port}','pid':os.getpid(),'create_time':process.create_time(),'command':process.cmdline(),'started_epoch':time.time(),'read_only_experiment_access':True},indent=2),encoding='utf-8')
    print(f'Dashboard ready: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
