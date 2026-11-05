"""Bound initial references and retained-checkpoint contracts for formal diagnostics."""
import json
from pathlib import Path
import sys

ROOT=Path('E:/AI/MOTE-quantum')
V12=ROOT/'runtime/formal-supervision-v12-release-20261031'
V12_HASH='54d23c18597285fc9d35f8bce3c5dde7e34c1bf978da6933b832ace0b3e5f1f2'
OLD=ROOT/'runtime/bf16-followup-v6r1-release-20261031'
OLD_HASH='b55fcea823f133d944147ffefbcc8d756ae6e44efa165caa26234e75c206c719'
WORKERS=ROOT/'runtime/formal-diagnostic-workers-v14-release-20261031'
INITIAL=ROOT/'diagnostics/initial-local-fp16-v14-20261031'
FORMAL=ROOT/'runtime/formal-diagnostics-v14-20261031'
TRAINING=ROOT/'runtime/formal-supervision-v12-20261031'
sys.path.insert(0,str(V12))
import formal_runtime_v12 as core
from watch_pro_product_exit_v9 import atomic_json

KINDS=('timing','local_precision','full_model')
TOKENS=(3276800,16384000,32768000)
ENTRIES=dict(timing='isolated_inference_timing_v4.py',local_precision='local_checkpoint_precision_v4.py',full_model='full_checkpoint_diagnostics_v4.py')


def verify_bundle(path,expected):
    path=Path(path);core.bound(path,expected);value=core.read(path)
    for n,h in value['files'].items():core.bound(path.parent/n,h)
    return value


def dependencies():
    verify_bundle(V12/'MANIFEST.json',V12_HASH)
    value=verify_bundle(OLD/'MANIFEST.json',OLD_HASH)
    for item in value['diagnostic_bundles'].values():verify_bundle(item['manifest'],item['sha256'])
    return value


def audit_measurement(output,kind,initial=False):
    # Import only the already-frozen independent auditors, never a mutable copy.
    verify_bundle(OLD/'MANIFEST.json',OLD_HASH)
    sys.path.insert(0,str(OLD))
    if kind=='local_precision':
        from audit_local_checkpoint_precision_v4 import audit_result as audit
    elif kind=='timing':
        from audit_inference_timing_v4 import audit
    elif initial:
        from audit_initial_full_model_diagnostics_v5 import audit
    else:
        from audit_full_checkpoint_diagnostics_v4 import audit
    value=audit(Path(output));core.require(value['passed'],'Independent raw-array audit failed');return value


def measurement_cell(output,tokens,kind,result_hash=None,audit_hash=None):
    output=Path(output)
    rb=core.bound(output/'RESULT.json',result_hash);ab=core.bound(output/'AUDIT.json',audit_hash)
    audit=core.read(ab['path']);core.require(audit['passed'] and audit['result_sha256']==rb['sha256'],'Unbound independent audit')
    return dict(tokens=tokens,kind=kind,result=rb['path'],result_sha256=rb['sha256'],audit=ab['path'],audit_sha256=ab['sha256'])


def validate_cell(row,cell,checkpoint=None):
    core.bound(cell['result'],cell['result_sha256']);core.bound(cell['audit'],cell['audit_sha256'])
    result=core.read(cell['result']);audit=core.read(cell['audit'])
    core.require(result['phase']=='complete' and result['passed'] and audit['passed'] and audit['result_sha256']==cell['result_sha256'],'Incomplete diagnostic')
    core.require(result['method']==row['method'] and result['seed']==row['seed'],'Diagnostic method/seed differs')
    core.require(result['release_sha256']==core.RELEASE_HASH and result['protocol_sha256']==core.PROTOCOL_HASH,'Diagnostic release/protocol differs')
    precision=result.get('native_precision',result.get('native_checkpoint_precision'))
    core.require(precision==('torch.bfloat16' if row['cohort']=='bf16' else 'torch.float16'),'Diagnostic precision differs')
    allowed=core.GPUS[:1] if row['cohort']=='bf16' else core.GPUS[1:]
    core.require(result['physical_device']['uuid'].removeprefix('GPU-') in [g.removeprefix('GPU-') for g in allowed],'Physical GPU cohort differs')
    actual_tokens=result.get('checkpoint_tokens',result.get('binding',{}).get('tokens'))
    core.require(actual_tokens==cell['tokens'],'Diagnostic checkpoint tokens differ')
    if cell['kind']=='local_precision':core.require(result.get('gpu_used') and result.get('native_arithmetic_measured'),'Local arithmetic was not native')
    if checkpoint is None:
        core.require(cell['tokens']==0 and 'checkpoint' not in result and 'checkpoint' not in result.get('binding',{}),'Initial state used trained values')
    else:
        actual=result.get('checkpoint',result.get('binding',{}).get('checkpoint'))
        payload=result.get('checkpoint_payload_sha256',result.get('binding',{}).get('payload_sha256'))
        core.require(Path(actual).resolve()==Path(checkpoint['checkpoint']).resolve() and payload==checkpoint['payload']['sha256'],'Different trained checkpoint')
    return cell


def initial_cells(row):
    bf=ROOT/'diagnostics/bf16-followup-v6r1-pro4000-20261031/CONTROLLER.json'
    fp=ROOT/'diagnostics/initial-full-model-v5-v100a-20261031/CONTROLLER.json'
    controller=core.read(bf if row['cohort']=='bf16' else fp)
    cells=[]
    for kind in ('full_model','local_precision'):
        if row['cohort']=='fp16' and kind=='local_precision':
            index=core.read(INITIAL/'INITIAL_LOCAL_INDEX.json')
            core.require(index['phase']=='complete_audited' and len(index['measurements'])==15,'FP16 initial local cohort incomplete')
            matches=[v for v in index['measurements'] if v['name']==row['name']]
            core.require(len(matches)==1,'Missing/duplicate FP16 initial reference')
            cell=matches[0]['cell']
        else:
            matches=[j for j in controller['jobs'].values() if j['method']==row['method'] and j['seed']==row['seed'] and j.get('source','initial')=='initial' and
                (j.get('kind','initial_full_model') in ('initial_full_model','full_model') if kind=='full_model' else j.get('kind')=='local_precision')]
            core.require(len(matches)==1 and matches[0]['phase']=='complete','Initial diagnostic missing/ambiguous')
            job=matches[0];cell=measurement_cell(job['output'],0,kind,job['result_sha256'],job['audit_sha256'])
        cells.append(validate_cell(row,cell))
    return cells


def checkpoint_plan(row,audit):
    core.require(audit['passed'] and audit['phase']=='training_complete_audited' and audit['name']==row['name'],'Training audit differs')
    core.require((audit['seed'],audit['method'],audit['cohort'])==(row['seed'],row['method'],row['cohort']),'Training audit cohort differs')
    core.require(audit['counts']['optimizer_updates']==1000 and audit['primary_tokens']==32768000,'Formal budget incomplete')
    core.require(Path(audit['result']['path']).resolve()==(Path(row['output'])/'RESULT.json').resolve(),'Training result belongs to another run')
    core.bound(audit['result']['path'],audit['result']['sha256'])
    checkpoints={c['tokens']:c for c in audit['retained']}
    core.require(len(checkpoints)==len(audit['retained']) and set(TOKENS)<=set(checkpoints),'Protected checkpoints missing')
    planned=[]
    for token in TOKENS:
        checkpoint=checkpoints[token];path=Path(checkpoint['checkpoint'])
        core.require(path.resolve()==(Path(row['output'])/'checkpoints'/f'checkpoint-{token:010d}').resolve(),'Checkpoint path escapes run')
        core.require(Path(checkpoint['marker']['path']).resolve()==(path/'complete.json').resolve() and Path(checkpoint['payload']['path']).resolve()==(path/'training.pt').resolve(),'Checkpoint member escapes')
        core.bound(path/'complete.json',checkpoint['marker']['sha256'])
        core.bound(path/'training.pt',checkpoint['payload']['sha256'])
        for kind in KINDS:planned.append(dict(tokens=token,kind=kind,checkpoint=checkpoint,output=str(core.DIAGNOSTICS/row['name']/f'{token:010d}-{kind}')))
    return planned


def worker_command(row,kind,output,checkpoint=None):
    if checkpoint is None:
        core.require(row['cohort']=='fp16' and row['gpu_uuid'] in core.GPUS[1:] and kind=='local_precision','Only missing FP16 initial local cells may launch here')
        core.require(Path(output).resolve().parent==INITIAL.resolve(),'Unexpected initial output')
        item=core.read(OLD/'MANIFEST.json')['diagnostic_bundles'][kind]
        command=[str(ROOT/'envs/qwen-native-py312-cu128/Scripts/python.exe'),'-u',str(Path(item['manifest']).parent/item['entry']),
                 '--initial','--method',row['method'],'--seed',str(row['seed'])]
    else:
        core.require(kind in KINDS,'Unknown diagnostic kind')
        core.require(Path(output).resolve().parent==(core.DIAGNOSTICS/row['name']).resolve(),'Unexpected formal diagnostic output')
        command=[str(ROOT/'envs/qwen-native-py312-cu128/Scripts/python.exe'),'-u',str(WORKERS/ENTRIES[kind]),'--checkpoint',checkpoint['checkpoint']]
    return command+['--gpu',row['gpu_uuid'],'--output',str(output)]


def publish_barrier(row,audit_path,cells):
    expected={(0,'full_model'),(0,'local_precision')}|{(t,k) for t in TOKENS for k in KINDS}
    core.require(len(cells)==11 and {(c['tokens'],c['kind']) for c in cells}==expected,'Missing or duplicate formal diagnostics')
    audit=core.read(audit_path);checkpoint_plan(row,audit)
    retained={v['tokens']:v for v in audit['retained']}
    for cell in cells:validate_cell(row,cell,retained.get(cell['tokens']))
    path=core.DIAGNOSTICS/row['name']/'DIAGNOSTICS_COMPLETE.json'
    core.require(not path.exists(),'Diagnostic receipt already exists')
    value=dict(schema='qwen-formal-diagnostics-v12',passed=True,name=row['name'],training_audit_sha256=core.sha(audit_path),measurements=cells)
    atomic_json(path,value)
    from run_formal_matrix_v12 import verify_diagnostic_barrier
    return verify_diagnostic_barrier(row,audit_path)
