"""Read-only formal validation, shared capacity accounting and result audits."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT=Path('E:/AI/MOTE-quantum')
WORK=Path('D:/AI/MOTE')
MATRIX=ROOT/'configs/formal-matrix-v11-20261031/MATRIX.json'
MATRIX_HASH='101b00fa3696d639d4fc91051aad93c7bbf46e2ebaac6332c2c8d99f1a6691a5'
RELEASE=ROOT/'releases/integration-v3.json'
RELEASE_HASH='2186bce5f9ad702d65861dc40fff5e20a59b0211bdd32f3c5812b15ef0ad4685'
PROTOCOL_HASH='d4299837c68ce0e6dcc80c181468bc5cfd44cdbfbd0431a1d35c92dcc8dd8bc1'
ACCEPTANCE=ROOT/'releases/INTEGRATION_ACCEPTANCE.json'
RUNS=ROOT/'runs/formal-matrix-v11'
DIAGNOSTICS=ROOT/'diagnostics/formal-matrix-v12'
PRO='GPU-67f0c332-3f01-a733-4651-5532e94fc051'
GPUS=(PRO,'GPU-f145a48f-ccc7-cccf-94ad-acf680ff0891','GPU-3753e6c9-3189-164f-a329-507d9a4a16f6')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def bound(path,expected=None):
    path=Path(path);h=sha(path)
    require(expected is None or h==expected,'Evidence hash mismatch: '+str(path))
    return dict(path=str(path),sha256=h,bytes=path.stat().st_size)


def matrix_rows():
    bound(MATRIX,MATRIX_HASH);manifest=read(MATRIX)
    rows=[]
    for item in manifest['configurations']:
        path=Path(item['path']);require(path.resolve().parent==MATRIX.parent.resolve(),'Config path escape')
        bound(path,item['sha256']);rows.append(read(path))
    require(len(rows)==30 and len({r['name'] for r in rows})==30,'Incomplete matrix')
    return rows,manifest


def validate_acceptance(acceptance_path=ACCEPTANCE):
    """Refuse before making a directory, lock, worker or CUDA context."""
    require(Path.cwd().resolve()==WORK.resolve(),'Wrong experiment workspace')
    acceptance_path=Path(acceptance_path)
    require(acceptance_path.resolve()==ACCEPTANCE.resolve(),'Unexpected acceptance path')
    require(acceptance_path.is_file(),'Final integration acceptance is not published')
    approval=read(acceptance_path)
    require(approval.get('formal_approved') is True,'Final integration acceptance is incomplete')
    require(approval.get('formal_matrix_sha256')==MATRIX_HASH,'Formal matrix not bound by acceptance')
    rows,manifest=matrix_rows();bound(RELEASE,RELEASE_HASH);release=read(RELEASE)
    from fitmotn.experiments.qwen_joint.acceptance import source_identity
    from fitmotn.experiments.qwen_joint.validation import formal_budget
    require(source_identity()==release['code'],'Frozen checkout changed')
    for row in rows:
        target=formal_budget(approval,release['code'],release['data_sha256'],release['model_manifest_sha256'],release['config_sha256'],
            row['method'],row['seed'],row['gpu_uuid'],(row['initialization']['common'] or {}).get('sha256'),ROOT)
        require(target==row['training']['target_supervised_tokens']==32768000,'Accepted target differs')
    require(approval.get('formal_controller_manifest_sha256') is not None,'Formal supervision bundle not bound')
    return rows,manifest,approval,bound(acceptance_path)


def capacity_math(processing_actual,formal_processing_written,formal_checkpoint_written,free_bytes,checkpoint_envelope,
                  processing_envelope,other_remaining,legacy_failure_allowance,headroom=2_000_000_000):
    numbers=[processing_actual,formal_processing_written,formal_checkpoint_written,free_bytes,checkpoint_envelope,
             processing_envelope,other_remaining,legacy_failure_allowance,headroom]
    require(all(type(v) is int and v>=0 for v in numbers),'Invalid capacity values')
    require(formal_processing_written<=processing_actual,'Formal processing cannot exceed actual processing')
    require(formal_checkpoint_written<=checkpoint_envelope,'Formal checkpoint/failure envelope exhausted')
    require(formal_processing_written<=processing_envelope,'Formal log/diagnostic envelope exhausted')
    pending_checkpoints=checkpoint_envelope-formal_checkpoint_written
    pending_processing=processing_envelope-formal_processing_written+other_remaining
    processing_total=processing_actual+pending_processing
    pending=pending_checkpoints+pending_processing+legacy_failure_allowance+headroom
    return dict(processing_actual_plus_reserved_bytes=processing_total,processing_fits=processing_total<=20_000_000_000,
                formal_checkpoint_bytes_written=formal_checkpoint_written,formal_processing_bytes_written=formal_processing_written,
                pending_checkpoint_bytes=pending_checkpoints,pending_processing_bytes=pending_processing,
                pending_all_categories_plus_headroom=pending,free_bytes=free_bytes,disk_fits=pending<=free_bytes,
                fits=processing_total<=20_000_000_000 and pending<=free_bytes)


def capacity_snapshot(rows,approval):
    """Reserve all thirty runs once; subtract only bytes already in owned formal outputs."""
    processing=formal_processing=formal_checkpoints=0
    run_paths={Path(r['output']).resolve() for r in rows}
    for directory,folders,files in os.walk(ROOT,followlinks=False):
        base=Path(directory)
        require(not base.is_symlink() and not base.is_junction(),'Linked inventory directory')
        for folder in folders:
            p=base/folder;require(not p.is_symlink() and not p.is_junction(),'Linked inventory child')
        for name in files:
            p=base/name;require(not p.is_symlink(),'Linked inventory file')
            try:size=p.stat().st_size
            except FileNotFoundError:continue
            rel=p.relative_to(ROOT);parts=rel.parts
            if parts[0]=='envs' or parts[0]=='initialization':continue
            own=None
            if p.is_relative_to(RUNS):
                relative=p.relative_to(RUNS)
                require(len(relative.parts)>=2,'Unexpected formal root file')
                own=RUNS/relative.parts[0]
                require(own.resolve() in run_paths,'Unknown formal run directory')
                if len(relative.parts)>=3 and relative.parts[1] in ('checkpoints','failure-checkpoint'):
                    formal_checkpoints+=size;continue
            if parts[0]=='runs' and (name=='training.pt' or name=='training.pt.partial') and not parts[1].startswith('mechanism-'):
                continue
            processing+=size
            if own is not None or p.is_relative_to(DIAGNOSTICS):formal_processing+=size
    resources=approval['resources']
    result=capacity_math(processing,formal_processing,formal_checkpoints,shutil.disk_usage(ROOT).free,
                         resources['formal_checkpoint_envelope_bytes'],resources['formal_processing_envelope_bytes'],
                         resources['other_remaining_processing_bytes'],resources['legacy_failure_allowance_bytes'])
    result.update(observed_epoch=time.time(),actual_processing_bytes=processing,
                  note='All unfinished and finished matrix runs share one global envelope. Committed/partial owned formal bytes count as consumed reservation; old assets never deleted.')
    return result


def initial_reference(row):
    controller=(ROOT/'diagnostics/bf16-followup-v6r1-pro4000-20261031/CONTROLLER.json' if row['cohort']=='bf16'
                else ROOT/'diagnostics/initial-full-model-v5-v100a-20261031/CONTROLLER.json')
    state=read(controller);require(state['phase']=='complete_audited' and not state.get('error'),'Initial diagnostics incomplete')
    matches=[j for j in state['jobs'].values() if j['seed']==row['seed'] and j['method']==row['method'] and
             j.get('source','initial')=='initial' and j.get('kind','initial_full_model') in ('initial_full_model','full_model')]
    require(len(matches)==1,'Ambiguous initial model reference')
    job=matches[0];require(job['phase']=='complete','Incomplete initial reference')
    out=Path(job['output']);bound(out/'RESULT.json',job['result_sha256']);bound(out/'AUDIT.json',job['audit_sha256'])
    result=read(out/'RESULT.json');audit=read(out/'AUDIT.json')
    require(result['passed'] and audit['passed'] and audit['result_sha256']==job['result_sha256'],'Initial reference audit differs')
    require(result['release_sha256']==RELEASE_HASH and result['seed']==row['seed'] and result['method']==row['method'],'Initial reference release differs')
    return result['initialization'],bound(out/'RESULT.json')


def initialization_checks(row,identity,parameters,placement,initial,common_metadata,acceptance_hash):
    require(identity['stage']=='formal' and identity['seed']==row['seed'] and identity['method']==row['method'],'Wrong actual formal initialization')
    require(identity['gpu']==row['gpu_uuid'] and identity['acceptance_sha256']==acceptance_hash,'Wrong GPU or acceptance')
    require(identity['target_tokens']==32768000 and identity['data_sha256']==row['data']['release']['sha256'],'Wrong targets or data')
    require(identity['model_manifest_sha256']==row['model']['manifest']['sha256'] and identity['config_sha256']==row['release']['config_sha256'],'Model/config differs')
    require(parameters['method']==row['method'] and parameters['adapter']['seed']==row['seed'],'Actual adapter seed differs')
    require(parameters['layers']==list(range(21,28)) and parameters['parameters']==row['architecture']['parameters'],'Layer/parameter budget differs')
    require(placement['exact'] and placement['before']==placement['after']==initial['router_tensor_hash'],'Actual initial router values differ')
    expected_precision='torch.bfloat16' if row['cohort']=='bf16' else 'torch.float16'
    require(initial['native_precision']==expected_precision and initial['zero_training_updates'],'Wrong initial precision/reference')
    require(identity['cuda_device']['visible_count']==1 and identity['cuda_device']['uuid'].removeprefix('GPU-')==row['gpu_uuid'].removeprefix('GPU-'),'Physical GPU differs')
    if row['method']=='dense':
        require(identity['common_sha256'] is None and parameters['projections']=={},'Dense did not use original FFNs')
    else:
        require(identity['common_sha256']==row['initialization']['common']['sha256']==common_metadata['checkpoint_sha256'],'Common weights differ')
        require(parameters['projections']==common_metadata['projections'],'Actual common projection metadata differs')
    return dict(passed=True,observed_router_tensor_hash=placement['after'],reference_native_state_tree_digest=initial['native_state_tree_digest'],
                full_training_start_tensor_hash_directly_observed=False,
                nonrouter_binding='Frozen loader verifies common payload SHA256, exact parameter inventory and tensor state_hash, then copies into the actual model. Runtime identity and projection witnesses match this per-seed payload.',
                scope='Actual runtime configuration/common-loader provenance and directly observed FP32 router hash; reference full-state hash is from the independently audited initial diagnostic, not a second in-worker full tensor read.')


def audit_initialization(row,acceptance_hash):
    output=Path(row['output']);release=read(RELEASE)
    files={name:read(output/name) for name in ('RUN_CONFIG.json','PARAMETERS.json','ROUTER_PLACEMENT.json')}
    identity=files['RUN_CONFIG.json'];require(identity['code']==release['code'],'Actual source identity differs')
    initial,reference=initial_reference(row)
    common=row['initialization']['common'];metadata=None
    if common:
        bound(common['path'],common['sha256']);metadata=read(Path(common['path']).parent/'COMMON_INITIALIZATION.json')
    result=initialization_checks(row,identity,files['PARAMETERS.json'],files['ROUTER_PLACEMENT.json'],initial,metadata,acceptance_hash)
    return dict(**result,seed=row['seed'],method=row['method'],cohort=row['cohort'],reference=reference,
                observed_files={n:bound(output/n) for n in files},observed_epoch=time.time())


def update_checks(updates,samples,target=32768000):
    require(len(updates)==len(samples)==target//32768,'Incomplete update/sample count')
    changing=0;zero=[];overflow=0
    for i,(u,s) in enumerate(zip(updates,samples),1):
        require(u['tokens']==s['tokens']==i*32768 and u['update_tokens']==32768,'Nonmonotonic token stream')
        require(sum(v['valid_targets'] for v in s['samples'])==32768,'Effective tokens differ')
        require(u['optimizer_update']['updated'] is True and math.isfinite(u['ce']) and math.isfinite(u['grad_norm']),'Invalid optimizer or loss')
        deltas=[]
        for g in u['optimizer_update']['groups'].values():
            require(all(math.isfinite(g[k]) and g[k]>=0 for k in ('unscaled_grad_norm','model_update_norm')),'Nonfinite group statistic')
            deltas.append(g['model_update_norm'])
        require(deltas,'No trainable parameter groups')
        if any(v>0 for v in deltas):changing+=1
        else:zero.append(i)
        if i==1:require(all(v>0 for v in deltas),'First update did not jointly change all groups')
        if i==len(updates):require(all(v==0 for v in deltas),'Frozen terminal zero-LR behavior differs')
        overflow+=u['overflow_retries']
    return dict(optimizer_updates=len(updates),parameter_changing_updates=changing,zero_change_updates=zero,overflow_retries=overflow,
                matches_expected_999_parameter_changes=changing==999,
                interpretation='Additional zero changes at nonzero LR are retained as precision/convergence findings, never silently removed.')


def audit_training(row,acceptance_hash):
    output=Path(row['output']);result=read(output/'RESULT.json');identity=read(output/'RUN_CONFIG.json')
    require(result['identity']==identity and result['phase']=='complete' and result['start_tokens']==0,'Incomplete or resumed formal run')
    require(result['tokens']==result['target']==32768000,'Wrong final budget')
    require(result['precision']==('torch.bfloat16' if row['cohort']=='bf16' else 'torch.float16'),'Wrong actual precision')
    start=audit_initialization(row,acceptance_hash)
    updates=[json.loads(line) for line in (output/'updates.jsonl').read_text().splitlines()]
    samples=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    expected_groups={'residual_tail_or_dense'} if row['method']=='dense' else {'experts','residual_tail_or_dense','router'}
    require(all(set(u['optimizer_update']['groups'])==expected_groups for u in updates),'Missing or unexpected trainable parameter groups')
    counts=update_checks(updates,samples)
    require(sha(output/'samples.jsonl')==result['samples_sha256']==row['data']['formal_samples_windows_crlf_sha256'],'Full sample log differs')
    require(updates[-1]['cursor']==row['data']['expected_final_cursor'],'Final data cursor differs')
    dev=[]
    for token in row['evaluation']['save_and_dev_tokens']:
        path=output/f'dev-{token:010d}.json';d=read(path)
        require(d['valid_targets']==1048576 and d['sample_hash']==row['evaluation']['expected_dev_sample_hash'] and math.isfinite(d['ce']),'Dev data/result differs')
        dev.append(dict(tokens=token,**d,evidence=bound(path)))
    require(result['internal_dev']==read(output/f'dev-{32768000:010d}.json'),'Final dev result differs')
    checkpoint_root=output/'checkpoints';owner=read(checkpoint_root/'RETENTION_OWNER.json');index=read(checkpoint_root/'RETENTION_INDEX.json')
    require(owner['identity']==identity and index['owner_nonce']==owner['nonce'],'Retention ownership differs')
    from fitmotn.experiments.qwen_joint.retention import Retention,FORMAT
    Retention(checkpoint_root,owner['nonce'])  # Read-only validation of absolute ownership and identity hash.
    entries=index['entries'];expected_tokens=row['evaluation']['save_and_dev_tokens']
    require(sorted(e['tokens'] for e in entries.values())==expected_tokens,'Retention inventory missing checkpoint events')
    best=min(dev,key=lambda d:(d['ce'],d['tokens']))['tokens']
    keep={3276800,16384000,32768000,best};active={e['tokens'] for e in entries.values() if e['status']=='active'}
    require(active==keep and len(active)<=4,'Protected checkpoint policy differs')
    require({p.name for p in checkpoint_root.iterdir()}=={'RETENTION_OWNER.json','RETENTION_INDEX.json'}|{n for n,e in entries.items() if e['status']=='active'},'Unexpected retained checkpoint artifacts')
    dev_scores={d['tokens']:d['ce'] for d in dev}
    retained=[]
    for name,e in entries.items():
        path=checkpoint_root/name
        require(path.resolve().parent==checkpoint_root.resolve(),'Checkpoint path escaped')
        require(e['dev_ce']==dev_scores[e['tokens']],'Registered dev score differs')
        if e['status']=='pruned':require(not path.exists(),'Pruned payload still present');continue
        require(e['status']=='active','Interrupted retention must be reviewed')
        marker=read(path/'complete.json');require(marker['format']==FORMAT and marker['identity']==identity and marker['tokens']==e['tokens'],'Checkpoint identity differs')
        require(set(marker['files'])=={'training.pt'},'Unexpected checkpoint inventory')
        require({p.name for p in path.iterdir()}=={'training.pt','complete.json'},'Unexpected checkpoint member')
        marker_binding=bound(path/'complete.json')
        for filename,spec in marker['files'].items():
            actual=bound(path/filename,spec['sha256']);require(actual['bytes']==spec['bytes'],'Payload size differs')
        require(e['files']=={n:{k:v[k] for k in ('sha256','bytes')} for n,v in [('training.pt',actual),('complete.json',marker_binding)]},'Registered checkpoint evidence changed')
        retained.append(dict(tokens=e['tokens'],checkpoint=str(path),marker=marker_binding,payload=actual))
    require(Path(result['checkpoint']).resolve()==(checkpoint_root/'checkpoint-0032768000').resolve(),'Final checkpoint path differs')
    execution=read(output/'SOURCE_EXECUTION.json');release=read(RELEASE)
    require(execution['mode']=='promoted_checkout' and execution['release_sha256']==RELEASE_HASH and execution['wrapper_sha256']==row['release']['entry']['sha256'],'Entry attestation differs')
    canonical={n.replace('\\','/'):h for n,h in release['code'].items()}
    for module,spec in execution['modules'].items():
        require(spec['sha256']==canonical['experiments/qwen_joint/'+module.rsplit('.',1)[-1]+'.py'],'Executed module changed')
    return dict(passed=True,phase='training_complete_audited',name=row['name'],seed=row['seed'],method=row['method'],cohort=row['cohort'],
                result=bound(output/'RESULT.json'),source_execution=bound(output/'SOURCE_EXECUTION.json'),initialization=start,counts=counts,
                samples_sha256=result['samples_sha256'],internal_dev=dev,retained=retained,best_dev_tokens=best,primary_tokens=32768000,
                diagnostics_required=True,reserved_test_accessed=False)
