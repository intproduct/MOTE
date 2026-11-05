"""Independently audit retained timing samples; no model/CUDA imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as file: return hashlib.file_digest(file,'sha256').hexdigest()


def audit(output):
    output=Path(output); result=read(output/'RESULT.json')
    assert result['passed'] and result['phase']=='complete'
    assert result['warmup_repeats']==2 and result['measured_repeats']==10
    assert result['batch_size']==1 and result['windows_per_repeat']==2 and result['sequence_length']==1024
    assert result['valid_targets_per_repeat']==2046
    assert result['native_precision'] in ('torch.float16','torch.bfloat16')
    assert result['router_precision']=='torch.float32' and result['host_exclusive'] is False
    assert result['selected_device_no_foreign_compute_at_observations']
    assert result['kv_cache_policy'].startswith('disabled;')
    checkpoint=Path(result['checkpoint']); marker=read(checkpoint/'complete.json')
    assert sha(checkpoint/'complete.json')==result['checkpoint_marker_sha256']
    assert sha(checkpoint/'training.pt')==result['checkpoint_payload_sha256']
    assert marker['files']['training.pt']['sha256']==result['checkpoint_payload_sha256']
    assert marker['tokens']==result['checkpoint_tokens']
    for key in ('method','seed'): assert marker['identity'][key]==result[key]
    rows=result['rows']; assert len(rows)==12
    evidence={}; seconds=[]
    for offset,row in enumerate(rows):
        phase='warmup' if offset<2 else 'measured'; index=offset if offset<2 else offset-2
        assert row['phase']==phase and row['index']==index
        assert math.isfinite(row['seconds']) and row['seconds']>0
        path=output/(phase+'-'+str(index)+'.json')
        assert read(path)==row; evidence[str(path)]=sha(path)
        for key in ('observation_before','observation_after'):
            observation=row[key]; snap=observation['snapshot']
            assert observation['foreign_compute_observed'] is False
            assert snap['uuid']==result['lease']['uuid']
            assert all(app[0]==snap['uuid'] and app[1]==str(result['lease']['pid']) for app in snap['compute_apps'])
            assert math.isfinite(snap['epoch'])
        assert row['observation_after']['snapshot']['epoch']>=row['observation_before']['snapshot']['epoch']
        assert row['observation_after']['peak_reserved_bytes']>=row['observation_after']['peak_allocated_bytes']>0
        if phase=='measured': seconds.append(row['seconds'])
    expected=dict(repeats=10,input_tokens_per_repeat=2048,total_input_tokens=20480,
        total_measured_seconds=math.fsum(seconds),mean_seconds=statistics.mean(seconds),
        median_seconds=statistics.median(seconds),min_seconds=min(seconds),max_seconds=max(seconds),
        input_tokens_per_second=20480/math.fsum(seconds))
    assert expected==result['summary']
    assert len(result['logit_validation'])==2
    assert all(row['shape']==[1,1024,151936] and row['dtype']==result['native_precision'] for row in result['logit_validation'])
    return dict(passed=True,result=str(output/'RESULT.json'),result_sha256=sha(output/'RESULT.json'),
        method=result['method'],seed=result['seed'],checkpoint_tokens=result['checkpoint_tokens'],
        native_precision=result['native_precision'],summary=expected,raw_evidence=evidence,
        auditor_sha256=sha(__file__),host_exclusive=False,formal_approved=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    value=audit(args.output); path=Path(args.output)/'AUDIT.json'
    with path.open('x',encoding='utf8') as file: json.dump(value,file,indent=2)
    print(json.dumps(dict(method=value['method'],precision=value['native_precision'],passed=True,summary=value['summary'])))
