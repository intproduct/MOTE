"""Evaluation-only trained-checkpoint mechanism probes, with owned GPU lease."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path('E:/AI/MOTE-quantum')
REF=ROOT/'diagnostics/mechanism-v1-20261031'


def main(args):
    # Device masking precedes torch and legacy model imports.
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from fitmotn.experiments.qwen_joint.acceptance import source_identity
    from fitmotn.experiments.qwen_joint.resources import GPULease,PRO4000,V100
    from fitmotn.experiments.qwen_joint.state import write_json
    from fitmotn.experiments.qwen_joint.checkpoint import decode
    from checkpoint_mechanism_v4 import read,digest,load_checkpoint
    from production_resource_preflight_v3 import check as resource_check
    out=Path(args.output)
    if not out.resolve().is_relative_to((ROOT/'diagnostics').resolve()):raise ValueError('Output outside diagnostic root')
    if args.gpu not in (PRO4000,*V100):raise ValueError('Unknown physical GPU')
    out.mkdir(parents=True,exist_ok=False)
    def status(phase,**details):
        write_json(out/'STATUS.json',dict(phase=phase,pid=os.getpid(),gpu=args.gpu,time=time.time(),**details))
    started=time.perf_counter()
    try:
        status('verifying_checkpoint')
        release=read(ROOT/'releases/integration-v3.json');assert source_identity()==release['code']
        marker,packed,source_mode=load_checkpoint(args.checkpoint,release);identity=marker['identity']
        native=torch.bfloat16 if args.gpu==PRO4000 else torch.float16
        original_native=torch.bfloat16 if identity['gpu']==PRO4000 else torch.float16
        if native!=original_native:raise ValueError('Checkpoint precision cohort differs from diagnostic device')
        dtype_names={value.dtype for name,value in packed['model'].items() if '.router.' not in name}
        assert dtype_names=={native}
        budget=resource_check();reserve=50_000_000
        assert budget['acceptance_launch_fits'] and budget['processing_actual_plus_reserved_bytes']+reserve<20_000_000_000
        budget.update(additional_diagnostic_reservation_bytes=reserve);write_json(out/'RESERVATION.json',budget)
        protocol=read(REF/'MECHANISM_PROTOCOL.json')
        assert digest(REF/'fixed_windows.npz')==protocol['arrays_sha256']
        assert protocol['model_manifest_sha256']==identity['model_manifest_sha256'] and protocol['data_release_sha256']==identity['data_sha256']
        arrays=np.load(REF/'fixed_windows.npz',allow_pickle=False)
        labels=np.concatenate([arrays[f'labels_{i}'] for i in range(2)]);valid=labels!=-100
        assert int(valid.sum())==2046
        seeds=read(ROOT/'initialization/seed-release-v2-20261031/SEED_RELEASE.json')
        initial=next(row for row in seeds['initializations'] if row['seed']==identity['seed'])
        assert digest(initial['common_path'])==initial['checkpoint_sha256']
        assert identity['common_sha256']==(None if identity['method']=='dense' else initial['checkpoint_sha256'])
        common=torch.load(initial['common_path'],map_location='cpu',weights_only=False,mmap=True)
        restored=decode(packed);del packed
        if any(p.exists() for p in (ROOT/'STOP',ROOT/'queue-20261030/STOP',out/'STOP')):raise RuntimeError('STOP marker')
        with GPULease(ROOT,args.gpu,'trained_checkpoint_mechanism_v4',out) as lease:
            from transformers import AutoModelForCausalLM
            from fitmotn.experiments.qwen_joint.model import AdapterConfig,patch_qwen,set_model_precision,tensor_hash
            from fitmotn.experiments.qwen_joint.reproducibility import configure
            from fitmotn.experiments.qwen_joint.train import make_guard
            from mechanism_metrics_candidate_v1 import switch_risk
            from risk_probe_v4 import aligned_probe
            configure();torch.set_num_threads(8)
            guard=make_guard(SimpleNamespace(root=ROOT,gpu=args.gpu),out)
            def check():guard.check('diagnostic')
            check();status('loading_model')
            manifest=read(ROOT/'queue-20261030/model-manifest.json')
            model=AutoModelForCausalLM.from_pretrained(manifest['path'],local_files_only=True,
                trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa')
            patch_qwen(model,identity['method'],AdapterConfig(**identity['adapter']),common=common if identity['method']!='dense' else None)
            model.load_state_dict(restored['model'],strict=True)
            model.requires_grad_(False).eval();model.config.use_cache=False
            original_f32={key:value.float() for key,value in restored['model'].items()}
            assert tensor_hash(model.state_dict())==tensor_hash(original_f32)
            baseline_hash=tensor_hash(model.state_dict())
            del original_f32,restored,common;gc.collect()
            set_model_precision(model,torch.float32,'cuda')
            router_before=tensor_hash({key:value for key,value in model.state_dict().items() if '.router.' in key})
            variants=[('fp32_reference',None,False),('fp32_repeat',None,False),('batch_two',None,False),
                ('input_float16_roundtrip',None,False),('input_bfloat16_roundtrip',None,False)]
            if identity['method']!='dense':
                for name in protocol['controlled_replacement']['projections']:
                    for farthest in (False,True):
                        variants.append((f'replace_layer{name.split(".")[2]}_'+('farthest' if farthest else 'nearest'),name,farthest))
            variants.append(('native_bfloat16' if native==torch.bfloat16 else 'native_float16',None,False))
            rows=[];reference=None
            for variant,replacement,farthest in variants:
                check();status('evaluating_variant',variant=variant,method=identity['method'],checkpoint_tokens=marker['tokens'])
                if variant.startswith('native_'):set_model_precision(model,native,'cuda')
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
                raw,records=aligned_probe(model,arrays,protocol['reference_activations']['projections'],variant,check,replacement,farthest)
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                path=out/(variant+'.npz')
                with path.open('xb') as file:np.savez(file,logprobs=raw['logprobs'],topk=raw['topk'],labels=labels,valid=valid)
                row=dict(variant=variant,valid_targets=int(valid.sum()),ce=float(-raw['logprobs'][valid].astype(np.float64).mean()),
                    elapsed_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),raw_path=str(path),raw_sha256=digest(path),raw_bytes=path.stat().st_size,
                    router_names=raw['router_names'],routing_applicable=identity['method']!='dense',
                    native_precision=str(native) if variant.startswith('native_') else 'float32',
                    input_quantization_only=variant.startswith('input_'),isolated_timing=False)
                if variant=='fp32_reference':reference=raw
                else:
                    row['risk']=switch_risk(reference['logprobs'],raw['logprobs'],reference['topk'],raw['topk'],valid)
                    row['mean_nll_change']=float((reference['logprobs'][valid].astype(np.float64)-raw['logprobs'][valid]).mean())
                    if variant=='fp32_repeat':
                        assert np.array_equal(raw['logprobs'],reference['logprobs']) and np.array_equal(raw['topk'],reference['topk'])
                        row['exact_repeat']=True
                if records:
                    assert len(records)==2
                    positions=np.concatenate([arrays[f'positions_{i}']+i*1024 for i in range(2)])
                    row['affected_position_count']=len(positions)
                    row['affected_position_mean_nll_change']=float((reference['logprobs'][positions].astype(np.float64)-raw['logprobs'][positions]).mean())
                    details=out/(variant+'-intervention.pt')
                    with details.open('xb') as file:torch.save(records,file)
                    row['intervention_path']=str(details);row['intervention_sha256']=digest(details);row['intervention_bytes']=details.stat().st_size
                assert router_before==tensor_hash({key:value for key,value in model.state_dict().items() if '.router.' in key})
                rows.append(row);write_json(out/(variant+'.json'),row)
            set_model_precision(model,torch.float32,'cuda')
            assert baseline_hash==tensor_hash(model.state_dict()),'Diagnostic changed checkpoint values'
            assert source_identity()==release['code']
            payload=dict(passed=True,phase='complete',method=identity['method'],seed=identity['seed'],checkpoint_tokens=marker['tokens'],
                checkpoint=str(Path(args.checkpoint)),checkpoint_payload_sha256=marker['files']['training.pt']['sha256'],
                native_precision=str(native),physical_device=lease.record['cuda_device'],lease=lease.record,
                variants=rows,reference='stored native checkpoint values converted to FP32, including backbone; not FP32-trained weights',
                protocol_sha256=digest(REF/'MECHANISM_PROTOCOL.json'),arrays_sha256=protocol['arrays_sha256'],
                source_identity_mode=source_mode,source_code=release['code'],release_sha256=digest(ROOT/'releases/integration-v3.json'),
                helper_sha256={name:digest(Path(__file__).parent/name) for name in ['full_checkpoint_diagnostics_v4.py','risk_probe_v4.py','checkpoint_mechanism_v4.py','mechanism_metrics_candidate_v1.py']},
                elapsed_seconds=time.perf_counter()-started,isolated_timing=False,formal_approved=False,
                interpretation='Precision changes routing and other arithmetic. Switch-associated loss is not a routing-only causal effect.',
                native_bfloat16='measured' if native==torch.bfloat16 else 'unsupported_on_sm70')
            write_json(out/'RESULT.json',payload)
            actual=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert actual<reserve
            status('complete',result=str(out/'RESULT.json'),bytes=actual)
    except BaseException:
        write_json(out/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time(),pid=os.getpid()))
        status('failed_preserved');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--gpu',required=True)
    parser.add_argument('--output',required=True);main(parser.parse_args())
