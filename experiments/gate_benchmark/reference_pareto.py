"""Copy audited historical controls with explicit provenance, never relabel training."""
import hashlib
import json
from pathlib import Path
import shutil
from .config import load_config, digest
from .runner import write_json

NON_PROTOCOL_KEYS={'output_dir','gates','variants','study_protocol','pilot_selection_sha256'}


def protocol(cfg):
    return {k:v for k,v in cfg.items() if k not in NON_PROTOCOL_KEYS}


def import_controls(formal_config='experiments/gate_benchmark/configs/pareto_formal.json'):
    c=load_config(formal_config)
    root=Path(c['output_dir'])/digest(c)[:12]
    old=Path('outputs/gate_benchmark/45c21049b433')
    old_cfg=json.loads((old/'config.json').read_text())
    assert protocol(c)==protocol(old_cfg),'historical training/evaluation protocol differs'
    previous_audit=json.loads((old/'audit.json').read_text())
    assert previous_audit['passed'] and previous_audit['completed_audited']==54
    repeatability=json.loads(Path('outputs/gate_benchmark_round2/baseline_repeatability.json').read_text())
    assert repeatability['passed'] and len(repeatability['checks'])==4
    root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():assert json.loads((root/'config.json').read_text())==c
    else:write_json(root/'config.json',c)
    imported=[]
    for stage in ['mlp','bert']:
        for task in c[stage]['tasks']:
            for seed in c['seeds']:
                for gate in ['mlp','compact_born']:
                    name=f'{stage}_{task}_{gate}_seed{seed}'
                    source,dest=old/name,root/name
                    reference=dict(kind='audited_historical_control',source_run=str(source.resolve()),
                        source_group=str(old.resolve()),reason='same protocol; four repeated pilot trajectories exactly match',
                        protocol_sha256=digest(protocol(c)),files_sha256={})
                    files=['manifest.json','initial_dev.json','curves.jsonl','best.pt','result.json']
                    for filename in files:
                        reference['files_sha256'][filename]=hashlib.sha256((source/filename).read_bytes()).hexdigest()
                    if dest.exists():
                        assert (dest/'reference.json').exists(),f'incomplete reference retained: {dest}'
                        assert json.loads((dest/'reference.json').read_text())==reference
                        assert all(hashlib.sha256((dest/f).read_bytes()).hexdigest()==h for f,h in reference['files_sha256'].items())
                    else:
                        dest.mkdir()
                        # result.json is copied last; incomplete imports cannot look complete.
                        for filename in files[:-1]:shutil.copy2(source/filename,dest/filename)
                        write_json(dest/'reference.json',reference)
                        shutil.copy2(source/'result.json',dest/'result.json')
                    imported.append(name)
    write_json(root/'historical_controls.json',dict(imported_runs=imported,count=len(imported),
                                                   newly_trained=False,protocol_verified=True))
    print(json.dumps(dict(reused=len(imported),new_runs=90,analysis_matrix=126,group=str(root)),indent=2))


if __name__=='__main__':import_controls()
