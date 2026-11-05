"""Run the committed canonical release, with actual module-byte attestations."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path('E:/AI/MOTE-quantum')
assert Path.cwd().resolve()==Path('D:/AI/MOTE').resolve()
gpu=sys.argv[sys.argv.index('--gpu')+1]
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from fitmotn.experiments.qwen_joint import acceptance,state
release_path=ROOT/'releases/integration-v3.json'
release=json.loads(release_path.read_text())
assert acceptance.source_identity()==release['code'],'Checkout differs from committed release'
canonical={name.replace('\\','/'):sha for name,sha in release['code'].items()}
assert len(canonical)==len(release['code'])
output=Path(sys.argv[sys.argv.index('--output')+1]);existed=output.exists()
try:
    from fitmotn.experiments.qwen_joint import train
    train.main()
    modules={name:dict(path=str(Path(module.__file__).resolve()),sha256=state.digest(module.__file__))
        for name,module in sys.modules.items()
        if name.startswith('fitmotn.experiments.qwen_joint.') and getattr(module,'__file__',None)}
    for name,item in modules.items():
        relative='experiments/qwen_joint/'+name.rsplit('.',1)[-1]+'.py'
        assert item['sha256']==canonical[relative],name
    state.write_json(output/'SOURCE_EXECUTION.json',dict(mode='promoted_checkout',modules=modules,
        working_directory=str(Path.cwd()),release_sha256=state.digest(release_path),
        staged_release_sha256=release['staged_release_sha256'],
        source_canonicalization_sha256=release['source_canonicalization_sha256'],
        wrapper_sha256=state.digest(__file__),formal_approved=False,time=time.time()))
except BaseException:
    if not existed and output.is_dir():
        state.write_json(output/'ENTRY_FAILURE.json',dict(error=traceback.format_exc(),pid=os.getpid(),time=time.time()))
    raise
