"""CPU-only input-roundtrip acceptance on ten retained first-update states."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('E:/AI/MOTE-quantum')
OUT = ROOT / 'diagnostics/local-checkpoint-precision-v4-cpu-20261031'
sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_mechanism_v4 import read, digest
from fitmotn.experiments.qwen_joint.state import write_json
from production_resource_preflight_v3 import check
import local_checkpoint_precision_v4

manifest = read(Path(__file__).parent / 'MANIFEST.json')
assert all(digest(Path(__file__).parent / name) == sha for name, sha in manifest['files'].items())
assert Path.cwd().resolve() == Path('D:/AI/MOTE').resolve()
OUT.mkdir(exist_ok=False)
nonce = str(uuid.uuid4()); lock = ROOT / 'locks/local-checkpoint-precision-v4-cpu.lock'
with lock.open('x', encoding='utf8') as file:
    json.dump(dict(pid=os.getpid(), nonce=nonce, gpu_used=False, run_directory=str(OUT)), file)
state = dict(phase='running', pid=os.getpid(), gpu_used=False, jobs={}, formal_approved=False,
    frozen_manifest=str(Path(__file__).parent / 'MANIFEST.json'), frozen_manifest_sha256=digest(Path(__file__).parent / 'MANIFEST.json'))
def save():
    state['time'] = time.time(); write_json(OUT / 'CONTROLLER.json', state)
try:
    budget = check(); reserve = 300_000_000
    assert budget['acceptance_launch_fits'] and budget['processing_actual_plus_reserved_bytes'] + reserve <= 20_000_000_000
    assert budget['free_bytes'] > budget['pending_all_categories_bytes'] + reserve + 2_000_000_000
    budget['additional_diagnostic_stage_reservation_bytes'] = reserve
    write_json(OUT / 'RESERVATION.json', budget)
    for cohort in ('pro4000','v100b'):
        audit_path = ROOT / 'runtime' / f'production-replay-artifact-audit-v3-{cohort}.json'
        audit = read(audit_path); assert audit['passed'] and len(audit['rows']) == 5
        for row in audit['rows']:
            checkpoint = Path(row['checkpoint_files'][0]['path']).parent
            name = cohort + '-' + row['method']; output = OUT / name
            state['jobs'][name] = dict(phase='running', checkpoint=str(checkpoint), output=str(output),
                audit_sha256=digest(audit_path)); save()
            local_checkpoint_precision_v4.run(SimpleNamespace(checkpoint=str(checkpoint), output=str(output), gpu=None))
            state['jobs'][name].update(phase='complete', result_sha256=digest(output / 'RESULT.json')); save(); gc.collect()
    state['phase'] = 'complete'; save()
except BaseException:
    state.update(phase='failed_preserved', error=traceback.format_exc()); save()
    write_json(OUT / 'FAILURE.json', state); raise
finally:
    if lock.exists() and read(lock).get('nonce') == nonce: lock.unlink()
