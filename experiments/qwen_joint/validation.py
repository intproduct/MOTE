"""Fail-closed formal release checks, independent of GPU initialization."""
from pathlib import Path
import json
import shutil
from .state import digest

METHODS = ('mlp', 'affine_born', 'reencode_cz', 'reencode_product', 'dense')

def formal_budget(acceptance, code, data_hash, model_hash, config_hash, method, seed, gpu, common_hash, root):
    if acceptance.get('schema') != 'qwen-integration-acceptance-v3' or acceptance.get('formal_approved') is not True:
        raise ValueError('Integration/pilot/resource acceptance is incomplete')
    required = dict(code=code, data_sha256=data_hash, model_manifest_sha256=model_hash, config_sha256=config_hash)
    if any(acceptance.get(k) != v for k,v in required.items()):
        raise ValueError('Formal release differs from accepted source/model/data/configuration')
    if acceptance.get('methods') != list(METHODS) or acceptance.get('seeds') != [17,29,43]:
        raise ValueError('The complete method/seed matrix must be accepted')
    expected_common = acceptance['common_sha256_by_seed'][str(seed)]
    if method != 'dense' and common_hash != expected_common:
        raise ValueError('Formal common initialization differs from acceptance')
    if gpu not in acceptance['accepted_gpu_uuids']:
        raise ValueError('GPU has not passed this frozen release')
    target = acceptance['formal_tokens']
    if type(target) is not int or target <= 0 or target % 327680:
        raise ValueError('Formal target must have exact update-aligned deciles')
    resources = acceptance['resources']
    if resources.get('approved') is not True or resources['new_data_cache_reserved_total_bytes'] > 20000000000:
        raise ValueError('Resource reservation is incomplete or over budget')
    if resources['maximum_new_run_peak_bytes'] <= 0 or shutil.disk_usage(root).free < resources['maximum_new_run_peak_bytes']:
        raise ValueError('Insufficient reserved peak capacity for one new run')
    for evidence in acceptance['required_evidence']:
        path = Path(evidence['path'])
        if not path.is_file() or digest(path) != evidence['sha256']:
            raise ValueError('Required acceptance evidence changed')
    if not acceptance['required_evidence']:
        raise ValueError('Acceptance evidence inventory is empty')
    return target
