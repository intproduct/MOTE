"""Deterministic execution and complete recovery fingerprints."""
import os
import hashlib
import numpy as np
import torch
from .state import rng_state

def tree_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(b'T' + str(x.dtype).encode() + str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b'N' + str(x.dtype).encode() + str(x.shape).encode() + x.tobytes())
        elif isinstance(x, dict):
            h.update(b'D')
            for k in sorted(x, key=repr):
                visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x: visit(v)
        else:
            h.update(type(x).__name__.encode() + repr(x).encode() + b'\0')
    visit(value)
    return h.hexdigest()


PROFILE = dict(deterministic_algorithms=True, cublas_workspace_config=':4096:8',
               tf32=False, cpu_threads=8, attention='sdpa', microbatch=1)

def configure():
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Configure deterministic cuBLAS before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)
    return dict(PROFILE)

def state_hashes(model, optimizer, schedule, scaler, stream):
    return dict(model=tree_digest(model.state_dict()), optimizer=tree_digest(optimizer.state_dict()),
                schedule=tree_digest(schedule.state_dict()), scaler=tree_digest(scaler.state_dict()),
                rng=tree_digest(rng_state()), cursor=tree_digest(stream.state_dict()))
