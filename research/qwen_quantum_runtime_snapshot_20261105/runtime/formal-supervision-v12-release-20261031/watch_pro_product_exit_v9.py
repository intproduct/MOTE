"""Observe two already-running Windows processes using held kernel handles.

No GPU use, no signals, no training launches, no edits to the original run.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid

ROOT = Path('E:/AI/MOTE-quantum')
OUT = ROOT/'runtime/pro-product-exit-v9-20261031'
GPU = 'GPU-67f0c332-3f01-a733-4651-5532e94fc051'
RUN = ROOT/'runs/production-acceptance-v3-pro4000-20261031/pilot-reencode_product'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value, replace=os.replace, attempts=100, delay=.05):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    for attempt in range(attempts):
        try:
            replace(temporary, path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise  # Keep the complete unique temporary as failure evidence.
            time.sleep(delay)


class HeldProcess:
    def __init__(self, pid):
        import psutil
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.api.GetExitCodeProcess.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        before = psutil.Process(pid)
        created = before.create_time()
        self.handle = self.api.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.record = dict(pid=pid, create_time=created, command=before.cmdline())
        if psutil.Process(pid).create_time() != created or self.poll() is not None:
            self.close()
            raise RuntimeError('PID identity changed or exited before observation')

    def poll(self):
        result = self.api.WaitForSingleObject(self.handle, 0)
        if result == 258:
            return None
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.api.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(code.value)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle); self.handle = None


def validate_command(words):
    from pathlib import PureWindowsPath
    names = [PureWindowsPath(w).name for w in words]
    if 'production_v3_entry_r1.py' not in names or 'pilot' not in words:
        raise ValueError('Unexpected worker entry or stage')
    for flag, expected in [('--method', 'reencode_product'), ('--gpu', GPU), ('--seed', '17')]:
        if words.count(flag) != 1 or words[words.index(flag)+1] != expected:
            raise ValueError('Wrong ' + flag)
    if words.count('--output') != 1 or Path(words[words.index('--output')+1]).resolve() != RUN.resolve():
        raise ValueError('Unexpected worker output')


def main():
    assert Path.cwd().resolve() == Path('D:/AI/MOTE').resolve()
    assert not OUT.exists()
    OUT.mkdir()
    held = []
    try:
        for pid in (77160, 78436):
            process = HeldProcess(pid); held.append(process)
            validate_command(process.record['command'])
        state = dict(phase='observing_existing_workers', pid=os.getpid(), watcher_sha256=sha(__file__),
                     workers=[p.record for p in held], started_epoch=time.time(), no_gpu_use=True,
                     no_training_launch=True, run_directory=str(RUN))
        atomic_json(OUT/'ATTACHED.json', state)
        while True:
            exits = [p.poll() for p in held]
            if all(code is not None for code in exits):
                state.update(phase='exited', finished_epoch=time.time(), exit_codes=exits,
                             passed=all(code == 0 for code in exits))
                atomic_json(OUT/'RESULT.json', state)
                break
            if time.time() - state['started_epoch'] > 4*3600:
                raise RuntimeError('Observation exceeded four hours; no process was signalled')
            time.sleep(2)
    except BaseException:
        atomic_json(OUT/'FAILURE.json', dict(error=traceback.format_exc(), time=time.time(), pid=os.getpid()))
        raise
    finally:
        for p in held:
            p.close()


if __name__ == '__main__':
    main()
