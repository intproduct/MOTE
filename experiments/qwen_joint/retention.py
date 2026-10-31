"""Owned-new-run checkpoint retention candidate; never adopts existing assets."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
from .state import digest,write_json

FORMAT='qwen-immutable-base-full-state-v1'
NAME=re.compile(r'checkpoint-[0-9]{10}\Z')

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def identity_hash(identity):return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Retention:
    @classmethod
    def create(cls, allowed_root, run, identity, target_tokens):
        allowed=Path(allowed_root).resolve(strict=True)
        path=Path(run).absolute()
        if path.parent.resolve(strict=True)!=allowed or path.exists() or path.is_symlink():
            raise ValueError('Only a fresh direct child of the explicit experiment root can be owned')
        if target_tokens<=0 or target_tokens%10:raise ValueError('Target must permit exact 10% and 50% checkpoints')
        path.mkdir(exist_ok=False)
        owner=dict(schema=1,nonce=str(uuid.uuid4()),run=str(path.resolve()),allowed_root=str(allowed),
            identity=identity,identity_hash=identity_hash(identity),target_tokens=target_tokens,created_epoch=time.time())
        write_json(path/'RETENTION_OWNER.json',owner)
        write_json(path/'RETENTION_INDEX.json',dict(owner_nonce=owner['nonce'],entries={},events=[]))
        return cls(path,owner['nonce'])

    def __init__(self, run, nonce):
        self.path=Path(run).resolve(strict=True)
        self.owner=read(self.path/'RETENTION_OWNER.json')
        if self.owner['nonce']!=nonce or self.owner['run']!=str(self.path) or self.path.parent!=Path(self.owner['allowed_root']).resolve(strict=True):
            raise ValueError('Run ownership mismatch')
        if identity_hash(self.owner['identity'])!=self.owner['identity_hash']:raise ValueError('Identity changed')
        self.index_path=self.path/'RETENTION_INDEX.json'

    @contextmanager
    def locked(self):
        lock=self.path/'RETENTION_MUTATION.lock'
        with lock.open('x',encoding='utf8') as f:
            json.dump(dict(pid=os.getpid(),nonce=self.owner['nonce']),f);f.flush();os.fsync(f.fileno())
        try:
            index=read(self.index_path)
            if index['owner_nonce']!=self.owner['nonce']:raise ValueError('Index owner mismatch')
            yield index
        finally:
            if lock.exists() and read(lock).get('nonce')==self.owner['nonce']:lock.unlink()

    def checkpoint_path(self,name):
        if not NAME.fullmatch(name):raise ValueError('Unexpected checkpoint name')
        path=self.path/name
        if path.is_symlink() or path.is_junction() or path.resolve().parent!=self.path:
            raise ValueError('Checkpoint escapes owned run')
        return path

    def verify(self,name,expected=None,partial=False):
        path=self.checkpoint_path(name)
        if not path.exists() and partial and expected is not None:return expected
        if not path.is_dir():raise ValueError('Missing checkpoint directory')
        names={p.name for p in path.iterdir()}
        required={'training.pt','complete.json'}
        if not partial and names!=required or partial and not names<=required:
            raise ValueError('Incomplete checkpoint or unrecognized evidence; preserve directory')
        if expected is None:
            manifest=read(path/'complete.json')
            if manifest.get('format')!=FORMAT or manifest['identity']!=self.owner['identity'] or set(manifest['files'])!={'training.pt'}:
                raise ValueError('Checkpoint format or identity mismatch')
            if manifest['tokens']!=int(name.removeprefix('checkpoint-')):raise ValueError('Token/name mismatch')
            expected={key:dict(sha256=digest(path/key),bytes=(path/key).stat().st_size) for key in names}
            if expected['training.pt']!=manifest['files']['training.pt']:raise ValueError('Payload checksum mismatch')
        for key in names:
            file=path/key
            if file.is_symlink() or file.resolve().parent!=path.resolve() or not file.is_file():raise ValueError('Unsafe checkpoint member')
            if file.stat().st_size!=expected[key]['bytes'] or digest(file)!=expected[key]['sha256']:
                raise ValueError('Registered checkpoint has changed')
        return expected

    def register(self,name,dev_ce):
        if not isinstance(dev_ce,(int,float)) or not math.isfinite(dev_ce):raise ValueError('Require finite frozen-dev score')
        with self.locked() as index:
            if name in index['entries']:raise ValueError('Never overwrite checkpoint registration')
            files=self.verify(name)
            token=int(name.removeprefix('checkpoint-'))
            previous=[e['tokens'] for e in index['entries'].values()]
            if token<=0 or token>self.owner['target_tokens'] or previous and token<=max(previous):raise ValueError('Nonmonotonic checkpoint')
            index['entries'][name]=dict(tokens=token,dev_ce=float(dev_ce),files=files,status='active')
            index['events'].append(dict(action='registered',name=name,time=time.time()))
            write_json(self.index_path,index)

    def protected(self,index):
        entries=index['entries']
        if not entries:return set()
        target=self.owner['target_tokens']
        keep={name for name,e in entries.items() if e['tokens'] in (target//10,target//2,target)}
        keep.add(max(entries,key=lambda name:entries[name]['tokens']))
        # Earliest checkpoint wins ties; this policy is frozen before scores arrive.
        keep.add(min(entries,key=lambda name:(entries[name]['dev_ce'],entries[name]['tokens'])))
        if any(entries[name]['status']!='active' for name in keep):raise ValueError('A protected recovery checkpoint is not active')
        return keep

    def prune(self,fault=None):
        """Delete only verified files of registered, dispensable checkpoints.

        No recursive deletion or cross-directory moves. Interrupted deletions are
        recorded before the first unlink; recovery only finishes that exact intent.
        """
        with self.locked() as index:
            if any((p/marker).exists() for p in (self.path,self.path.parent) for marker in ('FAILURE.json','TRAINING_FAILURE_STATE.json','STOP')):raise ValueError('Preserve failed/stopped run')
            keep=self.protected(index)
            # Recheck latest/best/milestones before touching any predecessor.
            for name in keep:self.verify(name,index['entries'][name]['files'])
            candidates=[name for name,e in index['entries'].items() if name not in keep and e['status'] in ('active','pruning')]
            for name in candidates:
                entry=index['entries'][name]
                self.verify(name,entry['files'],partial=entry['status']=='pruning')
            removed=[]
            for name in candidates:
                entry=index['entries'][name]
                if entry['status']=='active':
                    entry['status']='pruning'
                    index['events'].append(dict(action='deletion_intent',name=name,protected=sorted(keep),time=time.time()))
                    write_json(self.index_path,index)
                if fault=='after_intent':raise OSError('Injected interruption before unlink')
                path=self.checkpoint_path(name)
                # Check the resolved absolute directory is still this owned child.
                self.verify(name,entry['files'],partial=True)
                payload=path/'training.pt'
                if payload.exists():payload.unlink()
                if fault=='after_payload':raise OSError('Injected interruption after payload unlink')
                marker=path/'complete.json'
                if marker.exists():marker.unlink()
                if path.exists():path.rmdir()  # Empty owned checkpoint only; never recursive.
                if fault=='after_directory':raise OSError('Injected interruption after directory removal')
                entry['status']='pruned'
                index['events'].append(dict(action='pruned',name=name,time=time.time()))
                write_json(self.index_path,index);removed.append(name)
            return dict(protected=sorted(keep),pruned=removed)
