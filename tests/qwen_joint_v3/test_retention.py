import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import pytest
from fitmotn.experiments.qwen_joint.retention import Retention,FORMAT,read,digest,write_json

IDENTITY={'code':'candidate-v3','model':'frozen','data':'frozen','seed':17,'method':'mlp'}
@pytest.fixture
def manager(tmp_path):return Retention.create(tmp_path,tmp_path/'new-run',IDENTITY,1000)
def checkpoint(manager,tokens,score=9.):
    name=f'checkpoint-{tokens:010d}';path=manager.path/name;path.mkdir()
    data=path/'training.pt';data.write_bytes(f'owned dummy state at {tokens}'.encode())
    write_json(path/'complete.json',dict(format=FORMAT,identity=IDENTITY,tokens=tokens,cursor=tokens,
        files={'training.pt':dict(bytes=data.stat().st_size,sha256=digest(data))}))
    manager.register(name,score);return path

def test_retains_milestones_best_and_latest(manager):
    for token in range(100,1001,100):
        checkpoint(manager,token,1. if token==200 else 2.)
        manager.prune()
    expected={100,200,500,1000}
    assert {int(p.name[11:]) for p in manager.path.glob('checkpoint-*')}==expected
    index=read(manager.index_path)
    assert sum(e['status']=='pruned' for e in index['entries'].values())==6
    assert all(e['files']['training.pt']['sha256'] for e in index['entries'].values())

def test_final_best_deduplicates_and_ties_choose_earliest(manager):
    for token in range(100,1001,100):
        checkpoint(manager,token,1. if token==1000 else 2.);manager.prune()
    assert len(list(manager.path.glob('checkpoint-*')))==3

def test_new_corrupt_checkpoint_cannot_retire_previous(manager):
    first=checkpoint(manager,100)
    broken=manager.path/'checkpoint-0000000200';broken.mkdir()
    (broken/'training.pt').write_bytes(b'bad')
    write_json(broken/'complete.json',dict(format=FORMAT,identity=IDENTITY,tokens=200,files={'training.pt':dict(bytes=3,sha256='0'*64)}))
    with pytest.raises(ValueError,match='checksum'):manager.register(broken.name,1.)
    manager.prune();assert first.exists() and broken.exists()

@pytest.mark.parametrize('marker',['FAILURE.json','STOP'])
def test_failed_or_stopped_run_is_preserved(manager,marker):
    for token in (100,200,300):checkpoint(manager,token)
    (manager.path/marker).write_text('failure or stop')
    with pytest.raises(ValueError,match='failed/stopped'):manager.prune()
    assert len(list(manager.path.glob('checkpoint-*')))==3

def test_incomplete_and_foreign_files_never_removed(manager):
    for token in (100,200,300):checkpoint(manager,token)
    extra=manager.path/'checkpoint-0000000200'/'failure-extra.log';extra.write_text('evidence')
    with pytest.raises(ValueError,match='unrecognized'):manager.prune()
    assert extra.exists() and (extra.parent/'training.pt').exists()

def test_changed_registered_checkpoint_preserved(manager):
    for token in (100,200,300):checkpoint(manager,token)
    changed=manager.path/'checkpoint-0000000200'/'training.pt';changed.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='changed'):manager.prune()
    assert changed.exists()

def test_bad_newest_blocks_any_retirement(manager):
    for token in (100,200,300):checkpoint(manager,token)
    (manager.path/'checkpoint-0000000300'/'training.pt').write_bytes(b'bad')
    with pytest.raises(ValueError):manager.prune()
    assert (manager.path/'checkpoint-0000000200'/'training.pt').exists()

@pytest.mark.parametrize('fault',['after_intent','after_payload','after_directory'])
def test_interrupted_deletion_keeps_recovery_and_resumes_exact_intent(manager,fault):
    for token in (100,200,300):checkpoint(manager,token)
    with pytest.raises(OSError):manager.prune(fault=fault)
    manager.verify('checkpoint-0000000100');manager.verify('checkpoint-0000000300')
    assert read(manager.index_path)['entries']['checkpoint-0000000200']['status']=='pruning'
    resumed=Retention(manager.path,manager.owner['nonce'])
    result=resumed.prune()
    assert result['pruned']==['checkpoint-0000000200']
    assert not (manager.path/'RETENTION_MUTATION.lock').exists()

def test_existing_run_or_wrong_owner_or_escape_rejected(manager,tmp_path):
    with pytest.raises(ValueError):Retention.create(tmp_path,manager.path,IDENTITY,1000)
    with pytest.raises(ValueError):Retention(manager.path,'foreign')
    with pytest.raises(ValueError):manager.checkpoint_path('../other/checkpoint-0000000200')

def test_mutation_lock_never_stolen(manager):
    lock=manager.path/'RETENTION_MUTATION.lock';lock.write_text('{"nonce":"foreign"}')
    with pytest.raises(FileExistsError):manager.prune()
    assert lock.exists()
