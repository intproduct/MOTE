import copy
import psutil
import pytest
from formal_health_processes_v24 import process_identity, held_exits_match

RECORD = {'pid': 10, 'create_time': 100., 'command': ['C:/Python/python.exe', 'worker.py']}


class Process:
    def __init__(self, pid):
        self.pid = pid
    def cmdline(self):
        return RECORD['command']
    def create_time(self):
        return RECORD['create_time']


def test_exact_identity():
    assert process_identity(RECORD, Process)['matches']


@pytest.mark.parametrize('exception', [psutil.NoSuchProcess, psutil.AccessDenied])
def test_inaccessible_process_is_evidence_not_crash(exception):
    def missing(pid):
        raise exception(pid)
    result = process_identity(RECORD, missing)
    assert not result['matches'] and result['reason']


@pytest.mark.parametrize('case', ['pid_reused', 'wrong_command', 'missing_create_time', 'nonfinite_create_time'])
def test_identity_rejected(case):
    record = copy.deepcopy(RECORD)
    if case == 'pid_reused': record['create_time'] = 99.
    if case == 'wrong_command': record['command'] = ['wrong.py']
    if case == 'missing_create_time': record.pop('create_time')
    if case == 'nonfinite_create_time': record['create_time'] = float('nan')
    assert not process_identity(record, Process)['matches']


def fixtures():
    attached = {'workers': [RECORD, dict(RECORD, pid=11, create_time=101.)]}
    receipt = {'phase': 'exited', 'passed': True,
               'workers': [dict(r, exit_code=0) for r in attached['workers']]}
    return copy.deepcopy(attached), copy.deepcopy(receipt)


def test_held_exits_prove_exit_even_with_failure_code():
    a, r = fixtures()
    assert held_exits_match(a, r)
    r['workers'][1]['exit_code'] = 1
    assert held_exits_match(a, r)  # caller must still check failed controller state


@pytest.mark.parametrize('case', ['missing_worker', 'duplicate_worker', 'wrong_create', 'wrong_command',
                                'missing_code', 'bool_code', 'wrong_phase', 'failed_receipt'])
def test_bad_receipt(case):
    a, r = fixtures()
    if case == 'missing_worker': r['workers'].pop()
    if case == 'duplicate_worker': r['workers'][1] = r['workers'][0]
    if case == 'wrong_create': r['workers'][0]['create_time'] += 1
    if case == 'wrong_command': r['workers'][0]['command'] = ['other.py']
    if case == 'missing_code': r['workers'][0].pop('exit_code')
    if case == 'bool_code': r['workers'][0]['exit_code'] = False
    if case == 'wrong_phase': r['phase'] = 'running'
    if case == 'failed_receipt': r['passed'] = False
    assert not held_exits_match(a, r)
