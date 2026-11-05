"""Observe live identities and held exit receipts without controlling processes."""
import math
import psutil


def norm(command):
    return [str(x).replace('\\', '/').lower() for x in command]


def process_identity(record, lookup=psutil.Process):
    observed = {'pid': record['pid'], 'present': False, 'matches': False,
                'expected_command': record['command'],
                'expected_create_time': record.get('create_time')}
    try:
        p = lookup(record['pid'])
        command, created = p.cmdline(), p.create_time()
        expected = record.get('create_time')
        observed.update(present=True, command=command, create_time=created,
                        matches=norm(command) == norm(record['command']) and
                        expected is not None and math.isfinite(expected) and
                        abs(created-expected) < .01)
    except psutil.NoSuchProcess:
        observed['reason'] = 'process_absent'
    except psutil.AccessDenied:
        observed['reason'] = 'identity_access_denied'
    return observed


def held_exits_match(attached, receipt):
    """An exit receipt proves exits only, never successful experiment completion."""
    if receipt.get('phase') != 'exited' or receipt.get('passed') is not True:
        return False
    expected, observed = attached.get('workers', []), receipt.get('workers', [])
    if len(expected) < 2 or len(expected) != len(observed):
        return False
    identities = set()
    for e in expected:
        key = (e['pid'], e['create_time'])
        if key in identities:
            return False
        identities.add(key)
        matches = [o for o in observed if o.get('pid') == e['pid'] and
                   o.get('create_time') == e['create_time']]
        if len(matches) != 1:
            return False
        o = matches[0]
        if norm(o.get('command', [])) != norm(e['command']) or type(o.get('exit_code')) is not int:
            return False
    return True
