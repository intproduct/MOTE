"""Pure validation of active evaluation evidence at a post-read observation time."""
import math


def active_evaluation_evidence(status, checkpoint_exists, dev_exists, thermal_rows, observed_epoch):
    try:
        tokens, target = status['tokens'], status['target']
        if status['phase'] != 'running' or target != 32768000 or type(tokens) is not int or not (0 < tokens <= target) or tokens % (target // 10):
            return None
        if not checkpoint_exists or dev_exists or not math.isfinite(observed_epoch) or not math.isfinite(status['time']):
            return None
        evaluation = [r for r in thermal_rows if r.get('stage') == 'evaluation' and r.get('time', 0) > status['time']]
        if len(evaluation) < 2 or thermal_rows[-1] != evaluation[-1]:
            return None
        a, b = evaluation[-2:]
        if not (math.isfinite(a['time']) and math.isfinite(b['time']) and a['time'] < b['time'] <= observed_epoch and observed_epoch-b['time'] < 60):
            return None
        if any(r.get('wait_seconds') != 0 or r.get('event') for r in (a,b)):
            return None
        return {'kind':'internal_dev_evaluation','tokens':tokens,'observed_epoch':observed_epoch,'last_evaluation_guards':[a,b]}
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
