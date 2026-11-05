"""Bounded post-evaluation evidence. Pure CPU validation, no process control."""
import math

def post_evaluation_evidence(status, config, checkpoint, dev, dev_mtime, thermal_rows, now, acceptance, sample_hash):
    try:
        tokens, target = status['tokens'], status['target']
        if status['phase'] != 'running' or target != 32768000 or type(tokens) is not int or not (0 < tokens <= target) or tokens % (target // 10):
            return None
        times = (now, status['time'], dev_mtime, checkpoint['created_epoch'])
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in times):
            return None
        if not (status['time'] <= checkpoint['created_epoch'] <= dev_mtime <= now and now-dev_mtime < 60):
            return None
        identity = checkpoint['identity']
        if checkpoint['tokens'] != tokens or checkpoint['cursor'] != status['cursor'] or identity['acceptance_sha256'] != acceptance:
            return None
        if any(identity[k] != config[k] for k in ('method', 'seed', 'gpu')):
            return None
        if dev['valid_targets'] != 1048576 or dev['sample_hash'] != sample_hash or dev['evaluation_order_seed'] != 17:
            return None
        if not (math.isfinite(dev['ce']) and math.isfinite(dev['nll_sum']) and math.isclose(dev['ce'], dev['nll_sum']/dev['valid_targets'], rel_tol=1e-9, abs_tol=1e-9)):
            return None
        evaluation = [r for r in thermal_rows if r.get('stage') == 'evaluation' and status['time'] < r.get('time', 0) <= dev_mtime]
        if len(evaluation) < 2:
            return None
        last = thermal_rows[-1]
        if not (evaluation[-2]['time'] < evaluation[-1]['time'] <= dev_mtime <= last['time'] <= now and dev_mtime-evaluation[-1]['time'] < 60 and now-last['time'] < 60):
            return None
        if last.get('stage') != 'update' or any(r.get('wait_seconds') != 0 or r.get('event') for r in [*evaluation[-2:], last]):
            return None
        return {'kind':'post_evaluation_handoff', 'tokens':tokens, 'dev_mtime':dev_mtime, 'dev_age_seconds':now-dev_mtime, 'maximum_age_seconds':60, 'last_evaluation_guards':evaluation[-2:], 'next_update_guard':last, 'dev':dev}
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
