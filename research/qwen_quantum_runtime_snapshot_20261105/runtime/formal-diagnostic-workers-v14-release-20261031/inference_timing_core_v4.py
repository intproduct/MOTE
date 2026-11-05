"""Testable timing boundaries; no CUDA initialization or model imports."""
import math
import statistics


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')


def summarize(seconds, input_tokens_per_repeat):
    positive_integer(input_tokens_per_repeat, 'input_tokens_per_repeat')
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError('All measured durations must be finite and positive')
    total = math.fsum(seconds)
    return dict(repeats=len(seconds), input_tokens_per_repeat=input_tokens_per_repeat,
        total_input_tokens=len(seconds) * input_tokens_per_repeat,
        total_measured_seconds=total, mean_seconds=statistics.mean(seconds),
        median_seconds=statistics.median(seconds), min_seconds=min(seconds), max_seconds=max(seconds),
        input_tokens_per_second=len(seconds) * input_tokens_per_repeat / total)


def serial_measure(forward, synchronize, clock, before, after, *, warmup=2, repeats=10):
    """One forward call processes every fixed window serially, including lm_head.

    Safety observations, validation and garbage collection stay outside timers.
    The after callback persists each raw sample before another repeat begins.
    """
    positive_integer(warmup, 'warmup'); positive_integer(repeats, 'repeats')
    rows = []
    for phase, count in (('warmup', warmup), ('measured', repeats)):
        for index in range(count):
            observation = before(phase, index)
            synchronize()
            start = clock()
            forward()
            synchronize()
            seconds = clock() - start
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Nonpositive or nonfinite forward duration')
            row = dict(phase=phase, index=index, seconds=seconds,
                observation_before=observation)
            extra = after(row)
            if extra is not None: row['observation_after'] = extra
            rows.append(row)
    return rows


def assert_plain_evaluation(model):
    if getattr(model.config, 'use_cache', None) is not False:
        raise ValueError('KV cache must be explicitly disabled')
    if getattr(model, 'is_gradient_checkpointing', False):
        raise ValueError('Gradient checkpointing is not an inference benchmark')
    for name, module in model.named_modules():
        if module.training: raise ValueError('Training module: ' + name)
        for hook_kind in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'):
            if getattr(module, hook_kind, {}): raise ValueError('Instrumented module: ' + name)
        if getattr(module, 'capture_metrics', False):
            raise ValueError('Routing metrics enabled: ' + name)


def assert_owned_observation(snapshot, lease_record, lock_record, owner_pid):
    if (lock_record.get('pid') != owner_pid or lock_record.get('token') != lease_record['token']
            or lock_record.get('uuid') != snapshot['uuid'] or snapshot['uuid'] != lease_record['uuid']):
        raise RuntimeError('Device lease ownership changed')
    foreign = []
    for app in snapshot['compute_apps']:
        if len(app) < 3 or app[0] != snapshot['uuid'] or app[1] != str(owner_pid):
            foreign.append(app)
    if foreign: raise RuntimeError('Foreign or unverifiable device compute: ' + repr(foreign))
    return dict(foreign_compute_observed=False, desktop_processes_observed=bool(snapshot.get('display_apps')),
        snapshot=snapshot)
