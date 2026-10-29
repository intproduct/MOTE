from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from ..autoregressive_gate.data import digest, write_json, prepare, training_batch, evaluation_batches
from .model import initialized_model, seed_all, CausalLM, full_state_hash
from ..gate_benchmark.models import (
    shared_state, state_hash, parameter_report, auxiliary_loss,
    routing_snapshot, routed_modules, is_router_key,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE_PATHS = [
    'ADTN.py', 'gate.py', 'losses.py', 'sparse_mixt.py', 'structured_routers.py',
    'quantum_gate.py', 'advanced_routers.py', 'experiments/gate_benchmark/models.py',
    'experiments/autoregressive_gate/model.py', 'experiments/autoregressive_gate/data.py',
    'experiments/autoregressive_gate/runner.py',
    'experiments/routing_stability/routers.py', 'experiments/routing_stability/model.py',
    'experiments/routing_stability/runner.py', 'experiments/routing_stability/probe.py',
]


def timestamp():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    import tokenizers, transformers, datasets
    return {'time': timestamp(), 'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'numpy': np.__version__, 'tokenizers': tokenizers.__version__,
            'transformers': transformers.__version__, 'datasets': datasets.__version__,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True, encoding='utf-8', errors='replace'),
            'source_sha256': {p: digest(REPO/p) for p in SOURCE_PATHS},
            'precision': 'fp32', 'tf32': False, 'threads': torch.get_num_threads()}


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def atomic_checkpoint(path, value):
    tmp = Path(str(path)+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_tokens(data_root, split):
    return torch.from_numpy(np.load(Path(data_root)/f'{split}.npy').astype(np.int64))


@torch.no_grad()
def evaluate(model, tokens, cfg, device, collect_routing=False):
    model.eval()
    total, count, correct = 0.0, 0, 0
    routing = {}
    for x, y, valid in evaluation_batches(tokens, cfg['model']['context'], cfg['eval']['stride'],
                                         cfg['eval']['batch_size'], device):
        logits = model(x, valid)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              ignore_index=-100, reduction='sum')
        if not torch.isfinite(nll):
            raise FloatingPointError('non-finite evaluation NLL')
        total += float(nll.double().item())
        count += int((y != -100).sum().item())
        correct += int(((logits.argmax(-1) == y) & (y != -100)).sum().item())
        if collect_routing:
            for name, r in routing_snapshot(model).items():
                item = routing.setdefault(name, {'tokens': 0, 'entropy_sum': 0.,
                                                 'load_sum': np.zeros(cfg['router']['experts'])})
                item['tokens'] += r['tokens']
                item['entropy_sum'] += r['H_tok'] * r['tokens']
                item['load_sum'] += np.asarray(r['load']) * r['tokens']
    assert count == len(tokens)-1, (count, len(tokens))
    result = {'ce': total/count, 'ppl': math.exp(total/count), 'nll_sum': total,
              'tokens': count, 'next_token_accuracy': correct/count}
    if collect_routing:
        result['routing'] = {}
        for name, r in routing.items():
            load = r['load_sum']/r['tokens']
            result['routing'][name] = {'entropy': r['entropy_sum']/r['tokens'],
                'load': load.tolist(), 'load_cv': float(load.std()/max(load.mean(), 1e-12)),
                'unused_experts': int((load == 0).sum()), 'routed_tokens_with_window_overlap': r['tokens']}
    return result


def lr_at(step, cfg):
    t = cfg['train']
    if step <= t['warmup']:
        return t['lr']*step/max(t['warmup'], 1)
    progress = (step-t['warmup'])/max(t['steps']-t['warmup'], 1)
    return t['lr']*(t['min_lr_ratio']+(1-t['min_lr_ratio'])*0.5*(1+math.cos(math.pi*progress)))


def train_one(cfg, data_root, run_dir, variant, seed):
    run_dir = Path(run_dir)
    if (run_dir/'result.json').exists():
        result = read_json(run_dir/'result.json')
        assert result['config_hash'] == canonical_hash(cfg)
        assert result['checkpoint_sha256'] == digest(run_dir/'best.pt')
        print(f'SKIP verified completed {run_dir.name}', flush=True)
        return result
    if run_dir.exists():
        raise RuntimeError(f'incomplete run retained: {run_dir}; archive deliberately before retry')
    run_dir.mkdir(parents=True)
    device = torch.device(cfg['device'])
    model, initial_shared = initialized_model(cfg, variant, seed)
    initial_router = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if is_router_key(k)}
    meta = read_json(Path(data_root)/'manifest.json')
    manifest = {'variant': variant, 'seed': seed, 'config': cfg, 'config_hash': canonical_hash(cfg),
                'initial_model_hash': full_state_hash(cpu_state(model)),
                'shared_init_hash': state_hash(initial_shared), 'data_manifest_sha256': digest(Path(data_root)/'manifest.json'),
                'parameters': parameter_report(model), 'environment': environment()}
    write_json(run_dir/'manifest.json', manifest)
    model.to(device)
    train = load_tokens(data_root, 'train')
    validation = load_tokens(data_root, 'validation')
    limit = cfg['eval'].get('validation_tokens')
    if limit:
        validation = validation[:limit+1]
    seed_all(seed+100000)
    generator = torch.Generator().manual_seed(seed+200000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    best_ce, best_step, compute_seconds = float('inf'), 0, 0.0
    batch_starts, finite_steps, nonzero_router_grad_steps = [], 0, 0
    t0 = time.perf_counter()
    initial_validation = evaluate(model, validation, cfg, device)
    append_json(run_dir/'curves.jsonl', {'step': 0, 'kind': 'validation', 'training_seconds': 0., **initial_validation})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    first_loss = None
    router_names = [name for name, _ in model.named_parameters() if is_router_key(name)]
    for step in range(1, cfg['train']['steps']+1):
        synchronize(device)
        start = time.perf_counter()
        model.training_step = step
        model.train()
        x, y, starts = training_batch(train, cfg['train']['batch_size'], cfg['model']['context'], generator, device)
        batch_starts.append(starts.tolist())
        rate = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        aux = auxiliary_loss(model)
        loss = ce + aux + model.regularizer()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'loss at step {step}: {loss}')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'], error_if_nonfinite=True)
        # Check router gradients without treating an individual temporarily inactive tensor as a failure.
        router_grad = sum(p.grad.detach().abs().sum() for name, p in model.named_parameters()
                          if name in router_names and p.grad is not None)
        if not torch.isfinite(router_grad):
            raise FloatingPointError('non-finite router gradient')
        nonzero_router_grad_steps += int(router_grad.item() > 0)
        optimizer.step()
        synchronize(device)
        compute_seconds += time.perf_counter()-start
        finite_steps += 1
        first_loss = float(ce.item()) if first_loss is None else first_loss
        if step == 1 or step % cfg['train']['log_every'] == 0:
            record = {'step': step, 'kind': 'train', 'ce': float(ce.item()), 'aux_loss': float(aux.item()),
                      'grad_norm': float(grad_norm.item()), 'router_grad_l1_after_clip': float(router_grad.item()),
                      'lr': rate, 'training_seconds': compute_seconds,
                      'tokens_seen': step*cfg['train']['batch_size']*cfg['model']['context'], **model.margin_metrics()}
            append_json(run_dir/'curves.jsonl', record)
            append_json(run_dir/'routing.jsonl', {'step': step, 'routing': routing_snapshot(model)})
            print(f'{run_dir.name} step={step} CE={record["ce"]:.4f} grad={record["grad_norm"]:.4f} train_s={compute_seconds:.1f}', flush=True)
        if step % cfg['train']['eval_every'] == 0 or step == cfg['train']['steps']:
            val = evaluate(model, validation, cfg, device)
            append_json(run_dir/'curves.jsonl', {'step': step, 'kind': 'validation',
                        'training_seconds': compute_seconds, **val})
            state = cpu_state(model)
            if not all(torch.isfinite(t).all() for t in state.values()):
                raise FloatingPointError('non-finite checkpoint')
            checkpoint = {'model': state, 'step': step, 'validation': val,
                          'variant': variant, 'seed': seed, 'config': cfg}
            if val['ce'] < best_ce:
                best_ce, best_step = val['ce'], step
                atomic_checkpoint(run_dir/'best.pt', checkpoint)
            atomic_checkpoint(run_dir/'last.pt', checkpoint)
            print(f'{run_dir.name} DEV step={step} CE={val["ce"]:.4f} PPL={val["ppl"]:.2f}', flush=True)
    parameter_changes = {name: float((p.detach().cpu()-initial_router[name]).abs().max().item())
                         for name, p in model.named_parameters() if name in initial_router}
    if any(value == 0 for value in parameter_changes.values()):
        raise RuntimeError(f'unchanged router tensors: {[k for k,v in parameter_changes.items() if v == 0]}')
    peak_allocated = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    peak_reserved = torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0
    shared_change = state_hash(shared_state(model)) != manifest['shared_init_hash']
    assert shared_change and nonzero_router_grad_steps == finite_steps
    batch_path = run_dir/'batch_starts.npy'
    np.save(batch_path, np.asarray(batch_starts, dtype=np.int64))
    # Reload, rather than trusting in-memory best weights, and independently recompute dev.
    best = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    restored = evaluate(model, validation, cfg, device, collect_routing=True)
    assert abs(restored['ce']-best_ce) < 1e-6
    result = {'variant': variant, 'seed': seed, 'phase': cfg['phase'],
              'config_hash': canonical_hash(cfg), 'steps': finite_steps, 'best_step': best_step,
              'initial_validation': initial_validation, 'best_validation': restored,
              'last_validation': val, 'first_train_ce': first_loss, 'last_train_ce': float(ce.item()),
              'training_seconds': compute_seconds, 'wall_seconds': time.perf_counter()-t0,
              'tokens_seen': finite_steps*cfg['train']['batch_size']*cfg['model']['context'],
              'training_tokens_per_second': finite_steps*cfg['train']['batch_size']*cfg['model']['context']/compute_seconds,
              'peak_allocated_mb': peak_allocated/2**20, 'peak_reserved_mb': peak_reserved/2**20,
              'parameters': manifest['parameters'], 'shared_init_hash': manifest['shared_init_hash'],
              'batch_order_sha256': digest(batch_path), 'data_manifest_sha256': manifest['data_manifest_sha256'],
              'finite_steps': finite_steps, 'nonzero_router_grad_steps': nonzero_router_grad_steps,
              'shared_parameters_updated': shared_change, 'router_parameter_max_changes': parameter_changes,
              'checkpoint_sha256': digest(run_dir/'best.pt'), 'completed_at': timestamp()}
    write_json(run_dir/'result.json', result)
    return result


def audit(group, cfg):
    group = Path(group)
    problems, entries = [], []
    shared, orders, data_hashes = {}, {}, set()
    sources = set()
    paired_initial = {}
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            name = f'{variant}_s{seed}'
            path = group/'runs'/name
            if not (path/'result.json').exists():
                problems.append({'run': name, 'problem': 'missing result'})
                continue
            r, m = read_json(path/'result.json'), read_json(path/'manifest.json')
            for f in ['best.pt', 'last.pt', 'curves.jsonl', 'routing.jsonl', 'batch_starts.npy']:
                if not (path/f).is_file():
                    problems.append({'run': name, 'problem': f'missing {f}'})
            if r['config_hash'] != canonical_hash(cfg) or r['steps'] != cfg['train']['steps']:
                problems.append({'run': name, 'problem': 'configuration/steps mismatch'})
            if r['checkpoint_sha256'] != digest(path/'best.pt'):
                problems.append({'run': name, 'problem': 'checkpoint hash mismatch'})
            if r['finite_steps'] != r['steps'] or r['nonzero_router_grad_steps'] != r['steps']:
                problems.append({'run': name, 'problem': 'gradient/finite audit failed'})
            if not r['shared_parameters_updated'] or not all(r['router_parameter_max_changes'].values()):
                problems.append({'run': name, 'problem': 'parameters did not update'})
            shared.setdefault(seed, set()).add(r['shared_init_hash'])
            orders.setdefault(seed, set()).add(r['batch_order_sha256'])
            data_hashes.add(r['data_manifest_sha256'])
            sources.add(canonical_hash(m['environment']['source_sha256']))
            paired_initial.setdefault((seed, cfg['designs'][variant]['base']), set()).add(m['initial_model_hash'])
            for file in ['curves.jsonl', 'routing.jsonl']:
                text = (path/file).read_text(encoding='utf-8')
                if any(t in text for t in ['NaN', 'Infinity']):
                    problems.append({'run': name, 'problem': f'non-finite {file}'})
            entries.append(name)
    if any(len(v) != 1 for v in [*shared.values(), *orders.values()]) or len(data_hashes) != 1 or len(sources) != 1:
        problems.append({'problem': 'inconsistent shared init, order, data, or source hashes'})
    if any(len(v) != 1 for v in paired_initial.values()):
        problems.append({'problem': 'margin/no-margin whole-model initialization mismatch'})
    result = {'passed': not problems, 'expected': len(cfg['seeds'])*len(cfg['gates']),
              'complete': len(entries), 'problems': problems, 'runs': entries, 'time': timestamp()}
    write_json(group/'audit.json', result)
    return result


@torch.no_grad()
def generate(model, tokenizer, device):
    examples = []
    for prompt in ['The history of science', 'In the early twentieth century', 'The city is located']:
        ids = tokenizer.encode(prompt).ids
        continuation = []
        for _ in range(64):
            x = torch.tensor([ids[-model.context:]], dtype=torch.long, device=device)
            token = int(model(x)[0, -1].argmax().item())
            ids.append(token)
            continuation.append(token)
            if token == tokenizer.token_to_id('<eos>'):
                break
        examples.append({'prompt': prompt, 'continuation': tokenizer.decode(continuation),
                         'token_ids': continuation, 'decoding': 'greedy, max_new_tokens=64, no KV cache'})
    return examples


def evaluate_all(group, cfg, data_root):
    from tokenizers import Tokenizer
    group = Path(group)
    if cfg['phase'] != 'formal' or not audit(group, cfg)['passed']:
        raise RuntimeError('held-out test evaluation requires the complete formal matrix')
    device = torch.device(cfg['device'])
    test = load_tokens(data_root, 'test')
    validation = load_tokens(data_root, 'validation')
    train = load_tokens(data_root, 'train')
    tokenizer = Tokenizer.from_file(str(Path(data_root)/'tokenizer.json'))
    for seed in cfg['seeds']:
        for variant in cfg['gates']:
            path = group/'runs'/f'{variant}_s{seed}'
            out = path/'evaluation.json'
            if out.exists():
                old = read_json(out)
                assert old['checkpoint_sha256'] == digest(path/'best.pt')
                continue
            ckpt = torch.load(path/'best.pt', map_location='cpu', weights_only=False)
            seed_all(seed)
            model = CausalLM(cfg, variant, seed).to(device)
            model.load_state_dict(ckpt['model'])
            dev = evaluate(model, validation, cfg, device)
            assert abs(dev['ce']-ckpt['validation']['ce']) < 1e-6
            result = {'variant': variant, 'seed': seed, 'checkpoint_sha256': digest(path/'best.pt'),
                      'validation_reload': dev, 'test': evaluate(model, test, cfg, device, True)}
            result['generation'] = generate(model, tokenizer, device)
            result['inference_benchmark'] = inference_benchmark(model, train, cfg, device)
            model.training_step = cfg['train']['steps']
            # Serial benchmark uses disposable copies of checkpoint weights and a fresh optimizer.
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
            generator = torch.Generator().manual_seed(987654)
            elapsed = []
            torch.cuda.reset_peak_memory_stats()
            for i in range(25):
                synchronize(device)
                start = time.perf_counter()
                x, y, _ = training_batch(train, cfg['train']['batch_size'], cfg['model']['context'], generator, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0,1), y.flatten()) + auxiliary_loss(model) + model.regularizer()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                synchronize(device)
                if i >= 5:
                    elapsed.append(time.perf_counter()-start)
            result['serial_benchmark'] = {'warmup_steps': 5, 'measured_steps': 20,
                'step_seconds': elapsed, 'mean_step_seconds': float(np.mean(elapsed)),
                'tokens_per_second': cfg['train']['batch_size']*cfg['model']['context']/float(np.mean(elapsed)),
                'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20,
                'scope': 'forward+aux+margin(if active)+backward+clip+AdamW+host batch transfer; fresh optimizer; weights discarded'}
            write_json(out, result)
            print(f'REEVAL {path.name} TEST CE={result["test"]["ce"]:.4f} PPL={result["test"]["ppl"]:.2f}', flush=True)
            del model, optimizer
            torch.cuda.empty_cache()



@torch.no_grad()
def inference_benchmark(model, tokens, cfg, device):
    model.eval()
    rows = []
    # Cache shared operators during warmup, all methods use identical shapes.
    for batch, length in [(16, cfg['model']['context']), (1, cfg['model']['context']), (1, 1)]:
        ids, _, _ = training_batch(tokens, batch, length, torch.Generator().manual_seed(12345), device)
        elapsed = []
        for i in range(35):
            synchronize(device)
            start = time.perf_counter()
            model(ids)
            synchronize(device)
            if i >= 10:
                elapsed.append(time.perf_counter()-start)
        rows.append({'batch': batch, 'length': length, 'warmup': 10, 'repeats': 25,
                     'seconds': elapsed, 'median_seconds': float(np.median(elapsed)),
                     'tokens_per_second': batch*length/float(np.median(elapsed)),
                     'scope': 'full forward, GPU-resident input; no KV cache; length1 is not cached decode'})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare','run','audit','evaluate'])
    p.add_argument('--config', required=True)
    p.add_argument('--root', default='outputs/routing_stability_round1')
    p.add_argument('--variant')
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    setup()
    cfg = read_json(args.config)
    root = Path(args.root).resolve()
    data_root = (REPO/cfg['data_root']).resolve()
    group = root/cfg['phase']/canonical_hash(cfg)[:12]
    if args.command == 'prepare':
        meta = prepare(data_root, cfg['model']['vocab_size'])
        print(json.dumps({k: {x:v for x,v in r.items() if x != 'article_sha256'} for k,r in meta['splits'].items()}, indent=2))
    elif args.command == 'run':
        group.mkdir(parents=True, exist_ok=True)
        if (group/'config.json').exists():
            assert read_json(group/'config.json') == cfg
        else:
            write_json(group/'config.json', cfg)
        # Verify cached dataset hashes and tokenizer before using them.
        prepare(data_root, cfg['model']['vocab_size'])
        gates = [args.variant] if args.variant else cfg['gates']
        seeds = [args.seed] if args.seed is not None else cfg['seeds']
        assert set(gates) <= set(cfg['gates']) and set(seeds) <= set(cfg['seeds'])
        for seed in seeds:
            for variant in gates:
                status = {'phase': cfg['phase'], 'state': 'training', 'run': f'{variant}_s{seed}',
                          'group': str(group), 'pid': os.getpid(), 'updated': timestamp()}
                write_json(root/'STATUS.json', status)
                try:
                    train_one(cfg, data_root, group/'runs'/status['run'], variant, seed)
                except Exception:
                    status.update(state='failed', traceback=traceback.format_exc(), updated=timestamp())
                    write_json(root/'STATUS.json', status)
                    raise
                torch.cuda.empty_cache()
        if not args.variant and args.seed is None:
            result = audit(group, cfg)
            assert result['passed'], result
            write_json(root/'STATUS.json', {'phase': cfg['phase'], 'state': 'training_complete',
                       'group': str(group), 'audit': result, 'updated': timestamp()})
    elif args.command == 'audit':
        result = audit(group, cfg)
        print(json.dumps(result, indent=2))
        if not result['passed']:
            raise SystemExit(1)
    elif args.command == 'evaluate':
        evaluate_all(group, cfg, data_root)


if __name__ == '__main__':
    main()
