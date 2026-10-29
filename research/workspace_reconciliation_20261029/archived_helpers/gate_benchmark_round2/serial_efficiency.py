"""Serial selected-checkpoint continuation timing; no original artifact changes.

Each run uses 3 untimed warmup updates followed by 10 measured training updates.
All gates for a task/seed see the same batches. These short continuations are
discarded and are NOT additional accuracy experiments or resumed main runs.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.update(HF_HOME=str(ROOT / 'outputs/hf_cache'), HF_HUB_OFFLINE='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
import csv
import gc
import json
import statistics
import time
import torch
from fitmotn.experiments.gate_benchmark.config import load_config, digest
from fitmotn.experiments.gate_benchmark.data import load_data, bert_path, TensorDictDataset
from fitmotn.experiments.gate_benchmark.models import ImageClassifier, BertClassifier, auxiliary_loss, warm_initialize, routed_modules
from fitmotn.experiments.gate_benchmark.runner import seed_all, metrics

c = load_config('experiments/gate_benchmark/configs/pareto_formal.json')
root = Path(c['output_dir'])
group = root / digest(c)[:12]
assert (group / 'execution_complete.json').exists(), 'Main GPU training must finish before isolated timing'
rows = []
for stage in ('mlp', 'bert'):
    for task in c[stage]['tasks']:
        data, meta = load_data(c, task)
        for seed in c['seeds']:
            for gate in (c['gates'][c['seeds'].index(seed):]+c['gates'][:c['seeds'].index(seed)]):
                name = f'{stage}_{task}_{gate}_seed{seed}'
                out = root / 'serial_efficiency' / name
                if (out / 'measurement.json').exists():
                    rows.append(json.loads((out / 'measurement.json').read_text()))
                    continue
                if out.exists():
                    raise FileExistsError(f'Partial timing measurement retained: {out}')
                out.mkdir(parents=True)
                seed_all(seed, c['deterministic'])
                model = (ImageClassifier(c, gate, seed, meta['input_dim'], meta['classes']) if stage == 'mlp' else
                         BertClassifier(c, gate, seed, meta['classes'], model_path=bert_path(c)[0])).cuda()
                checkpoint = torch.load(group / name / 'best.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model'])
                warm_initialize(model, 0, seed)
                # Independently reproduce saved final scores and retain class
                # predictions for diagnosing majority-class collapse. This
                # evaluation is outside the training-efficiency timer.
                model.eval()
                eval_logits, eval_labels, router_sums = [], [], {}
                with torch.no_grad():
                    for evaluation_batch in torch.utils.data.DataLoader(TensorDictDataset(data['test']), batch_size=c[stage]['batch_size'], shuffle=False):
                        evaluation_labels = evaluation_batch.pop('labels')
                        evaluation_output = model(**{k: v.cuda() for k,v in evaluation_batch.items()})
                        eval_logits.append(evaluation_output.float().cpu())
                        eval_labels.append(evaluation_labels)
                        for layer, module in routed_modules(model):
                            aux, tokens = module.last_aux, module.last_tokens
                            if aux is None:
                                continue
                            sums = router_sums.setdefault(layer, {'tokens': 0, 'H_tok': 0., 'drop_rate': 0.,
                                'importance': torch.zeros_like(aux['importance'], device='cpu'),
                                'load': torch.zeros_like(aux['load'], device='cpu')})
                            sums['tokens'] += tokens
                            for key in ('H_tok', 'drop_rate'):
                                sums[key] += float(aux[key])*tokens
                            for key in ('importance', 'load'):
                                sums[key] += aux[key].detach().cpu()*tokens
                eval_logits, eval_labels = torch.cat(eval_logits), torch.cat(eval_labels)
                reproduced = metrics(eval_logits, eval_labels, task)
                result = json.loads((group / name / 'result.json').read_text())
                recorded = result['final']
                for key in recorded:
                    assert abs(reproduced[key] - recorded[key]) <= (1e-6 if key == 'ce' else 0), (name, key)
                routes = {}
                for layer, sums in router_sums.items():
                    tokens = sums['tokens']
                    importance, load = sums['importance']/tokens, sums['load']/tokens
                    routes[layer] = {'tokens': tokens, 'H_tok': sums['H_tok']/tokens,
                        'H_dataset': float(-(importance*importance.clamp_min(1e-12).log()).sum()),
                        'drop_rate': sums['drop_rate']/tokens, 'importance': importance.tolist(), 'load': load.tolist(),
                        'load_cv': float(load.std(unbiased=False)/load.mean().clamp_min(1e-12)),
                        'unused_experts': int((load == 0).sum())}
                    for key, value in routes[layer].items():
                        expected = result['final_routing'][layer][key]
                        if isinstance(value, list):
                            assert len(value) == len(expected) and all(abs(a-b) <= 1e-6 for a,b in zip(value, expected)), (name, layer, key)
                        else:
                            assert abs(value-expected) <= (0 if key in ('tokens', 'unused_experts') else 1e-6), (name, layer, key)
                assert set(routes) == set(result['final_routing']), name
                (out / 'final_routing_reevaluation.json').write_text(json.dumps({'matches_recorded': True, 'routing': routes}, indent=2))
                predictions = eval_logits.argmax(-1)
                classes = meta['classes']
                confusion = torch.bincount(eval_labels*classes+predictions, minlength=classes*classes).reshape(classes,classes)
                (out / 'final_predictions.json').write_text(json.dumps({'metrics': reproduced, 'matches_recorded': True,
                    'source_split': meta['final_source_split'], 'confusion_matrix_rows_true_columns_pred': confusion.tolist(),
                    'label_counts': torch.bincount(eval_labels, minlength=classes).tolist(),
                    'prediction_counts': torch.bincount(predictions, minlength=classes).tolist(),
                    'labels': eval_labels.tolist(), 'predictions': predictions.tolist()}, indent=2))
                del eval_logits, eval_labels, evaluation_output, evaluation_batch, evaluation_labels, predictions, confusion
                del module, aux, router_sums, sums, importance, load, routes
                seed_all(seed, c['deterministic'])
                loader = torch.utils.data.DataLoader(TensorDictDataset(data['train']), batch_size=c[stage]['batch_size'],
                                                     shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
                iterator = iter(loader)
                opt = torch.optim.AdamW(model.parameters(), lr=c[stage]['lr'], weight_decay=c[stage]['weight_decay'])
                samples, seconds, timings = 0, 0.0, []
                torch.cuda.reset_peak_memory_stats()
                model.train()
                for step in range(13):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    batch = next(iterator)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    y = batch.pop('labels')
                    opt.zero_grad(set_to_none=True)
                    logits = model(**batch)
                    loss = torch.nn.functional.cross_entropy(logits, y) + auxiliary_loss(model)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(name)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if step >= 3:
                        seconds += elapsed
                        samples += len(y)
                        timings.append(elapsed)
                row = {'stage': stage, 'task': task, 'gate': gate, 'seed': seed,
                       'source_checkpoint_step': checkpoint['step'], 'warmup_steps': 3, 'measured_steps': 10,
                       'final_reevaluation_matches': True, 'routing_reevaluation_matches': True,
                       'training_seconds': seconds, 'seconds_per_step': seconds/10,
                       'step_time_sample_std': statistics.stdev(timings), 'examples_per_second': samples/seconds,
                       'peak_cuda_mib': torch.cuda.max_memory_allocated()/2**20, 'step_seconds': timings,
                       'protocol': 'serial selected-checkpoint continuation with fresh AdamW; includes data fetch, transfer, forward/backward, clipping and optimizer; 3 warmup + 10 measured; weights discarded'}
                (out / 'measurement.json').write_text(json.dumps(row, indent=2, allow_nan=False))
                rows.append(row)
                print(json.dumps(row), flush=True)
                del model, opt, checkpoint, logits, loss, batch, y, norm
                gc.collect()
                torch.cuda.empty_cache()
with (root / 'serial_efficiency_runs.csv').open('w', newline='', encoding='utf-8-sig') as f:
    fields = [k for k in rows[0] if k != 'step_seconds']
    w = csv.DictWriter(f, fields)
    w.writeheader()
    w.writerows({k: v for k, v in r.items() if k in fields} for r in rows)
aggregate = []
for stage in ('mlp', 'bert'):
    for task in c[stage]['tasks']:
        for gate in c['gates']:
            rr = [r for r in rows if (r['stage'], r['task'], r['gate']) == (stage, task, gate)]
            a = {'stage': stage, 'task': task, 'gate': gate, 'n': len(rr)}
            for metric in ('examples_per_second', 'seconds_per_step', 'peak_cuda_mib'):
                a[metric+'_mean'] = statistics.mean(r[metric] for r in rr)
                a[metric+'_std'] = statistics.stdev(r[metric] for r in rr)
            aggregate.append(a)
(root / 'serial_efficiency_summary.json').write_text(json.dumps({'runs': rows, 'groups': aggregate}, indent=2))
with (root / 'serial_efficiency_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, list(aggregate[0]))
    w.writeheader()
    w.writerows(aggregate)
