import os
from pathlib import Path
os.environ.update(HF_HOME=str(Path('outputs/hf_cache').resolve()), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import torch
from fitmotn.experiments.gate_benchmark.config import load_config, digest
from fitmotn.experiments.gate_benchmark.data import load_data, bert_path
from fitmotn.experiments.gate_benchmark.models import BertClassifier, routed_modules
from fitmotn.experiments.gate_benchmark.runner import seed_all, evaluate
c = load_config('experiments/gate_benchmark/configs/pilot.json')
root = Path(c['output_dir']) / digest(c)[:12]
data, meta = load_data(c, 'sst2')
rows = []
for gate in c['gates']:
    seed_all(17)
    model = BertClassifier(c, gate, 17, 2, model_path=bert_path(c)[0]).cuda()
    ckpt = torch.load(root / f'bert_sst2_{gate}_seed17/best.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(ckpt['model'])
    counts = {'tokens_across_layers': 0, 'assignments': 0, 'violations': 0, 'batches_across_layers': 0}
    def check(module, inputs, output):
        mask = output[1]
        per_token = (mask > 0).sum(-1)
        counts['tokens_across_layers'] += mask.shape[0]
        counts['assignments'] += int(per_token.sum())
        counts['violations'] += int((per_token != c['topk']).sum())
        counts['batches_across_layers'] += 1
    for _, m in routed_modules(model):
        m.mote.core.gate.register_forward_hook(check)
    score, route = evaluate(model, data['test'], 'sst2', c['bert']['batch_size'], torch.device('cuda'))
    original = json.loads((root / f'bert_sst2_{gate}_seed17/result.json').read_text())
    assert score == original['final']
    assert counts['violations'] == 0 and counts['assignments'] == counts['tokens_across_layers'] * c['topk']
    rows.append({'gate': gate, **counts, 'reported_drop_rates': [v['drop_rate'] for v in route.values()], 'reevaluated_final_exact_match': True})
    print(json.dumps(rows[-1]), flush=True)
    del model, ckpt
    torch.cuda.empty_cache()
(root / 'exact_mask_audit.json').write_text(json.dumps(rows, indent=2))
