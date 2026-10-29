"""Seal the reviewed delivery; no training/evaluation artifact is modified."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GROUP = ROOT / 'formal/9b3af7494afc'
NOW = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

assert not (ROOT / 'COMPLETION_AUDIT.json').exists(), 'Already sealed; inspect before modifying.'
assert not (ROOT / 'PIPELINE_LOCK.json').exists()
assert (ROOT / 'logs/formal_pipeline.err').stat().st_size == 0
status = read(ROOT / 'STATUS.json')
assert status['state'] == 'ready_for_review'
cfg = read(GROUP / 'config.json')
assert cfg == read(REPO / 'experiments/autoregressive_gate/configs/titan_v.json')
assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == status['config_hash']
frozen = read(GROUP / 'frozen_sources.json')
assert all(sha(REPO / p) == h for p, h in frozen.items())
audit = read(GROUP / 'audit.json')
assert audit['passed'] and audit['complete'] == audit['expected'] == 18 and not audit['problems']
missing = read(GROUP / 'missing_failed_unfinished.json')
assert not missing['training'] and not missing['missing_evaluation']
review = read(ROOT / 'REVIEW_AUDIT.json')
assert review['passed'] and all(review[k] == 18 for k in ['training_complete', 'checkpoint_reevaluated', 'serial_benchmarks'])
assert review['generation_samples'] == 54 and review['checkpoint_files_checked_on_cpu'] == 36
assert not any(review[k] for k in ['missing', 'failed', 'unfinished'])
for pilot in ['pilot/782e3576b60e', 'pilot/72a93c68e9ab']:
    p = read(ROOT / pilot / 'audit.json')
    assert p['passed'] and p['complete'] == 6
verification = read(ROOT / 'VERIFICATION.json')
assert verification['tests']['passed'] == 177 and verification['tests']['failed'] == 0
report = (ROOT / 'REPORT_zh.md').read_text(encoding='utf-8')
assert 'REVIEW_REQUIRED' not in report and '## 5. 实测解释' in report
assert '最终软路由概率' in report and '没有未修复的执行错误' in report

expected_rows = {'runs.csv': 18, 'aggregate.csv': 6, 'paired_deltas.csv': 30,
                 'paired_summary.csv': 10, 'curves.csv': 1620, 'routing.csv': 72,
                 'sample_variances.csv': 6, 'validation_thresholds.csv': 18}
row_counts = {}
for name, expected in expected_rows.items():
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
        row_counts[name] = len(list(csv.DictReader(f)))
    assert row_counts[name] == expected, (name, row_counts[name])

figure_notes = {
    'learning_steps.png': '完整轨迹与后期放大面板均已逐张查看；步数/CE坐标、实测点、样本SD及图例可读。',
    'learning_time.png': '完整与放大面板已查看；注明同一步的种子平均时间，不伪称固定秒数重评估；训练耗时排除验证保存。',
    'parameter_efficiency.png': '实际路由参数横轴，测试PPL及样本SD纵轴；full/half标签分列，未断轴。',
    'serial_efficiency.png': '串行端到端训练计时，5预热+20测量；单位及误差条可读，未解释为推理速度。',
    'paired_differences.png': '同种子配对CE差及未校正95% t区间；全部区间跨零，未据小均值差宣称显著优势。',
}
figures = []
for name, note in figure_notes.items():
    p = ROOT / 'figures' / name
    assert p.stat().st_size > 1000 and p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    figures.append({'file': f'figures/{name}', 'sha256': sha(p), 'passed': True,
                    'visual_inspection': note})
write('VISUAL_REVIEW.json', {'recorded_at': NOW, 'reviewer': 'Codex agent',
      'method': 'Each final PNG opened with view_image and visually inspected during this review turn.',
      'passed': True, 'figures': figures,
      'preserved_originals': 'review_history/automatic_20261028_1927/',
      'changes': 'Only plotting colors, label positions and full/late learning-curve panels; no raw measurements changed.'})

# Keep the initial source snapshot intact; retain the reviewed analysis separately.
reviewed_code = ROOT / 'review_history/reviewed'
reviewed_code.mkdir(exist_ok=True)
shutil.copy2(REPO / 'experiments/autoregressive_gate/analysis.py', reviewed_code / 'analysis.py')
review.update(report_and_visual_review_pending=False, report_review_completed_at=NOW,
              reviewer='Codex agent', report_sha256=sha(ROOT / 'REPORT_zh.md'),
              visual_review='VISUAL_REVIEW.json')
write('REVIEW_AUDIT.json', review)
delivery = read(ROOT / 'DELIVERY_CHECK.json')
delivery.update(human_review_pending=False, review_completed_at=NOW,
                reviewer='Codex agent (no separate human approval claimed)',
                report_and_figures_reviewed=True, missing=[], failed=[], unfinished=[])
write('DELIVERY_CHECK.json', delivery)

artifact_names = [*row_counts, 'raw_summary.json', 'generation_samples.json', 'REPORT_zh.md',
                  'REVIEW_METRICS.json', 'REVIEW_AUDIT.json', 'VISUAL_REVIEW.json',
                  'DELIVERY_CHECK.json', 'VERIFICATION.json', 'dependencies.txt',
                  'final_review.py', 'write_reviewed_report.py', 'complete_review.py',
                  'formal/9b3af7494afc/audit.json', 'formal/9b3af7494afc/config.json',
                  'formal/9b3af7494afc/frozen_sources.json',
                  'formal/9b3af7494afc/missing_failed_unfinished.json',
                  'review_history/reviewed/analysis.py']
completion = {
    'state': 'complete', 'completed_at': NOW, 'passed': True,
    'base_git_commit': 'c7d6690dfb13f2bf0a00ae145020ba7b0bb222fd',
    'branch': 'mote-quantum', 'new_autoregressive_code_committed': False,
    'group': 'formal/9b3af7494afc', 'config_hash': status['config_hash'],
    'training_complete': 18, 'checkpoint_reevaluated': 18, 'serial_benchmarks': 18,
    'generation_samples': 54, 'checkpoint_files_checked_on_cpu': 36,
    'pilot_complete': {'batch8': '6/6', 'batch32': '6/6'}, 'tests_passed': 177,
    'training_steps_per_run': 4000, 'training_tokens_per_run': 16384000,
    'validation_reload_max_ce_abs_error': max(x['validation_reload_ce_abs_difference'] for x in review['runs']),
    'frozen_sources_unchanged': True, 'raw_measurements_unchanged': True,
    'shared_init_expert_positions_batch_order_consistent': True,
    'independent_aggregate_and_pairs_recomputed': True,
    'report_reviewed': True, 'figures_reviewed': figures,
    'csv_rows': row_counts, 'missing': [], 'failed': [], 'unfinished': [],
    'unresolved_execution_errors': [],
    'delivery_helper_corrections': ['Initial CSV row assertion expected six aggregate threshold rows; file correctly contains eighteen per-run rows. Expected count corrected without changing CSV or training artifacts.'],
    'research_limitations': ['Three seeds, one small English corpus, small model and context128.',
        'All best checkpoints at step4000; not fully converged capacity comparison.',
        'BPE-token PPL is not word-level PPL; generated samples are repetitive.',
        'No isolated entanglement/chi ablation or quantum hardware advantage test.',
        'Short serial training benchmark on a Windows desktop GPU; no inference/profiler claim.',
        'Router parameter reduction is a much smaller total-model reduction.',
        'last.pt lacks optimizer/RNG; cannot claim seamless resume.'],
    'pipeline_started': status['started'], 'pipeline_finished': status['finished'],
    'artifacts_sha256': {name: sha(ROOT / name) for name in artifact_names},
}
write('COMPLETION_AUDIT.json', completion)
for name in ['STATUS.json', 'ACTIVE_PROCESS.json']:
    state = read(ROOT / name)
    state.update(state='complete', review_completed_at=NOW, process_running=False,
                 completion_audit='COMPLETION_AUDIT.json', report='REPORT_zh.md')
    write(name, state)
session = ROOT / 'SESSION_STATE.md'
text = session.read_text(encoding='utf-8')
banner = '\n**已完成：18/18 训练、18/18 完整 validation/test 重评估、18/18 串行计时、54 条生成、中文报告与五张图全部核验通过。不要重启训练或再次运行自动 analysis 主入口覆盖已复核报告。**\n'
text = text.replace('# MoTE 自回归实验执行状态\n', '# MoTE 自回归实验执行状态\n' + banner, 1)
text += f'\n{NOW}：最终报告与交付核验完成，见 REPORT_zh.md、COMPLETION_AUDIT.json、REVIEW_AUDIT.json、VISUAL_REVIEW.json。失败/缺失/未完成均为0；控制器和训练/评估进程已正常退出，锁已释放。冻结训练源码与配置未改变；仅分析绘图和中文解释经过复核修订，旧版保留在 review_history/automatic_20261028_1927/，当前分析副本在 review_history/reviewed/。新增自回归代码仍未提交，本轮未收到新的提交请求；所有大型产物均留在被忽略的 outputs。下一步仅删除已完成的 mote 自动跟进并通知用户。\n'
session.write_text(text, encoding='utf-8')
print(json.dumps({'state': 'complete', 'completed_at': NOW, 'csv_rows': row_counts,
                  'report_sha256': sha(ROOT/'REPORT_zh.md')}, ensure_ascii=False, indent=2))
