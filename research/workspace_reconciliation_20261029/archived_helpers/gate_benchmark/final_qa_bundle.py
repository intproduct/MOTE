"""Check and package the completed experiment, excluding all model/data payloads."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from PIL import Image

root = Path('outputs/gate_benchmark').resolve()
group = root / '45c21049b433'
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')


def read_json(p):
    def reject(value):
        raise ValueError(f'Nonfinite JSON value {value} in {p}')
    return json.loads(p.read_text(encoding='utf-8-sig'), parse_constant=reject)


audit = read_json(group/'audit.json')
assert audit['passed'] and audit['completed_audited'] == 54 and not audit['issues']
assert len(list(group.glob('*/result.json'))) == 54
assert len(list(group.glob('*/best.pt'))) == 54
summary = read_json(root/'summary.json')
assert not summary['missing_runs'] and not summary['incomplete_runs']
eff = read_json(root/'serial_efficiency_summary.json')
assert len(eff['runs']) == 54 and len(eff['groups']) == 18
assert all(x['final_reevaluation_matches'] and x['routing_reevaluation_matches'] for x in eff['runs'])
assert len(list((root/'serial_efficiency').glob('*/measurement.json'))) == 54
assert len(list((root/'serial_efficiency').glob('*/final_predictions.json'))) == 54
assert len(list((root/'serial_efficiency').glob('*/final_routing_reevaluation.json'))) == 54
counts = {}
for name, expected in [('formal_runs_extended.csv',54), ('formal_aggregate.csv',18),
                       ('formal_paired_deltas.csv',54), ('serial_efficiency_runs.csv',54),
                       ('serial_efficiency_summary.csv',18), ('learning_diagnostics.csv',18)]:
    with (root/name).open(encoding='utf-8-sig', newline='') as f:
        count = len(list(csv.DictReader(f)))
    assert count == expected, (name,count,expected)
    counts[name] = count
with (root/'formal_paired_deltas.csv').open(encoding='utf-8-sig',newline='') as f:
    paired_rows = list(csv.DictReader(f))
assert sum(r['gate'] != 'mlp' for r in paired_rows) == 36
assert sum(r['gate'] == 'mlp' for r in paired_rows) == 18
assert all(float(r['final_score_delta_pp']) == 0 for r in paired_rows if r['gate']=='mlp')
report = (root/'REPORT_zh.md').read_text(encoding='utf-8')
assert '\ufffd' not in report
assert '54/54' in report and 'validation' in report and 'ddof=1' in report
figure_links = re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)',report)
assert len(figure_links) == 11 and len(set(figure_links)) == 11
for link in figure_links:
    with Image.open(root/link) as img:
        assert img.width >= 1000 and img.height >= 500
        img.verify()
commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
branch = subprocess.check_output(['git','branch','--show-current'],text=True).strip()
status = subprocess.check_output(['git','status','--short'],text=True)
assert commit == '807366cb8afc58b1fc37086f998ae791e8bb16a6' and branch == 'mote-quantum'
assert not status.strip(),status
status_record = read_json(root/'STATUS.json')
status_record.update(time=now, phase='complete', report='outputs/gate_benchmark/REPORT_zh.md',
                     independent_final_reevaluations=54, isolated_efficiency_measurements=54,
                     audit_passed=True, failed_runs=[], missing_runs=[], incomplete_runs=[])
(root/'STATUS.json').write_text(json.dumps(status_record,indent=2,ensure_ascii=False),encoding='utf-8')
phase = {'phase':'complete','updated_at':now,'driver_pid':None,'completed_runs':54,'expected_runs':54,
         'report':'outputs/gate_benchmark/REPORT_zh.md','status_file':'outputs/gate_benchmark/STATUS.json'}
(root/'logs/active_phase.json').write_text(json.dumps(phase,indent=2),encoding='utf-8')
with (root/'logs/phase_history.jsonl').open('a',encoding='utf-8') as f:
    f.write(json.dumps(phase)+'\n')
log=root/'RUN_LOG_zh.md'
with log.open('a',encoding='utf-8') as f:
    f.write('\n\n## 默认矩阵与交付完成\n\n')
    f.write('- 08:20:11：正式矩阵 54/54 全部完成，43,200 个计划训练步，所有子进程退出码 0；无失败、OOM、缺失或中断 run。MLP 27 组完成后才运行 BERT 27 组。\n')
    f.write('- 完整独立审计 54/54 通过：有限值、梯度、共享初始化及数据 hash、数据划分、参数更新、选模与文件完整性均通过。所有 gate 参数张量在最佳 checkpoint 中均相对初态发生更新。\n')
    f.write('- 08:38:37：完成 54 个最佳 checkpoint 的串行重新评估及单训练进程效率诊断；accuracy/F1/score 精确复现，逐层路由统计容差内一致。此处“单进程”指本实验同时只有一个训练进程，桌面 GPU 进程始终保留，并非完全独占设备。\n')
    f.write('- 11 张图已逐张目视核验，完整中文报告、配对差值、均值/样本 SD、学习曲线、路由向量、逐样本预测和混淆矩阵均已保存。仓库 summarize 与正式独立 CSV/JSON 均保留。\n')
    f.write('- 交付校验首次误把配对 CSV 预期行数写为 36；文件实际包含 36 条 Born/MPS 对照与 18 条 MLP 自身零差值，共 54 行。保留 final_delivery_attempt1.log，修正校验器并分别验证行数及 MLP 零差值；未修改或删除原始结果。\n')
    f.write('- 依赖最终检查通过，Git 仍处于原 mote-quantum commit 且工作区干净。仅停止本任务创建的 nvidia-smi 遥测进程 PID 50260，未终止其他用户进程。\n')
    f.write(f'- 交付核验时间：{now}。正式训练约 5.77 小时，包含准备与复核总计约 7 小时；没有以缩减默认矩阵换取规划时限。轻量交付包排除模型、数据、环境与大型 profiler trace。\n')
qa = {'passed':True, 'time':now, 'completed_formal_runs':54, 'final_reevaluation_matches':54,
      'routing_reevaluation_matches':54,'figures_verified':11,'figure_visual_review':'11/11 reviewed by assistant',
      'csv_rows':counts,'git_commit':commit,'git_branch':branch,'git_status_short':status,
      'failed_runs':[],'missing_runs':[],'incomplete_runs':[]}
(root/'FINAL_QA.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')

# Explicit file whitelist: no .pt weights, cache, datasets, environment or profile trace.
selected = {}
def add(path, archive_name=None):
    assert path.is_file(),path
    assert path.suffix.lower() not in ('.pt','.pth','.safetensors','.bin','.zip'),path
    archive_name = archive_name or path.relative_to(root).as_posix()
    selected[archive_name] = path

for name in ['REPORT_zh.md','RUN_LOG_zh.md','FINAL_QA.json','STATUS.json','runs.csv','summary.json',
             'formal_runs_extended.csv','formal_aggregate.csv','formal_paired_deltas.csv','formal_analysis.json',
             'formal_dev_curves.csv','formal_routing_per_layer.csv','serial_efficiency_runs.csv',
             'serial_efficiency_summary.csv','serial_efficiency_summary.json','learning_diagnostics.csv',
             'learning_diagnostics.json','audit_results.py','analyze_results.py','serial_efficiency.py',
             'plot_serial_efficiency.py','make_report.py','learning_diagnostics.py','final_qa_bundle.py']:
    add(root/name)
for link in figure_links:
    add(root/link)
for folder in ['45c21049b433','7d5c8d3cbd27']:
    for pattern in ['config.json','audit.json','exact_mask_audit.json','execution.jsonl','execution_complete.json',
                    '*/manifest.json','*/initial_dev.json','*/curves.jsonl','*/result.json']:
        for p in (root/folder).glob(pattern):
            add(p)
for p in (root/'serial_efficiency').glob('*/*.json'):
    add(p)
for p in (root/'logs').iterdir():
    if p.is_file() and p.suffix.lower() in ('.log','.txt','.json','.csv','.jsonl','.ps1') and p.name not in ('step_profile_trace.json','final_delivery.log'):
        add(p)
for name in ['README.md','configs/pilot.json','configs/titan_v.json','requirements-titan-v.txt']:
    p=Path('experiments/gate_benchmark')/name
    if p.is_file():
        add(p,'protocol/'+name)
p=Path('outputs/gate_parallel_pilot/8463c90d231e/serial_equivalence.json')
add(p,'parallel_pilot/serial_equivalence.json')
manifest = {'created_at':now, 'git_commit':commit, 'note':'No model weights, raw datasets, environment, or profile trace included.',
            'files':[{'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for name,p in sorted(selected.items())]}
(root/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
bundle=root/'gate_benchmark_results_20261027.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(selected.items()):
        z.write(p,name)
    z.write(root/'DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.json')
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(selected)+1
print(json.dumps({'qa':qa,'bundle':str(bundle),'bundle_bytes':bundle.stat().st_size,'file_count':len(selected)+1},ensure_ascii=False,indent=2))
