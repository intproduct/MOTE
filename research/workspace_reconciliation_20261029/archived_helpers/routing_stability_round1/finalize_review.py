"""Sign off the completed experimental delivery, keeping prior state snapshots."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re

r=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=dt.datetime.now().astimezone().isoformat()
d=read(r/'REVIEW_DATA_AUDIT.json');f=read(r/'FIGURE_REVIEW.json')
assert d['passed'] and d['training']==d['best_checkpoint_reevaluations']==d['stability_records']==18
assert d['generation_samples']==54 and d['cpu_loaded_checkpoints']==36
assert not d['missing'] and not d['failed'] and not d['unfinished']
assert f['all_six_visually_reviewed'] and len(f['figures'])==6
for x in f['figures']:assert x['passed'] and sha(r/'figures'/x['file'])==x['sha256']
report=(r/'REPORT_zh.md').read_text(encoding='utf-8')
assert 'REVIEW_REQUIRED' not in report and '\ufffd' not in report
images=re.findall(r'!\[.*?\]\((.*?)\)',report)
assert len(images)==6 and all(Path(p).exists() for p in images)
assert not (r/'PIPELINE_LOCK.json').exists()
assert (r/'logs/formal_pipeline.err').stat().st_size==0
for name,value in d['raw_artifact_sha256'].items():assert sha(r/name)==value,name
old=read(r/'review_history/STATUS.json')
assert old['state']=='ready_for_review'
new={**old,'pipeline_state':'ready_for_review','pipeline_finished':old['finished'],
     'state':'completed','task_complete':True,'active':False,'review_finished':now,
     'report':str(r/'REPORT_zh.md'),'completion_audit':str(r/'COMPLETION_AUDIT.json'),
     'counts':{'training':18,'reevaluations':18,'serial_benchmarks':18,'stability_probes':18,'generation_samples':54,'figures_reviewed':6},
     'scientific_target_met':False,'note':'All authorized work complete; no candidate met combined quality/latency/50% flip reduction target. PID is historical, not active.'}
write(r/'STATUS.json',new);write(r/'ACTIVE_PROCESS.json',new)
delivery=read(r/'DELIVERY_CHECK.json')
delivery.update(review_pending=False,review_finished=now,task_complete=True,generation_samples=54,
                report_sha256=sha(r/'REPORT_zh.md'),missing=[],failed=[],unfinished=[])
write(r/'DELIVERY_CHECK.json',delivery)
session=f'''# MoTE 路由稳定性正式实验：已完成

**训练、核验、中文报告与六图复核全部完成。结束时间：{now}（Asia/Shanghai）。不要重启本轮或旧分类/自回归实验。**

正式组 `D:/files/MOTE/outputs/routing_stability_round1/formal/756b04397b2d`，配置 `experiments/routing_stability/configs/titan_v.json`，hash `{d['config_hash']}`。六组 × seeds17/29/43，共18个新增run；每run4000步、16,384,000采样token，WikiText-2 raw、train-only BPE4096、2层hidden256、context128、batch32、TITAN V FP32。margin τ=.05/λ=.01，前2000步关闭，3000步满强度。冻结代码/配置未改。

## 完成核验

- 18/18训练、18/18最佳checkpoint完整validation/test重评估、18/18串行训练计时、54/54推理形状计时、18/18稳定性探针、54/54固定提示生成、6/6图表逐张查看。
- 全部72,000步有限、router梯度非零，router与共享参数更新。共享初始化/专家位置/采样顺序一致，9组margin配对完整初始化相同、前2000步41个日志点CE相同。
- 36个best/last checkpoint CPU严格加载、所有tensor有限，实际参数计数一致。全部重评估和探针hash与best一致；validation重载CE差为0。
- `REVIEW_DATA_AUDIT.json`核对8份CSV与原始JSON、1350个推理计时值、360个串行训练计时值、4608条探针比较记录。缺失、失败、未完成清单为空。
- 正式pipeline于2026-10-29 06:52:59正常完成自动阶段，PID42008已退出、锁释放、stderr为空。`ACTIVE_PROCESS.json`保留历史PID，active=false。

## 实测结论

- half MLP Test PPL 79.48±1.98；MLP+margin 79.47±2.00；原Born 79.84±0.93；原Born+margin 79.85±0.92；共享Born 80.39±2.47；共享Born+margin 80.38±2.46（3seed样本SD）。
- 共享Born相对原Born的b16×128整模型前向延迟配对平均降低24.5%；但相对half MLP，PPL平均高1.135%、局部BF16输入舍入切换率平均高43.4%、串行训练step耗时高42.3%。
- margin相对各自基线局部BF16切换率：MLP下降10.3%、共享Born下降9.1%、原Born增加3.9%。没有候选达到质量/延迟/切换减半组合目标。
- MLP+margin的BF16目标logprob差>0.1事件42→64，说明局部切换减少不等于尾部改善。共享Born在BF16的部分尾部改善，但仍不优于MLP，并且batch变化和FP16下证据混合。
- 正式末期无固定top4退化；仅mlp_margin_s29第一up模块有1个未用专家。共享Born早期暂时集中已在报告披露。全部生成有明显重复/主题漂移，未宣称语言质量提升。

## 交付

- 主报告：`REPORT_zh.md`，含质量/实际参数、全部配对差、计时/显存/路由/尾部、目标核对、逐seed明细和六图解释。
- 最终签核：`COMPLETION_AUDIT.json`；数据复核：`REVIEW_DATA_AUDIT.json`；图表复核：`FIGURE_REVIEW.json`；结论原始数值：`REVIEW_FINDINGS.json`。
- `runs.csv/aggregate.csv/paired_deltas.csv/paired_summary.csv/curves.csv/routing.csv/stability.csv/error_tails.csv/raw_summary.json`；全部生成`generation_samples.json/.md`。
- 每run全部原始结果及checkpoint保留在正式组；pilot_v1/v2/v3、训练侧校准、选择记录全部保留。强正则pilot固定top4已拒绝，不计为成功。
- 报告草稿、完成前状态和原进度记录在`review_history/`；`TRAINING_COMPLETION_AUDIT.json`和`REVIEW_DATA_AUDIT.json`的task_complete=false属于当时阶段快照，最终以`COMPLETION_AUDIT.json`为准。

## 限制与后续

仅router输入FP16/BF16 roundtrip后FP32运算，不是原生低精度、跨引擎、分布式或RL训练—推理对齐；b1×1无KV cache。单小模型/语料/3seed、旧轮见过test；无无U/无CRY/经典正交正式消融，不能归因于量子或纠缠优势。共有参数量差异已报告；本轮没有full MLP或MPO组。没有按正式结果追加调参或训练。

定期跟进`mote`在最终签核后删除，删除回执见`AUTOMATION.json`。实验未新增Git提交/推送。原始数据、失败pilot和冻结源码均继续保留。若以后开展新实验，应使用独立配置与输出组；last.pt无optimizer/RNG，不支持无缝恢复。
'''
(r/'SESSION_STATE.md').write_text(session,encoding='utf-8')
artifacts=['REPORT_zh.md','generation_samples.md','generation_samples.json','FIGURE_REVIEW.json','REVIEW_FINDINGS.json',
           'REVIEW_DATA_AUDIT.json','DELIVERY_CHECK.json','STATUS.json','ACTIVE_PROCESS.json','SESSION_STATE.md',
           'raw_summary.json','runs.csv','aggregate.csv','paired_deltas.csv','paired_summary.csv','curves.csv','routing.csv','stability.csv','error_tails.csv',
           'review_audit.py','review_report.py','finalize_review.py','FORMAL_SELECTION.json','PROTOCOL_FROZEN.md','VERIFICATION.json',
           'formal/756b04397b2d/audit.json','formal/756b04397b2d/missing_failed_unfinished.json','formal/756b04397b2d/config.json','formal/756b04397b2d/frozen_sources.json',
           'logs/formal_pipeline.log','logs/formal_pipeline.err']+[f'figures/{x["file"]}' for x in f['figures']]
audit={'time':now,'state':'completed','task_complete':True,'config_hash':d['config_hash'],
       'group':'formal/756b04397b2d','new_training_runs':18,'reused_training_runs':0,
       'training':18,'checkpoint_reevaluations':18,'serial_benchmarks':18,'stability_probes':18,
       'generation_samples':54,'figures_visually_reviewed':6,'cpu_loaded_finite_checkpoints':36,
       'related_tests_preflight_passed':61,'finite_training_steps':72000,'nonzero_router_gradient_steps':72000,
       'raw_data_audit_passed':True,'report_review_passed':True,'report_has_no_placeholders':True,
       'validation_reload_max_ce_difference':0.0,'csv_row_counts':d['csv_row_counts'],
       'missing':[],'failed':[],'unfinished':[],'pipeline_finished':old['finished'],
       'no_active_experiment_worker_observed':True,'pipeline_lock_absent':True,'pipeline_stderr_empty':True,
       'training_sources_config_unchanged':True,'prior_drafts_and_states_preserved':'review_history/',
       'pilots_preserved':['pilot/5b7d1b6a5189','pilot/2c58817d07ec','pilot/1361f2f058eb'],
       'combined_engineering_target_met':False,'quantum_or_entanglement_advantage_claimed':False,
       'limitations':read(r/'REVIEW_FINDINGS.json')['limitations'],
       'artifact_sha256':{name:sha(r/name) for name in artifacts},
       'raw_run_artifact_sha256':d['raw_artifact_sha256'],
       'automation':{'id':'mote','delete_pending':True}}
write(r/'COMPLETION_AUDIT.json',audit)
print(json.dumps({k:v for k,v in audit.items() if k not in ['artifact_sha256','raw_run_artifact_sha256']},ensure_ascii=False,indent=2))
