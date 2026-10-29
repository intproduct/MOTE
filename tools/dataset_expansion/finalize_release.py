"""Publish the validated local release, manifests and plain-language handoff."""
import collections,hashlib,json,shutil,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'Datasets/expansion_20261016';DEST=ROOT/'Datasets/processed/shared_sft_v02_20261016'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 m=load(DEST/'audit/merge_summary.json');v=load(DEST/'audit/validation_report.json');assert v['passed'] and v['rows']==m['input_records'];assert m['input_records']==1047116,'Selected source row total differs from the frozen collection';assert not m['errors'];files=m['files'];n=sum(files.get(x,0) for x in ['train','validation','test','smoke'])
 # Idempotently align the searchable index with family exclusions in the exported files.
 db=sqlite3.connect(DEST/'audit/index.sqlite');db.execute("UPDATE records SET status='quarantine' WHERE final_split='excluded' AND status!='duplicate'");db.commit();actual=dict(db.execute("SELECT CASE WHEN status='eligible' THEN final_split WHEN status='duplicate' THEN 'duplicates' ELSE status END, COUNT(*) FROM records GROUP BY 1"));assert actual==files,(actual,files)
 source_stats=[{'source':s,'status':status,'split':split,'records':count,'distinct_questions':questions,'connected_families':families} for s,status,split,count,questions,families in db.execute('SELECT source,status,final_split,COUNT(*),COUNT(DISTINCT problem_hash),COUNT(DISTINCT group_root) FROM records GROUP BY source,status,final_split')];db.close();write(DEST/'audit/source_question_counts.json',source_stats)
 shutil.copyfile(ROOT/'tools/dataset_expansion/schema_v2.json',DEST/'schema.json');shutil.copyfile(EXP/'DATA_CONTRACT_zh.md',DEST/'DATA_CONTRACT_zh.md');shutil.copyfile(EXP/'release_config.json',DEST/'audit/release_config.json')
 policy=load(EXP/'source_admission_policy.json');write(DEST/'audit/source_admission_policy.json',policy)
 scripts=[]
 for p in sorted((ROOT/'tools/dataset_expansion').rglob('*.py')):scripts.append({'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
 write(DEST/'audit/processing_scripts_manifest.json',{'scope':'Final scripts at publication; recovery from an earlier compatible normalization pass, if any, is described by resume audit.','files':scripts})
 dispositions=[]
 for p in sorted((DEST/'audit').glob('applied_*findings.json')):
  for r in load(p).get('applied',[]):dispositions.append({'id':r['id'],'action':r['action'],'applied_release_status':r['applied_release_status'],'classification':r.get('classification'),'reason':r.get('reason_zh') or r.get('conclusion_zh'),'input_sha256':r['input_sha256'],'output_sha256':r['output_sha256'],'audit_file':str(p.relative_to(DEST)).replace('\\','/')})
 with (DEST/'audit/record_dispositions.jsonl').open('w',encoding='utf-8') as f:
  for r in dispositions:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 domains=m['domain_counts'].get('train',{});names={'math':'数学','code':'代码','logic':'逻辑','reading':'阅读理解','physics':'物理','instruction':'指令遵循','instruction_following':'指令遵循','writing':'写作','general_chat':'通用对话','science_grounded':'有上下文的基础科学','chemistry':'化学','biology':'生物','stem':'其他 STEM','language':'语言教育','social_science':'社会科学','science_foundation':'科学基础','earth_science':'地球科学'}
 domain_lines='\n'.join(f'- {names.get(k,k)}：{count:,} 条训练记录。' for k,count in sorted(domains.items(),key=lambda x:-x[1]))
 tiers=collections.Counter()
 for split in ['train','validation','test','smoke']:tiers.update(m['quality_tiers'].get(split,{}))
 pilot=load(DEST/'inspection/manifest.json');views=load(DEST/'audit/domain_views_manifest.json')
 readme=f'''# 共享 SFT 数据母库 v2（2026-10-16）

本轮已完成来源固定、内容筛查、统一 JSONL、跨来源合并、题族划分、基准排除和逐条格式/完整性校验。共整理 **{m['input_records']:,} 条来源记录**；其中 **{n:,} 条进入按证据分级的可用候选**，包括训练 **{files.get('train',0):,}**、验证 **{files.get('validation',0):,}**、测试 **{files.get('test',0):,}**、smoke **{files.get('smoke',0):,}**。这些是项目内部保留集，不是标准测评成绩。

“可用候选”表示通过本轮明确的门禁，**不表示每条答案或每一步长推理都被独立证明正确**。来源判题、参考匹配、本地独立重算、人工撰写和逐条可见内容审查分别记录。没有为了凑数量把缺乏证据的科学/聊天数据混入训练。

本轮逐条内容复核由 Agent 执行，并保存具体错例与验算依据；审计文件中的 manual 命名不表示人类专家认证。`human_authored` 表示来源中的人类示范，与本轮复核方式分开。

## 先打开哪些文件

- `jsonl/train.jsonl`：总训练候选，完整消息与回答。
- `by_domain/<领域>/train.jsonl`：同一训练库按领域拆分，便于后续配比例；不要与总文件重复相加。
- `inspection/stratified_train_sample.jsonl`：{pilot['rows']:,} 条完整抽样，便于快速检查；不是新的独立数据，也不是推荐训练比例。
- `transfer/shared_sft_v02_training.zip`：验证后生成的便携包，含训练/保留集与完整抽样，不含庞大的待审、隔离和原始下载池。
- `DATA_CONTRACT_zh.md`：怎样读取 messages、监督掩码、推理、final、许可及证据分级。
- `../../expansion_20261016/SOURCES_zh.md`：今晚搜集的公开来源、官方链接、采集范围及接纳/暂缓理由。
- `audit/validation_report.json`、`audit/merge_summary.json`：全量校验、数量、领域、长度和排除原因。

本轮新增数据、缓存与结果均保存在本项目 D 盘 `Datasets` 内。上一版和原始数据保留。

公开大库采用固定版本、明确分片/前缀的有界采集，不代表把每个来源全集下载完毕。实际选择范围和哈希留在各来源下载清单；本轮审查结论针对取得的数据，不能推广成整份上游数据集的准确率。

## 按领域的训练数量

{domain_lines}

数量是导出记录数，独立问题/题族另见 `audit/validation_report.json` 的 `distinct_questions_and_families`。比如同一篇阅读文章可以包含多道问题，同一逻辑理论可以有多个真假未知查询；不能把它们都算成独立背景。

## 已落实的质量处理

数学保留完整 CoT 和最终交付，检查最终 LaTeX 结构、答案缺失、题干泄题和题意修补风险。对可处理的最终表达式另做符号等价检查；无法证等价仅作待审，不等同于证明错误。逐条内容复核发现并隔离了推理结束但 final 截断的样本，以及题干自带答案的样本。

本轮取得的 NVIDIA Nemotron-SFT-Math-v4 分片整体留在待审区：新增门禁之后的20条诊断样本仍暴露明确错误数学断言、构造条件遗漏与无效证明步骤。因此未将这批大规模长推理计入默认训练候选。具体逐条记录见数学审查文件；这不是对整个上游库的错误率估计。来源级决定见 `audit/source_admission_policy.json`，即使某条记录自身没有额外标志，来源级暂缓也依然生效。

旧版本地 OpenMathReasoning 分片也重新接受了同样审查：10条诊断样本中，2条最终讲解有明确错误，3条证明需要复核，故该收集分片整体待审；不是断言整个上游库或每条记录都错。官方来源已按原文件 SHA256 确认。此次默认数学候选主要来自经完整性、答案参考、污染和具体错例门禁筛查的 OpenR1；也不把它的未抽查推理宣称为全部正确。

代码除发布方测试通过外，还核对题目恢复来源、原 train 划分、答案代码绑定、Python 语法、有效断言证据及部分题面示例冲突。同题不同呈现按源题族处理。**本机没有重新执行所有来源代码**，因此不把来源执行通过写成本地重新通过。

逻辑以独立开放世界规则闭包重算标签；阅读保留完整给定上下文，并核对源答案跨度。指令数据保留 system/历史对话及源监督掩码；严格格式任务的辅助推理不会被插到用户可见 JSON/XML 正文前面。

科学领域从严：OpenScienceReasoning-2 的 20 条诊断抽样发现 8 条明确实质缺陷，另 6 条未决，因此本轮该部分全部留待审。该数字不是全库错误率。TextbookReasoning、PhysicsEval、缺独立参考的 Nemotron 科学合成数据也不因品牌或 boxed 对上就自动放行。可用基础科学以筛查后的 SciQ、ScienceQA 和从官方 OpenStax 恢复的训练题为主；不足的物理数量没有用低质长 CoT 补齐。

## 待审与隔离不是训练文件

- `jsonl/review_required.jsonl`：{files.get('review_required',0):,} 条，缺少充分证据、存在未决问题或需要专业复核。
- `jsonl/quarantine.jsonl`：{files.get('quarantine',0):,} 条，存在内容/完整性风险、原评测来源或可能的基准重合等。
- `jsonl/duplicates.jsonl`：{files.get('duplicates',0):,} 条重复/备选呈现，保留追溯，不用于虚增独立题目数量。

这些文件也使用相同 schema，但不能仅因为格式正确就回流训练。父目录 `expansion_20261016/*/candidates.jsonl` 和科学 `normalized/` 是各处理阶段的中间文件，中央门禁之前的“候选”不等于最终训练集。

合并期间追加的内容复核处分另见 `audit/record_dispositions.jsonl`，按 ID 和输入/输出 SHA 关联；来源整体暂缓见 `audit/source_admission_policy.json`。判断可用性应先看 `release_status` 和文件位置，不能只看参考匹配字段或单个 `review_flags` 列表。

## 划分与校验边界

基准排除索引包含 {m['benchmark_index_rows']:,} 条记录，涵盖具体下载版本的 GSM8K、MATH、HumanEval、MBPP、MMLU、MMLU-Pro、IFEval、ARC、HellaSwag、AIME、BigCodeBench 和 LiveCodeBench。采用规范化匹配、包含匹配、保守近似与题族传播；没有声称能识别所有语义改写或未知测评集。

完整性检查覆盖每条导出记录的 schema、唯一 ID、原消息字段哈希、转换可逆性、监督角色、候选内容重复及训练/保留集题族隔离。长度报告单位为**字符**，不是 token。没有执行 tokenizer、截断、packing、聊天模板转换或模型训练。

## 下一步

先用完整抽样检查学生所需的任务类型，再按 Gemma3、Llama3.1、Qwen3、Qwen3.5 各自 tokenizer 和原生模板统计长度。装不下的完整样本应单列或重新生成精炼完整解答，不能裁掉答案和结束部分。数据母库的自然数量比例不是训练配方；后续需要明确领域比例、token 预算和长度策略。

科学补充还准备了 2,000 条独立解答队列，位于 `../../expansion_20261016/science/teacher_queue/benchmark_filtered/`。只把 `physics_blind_teacher.filtered.jsonl` 交给 teacher，参考证据 sidecar 留给验证端。队列通过本轮基准索引检查；尚未调用模型生成，也不计入训练数量。提示词要求先检查题目可解性，允许返回 invalid/ambiguous；再验证单位、数值、边界及结论，不能为了匹配参考答案改题。

本库保留逐条原许可与归属，没有统一换成一个许可证。非商业来源和许可未声明的来源可以单独筛选。正式使用请依据记录与源数据说明；本轮没有推断一个不存在的统一许可。

## 溯源和复现

数学：`../../expansion_20261016/math/audit/`；代码：`../../expansion_20261016/code/README_zh.md`；通用：`../../expansion_20261016/general/README_zh.md`；科学：`../../expansion_20261016/science/README_zh.md`。这里包括固定 revision、原始下载/投影证据、SHA、处理脚本和实际错例审查。

处理代码位于项目 `tools/dataset_expansion/`。中央配置见 `audit/release_config.json`；最终文件哈希见 `MANIFEST.json`。原数据只读保留，所有清洗通过新版本文件呈现。
'''
 (DEST/'README_zh.md').write_text(readme,encoding='utf-8')
 (ROOT/'Datasets/processed/local_sft_v01_20261016/STATUS_20261016.md').write_text('旧版本保留用于复现。新一轮复审已将该版本使用的本地 OpenMathReasoning 单分片整体转入待审，并隔离具体错误记录；不能沿用旧版 eligible 标签判断当前可训练性。请以 ../shared_sft_v02_20261016/README_zh.md 和新版 jsonl/train.jsonl 为准。原始旧版文件未改写。\n',encoding='utf-8')
 manifest={'release':'shared_sft_v02_20261016','schema_version':'mote-sft-jsonl-2.0','status':'validated','input_records':m['input_records'],'eligible_records':n,'eligible_evidence_tiers':dict(tiers),'canonical_files':v['output_files'],'domain_views':views,'inspection_subset':pilot,'quality_is_evidence_graded_not_universal_correctness':True,'tokenizer_processing_performed':False,'source_evidence_directories':['../../expansion_20261016/math/audit','../../expansion_20261016/code','../../expansion_20261016/general','../../expansion_20261016/science'],'input_manifest':m['input_files']};write(DEST/'MANIFEST.json',manifest)
 (ROOT/'Datasets/开始使用_20261016.md').write_text(f'''# 本轮数据入口\n\n最新已校验版本：`processed/shared_sft_v02_20261016/README_zh.md`。\n\n默认训练候选共 {files.get('train',0):,} 条，统一 JSONL，完整消息和回答；按领域文件位于 `by_domain/`。先看该版本的说明和 {pilot['rows']:,} 条完整抽样。\n\n`expansion_20261016` 保存原始下载、阶段性转换、审查和隔离资料，其中名称为 candidates 的阶段文件也可能尚未通过中央去重/去污染，不要直接混入训练。\n\n本轮未做 tokenizer 切分、截断或模型训练。质量按证据分级，不宣称所有长推理都正确。\n''',encoding='utf-8')
 print(json.dumps({'published':True,'path':str(DEST),'train':files.get('train',0),'eligible':n,'canonical_rows':v['rows'],'inspection':pilot['rows']}))
if __name__=='__main__':main()
