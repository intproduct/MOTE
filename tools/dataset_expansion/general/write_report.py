"""Create auditable source inventory and delivery manifest after successful full validation."""
import json,hashlib
from pathlib import Path
from download import ROOT,BASE,RAW,AUDIT
from validate import file_sha

def main():
 validation=json.loads((AUDIT/'final_validation.json').read_text(encoding='utf-8'))
 conversion=json.loads((AUDIT/'conversion_summary.json').read_text(encoding='utf-8'))
 downloads=[]
 for p in sorted(AUDIT.glob('*.download.json')):
  m=json.loads(p.read_text(encoding='utf-8'));local=m.get('local_path')
  if local:
   path=ROOT/local
   if not path.exists():path=RAW/Path(local).name
  elif 'proofwriter' in p.name:path=RAW/'proofwriter-dataset-V2020.12.3.zip'
  else:continue
  m['hash_verification_scope']='download_time_not_rescanned';m['actual_bytes']=path.stat().st_size
  if m.get('repo'):
   stem=m['repo'].split('/')[-1];metadata=AUDIT/(stem+'.metadata.json');card=AUDIT/(stem+'.README.md')
   assert json.loads(metadata.read_text(encoding='utf-8'))['sha']==m['revision']
   m['source_metadata_sha256']=file_sha(metadata)
   if card.exists():m['source_card_sha256']=file_sha(card)
  downloads.append(m)
 (AUDIT/'source_download_inventory.json').write_text(json.dumps(downloads,ensure_ascii=False,indent=2),encoding='utf-8')
 levels={
 'source_verified':'Source reports instruction validation/model judging (Nemotron v1 IF) or human absolute ratings (HelpSteer review pool). This does NOT mean local semantic correctness or full reasoning proof. Inspect verification.method.',
 'source_curated':'Publisher SFT curation (Nemotron v3) OR human-authored demonstrations (NoRobots) OR explicitly spot-reviewed HelpSteer. Distinguish verification.method; publisher-only synthetic v3 should default to review.',
 'reference_matched':'SQuAD train human reference span text matches supplied context at the published character offset. All exported reference spans were checked, but question/answer semantic annotation correctness is inherited.',
 'independently_recomputed':'ProofWriter OWA final label independently recomputed with finite Horn-rule closure, explicit negation, and published gold/closure agreement. Natural-language/formal representation correspondence is inherited; six visible examples manually checked.'}
 ready={'status':'READY','files':validation['files'],'candidate_rows_before_central_policy':validation['candidate_total'],'candidate_distinct_family_ids':validation['candidate_distinct_family_ids'],'candidate_sources':validation['candidate_sources'],'candidate_domains':validation['candidate_domains'],'quality_level_definitions':levels,'license_policy':{'AI2/ProofWriter':'rights_pending; official archive has no declared dataset license; do not infer code-repo license','HuggingFaceH4/no_robots':'cc-by-nc-4.0; noncommercial','rajpurkar/squad':'cc-by-sa-4.0; preserve attribution/share-alike obligations','nvidia/Nemotron-Instruction-Following-Chat-v1':'odc-by-1.0 for chat_if rows; inspect row-level license','nvidia/HelpSteer2':'cc-by-4.0','nvidia/HelpSteer3':'cc-by-4.0'},'source_curated_default_review':['nvidia/Nemotron-SFT-Instruction-Following-Chat-v3'],'source_curated_distinct_evidence':{'HuggingFaceH4/no_robots':'Human-authored demonstrations, not synthetic source-only curation; semantic spot-check failures still isolated.','nvidia/HelpSteer2':'Only two manually accepted visible examples exported as candidates.','nvidia/HelpSteer3':'Only three manually accepted visible examples exported as candidates.'},'training_representation':'messages authoritative; preserve system and all previous turns; supervise only metadata.assistant_loss_mask or final assistant; reasoning_content separate for constrained instruction tasks','tokenizer_or_length_cutting_performed':False,'all_candidate_records_schema_and_self_hash_checked':True,'source_roundtrip_scope':'first_16_candidates_per_source; originals retained for central checks','validation_report':'audit/final_validation.json','raw_download_inventory':'audit/source_download_inventory.json','manual_audits':['audit/manual_content_reviews.json','audit/helpsteer_manual_reviews.json','audit/proofwriter_manual_reviews.json'],'unit_tests_passed':27,'mutations_after_ready':'none'}
 lines=['# 通用、阅读与逻辑数据增补审计','',
 '这是一套可审查的候选与隔离池，不能把目录内所有 JSONL 一起喂给训练。原始下载、许可、版本、完整回答和隔离原因均保留。最终训练接纳、跨来源去重及题族划分由中央流水线决定。','',
 '## 本次交付','',
 '| 文件 | 条目 | 独立题族 | 用途 |','|---|---:|---:|---|']
 for r in validation['files']:
  name=Path(r['file']).name;use='候选；仍须按来源证据/许可筛选'if name.endswith('candidates.jsonl')else'不进入默认训练'
  lines.append(f"| {name} | {r['rows']:,} | {r['distinct_families'] if r['distinct_families'] is not None else 'N/A'} | {use} |")
 lines+=['','各文件 SHA-256、字节数、来源计数、候选自洽性及每来源前16条 messages 回读结果见 `READY.json` 和 `audit/final_validation.json`。候选总量是中央规则过滤之前的数量，不能直接称为最终高质量训练量。','',
 '## 能保证什么','',
 '- 全部回答保持完整；未 tokenize、未按长度截断、未删除 reasoning_content。源系统提示、多轮上下文、监督轮掩码全部保留；候选逐条检查字段哈希，并对每来源前16条候选回读原始文件核对。完整原始文件供中央继续检验。',
 '- v3 的 train_turns 是与 messages 同长的布尔列表，原样保存为 assistant_loss_mask。其余来源仅训练最后 assistant；上下文中的历史助手回答不自动变成监督目标。',
 '- SQuAD 仅使用官方 train，共核对全部金标准答案的字符位置。按 article title 建题族，同时保留 context_sha256。输出是直接答案，不伪称长推理蒸馏。',
 '- ProofWriter 仅导出 OWA train，独立重算 10,736 个理论、122,642 个题目；选出 True/False/Unknown 各 10,000 条，五个深度各 6,000 条。显式负事实与 Unknown 严格区分；矛盾闭包拒绝。以规范化事实/规则的整组哈希隔离题族，官方 dev/test 理论只作排除索引。',
 '- 原始 ZIP 含官方 dev/test 文件，但不作为训练导出。其他来源只下载 train 或明确 SFT 训练数据。两份大型 NVIDIA 文件按完整行取固定前缀，记录前缀 SHA；没有假称下载或校验了整个多 GB 源文件。','',
 '## 不能保证什么','',
 '- 发布方的 verified/高分不等于我们独立证明了推理、知识和事实正确。JSON/XSD 通过也不能证明填入字段的事实有来源。',
 '- Nemotron v3 的 source_curated 仅有来源筛选证据，建议默认 review；它的少量本地格式检查不覆盖全部自然语言约束。NoRobots 同枚举但有人类撰写证据，应通过 verification.method 区分。',
 '- v1 IF 的 source_verified 指数据卡声明的指令校验与 judge 流程，answer_pass 保持 null；不应称本地逐题事实验证。',
 '- SQuAD 的 span 检查只证明来源金标准出现在上下文的指定位置；ProofWriter 的形式逻辑重算不等于对全量自然语言/形式表达映射进行了第二次人工标注。',
 '- 没有针对用户最终全部评测集进行全局去污染；中央应继续用题目指纹和题族做交叉隔离。','',
 '## 明确降级的来源','',
 '- Tulu-3 IF：抽样发现精确词频、每句关键词等约束失败，且源数据没有可直接执行的完整 kwargs。整源 review，不因检查少数规则通过而放行。',
 '- NVIDIA structured_outputs：保留 schema 检查结果，但无根据的字段填充不能当真实知识，整体 review；解析或 schema 明确失败进 quarantine。',
 '- 普通 Llama-Nemotron/chat_if 的 chat：未独立语义审查，整体 review。v3 chat 的部分上下文缺失，因此未下载并未补猜缺失轮次。',
 '- HelpSteer2/3：仅先筛选三项满分或至少三名评审一致 perfectly helpful，固定种子抽检仍发现中断指令解释错误、压痕公式错误、黑洞返回错误前提和星战剧情混写。满分池整体降 review，只有 5 条逐条审过的阅读/写作实例放行。EditQuality 的 good 仅表示按反馈编辑，并非正确性认证，原始文件归档不导出训练。',
 '- NoRobots：Chat 角色扮演、未执行代码及识别出的具体内容问题降 review；人类撰写也不假称全量无误。','',
 '## 许可与使用','',
 'NoRobots 为 CC-BY-NC-4.0；SQuAD 为 CC-BY-SA-4.0。ProofWriter 官方发布包未声明数据许可，单列 rights_pending，不从关联代码仓库借用 Apache 许可。其余许可按逐条来源元数据保留；本目录不是无条件可重新发布的统一许可合集。','',
 '## 复现','',
 '脚本在 `tools/dataset_expansion/general/`。下载脚本固定 revision 并验证源 LFS SHA（完整文件）；convert.py / proofwriter.py / helpsteer.py 生成各池；validate.py 检查候选自洽性、文件哈希及源消息抽样保真；write_report.py 生成 READY。缓存和第三方依赖位于本目录 tmp/vendor，不写 C 盘。',
 '测试为 14 个格式/约束边界测试、9 个显式否定/闭包测试、4 个 HelpSteer 上下文/评审规则测试；27 个均通过。人工审查分别有 34、26、6 条哈希绑定记录，审查通过不升级为全量或隐藏推理正确性证明。','']
 (BASE/'README_zh.md').write_text('\n'.join(lines),encoding='utf-8')
 (BASE/'READY.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':'READY','rows':validation['candidate_total'],'families':validation['candidate_distinct_family_ids']}),flush=True)
if __name__=='__main__':main()
