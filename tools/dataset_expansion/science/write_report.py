import json,collections
from pathlib import Path
from process import ROOT
report=json.loads((ROOT/'audit/final_summary.json').read_text(encoding='utf-8'))
lines=['# 科学数据扩展与审查记录（2026-10-16）','',
'这次完成了七类来源的下载、版本固定、完整回答保留、统一 JSONL 和分级筛查。**没有把搜集到的数据全部宣称为高质量训练集**：复杂科学题的同源参考答案可能也错，官方发布或论文过滤不足以代替逐题验证。','',
'最终文件在 `release/<来源>/`。`normalized/` 是中间产物；不要从 `raw/` 直接抓取文件混入训练。后续全局去重、基准污染检查和训练/验证划分由主流程完成。','',
'## 实际交付数量','',
'| 来源 | 候选 | 待审 | 隔离 | 工具轨迹待审 |','|---|---:|---:|---:|---:|']
for name,s in report['sources'].items():
 c=s['counts'];lines.append('| '+name+' | '+' | '.join(str(c.get(k,0)) for k in ['candidates','review','quarantine','tool_trajectories_review'])+' |')
lines += ['', '候选也保留证据等级；不是同一个“正确率保证”。计数是科学工作流内部结果，尚未扣除与其他工作流、旧数据或评测题的重复。', '',
'## 各层的含义','',
'- **SciQ**：只用官方 train，输出是原有众包正确答案，输入提供完整 support。放行前要求答案在 support 中按词边界出现，且无结构/图示问题。它是基础科学阅读和简答数据，没有杜撰长推理。出处：[Ai2 数据卡](https://huggingface.co/datasets/allenai/sciq)、[原论文](https://arxiv.org/abs/1707.06209)。',
'- **ScienceQA**：用官方 GitHub 固定版本及 train ID，排除所有带图片的题和仍依赖未提供视觉内容的题。原 lecture、solution 完整保留，最后从原 answer index 明确输出答案。答案序号映射经过本地验证，科学正确性不是重新证明。出处：[官方仓库](https://github.com/lupantech/ScienceQA)、[原论文](https://arxiv.org/abs/2209.09513)。',
'- **OpenScienceReasoning-2**：固定 HF 转换版本中完整 partial-train 分片，保留完整 R1-0528 输出。最终 boxed 与 expected_answer 一致只能证明标签匹配。固定种子语义抽样发现明确错题/错解后，**整池降为待审**，不是可直接训练候选。没有原生学科标签，science_mixed 不能计作十万条物理。出处：[NVIDIA 数据卡](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2)。',
'- **Nemotron-SFT-Science-v2 syn_mcq**：整个 96,610 条子集已下载，保留原始 messages、reasoning_content、tools 和真实 topic。缺少逐条 gold/正确性证据，全部待审；工具轨迹另外存放，避免丢掉工具结果后误作普通 SFT。出处：[NVIDIA 数据卡](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2)。',
'- **TextbookReasoning**：完整扫描唯一分片，处理所有 227,126 条非数学记录，424,714 条数学留给数学工作流。论文中的 reference_answer 是从答案生成的摘要，并非独立真值。本地抽查暴露了严重错解及缺上下文，整个来源降为研究/待审池。出处：[数据卡](https://huggingface.co/datasets/MegaScience/TextbookReasoning)、[论文](https://arxiv.org/html/2507.16812v2)。',
'- **PhysicsEval**：只下载官方 train 17,647 条。发现明确的电荷/重子数不守恒错解；整源保持待审，已识别错误隔离。出处：[数据卡](https://huggingface.co/datasets/IUTVanguard/PhysicsEval)、[论文](https://arxiv.org/abs/2508.00079)。', '',
'- **OpenStax 物理**：从固定 PhysCorp-A train 中仅选择全部 1,429 条 OpenStax 记录，重新抓取 156 个官方题目/答案页，按原答案链接和 DOM ID 对齐，恢复 MathML 指数、分式和上下标。二次整理版本曾把 10^6 摊平成 10 6，所以没有沿用它的损坏公式。候选排除图示、缺表格常数、核物理符号、缺解释和已识别歧义；答案来自官方 answer key，不伪造 CoT。其他 5,003 条来源混合不导出。出处：[选择来源](https://huggingface.co/datasets/shanyangmie/physcorp-a)、[OpenStax](https://openstax.org/subjects/science)。', '',
'## 本轮最重要的质量发现','',
'TextbookReasoning 的初筛池按每个来源学科的题目 SHA256 取最小十条，共抽出 60 条。物理十条经阅读发现七条存在错答案、错推导或缺关键条件；这是小规模诊断样本，不能当作全库精确错误率。完整题目、输出、来源行号、版本和内容哈希保存在 `audit/manual_findings.json`。', '',
'典型例子：130 N 与 110 N 的合力，夹角正切为 12/5，正确大小为 200 N。源输出已经算出 200 N，却为了服从参考答案改为 185.7 N。另有水饱和度题缺毛细压力曲线，输出自行假定 40%；还有长螺线管把单个环中心场乘以每米环数，量纲直接错误。', '',
'因此本轮没有靠这些来源把“已核验物理”凑到几万条。已完成的是可追溯的大研究池和较小、证据更清楚的基础层。大量物理题仍需要独立 teacher/公式与单位检查、人工抽检后才能晋级。', '',
'OpenScienceReasoning-2 对整个已下载分片以固定种子题目 SHA256 取 20 条，读取完整题干和最终解释并保留全部原 CoT：8 条明确实质缺陷，6 条未决，6 条最终解释暂未见明确错误但不予全流程认证。**这是小规模筛查信号，不能推成总体准确率**。审查范围和逐条完整证据在 `audit/osr2_semantic_findings.json`，没有把未发现错误视为已证明正确。', '',
'其中辐射题原单位为 Ci，按已给贡献算出的年剂量至少约 17,165 mSv，模型却擅自改为 mCi，再凑到 18.4 mSv 的选项；AFM 题问 10%–90% 的过渡宽度，模型算出不在选项里，转而回答 0%–100% 全宽；化学题没有 Claisen 所需的 vinyl 底物，回答仍编造重排和类似 SN2 的反转。', '',
'基础层也接受相同审查：ScienceQA 中解释按旧顺序称第一/第二选项而 choices 已换序的记录，全部待审；SciQ 的答案单词出现在 support 中但 support 不支持题意，不能算通过，因此增加残句/外部上下文筛查，并隔离已发现的生物学简化错误。`audit/foundation_semantic_audit.json` 保存 30 条更新后样本和 8 条具体发现。', '',
'## 格式与保留原则','',
'每行包含 id、problem_id、family_id、input、output、final、solution、domain、source、source_revision、source_split、license、verification、review_flags、provenance、metadata 等。所有模型无关，不调用 tokenizer，不截断、不补造推理、不插入学生模型 EOS。', '',
'原始长推理完整保留。`final` 是已存在的可见最终回答，TextbookReasoning 没有独立 final 字段时保留完整 source answer；PhysicsEval 的简短答案来自其原字段，未拼接进原解答。ScienceQA 组合原 lecture/solution 并附上明确答案，其转换已记录。工具轨迹的训练应使用原 messages；不得只取最后一段输出。', '',
'部分 provenance 仍有兼容字段 `raw_output_sha256`，它校验本流程最初构造的 output；在 Nemotron 的 think 包装等情形下，不能把它误解为原文件中某个 output 字段的原字节。原始文件的 SHA256 独立保存，Nemotron 的原 messages 也完整保留。ScienceQA 使用明确的 normalized_output_sha256，并另存原 lecture/solution 哈希。', '',
'## 来源、许可与复现','',
'- 下载与缓存、临时文件全部指向本目录或 D 盘工具目录；没有将数据下载到 C 盘。',
'- 版本、URL、文件 SHA256：`audit/all_downloads.json`；最终 JSONL SHA256：`audit/release_manifest.json`。',
'- 许可分别保留：OSR2 为 CC-BY-4.0；Nemotron v2 / PhysicsEval 为 CC-BY-SA-4.0；TextbookReasoning 为 CC-BY-NC-SA-4.0；SciQ 为 CC-BY-NC-3.0。',
'- ScienceQA 官方 `LICENSE-DATA` 实际为 **CC-BY-NC-SA-4.0**，与第三方 HF 卡省略 NC 的写法不同。本批采用官方许可，文件存于 `audit/scienceqa_LICENSE-DATA.txt`。代码 MIT 许可不能代替数据许可。',
'- OpenStax 原教材 CC-BY-4.0；选择集合 PhysCorp-A 为 CC-BY-NC-4.0，本批行级许可保守保留 NC，同时注明原文许可。官方 HTML 的逐页哈希另在 `audit/openstax_html_manifest.json`。',
'- 下载和处理脚本在 `tools/dataset_expansion/science/`；11 项解析/筛查测试与 6 项 MathML 转换测试通过。初次全量检查唯一 ID、train 来源、内容哈希和字段类型；后置语义降级在同一次写出中计算 SHA，避免机械盘重复全库扫描。中央合并再验证全量格式/重复。',
'- 未运行下载的数据代码，未调用付费模型或重新生成推理。', '',
'## 后续物理盲审队列','',
'`verification_queue/physics_2000_blind_teacher.jsonl` 是 2,000 条按标准化物理子学科轮转配额、题目哈希抽取的待验证题。源答案放在单独的 `physics_2000_source_evidence.jsonl`；给 teacher 时只给 blind_teacher 文件，避免看到参考后强行凑答案。尚未调用 teacher，结果不能当作已生成或已验证数据。分歧需要第三方验算；一致也只提高证据，不能证明推理全对。','',
'## 证据字段如何理解','',
'候选使用 `quality_level=source_verified`，具体含义是原教材/教育数据明确给出了答案并通过本地结构门禁，不表示整个回答经过重新求解。`reference_matched` 仅表示输出最终标签与同源参考一致，本轮 OSR2/Textbook 的这一层仍属待审。`source_curated` 和 `structural_only` 不能直接当训练合格标记。务必同时检查 `release_status`、`verification` 和 `review_flags`；本批所有记录的 `process_verified` 均为 false。','',
'## 未引入的来源','',
'没有使用 PHYSICS-Verified、UGPhysics 等评测集来补训练数；没有重复下载 MegaScience 大混合来重复计算 TextbookReasoning；没有取没有解答的 RL 环境配置冒充 SFT；不采用来源不明的百万条科学合成集合充数。']
(ROOT/'README_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(str(ROOT/'README_zh.md'))
