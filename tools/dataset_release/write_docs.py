"""Write a dataset card from actual export statistics."""
import json,shutil
from common import DEST

def main():
 summary=json.loads((DEST/'audit/summary.json').read_text(encoding='utf-8'));exp=json.loads((DEST/'audit/export_validation.json').read_text(encoding='utf-8'));v=json.loads((DEST/'audit/validation_final.json').read_text(encoding='utf-8'))
 counts=exp['files'];domains=counts['train']['domain_counts'];pd=exp['pilot_domains']
 text=f'''# MoTE 本地共享 SFT 数据集 v0.1

这是一份面向 Gemma3、Llama3.1、Qwen3.5、Qwen3 的**模型无关 JSONL 底稿**。本阶段完成内容整理、结构检查、答案验证分级、去重和训练/评估隔离；未进行 tokenizer 编码、长度分桶、切片、packing 或模型训练。原始文件保持不变。

已整理 {summary['canonical_records']:,} 条记录，来自本地蒸馏结果、物理/数理方法资料及一份数学推理 Parquet。**可交付的是分级的数据资产，不是“所有推理过程均已验证正确”的承诺。** 物理与写作仍缺足够核验，当前训练库明显偏数学。

## 先看哪些文件

- `pilot/train.jsonl`：{exp['pilot_rows']:,} 条答案核对通过的小规模混合样例，是正式训练候选的子集，每个题目族最多一条；完整保留所选回答。
- `jsonl/train.jsonl`：{counts['train']['rows']:,} 条答案核对通过的训练候选。
- `jsonl/validation.jsonl`：{counts['validation']['rows']:,} 条核对通过的验证数据，包含继承的 diagnostic_dev，逐条保留原 split。
- `jsonl/test.jsonl`：{counts['test']['rows']:,} 条内部保留集。它主要来自本地数学和物理，不是覆盖所有领域的权威测试集。
- `jsonl/smoke.jsonl`：{counts['smoke']['rows']:,} 条冒烟检查数据，不用于正式训练或统计显著性结论。
- `jsonl/review_required.jsonl`：{counts['review_required']['rows']:,} 条结构可用但证据不足的候选，尚未放行训练。
- `jsonl/quarantine.jsonl`：{counts['quarantine']['rows']:,} 条带明确风险标记的记录，包括生成未完成、缺答案、参考答案有分歧、缺图、重复题答案文本不同、基准重叠等。隔离不等于已经证明内容错误。

直接指定 `jsonl/train.jsonl` 或 `pilot/train.jsonl` 读取。不要递归合并整个目录：待审、隔离、测试文件和 pilot 子集都在其中。

训练候选的领域数量：{json.dumps(domains,ensure_ascii=False)}。

小样例的领域数量：{json.dumps(pd,ensure_ascii=False)}。

Pilot 优先选择独立计算通过及完整回答较短的条目，用于人工检查和后续小实验；它有选样偏好，并不代表原始长度分布，也不是统计显著性的保证。按题数计的比例也不等于训练 token 占比。

## 一行表示什么

基础字段借鉴 NVIDIA [OpenCodeReasoning 数据卡](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) 的 `id/input/output/source/solution` 思路；增加来源证据和质量分级。这不是 NVIDIA 官方数据，也不声称与其字段语义完全相同。

- `input`：完整题目；本阶段只规范换行、边缘空白和有记录的损坏转义。
- `output`：供后续适配训练的完整回答。有独立推理文本时表示为 `<think>...</think>` 加最终回复；没有时就是原有完整解答。
- `final`：teacher 的最终回复全文，可能包含解释，并不一定只是一个数值。`expected_answer` 才是参考答案；参考答案也可能有误。
- `solution`：代码题中可靠提取出的完整 Python 程序，其他题通常为 null。程序可提取或 AST 可解析，不代表测试通过。
- `auxiliary_reasoning`：对“只返回 JSON/精确文本”的指令题，将已有推理单独保存在这里，`output` 只保留符合要求的最终回复。推理未丢弃，也没有把思考文字强行拼进严格 JSON 输出。
- `output_sections`：reasoning/final 在 output 中的字符区间，左闭右开，按 Python Unicode 字符索引；不是 token 下标。
- `source/source_revision/source_split/license/provenance/metadata`：来源、原始记录位置、文件哈希及可获得的生成信息；未知值保留 null，不猜测。`source` 指原题数据源或本地资料，`dataset` 是本整理版名称。
- `problem_id/family_id`：规范化题目哈希/题目族。相同题的多条推理可以保留，但必须在同一 split。
- `quality_level/verification/review_flags`：放行依据与尚未解决的问题；`split` 是本地用途，`original_split` 保留蒸馏集已有用途。

`schema.json` 给出完整格式约束；`examples/sample_records.jsonl` 可快速查看各领域实例。它是 JSONL：每一物理行都是完整 JSON 对象，正文换行以 JSON 转义保存。

`<think>` 在这里是可解析的文本边界，**不是给四种模型直接指定统一 chat template**。将来根据各学生模型模板和训练目标再做适配；本阶段不插入模型 EOS，也不截掉长推理，不生成或截取任何 token 序列。仅有答案、没有推理的源记录也不会被伪造补全。

## 质量级别的边界

`independently_recomputed`：支持的库存、筛选排序和部分逻辑题由程序独立重算，输出和原参考值均匹配。

`reference_matched`：数学答案按平衡括号提取、规范文本或精确有理数比对；选择/阅读题按参考答案比对。未使用宽松的“字符串包含正确数字”判断。暂不支持的符号等价判定进入待审；与参考答案不同不自动认定 teacher 错，因为参考值也可能错。

`prior_execution_pass`：103 条代码在所有划分合计保留了**与当前程序 SHA-256 一致且所有已记录测试通过**的历史证据，其中 75 条在训练集。这些不是本轮重新执行的结果。另有 114 条可提取程序等待隔离执行，36 条提取存在歧义或缺失；任务与已有测试在 `code_review/jobs.jsonl`。

`assistant_reviewed_final`：18 条由本次助手逐项检查并以输入/回复哈希绑定的记录，包括 17 条物理/数理方法题和 1 条数学参考值修正。这不是领域专家背书，也未逐步审核整条隐藏推理。

`structural_only/unverified`：格式已整理但正确性不足以放行。原数据中“解析成功”“内部答案字母一致”“多数生成答案一致”均不自动提升为内容正确。

所有记录都没有宣称 CoT 全过程经过语义证明。答案核对通过只说明已有证据支持交付答案，不能排除冗长、绕路或中间错误后碰巧答对。应结合生成行为实验评估是否适合具体学生模型。

## 本轮做过的实际清洗

- 按完整解析器读取含多行单元格的 CSV；合并完全重复文件、完全相同题目/推理/回答，并保留来源。不同解答不凭多数票覆盖。
- 从多段 Markdown 包裹的 JSON 中恢复 371 条题目；修复可确认的非法 JSON 转义及 LaTeX 控制字符损坏。未知控制字符继续隔离。共记录 {summary['repair_actions']:,} 项修复动作，详细内容在 `audit/repairs.jsonl`。
- 重写代码围栏提取，避免把解释文字误作程序；仅以当前代码哈希匹配完整历史测试证据。
- 单独复核一条“年份末两位为 00，却要求模 1000 余数为 125 或 875”的题目：参考值 4 改为 0，保留原值和明确审查记录。没有批量用 teacher 输出改写 gold。
- 建立含 6,319 条 GSM8K/MATH 测试题的本地排除索引，25 条直接匹配记录被隔离。相同题目/数字模板/原始题目族采用同组划分，继承已有 heldout。

基准排除只覆盖这两个公开测试集的规范化精确匹配和数字模板匹配，尚不涵盖所有语义改写、HumanEval 或其他评测。不会据此宣称“完全无污染”。基准来源版本和下载文件哈希见 `audit/benchmark_sources.json`。

本地数学 Parquet 字段与 [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) 一致，但原始仓库 revision 未确认，因此来源仍写为本地分片。不能把公开数据卡的许可或来源信息无证据地套到该文件上；混合数据各条保留可获得的来源许可信息，未知项保持 null。

## 检查与复现

清洗逻辑的 19 项测试通过；全量 {v['record_count']:,} 条导出通过 Draft 2020-12 JSON Schema 验证、ID 唯一性、已放行题目族 split 隔离、pilot 子集关系和文件哈希检查。{v['source_files_checked_unchanged']:,} 个原始文件校验未变化。完整推理/最终回复的导出字符区间逐条比对通过；这里的保真针对规范化后的文本，不意味着原文件字节不经过格式清理。

本轮没有调用新的付费模型接口，也没有新生成 CoT。本地 Docker 引擎启动失败，因此未在主机上直接执行生成代码；历史通过证据与等待执行任务分别标注。

复现脚本位于原工程 `tools/dataset_release/`，依次运行 `build_release.py`、`export_jsonl.py`、`validate_release.py`。构建需要原始 Datasets 文件及已有 benchmark 索引；验证工具也可以对解压后的发布目录单独运行：

```bash
python validate_release.py /path/to/local_sft_v01_20261016
```

验证依赖 jsonschema（本轮使用 4.26.0）；构建额外需要 pyarrow。`MANIFEST.json` 记录标准 JSONL 文件的大小和 SHA-256；`audit/validation_final.json` 是本轮实测结果。重建会覆盖同版本派生文件，不会改原始文件。

## 下一阶段最值得做的事

先补物理的内容核验：优先无缺图依赖、参考推导明确、数值与量纲可独立计算的题；对书籍知识问答和数理方法概念题分别审查，不把它们都当成物理解题 CoT。写作需要单独的质量评审规则。对待审数学补严格符号验证，对 114 条代码在隔离环境重跑现有测试；通过后才按同一格式追加，并重新检查题目族隔离。

这些内容核验完成后，再决定各领域完整样本的配比，以及四个学生模型各自的长度策略。当前底稿保留长回答的全部内容，为后续选择保留余地。
'''
 (DEST/'README_zh.md').write_text(text,encoding='utf-8')
 shutil.copy2(__file__.replace('write_docs.py','schema.json'),DEST/'schema.json')
 print(str(DEST/'README_zh.md'))

if __name__=='__main__':main()
