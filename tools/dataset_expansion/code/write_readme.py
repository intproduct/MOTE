import json
from fetch_sources import DEST

def main():
 s=json.loads((DEST/'audit/processing_summary.json').read_text(encoding='utf-8'));c=s['counts'];sources=s['candidate_sources'];reasons=s['filter_reasons']
 text=f'''# 本地代码 SFT 候选集（2026-10-16）

本轮从固定版本的 NVIDIA OpenCodeReasoning-2 和 KodCode-V1-SFT-R1 采集 **{c['raw_rows']:,} 条原始记录**，经过来源验证筛选、完整性检查、去重及测试/题面审查，保留 **{c['candidates']:,} 个不同原题族的完整候选回答**。

这些是等待总管线进行跨领域去重与评测集排除的候选，并非声称所有内容已人工逐字证明正确的数据。`quality_level=source_verified` 表示来源侧执行验证通过，加上本轮本地静态检查；**本机没有运行这些程序或测试**。

## 交付文件

- `candidates.jsonl`：{c['candidates']:,} 条；完整 input/output/final/solution，含来源、许可、验证证据、原题族、原 split。KodCode 还保留完整 `unit_tests`。
- `review.jsonl`：{c['review']:,} 条；证据不足、题面格式/示例疑点、无法完整恢复等，不能直接混入训练。
- `quarantine.jsonl`：{c['quarantine']:,} 条；来源执行未通过、原评测 split、明显不完整或异常等。
- `duplicate_alternatives.jsonl`：{c['duplicate_alternatives']:,} 条；同题其他完整轨迹/呈现，不用它们虚增独立问题数。
- `MANIFEST.json` 和 `audit/final_validation.json`：最终文件行数、SHA256、检查结果。
- `raw/`：原始分片、原题投影、固定版本数据卡和必要字节范围缓存。原数据未覆盖。

## 两个许可组

- NVIDIA OpenCodeReasoning-2：{sources.get('nvidia/OpenCodeReasoning-2',0):,} 条。总集声明 CC BY 4.0，每条同时保留原题库许可。
- KodCode-V1-SFT-R1：{sources.get('KodCode/KodCode-V1-SFT-R1',0):,} 条。**CC BY-NC 4.0，非商业研究组**；不能把全包标成可商用。

## 实际做了哪些检查

1. 固定上游 revision，9 个完整原始文件逐一核对 HF 声明的 LFS SHA256。问题列投影保留单独的下载范围/投影哈希，不冒充完整文件哈希验证。
2. OCR2 要求 `pass_rate=1.0` 且 `judgement=right`；按官方原题 `dataset/split/index` 恢复问题，原始非 train split 不进入候选。KodCode 只采集 train，要求来源 R1 通过标记和通过序列。
3. 必须存在闭合推理边界、非空最终答案、闭合代码围栏，代码能够进行 Python AST 解析，最终代码与来源被验证的 solution 完全一致。来源未提供 finish_reason，不伪造该字段。
4. 排除明显长段重复、异常控制字符、缺失图片依赖等。完整推理保留原文，没有 tokenizer 操作、没有长度截断、没有合成补写答案。
5. KodCode 测试要有可见的结果验证：函数返回值/结果变量与断言存在静态联系；至少 3 个有效检查，或至少 3 个显式参数化用例。`assert True`、常量断言、自比较、结果被常量覆盖等不构成证据。本轮因此额外暂缓 **{reasons.get('insufficient_statically_linked_test_checks',0):,} 条**。这只是保守静态审查，不是形式化证明或测试覆盖率证明。
6. 对题面 doctest 做语法检查，对能解析成字面量的题面例子与单元测试期望进行比较。不能静态解析的例子保留“未比较”，不会被自动判错。例子期望不一致进入待审，可能还需要判断输出顺序是否自由。
7. 人工抽检发现 `Algorithm_3_C` 把 `(1,2,3)` 的乘积写成 120；按输入 SHA 绑定审查结论，保留原题进入待审，没有默默改写教师数据。
8. 按归一化问题去重，并把 KodCode `_I` / `_C` / `_OJ` 等同题呈现归入一个源题族，防止训练/验证串题。

## 质量边界与使用建议

代码通过给定测试仍可能有遗漏边界情况，合成测试也可能与题面共同出错。整条 CoT 没有逐步验证，字段 `process_verified=false`。筛选会改变题目难度分布，因此仍需以实际学生模型小批量试验确认收益。

`<think>` 是保留下来的数据文本边界，不代表已适配任何学生模型的原生聊天模板。本数据没有加入模型专属 EOS，也没有把长回答拆成片段。`output_sections` 是字符区间，不能当 token 区间。SFT 应读取 `input` / `output`；`unit_tests` 和验证元数据不应被无意拼入输入。

此阶段不作“已彻底去污染”承诺；总管线须对实际评测集继续做精确/近重复排除，且使用 `family_id` / `metadata.source_question_family_id` 分组。更多来源的取舍见 `audit/source_selection_notes.md`。

## 复现顺序

脚本位于 `tools/dataset_expansion/code/`。全部输出、原始下载与缓存写在 D 盘的本目录。

```text
fetch_sources.py
fetch_questions.py
fetch_kodcode.py
fetch_apps.py
process_code.py
refine_test_evidence.py
dedup_problem_families.py
refine_examples.py
apply_manual_reviews.py
validate_release.py
write_readme.py
```

下载可按脚本内的并发限制运行，后续处理按顺序运行。网络完整分片下载上限为同时 2 条。`test_curation.py` 是本轮清洗逻辑的针对性测试，不运行来源程序。
'''
 (DEST/'README_zh.md').write_text(text,encoding='utf-8')
 print(str(DEST/'README_zh.md'))
if __name__=='__main__':main()
