# MOTE SFT v2 第三阶段数据集层整改计划

版本：v1.0
日期：2026-07-29
目标分支：`mote-sft-v2`
目标发布协议：`math_raw_v2`

## 1. 一致性结论

本任务书与既定第三阶段判断基本一致。第三阶段处理的是必须结合真实来源 schema、答案语义和数据集关系才能完成的问题：OpenR1 verification 对齐、final-answer 可靠解析、来源适配、精确与同题分组去重、质量分层和生产发布统计。

任务书中的真实 tokenizer 长度、无截断、mask、EOS、冻结加载、确定性消费和恢复，是第二阶段工程不变量在真实数据上的生产验收。本轮复用并强化现有实现，不建立第二套训练时处理路径。

## 2. 交付边界

### 2.1 本轮完成

1. OpenR1 平行数组严格按索引组合，长度不一致和无 verified candidate 显式隔离。
2. final-answer 优先级解析、嵌套 LaTeX 支持、answer type 和生产硬门禁。
3. GSM8K main/Socratic、SVAMP、synthetic arithmetic、Hendrycks MATH、MetaMath、OpenR1、NuminaMath、OpenThoughts 的本地缓存 source adapter。
4. `math_raw_v2` 唯一 renderer/parser 协议和 round-trip 测试。
5. canonical schema 扩展、exact canonical dedup、normalized problem group 和每组暴露上限。
6. 真实 tokenizer 最终门禁、long-context 隔离信息和生产 frozen release。
7. 扩展 manifest：版本、来源、验证、质量层、去重、异常和 token 分布/占比。
8. source-manifest 驱动的离线 CLI、单卡 no-code 配置副本和使用说明。
9. fixture、集成、loader preflight 和全仓回归。

### 2.2 不伪造的外部产物

没有正式数据缓存、固定 revision 和生产 Qwen tokenizer时，不生成虚假的 `fitmotn-sft-v2-repaired-rc1`，不宣称完成真数据 1,000-batch preflight、100–500 update smoke 或 1,000–2,000 update canary。代码完成后，这些是发布节点上的准入步骤。

### 2.3 明确延期

- embedding semantic dedup；
- 多卡/rank 全局 sampler 新设计；
- packing、模型、loss、optimizer、RL 和 MOTE 核心；
- LLM judge、代码沙箱和完整 PII/许可证平台；
- 修改既有生产配置或原始缓存。

## 3. 生产不变量

- `generations[i]` 只与同索引 verification 字段组合；禁止 zip 截短或字段猜测。
- OpenR1 accepted 必须 `selected_math_verified=true`。
- final answer 必须独立、可分类、通过结构门禁；非 proof 不得等于整段 solution。
- 所有 source adapter 只在离线构建运行，训练只读冻结 token/label。
- 最终准入长度只由指定生产 tokenizer 对最终 render 结果判定。
- accepted 样本不截断 prompt/target、监督非空、mask 对齐、EOS 恰好一次。
- 相同 source identity 和 exact canonical content 均只能 accepted 一次。
- 同一 normalized problem group 默认最多保留两个高优先级解法。
- 所有拒绝、替换和去重均有 reason code、来源身份和 manifest 计数。
- 相同 source manifest、缓存、tokenizer、pipeline 和格式产生相同 release fingerprint。

## 4. 目标流水线

```text
local immutable source cache
  -> source adapter + strict source revision
  -> structured canonical reasoning sample
  -> answer/verification/quality gates
  -> math_raw_v2 render -> parse round-trip
  -> exact canonical dedup
  -> normalized problem grouping + ranked exposure cap
  -> production tokenizer materialization
  -> supervision/max-length gate
  -> accepted / quarantine / long-context index
  -> frozen manifest and quality report
  -> deterministic_strict single-card loader
```

## 5. 工作包

### WP1：OpenR1 严格对齐

- 将 generations、correctness_math_verify 和可选 correctness_llama 视为平行数组。
- 任一必需数组缺失或长度不一致，返回稳定 reason code。
- 只在 math verified 候选中选择；按格式完整、真实 tokenizer 长度和稳定索引排序。
- 保存选择索引、两类 verification、可用候选数、verified 数和策略。
- 动态旧路径同样 fail closed，避免 offline 与 legacy normalization 语义分叉。

### WP2：答案解析和质量门禁

- 解析顺序：显式字段、`####`、`Final Answer:`、`The answer is:`、Therefore/So、最后完整 boxed。
- 删除无条件最后一行回退。
- 支持嵌套 boxed、矩阵、集合、区间、单位和多 boxed。
- 规范化答案前缀并识别 numeric/expression/matrix/set/interval/unit/text/proof。
- 拒绝空答案、未知类型、异常超长、非 proof 的 answer==solution 和异常高 answer/solution 比例。

### WP3：canonical 与格式协议

- canonical 增加 problem、reasoning、final_answer、answer_type、problem_group_id、verification、quality tier 和 format version。
- 计算 normalized_problem_hash、problem_answer_hash 和 structured canonical hash。
- `math_raw_v2` 固定为 `Question` + `Solution` + `#### final_answer`。
- renderer/parser round-trip 不一致时隔离。

### WP4：真实来源 adapter

- source manifest 明确 adapter、local path/kind、split、revision 和 source policy。
- 支持九类当前来源并保持来源特有字段证据。
- revision 使用显式发布版本、commit 或缓存 fingerprint；拒绝 latest/main/master/unknown。
- synthetic arithmetic 保存 seed/template/difficulty，并程序复算答案与去重。
- MetaMath 使用 original_question 建 group；Numina/OpenThoughts 的未可靠验证内容进入 B/C，不自动提升。

### WP5：精确与 problem-group 去重

- source identity 冲突、canonical content 重复、problem+answer+reasoning 重复均隔离。
- normalized problem 相同进入同一 group。
- 每组默认最多两个解法，按 verified、quality tier、source authority、格式完整和真实长度排序。
- 被更优样本替换或超过 group cap 的记录进入 quarantine 并计数。

### WP6：冻结发布和质量报告

- release API 接受 format/adapter/tokenizer revision 和 source-build 元数据。
- manifest 报告 scanned/accepted/quarantine、verification、quality tier、exact/group dedup、各类异常。
- 报告 prompt/target/total/supervised token 的 mean/P50/P90/P95/P99/max。
- 报告每源序列占比和 supervised-token 占比。
- overlong canonical 记录进入 long-context 索引并从 seq1280 accepted 排除。
- The Stack 状态作为独立 pretrain data plane 元数据记录，不混入 SFT 统计。

### WP7：CLI、配置和准入

- 扩展构建 CLI：保留 canonical JSONL 兼容入口，新增 source manifest 生产入口。
- 新增单卡 `seq_len_run=1280`、frozen-exclusive、deterministic-strict、worker=0 的 no-code 配置副本。
- `source_max_epochs` 必须显式为正整数；实际值由计划序列数/accepted 数在发布后计算确认。
- 提供 release 审计和 loader preflight 命令/测试；不自动启动完整训练。

## 6. 测试矩阵

1. OpenR1：verified 优先、真实 token 最短、数组错位、无 verified、审计字段。
2. Parser：裸 `The answer is`、嵌套 boxed、矩阵/集合/区间、负小数、单位、多 boxed。
3. Gate：answer==solution、proof 例外、空/超长/未知答案。
4. Format：`math_raw_v2` render→parse round-trip。
5. Length：1279/1280/1281、prompt/target mask、单 EOS、无左右截断。
6. Dedup：source identity、canonical content、problem group cap 和优先级。
7. Release：重建 fingerprint、manifest 统计、long-context 隔离、损坏检测。
8. Pipeline：九类 adapter fixture、source revision、local cache、quality tier。
9. Loader：冻结记录无动态 tokenization、quarantine 不可见、状态恢复一致。

## 7. 完成定义

- 代码和 fixture 层面满足全部 P0 不变量；
- 新增 source-to-release 离线入口和版本化配置，不覆盖生产原文件；
- 定向测试与非平台全仓回归通过；
- Windows/POSIX 既有无关失败单独记录，不越界修复；
- 改动提交到 `mote-sft-v2` 并推送；
- 真实 rc1 构建、人工抽查、1,000-batch preflight、smoke 和 canary 作为持有数据/模型的一侧后续准入步骤。

## 8. 实施与验证记录

WP1-WP7 的代码、fixture、配置和说明已完成：

- canonical contract 升级为 v2，frozen release 升级为 v2，并保留 v1 release 校验兼容；
- OpenR1 动态路径拒绝近似长度选择，生产选择只能由 source builder 注入真实 tokenizer；
- 九类 adapter、`math_raw_v2`、答案门禁、质量层和 immutable revision 约束已接入；
- exact/problem-group 去重使用磁盘 SQLite staging，避免大规模候选全部驻留内存；
- manifest、long-context index、1000-record preflight、pure-SFT/no-code 配置和暴露计划已接入。

验证结果：

- 第三阶段及受影响路径综合回归：`107 passed`；
- 排除既有 Windows/POSIX 平台专属测试文件后的全仓回归：`377 passed, 3 skipped`；
- 完整测试：`404 passed, 3 skipped, 13 failed`。剩余 13 项仍是本轮未修改模块的既有平台差异：4 项 POSIX 路径语义断言和 9 项依赖 `os.getpgrp/os.getpgid` 的 vLLM actor 测试；
- Python `compileall` 和 `git diff --check` 通过。

本地没有任务书所指的正式九源缓存和生产 Qwen revision，因此没有生成或声称生成真实 rc1，也没有运行模型 smoke/canary；相应命令和准入顺序已写入使用说明。
