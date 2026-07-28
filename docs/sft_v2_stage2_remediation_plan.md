# MOTE SFT v2 第二阶段通用数据处理整改计划

版本：v1.0
日期：2026-07-29
目标分支：`mote-sft-v2`

## 1. 目标

本阶段建立一个确定性、无静默损坏、可审计、可复现的通用数据处理链路，使训练只消费通过显式契约验证的数据，并能准确说明每条样本的来源、监督范围、采样顺序和重复暴露情况。

本阶段改善的是训练数据的工程正确性、有效利用率和实验可解释性。它不承诺上游答案天然正确，也不替训练者决定最终格式、长度、混合权重或监督策略。

## 2. 边界

### 2.1 纳入范围

1. 语义保真的文本规范化，保留换行、LaTeX 和代码缩进。
2. 通用 canonical sample 契约、稳定源身份、内容 hash、来源和处理版本。
3. 结构化校验和统一 reason code；禁止无依据字段猜测和静默回退。
4. prompt、assistant target、labels、mask、BOS/EOS 和长度诊断。
5. 禁止无意识删除问题或 final answer；超长样本显式拒绝或由已配置策略处理。
6. accepted/quarantine/manifest/quality report 的轻量冻结发布能力。
7. 确定性、覆盖优先的源内采样；source epoch 内无放回遍历。
8. worker/rank 互斥分片、数据源耗尽行为、重复暴露与覆盖统计。
9. sampler state/fingerprint 接口，为 exact resume 和 checkpoint 集成提供可验证状态。
10. 配置意图与实际执行结果的可观测性和 fail-closed 行为。

### 2.2 明确不纳入

1. 具体数据集答案真实性、证明质量或 verifier 可信度判断。
2. OpenR1 等具体上游 schema 的专项修复；通用契约会为后续适配提供接口。
3. 跨源同题、语义近重复、benchmark 污染和数据许可法律判断。
4. embedding 去重、LLM judge、PII 平台、代码执行沙箱和工具调用回放。
5. 替训练者选择 raw/chat、1024/1280、answer-only/full-trace 或具体混合权重。
6. packing 和其他纯吞吐优化。

## 3. 理论依据

SFT 优化的是 supervised token 上的条件似然。输入被破坏、目标被截断、mask 错位或样本被意外重复，都会改变实际经验分布和梯度方向。因此本阶段优先建立以下不变量：

- 文本规范化不改变已知语义结构；
- 每个参与 loss 的 token 都有明确角色和来源；
- 问题与 final answer 不被无意识截断；
- 相同发布物、配置和 seed 产生相同样本顺序；
- source epoch 内的重复只能来自显式策略，不能来自迭代器重启或 worker/rank 冲突；
- 数据拒绝在训练前完成，训练路径不再承担质量筛选；
- 配置中的数据意图与训练真正消费的数据可以对照。

## 4. 目标架构

```text
raw source row
  -> lossless normalization
  -> canonical sample + source identity + content hash
  -> contract validation + reason codes
  -> accepted / quarantine
  -> renderer/tokenizer materialization
  -> supervision and length invariants
  -> frozen manifest/release
  -> deterministic source traversal and mixture sampling
  -> training observability
```

canonical 数据与训练策略分离。格式、tokenizer、序列长度、监督模式和采样策略是可配置派生参数，但其有效值必须进入发布或运行指纹，不能隐藏变化。

## 5. 实施工作包

### WP1：黄金回归测试与数据契约

- 新增文本保真、canonical identity、reason code 和 manifest 测试。
- 新增长度边界、supervised-token、final-answer 保留和 EOS 测试。
- 新增确定性排列、worker/rank 分片、source epoch、fingerprint 和 resume 测试。
- 所有已确认缺陷先形成失败测试，再修改实现。

验收：新增测试可以稳定复现旧实现的问题，并锁定新不变量。

### WP2：无损文本规范化

- 建立单一文本规范化实现，替换数据 adapter、reasoning helper 和 tokenization 中相互冲突的清洗逻辑。
- 统一 CRLF/LF，移除行尾空白，但保留换行、空行语义和行首缩进。
- wrapper token 清理不再连带压平正文。
- 结构化值转换保持字段间边界。

验收：LaTeX、多段推理和代码缩进 round-trip 通过；不存在全局 whitespace collapse。

### WP3：canonical sample、稳定身份和校验

- 增加 `source_sample_id`、`raw_content_hash`、`canonical_content_hash` 和 `derived_sample_id`。
- `source_sample_id` 只由 source/revision/split/row ID 决定；内容和 pipeline 变化使用独立 hash/version 表示。
- 定义最小 canonical 结构：来源、数据面、任务类型、messages 或 prompt/response、质量状态、处理版本。
- 定义集中 reason-code registry 和校验结果对象。

验收：相同输入跨进程得到相同 ID/hash；不同 pipeline 版本不会伪装成同一派生样本。

### WP4：监督安全和长度门禁

- tokenization 返回 prompt/target/total/supervised token 诊断。
- 默认策略不再通过左截 target 强行适配长度。
- prompt + target 超长时返回结构化拒绝；训练路径可配置为 fail closed。
- 强制 `supervised_tokens > 0`，labels 与 input 长度一致，prompt mask 为 `-100`。
- chat 路径同样执行监督和长度不变量。
- BOS/EOS 只由一个明确层负责。

验收：任何 accepted SFT 样本都不发生 target 左截；空监督和不一致 mask 为硬失败。

### WP5：轻量冻结发布

- 提供离线构建 API/CLI，将 canonical 样本分为 accepted 与 quarantine。
- 输出 manifest、文件 hash、处理版本、reason-code 汇总和 source 统计。
- 采用临时目录构建、校验后原子发布；默认拒绝覆盖已有 release。
- 训练侧提供 frozen JSONL/manifest 校验入口，输入损坏时 fail closed。

验收：相同输入与配置产生相同 manifest fingerprint；quarantine 不进入 accepted 输出。

### WP6：确定性覆盖优先采样

- 有限源使用 source epoch 内无放回确定性 permutation。
- source 耗尽后明确增加 epoch 并生成新排列，不做无记录的前缀重启。
- worker/rank 对同一全局排列进行互斥分片。
- 流式/未知长度源不能伪装为可无放回有限源；采用明确的受限策略或 fail closed。
- `max_samples` 不再默认等价于原始前缀；有限可索引数据先做确定性选择。

验收：同 seed 顺序一致，不同 seed 顺序可变；单 source epoch 内无意外重复；各 shard 无重叠且并集等于全局排列。

### WP7：采样状态、覆盖与重复统计

- 增加 source sampler fingerprint、epoch、position、order/RNG 状态和恢复校验。
- 数据内容、数量或策略改变时拒绝恢复旧 sampler state。
- 统计 total draws、unique samples、coverage、repeat count、max repeat、source epoch 和 worker/rank overlap。
- 将运行状态暴露给训练可观测性和未来 checkpoint metadata。

验收：中断恢复后的序列与不中断序列一致；状态不匹配明确失败。

### WP8：配置接口与 fail-closed 校验

- 增加通用数据处理和采样策略配置，但不规定实验最优值。
- 校验无放回策略与源能力、超长策略、worker/rank 参数和 frozen release 指纹。
- 禁止数据源加载失败时静默改变训练混合；需要显式 variant 或报错。
- 日志同时报告配置意图与实际消费统计。

验收：不兼容配置在训练启动前失败；实际有效策略可从日志和 manifest 还原。

## 6. 测试与可靠性策略

1. 单元测试：文本、ID/hash、校验、tokenization、采样、manifest。
2. 性质测试：输入顺序变化、seed、边界长度、state round-trip、shard 互斥。
3. 集成测试：小型多源 fixture 从 canonical 构建到 frozen release，再由 loader 消费。
4. 回归测试：现有 chat、extra dataset、custom reasoning、RL sampler 和训练配置测试。
5. 全量测试：在提交前运行完整 `pytest`；环境性不可运行项必须明确记录。
6. Git 审查：只提交本阶段相关文件，不替训练者修改格式、长度、混合权重等实验策略，不包含生成数据或缓存。

## 7. 交付物

- 通用文本规范化和 canonical contract 模块；
- 安全 SFT tokenization 与结构化长度诊断；
- frozen release builder/validator 和 manifest；
- 确定性 source sampler、分片和恢复状态；
- 数据覆盖、重复与拒绝统计；
- 配置校验和训练接入；
- 完整测试和实施说明；
- `mote-sft-v2` 分支提交并推送到 `origin`。

## 8. 完成定义

- 不再存在全局换行/缩进压平；
- accepted SFT 样本的 target 左截为 0、空监督为 0；
- 训练质量筛选可以完全前移至冻结发布阶段；
- source epoch 内无意外重复，worker/rank 分片无重叠；
- 相同发布物、策略和 seed 可复现；
- 重复、覆盖、拒绝和实际 token 贡献可报告；
- 具体数据集判断和训练策略选择仍保持在本阶段边界之外；
- 相关测试通过，变更已提交并推送。

## 9. 实施与验证记录

本计划中的 WP1-WP8 已实现。代码端提供了无损规范化、canonical contract、监督门禁、冻结发布、确定性无放回遍历、rank/worker 分片、采样状态恢复、覆盖/重复统计，以及与训练 checkpoint 和观测日志的连接。

验证结果：

- 第二阶段定向回归：`45 passed`；
- 排除既有 Windows 平台专属测试文件后的全仓回归：`334 passed, 3 skipped`；
- 完整测试：`361 passed, 3 skipped, 13 failed`。剩余 13 项均为本次未修改模块的既有 Windows 平台差异：4 项 POSIX 路径语义断言，以及 9 项依赖 `os.getpgrp/os.getpgid` 的 vLLM actor 测试；
- `git diff --check` 通过。

这些平台兼容问题未被夹带到第二阶段整改中，建议另立任务处理。
