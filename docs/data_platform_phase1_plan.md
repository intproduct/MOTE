# 通用数据处理平台第一阶段计划

分支：`mote-sft-v2`

## 1. 目标

在现有数学 reasoning SFT 冻结发布链路之外，建立可复用的两级数据发布结构：

```text
固定版本原始缓存
  -> 模型无关 clean release
  -> tokenizer/template/seq-length 绑定的 training release
  -> stage/mixture 配置消费
```

第一阶段必须支持五类数据平面：

- `pretraining`：Wiki、FineWeb 等连续文本；
- `general_sft`：通用 prompt/response 或 messages；
- `reasoning_sft`：数学和推理监督数据；
- `code_pretraining`：原始代码连续文本；
- `code_sft`：代码问题、解释、修改或测试反馈类监督数据。

## 2. 边界

本阶段实现：

1. 固定 Hugging Face revision 的独立下载/缓存入口；
2. 来源、split、revision、许可证、允许用途和获取时间登记；
3. 多数据平面 canonical clean record；
4. 基础文本规范化、敏感模式扫描和可审计质量 flags；
5. 原始哈希、规范化精确去重和跨来源去重；
6. 可选、确定性的 MinHash/LSH 近重复检测；
7. benchmark registry 的精确与近似污染检测；
8. accepted/quarantine/manifest/hash 的不可变 clean release；
9. tokenizer 绑定、SFT label mask、EOS、长度隔离和长度桶；
10. 冻结预训练 release 的训练端只读消费接口；
11. 现有 frozen SFT v1/v2 和数学专项 source pipeline 保持兼容。

本阶段不实现：

- embedding 语义去重；
- 神经质量分类器或 LLM judge；
- 自动 PII 实体识别平台；
- 不受信任代码执行或单元测试沙箱；
- 跨节点分布式调度；
- SFT 跨样本 packing。

这些能力必须通过明确的后续插件或独立执行节点接入，不能用简单启发式规则冒充。

## 3. 数据登记与协议

每个 source 必须声明：

- `source_name`、`kind`、`path`、`split`、不可变 `revision`；
- `data_plane` 和对应字段映射；
- `license`；
- 非空 `allowed_uses`；
- `acquired_at`；
- 可选语言、领域和上游元数据。

clean record 保留来源身份、原始与规范化哈希、规范化内容、结构化 messages/prompt/response、质量 flags、质量分数、许可证、allowed uses 和 provenance。tokenizer 信息不得进入 clean record 或 clean manifest。

## 4. 模型无关 clean release

构建器只读本地缓存，不在发布过程中联网。输出：

- `accepted.jsonl`：通过当前治理策略的 canonical clean records；
- `quarantine.jsonl`：记录完整 canonical record、reason code 和诊断；
- `manifest.json`：来源登记快照、策略、计数、哈希与 release fingerprint。

基础门禁包括空内容、控制字符/替换字符比例、最小字符数、重复行比例、schema 完整性和可配置敏感模式扫描。边界样本进入 quarantine，不进行静默丢弃。该扫描只覆盖确定性邮箱/密钥模式，不等同于完整 PII 实体识别平台。

去重顺序：

1. source identity；
2. raw content hash；
3. normalized content hash；
4. 可选 MinHash/LSH；
5. benchmark exact/near contamination。

## 5. Tokenizer 绑定 training release

绑定器输入经过验证的 clean release，并显式记录：

- clean release fingerprint；
- tokenizer revision/fingerprint；
- chat template fingerprint；
- max length、EOS 和 mask 策略；
- length bucket boundaries；
- packing 策略。

预训练数据使用 causal-LM labels；SFT 数据使用 prompt mask 或 chat-template mask。任何溢出、空监督、EOS 或 mask 异常均进入 quarantine。第一阶段 packing 只允许对 causal-LM 数据显式开启；SFT 默认且强制不跨样本 packing。

## 6. 训练端边界

- frozen pretraining release 在训练启动时校验 manifest、文件哈希、tokenizer fingerprint 和 max length；
- 训练时不得重新 tokenize；
- frozen SFT 继续沿用现有严格校验；
- 动态 Hugging Face 路径保留用于探索，但不属于生产 clean/training release。

## 7. 验收

测试必须覆盖：

1. 五类 data plane 的 canonicalization；
2. 缺少许可证、allowed uses 或固定 revision 时 fail closed；
3. clean release 不含 tokenizer 绑定字段；
4. raw、normalized、跨来源和 source identity 去重；
5. MinHash 近重复及 benchmark contamination；
6. accepted/quarantine/manifest 哈希与禁止覆盖；
7. tokenizer 绑定后的长度桶、EOS、SFT mask 和 overlong 隔离；
8. frozen pretraining release 的 tokenizer/max-length 不匹配拒绝；
9. 现有 stage2/stage3 SFT 测试不回退。

## 8. 实施结果

第一阶段已经实现：

- `data/clean_release.py`：五类数据平面、治理登记、基础清洗、敏感模式 flags、外部质量分数映射、精确去重、MinHash/LSH、benchmark 污染和不可变 clean release；
- `data/training_release.py`：tokenizer 绑定、causal/SFT objective 隔离、mask/EOS/长度门禁、长度桶、预训练 greedy packing、manifest 和 preflight；
- `cli/cache_hf_dataset.py`：固定 revision 的独立 Hugging Face 缓存；
- `cli/build_clean_release.py`、`cli/bind_training_release.py`、`cli/preflight_training_release.py`：两级发布命令；
- `FrozenTokenTask` 与 `extra_datasets.source=frozen_release`：训练端只读消费和 fingerprint/max-length/objective 校验；
- `configs/data_platform_registry.example.json` 和使用说明。

验证结果：

- 新平台、CLI 和扩展数据集测试：23 passed；
- 新平台与既有 stage2/stage3 数据链路组合：77 passed；
- 排除既有 Windows/POSIX 不兼容文件的全仓测试：397 passed，3 skipped；
- 完整 Windows 测试：424 passed，3 skipped，13 failed；13 项均为既有 POSIX 路径语义以及 Windows 缺少 `os.getpgrp/os.getpgid`，与本阶段无关；
- Python `compileall` 通过。

真实数据缓存、正式许可证登记和生产 tokenizer 不在本机工作区中，因此本阶段没有生成或声称生成真实 clean rc1 / bound rc1，也没有用真实 FineWeb、Wiki 或 The Stack 执行吞吐基准。
