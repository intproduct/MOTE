# SFT v2 通用数据处理使用说明

本说明对应 `mote-sft-v2` 分支的第二阶段实现。它只覆盖通用处理正确性、冻结发布和确定性采样，不替代具体数据集的正确性审计。

## 1. Canonical JSONL 最小输入

每行至少包含稳定来源身份和一种 SFT 内容表示：

```json
{
  "source_name": "example",
  "source_revision": "commit-or-release",
  "source_split": "train",
  "source_row_id": "42",
  "data_plane": "sft",
  "task_type": "single_turn",
  "prompt": "Question:\n...",
  "target": "Answer:\n..."
}
```

也可使用结构化 `messages`。来源身份缺失、内容非法、空监督或超过精确 token 长度的行会进入 quarantine。

## 2. 构建冻结发布

```bash
python -m MOTE.cli.build_sft_release \
  --input_jsonl canonical.jsonl \
  --output_dir /data/releases/math-sft-v2 \
  --tokenizer /models/Qwen3-8B \
  --max_length 1024 \
  --pipeline_version sft-v2
```

输出：

- `accepted.jsonl`：已经 materialize 的 `input_ids/labels` 和监督诊断；
- `quarantine.jsonl`：来源定位、reason code 和错误证据，不复制原始正文；
- `manifest.json`：文件 hash、tokenizer 指纹、长度、计数和 release fingerprint。

发布目录不可覆盖。构建在同级临时目录完成并通过 hash/行数校验后原子发布。

## 3. 训练消费冻结发布

```json
{
  "data": {
    "seq_len_run": 1024,
    "use_frozen_sft_release": true,
    "frozen_sft_release_dir": "/data/releases/math-sft-v2",
    "frozen_sft_exclusive": true,
    "wt_frozen_sft": 1.0,
    "source_sampling_mode": "deterministic_strict",
    "source_shuffle": true,
    "source_max_epochs": 1,
    "fail_on_dynamic_skip": true
  }
}
```

训练启动时会重新验证 release 文件、manifest、`seq_len_run` 和 tokenizer 指纹。任何不一致都会失败，不会退回动态 SFT 数据源。`frozen_sft_exclusive` 只控制 task/SFT 池；pretrain 数据是否参与仍由 stage 和既有权重配置决定。

## 4. 采样策略

- `deterministic_auto`：有限可索引源采用无放回确定性排列；不可索引源显式使用分片顺序遍历。
- `deterministic_strict`：任何不可索引源均拒绝启动，适合冻结 SFT 生产训练。
- `legacy_sequential`：兼容旧行为，仅用于对照实验。
- `source_max_epochs=0`：不限制 source epoch；正整数限制重复暴露轮数，超出时失败。
- `fail_on_dynamic_skip=true`：训练时出现缺字段或动态拒绝立即失败；旧实验需要兼容时必须显式关闭。

有限源的 `max_samples` 在确定性排列后选择，不再等同于读取源前缀。worker/rank 使用同一全局排列的互斥切片。

## 5. 恢复与可观测性

checkpoint 元数据保存：

- 数据与采样策略 fingerprint；
- source epoch、position 和已消费计数；
- task-choice RNG state；
- unique coverage、repeat count、最大重复和 source epoch；
- batch prompt/target/supervised/total token 统计。

精确数据恢复目前要求 `dataloader_num_workers=0`。多 worker 训练仍能互斥分片，但父进程无法可靠收集各 worker 的实时 cursor，因此 exact resume 会明确拒绝该组合。

## 6. 本阶段未解决的问题

- 上游答案真实性和 verifier 可靠性；
- 跨源同题、语义去重和 benchmark 污染；
- 数据许可、PII 和领域质量评级；
- 最优格式、序列长度、监督模式和混合权重；
- packing、代码执行和工具调用验证。

这些能力应建立在本阶段产生的 canonical identity、reason code、manifest 和冻结发布之上。
