# SFT v2 第三阶段离线构建与单卡准入

本说明对应 `math_raw_v2` 数据集层整改。source adapter、答案解析、质量筛选、去重和 tokenization 只允许在离线发布节点运行；训练节点只消费 frozen release。

## 1. 准备不可变来源清单

复制并填写：

```text
configs/sft_v2_rc1_sources.example.json
```

九类 adapter 为：`gsm8k_main`、`gsm8k_socratic`、`svamp`、`synthetic_arithmetic`、`hendrycks_math`、`metamath`、`openr1`、`numinamath`、`openthoughts`。

每个真实来源必须指向本地缓存，revision 必须是上游 commit、数据发布版本或缓存 fingerprint。`latest/main/master/unknown/REPLACE_WITH...` 不会作为发布 revision；本地文件和 Hugging Face Dataset 可从 SHA-256/`_fingerprint` 派生。构建过程不下载远端数据，来源不可访问时立即失败。

`planned_consumed_sequences` 必须填写为本次实验计划消费的 SFT 序列数；builder 会结合最终 `accepted_count` 输出预计 source epoch 和最大样本暴露次数。

默认质量下限为 Tier B：

- A：程序复算、权威参考或 `correctness_math_verify=true`；
- B：权威/上游参考可解析并通过基础门禁；
- C：仅 parser 成功但无法可靠验证，默认不进入生产 release；
- Q：隔离。

## 2. 构建生产候选 release

```bash
python -m MOTE.cli.build_sft_release \
  --source_manifest configs/sft_v2_rc1_sources.rc1.json \
  --output_dir /data/releases/fitmotn-sft-v2-repaired-rc1 \
  --tokenizer /models/Qwen3-0.6B \
  --tokenizer_revision qwen-immutable-commit-or-release \
  --max_length 1280 \
  --pipeline_version sft-v2-stage3-rc1
```

大规模候选不会全部保存在内存中。adapter 结果进入同级临时 SQLite staging，完成 exact/problem-group 排名后流式交给 frozen builder；成功或失败均清理 staging 文件。

输出文件：

- `accepted.jsonl`：最终 `input_ids/labels`、canonical 身份和质量证据；
- `quarantine.jsonl`：来源身份、reason code 和诊断，不进入训练；
- `long_context.jsonl`：通过内容门禁但超过 seq1280 的 canonical 候选；
- `manifest.json`：版本、hash、来源、验证、去重、质量层和 token 分布。

## 3. 核心策略

OpenR1 只接受 `correctness_math_verify=true` 的候选；generations 与 verification 数组长度不一致、无 verified 候选、格式不完整或显式参考答案不一致时隔离。多条 verified trace 使用生产 tokenizer 的真实长度选最短，索引作为稳定 tie-breaker。

`math_raw_v2` 固定格式：

```text
Question:
{problem}

Solution:
{reasoning}

#### {final_answer}
```

renderer 与 parser 必须 round-trip。final answer 不再使用“solution 最后一行”猜测；非 proof 的 `answer == solution`、空答案、异常长度/比例和未知答案类型均隔离。

去重策略：

- 相同 source identity 只能出现一次；
- 相同 problem+reasoning+answer+format 只保留一个；
- normalized problem 相同进入同一 group；
- 每组默认最多两个解法；
- 排名顺序：verified、quality tier、来源权威性、真实 token 长度、稳定来源身份。

## 4. Manifest 审计

发布前至少检查：

- OpenR1 accepted unverified 为 0；
- `answer_equals_solution`、`empty_supervision`、EOS/mask 异常为 0；
- accepted `total_tokens.max <= 1280`；
- exact duplicate 和 problem-group 拒绝数量符合政策；
- 每源 scanned/accepted/quarantine、质量层和 supervised-token 占比合理；
- `pretrain_data_plane.the_stack.status` 与实验配置一致。

`long_context_count` 不是 seq1280 accepted 数，不能混入当前训练。

## 5. Loader preflight

```bash
python -m MOTE.cli.preflight_sft_release \
  --release_dir /data/releases/fitmotn-sft-v2-repaired-rc1 \
  --min_records 1000
```

该命令重新验证 manifest/file hash，并顺序检查至少 1,000 条 accepted 记录的长度、labels、mask、监督、EOS、diagnostics、source/canonical 唯一性和 OpenR1 verification。它不会加载 tokenizer、不会动态解析数据、不会启动训练。

## 6. 单卡配置和暴露上限

配置副本：

```text
configs/fitmotn_k8_nodecay_global_8e_sft_v2_rc1.json
```

该副本选择 pure-SFT/no-code：Wiki、FineWeb 和 The Stack 均关闭，避免把 pretrain data plane 计入 SFT。`frozen_sft_exclusive` 只排除动态 SFT，因此这些开关仍被显式关闭。

配置中的 `source_max_epochs=1` 是保守占位上限，不允许无限重复。正式运行前计算：

```text
required_source_epochs = ceil(planned_consumed_sequences / manifest.accepted_count)
```

将结果显式写入实验配置和实验报告；如果不希望任何样本重复，保持 1 并相应降低计划消费序列数。

## 7. 实验准入顺序

1. 构建 rc1；
2. 审查完整 manifest 和 quarantine reason；
3. 分层人工抽查 OpenR1、MetaMath、Numina/OpenThoughts 等高风险来源；
4. 运行 1,000-record frozen preflight；
5. 运行 100–500 update smoke；
6. 运行 1,000–2,000 update canary；
7. 能力、稳定性和 resume 通过后再开始完整单卡训练。

Smoke 只验证持续运行、OOM/NaN/跳步、checkpoint/resume、manifest 对账和 eval parser，不用短程 loss 代替数据质量判断。
