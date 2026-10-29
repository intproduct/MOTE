# 通用数据平台第一阶段使用说明

## 1. 两级发布原则

`clean release` 与 tokenizer、chat template、最大长度和 label mask 无关，可以在训练配置冻结前提前构建。`bound training release` 必须在这些训练协议稳定后生成。两种发布都禁止覆盖已有目录。

## 2. 固定 Hugging Face 来源

生产数据先下载为固定 revision 的本地缓存：

```bash
python -m MOTE.cli.cache_hf_dataset \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --split train \
  --revision IMMUTABLE_COMMIT_OR_RELEASE \
  --output_dir /data/raw/fineweb_sample10bt \
  --license odc-by-1.0 \
  --allowed_use training
```

输出目录包含 `dataset/`、`source_receipt.json` 和 `_READY`。下载和发布是两个独立步骤；clean builder 不联网。

## 3. 构建模型无关 clean release

复制并填写 `configs/data_platform_registry.example.json`。所有 `REPLACE_WITH...` 必须换成真实值，尤其是 revision、许可证、允许用途和获取时间。

```bash
python -m MOTE.cli.build_clean_release \
  --registry configs/data_platform_registry.rc1.json \
  --output_dir /data/releases/clean-v1-rc1
```

输出：

- `accepted.jsonl`：canonical clean records；
- `quarantine.jsonl`：完整 record、reason code 和诊断；
- `manifest.json`：来源快照、质量策略、去重/污染策略、计数、文件哈希和 fingerprint。

完整原始字段继续保存在不可变 raw cache；canonical record 默认只保存训练所需字段和 source `preserve_fields` 指定的审计字段，避免在 TB 级 clean release 中复制整个原始对象。

clean manifest 不包含 tokenizer fingerprint、chat template 或 max length。近重复使用确定性 MinHash/LSH；在 TB 级数据正式运行前，必须先用 1% 数据测量吞吐和 SQLite 临时空间。

`execution.workers` 只并行 canonicalization、HTML/Unicode 规范化和基础质量计算，主进程仍按 registry 顺序执行去重与发布，因此输出顺序确定。EPYC 7A23 建议从 24–32 个 worker 起测；不要未经吞吐测试直接使用 96 个逻辑线程。

`quality_policy.reject_sensitive_flags` 可以对私钥、AWS key 和显式 credential assignment 执行 fail-closed 隔离；邮箱等模式始终记录在 `sensitive_flags`，但默认不自动删除。已有 GPU/LLM 质量分数可通过 source 的 `quality_score_fields` 映射进入 canonical record，平台不会把外部分数冒充为本地验证结果。

## 4. 绑定预训练数据

```bash
python -m MOTE.cli.bind_training_release \
  --clean_release_dir /data/releases/clean-v1-rc1 \
  --output_dir /data/releases/pretraining-qwen-1280-rc1 \
  --tokenizer /models/Qwen \
  --tokenizer_revision IMMUTABLE_TOKENIZER_REVISION \
  --max_length 1280 \
  --planes pretraining,code_pretraining \
  --length_buckets 256,512,768,1024,1280 \
  --packing_mode pretrain_greedy \
  --workers 24 \
  --prefetch_factor 4 \
  --trust_remote_code
```

`pretrain_greedy` 只连接完整 causal-LM 样本，每个成员保留独立 EOS，不切开单条记录。默认 `packing_mode=none`。

tokenizer worker 使用有界线程队列并保持输入顺序；生产 tokenizer 是否能随线程数线性扩展必须通过样本吞吐测试确认。若 tokenizer 实现不保证并发安全，使用默认 `--workers 1`。

## 5. 绑定 SFT 数据

```bash
python -m MOTE.cli.bind_training_release \
  --clean_release_dir /data/releases/clean-v1-rc1 \
  --output_dir /data/releases/general-sft-qwen-1280-rc1 \
  --tokenizer /models/Qwen \
  --tokenizer_revision IMMUTABLE_TOKENIZER_REVISION \
  --max_length 1280 \
  --planes general_sft \
  --length_buckets 256,512,768,1024,1280 \
  --trust_remote_code
```

一个 bound release 不能混合 causal-LM 与 SFT objective。SFT 跨样本 packing 在第一阶段被明确禁止；messages 数据必须由生产 tokenizer 提供有效 chat template。

绑定完成后至少预检 1,000 条实际记录：

```bash
python -m MOTE.cli.preflight_training_release \
  --release_dir /data/releases/pretraining-qwen-1280-rc1 \
  --min_records 1000
```

## 6. 训练配置消费

冻结预训练发布通过 `extra_datasets` 接入：

```json
{
  "name": "frozen_pretraining_rc1",
  "source": "frozen_release",
  "path": "/data/releases/pretraining-qwen-1280-rc1",
  "format": "text",
  "group": "pretrain",
  "bucket": "pretrain_general",
  "weight": 1.0
}
```

训练启动时会校验文件哈希、tokenizer fingerprint、max length 和 objective，训练期间不再 tokenize。通用/代码 SFT 可用同一 `frozen_release` source 并设置 `group=task`；现有数学 `frozen_sft` 发布路径保持兼容。

## 7. 尚未包含的高级能力

当前 MinHash 是 CPU 确定性近重复，不等于 embedding 语义去重。PII 实体识别、神经质量分类、LLM judge、代码执行沙箱和跨节点调度仍需独立实现和审计。原始 The Stack 级别代码库不应直接在单机上重新全量处理，应先按语言、许可证和仓库策略选择子集。

通用 `reasoning_sft` adapter 只提供结构化治理，不替代现有数学专项的答案提取、OpenR1 verified trace 和程序复算门禁。正式数学 release 仍应先经过 stage3 专项管线，或在后续将这些验证器作为 domain plugin 接入 clean builder。
