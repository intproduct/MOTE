# FitMoTN 全中文使用说明

FitMoTN 是一个面向研究实验的 MoTN 训练、强化学习、模型导出与评测仓库。项目的核心目标是在不改变原始 MoTN 方法语义的前提下，将旧实验代码整理成一套配置驱动、可恢复、可观测、可评测的工程流程。

本说明基于当前仓库的主 README 和实际代码整理，覆盖从环境安装到 Stage 4 vLLM 验收的完整使用方法。命令、Python 模块名和 JSON 配置字段保持代码中的英文名称，所有解释均使用中文。

## 一、当前能力概览

当前仓库主要支持：

- 对 Qwen 风格 MLP 层进行 MoTN patch；
- 冻结基础模型，只训练 patched MoTN 参数；
- 两阶段混合数据训练；
- bucketed reasoning recovery；
- baseline、中途和最终评测；
- patched checkpoint 保存与恢复；
- Hugging Face、lm-eval 和 EvalScope 评测；
- GSM8K GRPO 强化学习；
- MGPO 与 Long2Short 实验机制；
- 通用额外数据集接入；
- 完整 Hugging Face 模型导出；
- 通过 vLLM Transformers backend 进行推理；
- 使用 vLLM 生成 RL rollout；
- rollout policy 版本同步和滞后检查；
- vLLM 权重传输能力探测与实验性 NCCL adapter；
- 独立 vLLM rollout 子进程控制面；
- routing、expert usage、吞吐和资源观测。

当前仓库仍属于研究型 MVP，不是通用分布式训练平台。主要限制包括：

- 主要针对 Qwen 风格的 `model.layers[*].mlp`；
- 原始 MLP 需要包含 `gate_proj`、`up_proj`、`down_proj`；
- raw patched checkpoint 不能直接交给 vLLM；
- 没有原生 vLLM MoTN、expert parallel 或融合 CUDA kernel；
- 没有完成多机分布式专门适配；
- 独立 rollout actor 当前是同步请求模式，不是异步流水线；
- vLLM、CUDA 和 NCCL 必须在 NVIDIA 环境单独验收。

## 二、仓库获取

推荐将本地目录命名为 `fitmotn`：

```bash
git clone https://github.com/intproduct/MOTE.git fitmotn
cd fitmotn
```

使用 SSH：

```bash
git clone git@github.com:intproduct/MOTE.git fitmotn
cd fitmotn
```

## 三、安装环境

### 1. 基础开发环境

```bash
pip install -e ".[dev]"
```

基础依赖包括：

- Python 3.10 或更高版本；
- PyTorch；
- Transformers；
- Datasets；
- NumPy；
- tqdm；
- pytest。

验证基础环境：

```bash
python -c "import torch, transformers, datasets; print('基础环境正常')"
pytest -q
```

测试启动逻辑会把 `fitmotn` 和兼容包名 `MOTE` 绑定到当前 checkout，避免误用相邻目录中的旧 editable 安装。

### 2. lm-eval

```bash
pip install -U "lm_eval[hf]"
```

验证：

```bash
python -c "import lm_eval; print('lm-eval 正常')"
```

### 3. EvalScope

```bash
pip install -U evalscope
```

### 4. vLLM

仅在受支持的 NVIDIA Linux 环境安装：

```bash
pip install -e ".[vllm]"
```

验证：

```bash
python -c "import vllm; print(vllm.__version__)"
```

Mac 可以运行配置检查、模型导出、CPU/mock 测试和 actor 控制面测试，但不能据此认定真实 vLLM、CUDA 或 NCCL 已通过。

## 四、统一路径配置

建议在运行前设置：

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUT_ROOT=/path/to/outputs
export PROJECT_ROOT="$(pwd)"
```

各变量用途：

- `MODEL_ROOT`：基础模型目录；
- `DATA_ROOT`：原始数据和本地 shard；
- `CACHE_ROOT`：Hugging Face 数据集缓存；
- `OUTPUT_ROOT`：训练、评测、RL 和导出结果；
- `PROJECT_ROOT`：相对路径解析根目录。

正式实验建议显式设置这些变量。`runtime.dev_mode=true` 仅适用于本地开发和路径回退检查。

## 五、项目目录

```text
fitmotn/
├── cli/                    命令行入口
├── config/                 配置 schema、默认值和校验
├── data/                   数据加载、缓存、适配和 tokenization
├── diagnostics/            诊断工具
├── eval/                   HF、lm-eval、EvalScope 和 vLLM 评测
├── export/                 Stage 4 模型导出
├── rl/                     GRPO、rollout、vLLM 同步与 actor
├── scripts/                分析和验收脚本
├── tasks/                  训练与评测任务定义
├── tests/                  自动测试
├── train/                  SFT 和 RL 控制器
├── ADTN.py                 MoTN 核心实现镜像
├── gate.py                 gate 行为
├── patching.py             Qwen MLP patch
├── README.md               主说明
└── README_ZH.md            全中文使用说明
```

## 六、快速开始

### 1. 使用示例配置训练

```bash
python -m fitmotn.cli.train --config_json ./fitmotn_config.example.json
```

建议先复制配置：

```bash
cp fitmotn_config.example.json my_fitmotn_config.json
```

修改模型、数据和输出路径后运行：

```bash
python -m fitmotn.cli.train --config_json ./my_fitmotn_config.json
```

### 2. 使用命令行参数进行最小训练

```bash
python -m fitmotn.cli.train \
  --model_path '${MODEL_ROOT}/Qwen3-0.6B' \
  --output_root '${OUTPUT_ROOT}/fitmotn_runs' \
  --run_name demo_run \
  --batch_size 4 \
  --grad_accum 1 \
  --steps 100 \
  --seq_len_run 1024 \
  --lr 3e-5 \
  --layers_to_patch last_quarter \
  --device cuda:0
```

复杂实验应优先使用 JSON。专家数量、gate、capacity、初始化、数据混合和评测协议等深度参数不适合长期通过 CLI 管理。

## 七、核心配置说明

完整模板见：

- [fitmotn_config.example.json](./fitmotn_config.example.json)

### 1. `model`

常用字段：

- `model_path`：基础模型路径；
- `device`：训练设备；
- `torch_dtype`：模型精度；
- `use_amp`：是否启用自动混合精度；
- `trust_remote_code`：是否允许模型 remote code；
- `layers_to_patch`：patch 层范围；
- `E`：专家数量；
- `d`：TensorBlock 深度或结构参数；
- `k_in`：输入分解配置；
- `topk`：每个 token 激活的专家数量；
- `gate_type`：gate 类型；
- `temperature`：routing 温度；
- `capacity_factor`：专家容量倍率；
- `drop_tokens`：超过容量时是否丢弃 token；
- `aux_coeff`：负载均衡辅助损失系数；
- `zloss_coeff`：router z-loss 系数；
- `warmup_ratio`：专家 warmup 比例。

当前 patch 语义保持：

```text
gate_proj(MoTN) + up_proj(MoTN) + activation + down_proj(MoTN)
```

基础模型默认冻结，只训练 patched 参数。

### 2. `data`

常用字段：

- `seq_len_run`：训练序列长度；
- `dataloader_num_workers`：DataLoader worker 数量；
- `tok_shard_dir`：本地 token shard；
- `fineweb_cache_path`：FineWeb 缓存；
- `code_cache_path`：代码数据缓存；
- `gsm8k_cache_path`：GSM8K 缓存；
- `mmlu_cache_path`：MMLU 缓存；
- `math_cache_root`：数学数据缓存；
- `use_wiki_local`、`use_fineweb`、`use_code`：启用预训练数据；
- `use_gsm8k_train`、`use_math_train`：启用 reasoning 数据；
- `wt_*`：对应数据源采样权重；
- `reasoning_supervision_mode`：reasoning 监督形式；
- `extra_datasets`：额外通用数据集。

研究 MVP 推荐：

```json
{
  "data": {
    "dataloader_num_workers": 0
  }
}
```

原因是 stage 切换按全局 step 驱动，多 worker 预取可能造成切换滞后。

### 3. `train`

常用字段：

- `batch_size`；
- `grad_accum`；
- `steps`；
- `epochs`；
- `lr`、`block_lr`、`router_lr`；
- `max_grad_norm`；
- `save_every_updates`；
- `eval_every_updates`；
- `stage_a_ratio`；
- `stage_a_pretrain_ratio`、`stage_a_task_ratio`；
- `stage_b_mode`；
- `stage_b_pretrain_ratio`、`stage_b_task_ratio`；
- `gate_freeze_steps`；
- `begin_t`、`end_t`；
- usage 和 heavy runtime stats 相关频率；
- `benchmark_train_only`。

### 4. `eval`

常用字段：

- `eval_backend`：`lm_eval`、`evalscope` 或 `both`；
- `primary_eval_backend`：主结果来源；
- `final_tasks`：最终评测任务；
- `runtime.apply_chat_template`；
- `runtime.enable_thinking`；
- `generation.temperature`；
- `generation.top_p`、`top_k`；
- `fewshot`；
- `limits`；
- `max_gen_toks`；
- `task_overrides`。

### 5. `output`

```json
{
  "output": {
    "root_dir": "${OUTPUT_ROOT}/fitmotn_runs",
    "run_name": "experiment_name",
    "overwrite_output_dir": false
  }
}
```

### 6. `runtime`

```json
{
  "runtime": {
    "dev_mode": false
  }
}
```

## 八、两阶段训练

一次训练只调用一次 `trainer.train()`，Stage A 和 Stage B 是同一 run 内部的动态数据混合阶段。

### Stage A

通常以预训练数据为主，同时混入一部分任务数据，用于稳定 patched 模型并恢复基础能力。

### Stage B

提高 reasoning/task 数据比例，或者进入恢复性训练模式。

常见配置：

```json
{
  "train": {
    "stage_a_ratio": 0.6,
    "stage_a_pretrain_ratio": 0.7,
    "stage_a_task_ratio": 0.3,
    "stage_b_mode": "mixed",
    "stage_b_pretrain_ratio": 0.45,
    "stage_b_task_ratio": 0.55
  }
}
```

推荐参考：

- [fitmotn_reasoning_recovery_min_b.json](./fitmotn_reasoning_recovery_min_b.json)
- [fitmotn_reasoning_recovery_conservative_b.json](./fitmotn_reasoning_recovery_conservative_b.json)

## 九、Bucketed Reasoning Recovery

Bucketed 模式用于把 reasoning 数据按核心、辅助、不同来源和难度分桶控制，而不是只使用一个扁平 task pool。

启用方式：

```json
{
  "train": {
    "task_bucket_mode": "bucketed"
  }
}
```

推荐配置：

- [fitmotn_reasoning_recovery_gsm8k_core_bucketed.json](./fitmotn_reasoning_recovery_gsm8k_core_bucketed.json)

启动：

```bash
python -m fitmotn.cli.train \
  --config_json ./fitmotn_reasoning_recovery_gsm8k_core_bucketed.json
```

检查 reasoning 样本：

```bash
python -m fitmotn.cli.debug_reasoning_sample \
  --config_json ./fitmotn_reasoning_recovery_gsm8k_core_bucketed.json
```

## 十、额外数据集

`data.extra_datasets` 可以接入不需要专门编写 task 类的数据集。

支持的数据来源：

- `hf`；
- `local_jsonl`；
- `jsonl`；
- `jsonl_gz`；
- `load_from_disk`；
- `auto`。

支持的数据格式：

- `text`；
- `chat_messages`；
- `prompt_response`；
- `reasoning_qa`。

示例：

```json
{
  "data": {
    "extra_datasets": [
      {
        "name": "local_alpaca",
        "source": "local_jsonl",
        "path": "${DATA_ROOT}/alpaca.jsonl",
        "format": "prompt_response",
        "instruction_field": "instruction",
        "input_field": "input",
        "output_field": "output",
        "group": "task",
        "weight": 1.0
      }
    ]
  }
}
```

训练前检查：

```bash
python -m fitmotn.cli.inspect_dataset \
  --config ./my_fitmotn_config.json \
  --max-samples 3
```

带 tokenizer 检查：

```bash
python -m fitmotn.cli.inspect_dataset \
  --config ./my_fitmotn_config.json \
  --tokenizer '${MODEL_ROOT}/Qwen3-0.6B'
```

详细说明：

- [docs/extra_datasets.md](./docs/extra_datasets.md)

## 十一、训练输出

典型输出目录包含：

```text
run_dir/
├── train.log
├── train_light.jsonl
├── train.jsonl
├── usage.jsonl
├── mid_eval.jsonl
├── eval_summary.json
├── run_summary.json
├── checkpoints/
└── final_model/
```

`final_model/` 通常包含：

- `config.json`；
- tokenizer 文件；
- Trainer 模型权重；
- `fitmotn_state.pt`；
- `fitmotn_state.json`；
- `run_summary.json`。

`fitmotn_state.*` 保存：

- 基础模型路径；
- 实际 patch 层；
- MoTN 配置；
- FitMoTN 配置；
- checkpoint 格式；
- patched state dict；
- dtype 和 AMP；
- 可训练参数统计；
- baseline/mid/final 评测摘要；
- 环境快照。

新格式默认只保存 patched 层状态，不再重复保存整模型 state dict。旧 checkpoint 的完整 `state_dict` 仍可兼容读取。

## 十二、恢复 checkpoint

恢复顺序为：

1. 加载基础模型；
2. 按 checkpoint 中的真实 `layers_to_patch` 重新 patch；
3. 加载 `patch_state_dict`；
4. 旧格式回退到完整 `state_dict`；
5. 使用 `strict=False` 灌入状态。

这一步用于防止 patched 结构被普通 `save_pretrained()` 平铺或丢失。

## 十三、Hugging Face 与 lm-eval 评测

评测保存后的 checkpoint：

```bash
python -m fitmotn.cli.eval_hf \
  --model_or_ckpt /path/to/final_model \
  --device cuda:0
```

指定任务：

```bash
python -m fitmotn.cli.eval_hf \
  --model_or_ckpt /path/to/final_model \
  --device cuda:0 \
  --tasks gsm8k mmlu
```

统一评测入口：

```bash
python -m fitmotn.cli.eval_auto \
  --model_or_ckpt /path/to/final_model \
  --device cuda:0 \
  --eval_backend both \
  --primary_eval_backend lm_eval \
  --allow_backend_skip
```

## 十四、双评测后端

### lm-eval

定位为训练主链：

- 可直接评测内存中的 patched 模型；
- 支持 baseline、mid、final 连续评测；
- 不要求先保存 checkpoint；
- 推荐作为论文主结果后端。

### EvalScope

定位为路径/checkpoint 参考后端：

- 更适合已保存模型；
- 可作为官方口径或第二实现对照；
- 不保证支持训练中的内存模型；
- 不支持时会显式 skipped/warning，而不是静默吞掉。

推荐：

```json
{
  "eval": {
    "eval_backend": "both",
    "primary_eval_backend": "lm_eval"
  }
}
```

示例配置：

- [fitmotn_config.eval_dual_backend.example.json](./fitmotn_config.eval_dual_backend.example.json)
- [fitmotn_config.model_aligned_eval.example.json](./fitmotn_config.model_aligned_eval.example.json)

## 十五、模型原生评测协议

可以通过 JSON 控制 chat template、thinking、few-shot 和生成参数：

```json
{
  "eval": {
    "runtime": {
      "apply_chat_template": true,
      "enable_thinking": true,
      "think_end_token": "</think>"
    },
    "generation": {
      "do_sample": false,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 20
    },
    "fewshot": {
      "default": 0,
      "task_overrides": {
        "gsm8k": 8,
        "mmlu": 5
      }
    }
  }
}
```

当前 backend 不识别的字段不会静默丢弃，而会进入：

- `warnings`；
- `ignored_runtime_args`；
- `unsupported_runtime_args`；
- `unsupported_generation_args`。

## 十六、Stage 4 模型导出

### 1. metadata-only 导出

用于目录布局、manifest 和批量扫描，不是可推理模型：

```bash
python -m fitmotn.cli.export_hf \
  --checkpoint_dir /path/to/raw-fitmotn-checkpoint \
  --output_dir /path/to/metadata-export \
  --metadata_only \
  --validate
```

主要文件：

- `fitmotn_export_manifest.json`；
- `fitmotn_export_config.json`；
- `README.md`。

### 2. 完整 Hugging Face 导出

```bash
python -m fitmotn.cli.export_hf \
  --checkpoint_dir /path/to/raw-fitmotn-checkpoint \
  --output_dir /path/to/hf-export \
  --no-metadata_only \
  --base_model /path/to/base-model \
  --validate_layout
```

完整导出包含：

- `config.json`；
- `configuration_fitmotn.py`；
- `modeling_fitmotn.py`；
- 模型权重；
- export manifest；
- tokenizer 和 generation 配置。

加载方式：

```python
from transformers import AutoModel, AutoModelForCausalLM

decoder = AutoModel.from_pretrained("/path/to/hf-export", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/path/to/hf-export", trust_remote_code=True)
```

导出 wrapper 仍要求目标环境安装 `fitmotn`。

不要提交：

- 导出的大模型权重；
- raw checkpoint；
- 私有 tokenizer；
- 包含私有本地路径的 manifest。

## 十七、vLLM 离线评测

只能使用完整 Stage 4C Hugging Face export：

```bash
python -m fitmotn.cli.eval_vllm \
  --model_or_ckpt /path/to/hf-export \
  --model_impl transformers \
  --enforce_eager \
  --limit_per_task 32
```

支持范围：

- vLLM 原生兼容模型；
- 完整 FitMoTN HF export；
- GSM8K 和 MMLU 基本评测；
- 加载时间和生成吞吐记录。

不支持：

- raw patched checkpoint；
- metadata-only export；
- 原生 vLLM MoTN 注册；
- expert parallel；
- 融合 MoTN kernel；
- custom CUDA op。

错误路径会显式抛出 `NotImplementedError` 并输出 capability 信息，不会静默退化为基础模型。

## 十八、GRPO 强化学习

入口：

```bash
python -m fitmotn.cli.train_rl --config_json ./my_rl_config.json
```

基础配置：

```json
{
  "rl": {
    "enabled": true,
    "mode": "gsm8k_grpo",
    "max_steps": 100,
    "batch_size": 1,
    "group_size": 4,
    "grad_accum": 1,
    "lr": 5e-7,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "trainable_mode": "patch_only"
  }
}
```

当前 GRPO 中：

- rollout 可由 HF 或 vLLM 生成；
- reward、response mask、old/ref/new logprob 和 loss 仍由 PyTorch/HF 计算；
- policy version 按真实 optimizer update 维护；
- stale vLLM policy 默认拒绝；
- fallback 必须显式启用并记录原因。

## 十九、vLLM rollout

### 1. 进程内模式

```json
{
  "rl": {
    "rollout_backend": "vllm",
    "vllm_execution_mode": "in_process",
    "vllm_sync_strategy": "export_reload",
    "vllm_sync_every_updates": 1,
    "vllm_model_impl": "transformers",
    "vllm_enforce_eager": true,
    "vllm_gpu_memory_utilization": 0.7
  }
}
```

训练进程直接持有 vLLM engine。native runtime tensor inspection 和实验性 NCCL 只能使用该模式。

### 2. 独立子进程 actor

```json
{
  "rl": {
    "rollout_backend": "vllm",
    "vllm_execution_mode": "subprocess",
    "vllm_actor_start_method": "spawn",
    "vllm_device": "cuda:1",
    "vllm_sync_strategy": "export_reload",
    "vllm_sync_every_updates": 1
  }
}
```

该模式中：

- 训练进程保存和导出 policy；
- actor 子进程持有 vLLM engine；
- 有效 token IDs 通过同步控制通道发送；
- actor 返回生成 token IDs；
- actor 可执行加载、卸载、sleep、wake 和关闭；
- 推荐训练 GPU 与 rollout GPU 分离。

当前子进程模式只允许：

- `export_reload`；
- `weight_transfer_dryrun_static`。

明确禁止：

- 子进程 native NCCL；
- IPC transfer；
- runtime tensor dryrun；
- text prompt fallback。

本机控制面检查：

```bash
python scripts/smoke_vllm_actor_control.py
```

该命令只验证 spawn、请求和关闭，不加载 CUDA engine。

## 二十、vLLM policy 同步

### `export_reload`

稳定默认路径：

1. 保存当前训练 policy；
2. 生成完整 HF export；
3. 在 CPU 上验证 roundtrip；
4. 卸载旧 vLLM engine；
5. 从新 export 重建 engine；
6. 更新 policy version。

该路径正确性优先，但对大模型可能非常慢，必须把 checkpoint、export 和 rebuild 时间计入端到端 benchmark。

### dryrun

支持：

- `weight_transfer_dryrun_static`；
- `weight_transfer_dryrun_runtime`。

dryrun 会生成权重名字、shape、dtype、coverage、MoTN coverage 和 checksum 诊断，但不会执行原生 in-place transfer。为保证 rollout 可运行和 policy 新鲜，实际 policy 更新仍通过 `export_reload` 完成。

### NCCL adapter

默认严格门槛：

```json
{
  "rl": {
    "vllm_sync_strategy": "weight_transfer_nccl",
    "vllm_native_transfer_required_level": "four_phase"
  }
}
```

当前 request-style update-only API 必须显式启用：

```json
{
  "rl": {
    "vllm_execution_mode": "in_process",
    "vllm_sync_strategy": "weight_transfer_nccl",
    "vllm_native_transfer_required_level": "update_only"
  }
}
```

`update_only` 仍属于实验路径，只有在目标 NVIDIA 环境完成反复 transfer、checksum、rollout 和 teardown 验证后，才可用于正式训练。

## 二十一、Stage 4 CUDA 验收

完整方案：

- [docs/stage4_vllm_validation.md](./docs/stage4_vllm_validation.md)

两步 smoke 配置：

- [fitmotn_config.stage4_vllm_smoke.example.json](./fitmotn_config.stage4_vllm_smoke.example.json)

### 1. raw、HF export 与 vLLM 一致性

```bash
python scripts/validate_stage4_cuda.py \
  --raw-checkpoint /path/to/raw-checkpoint \
  --export-dir /path/to/hf-export \
  --output-json /path/to/evidence/stage4_parity.json \
  --device cuda:0 \
  --enforce-eager
```

### 2. 两步 RL smoke

```bash
python -m fitmotn.cli.train_rl \
  --config_json ./fitmotn_config.stage4_vllm_smoke.example.json
```

### 3. 验证 RL 日志

```bash
python scripts/validate_stage4_rl_run.py \
  /path/to/rl_train.jsonl \
  --min-updates 2 \
  --output-json /path/to/evidence/stage4_rl_smoke.json
```

正式验收至少需要：

- raw 与 HF export greedy 输出一致；
- HF 与 vLLM token match 达到设定阈值；
- 两个 optimizer update 完成；
- loss 有限；
- `policy_lag_updates=0`；
- 没有非预期 HF fallback；
- 保存环境版本、配置、manifest、日志和验收 JSON。

## 二十二、Usage 与论文级观测

主要文件：

- `train_light.jsonl`：高频轻量 usage；
- `train.jsonl`：训练 step/update 主记录；
- `usage.jsonl`：低频详细 usage；
- `mid_eval.jsonl`：中途评测；
- `run_summary.json`：run 级汇总。

三级频率：

- `usage_light_every`：GPU 侧轻量 usage snapshot；
- `usage_light_jsonl_every`：轻量 snapshot 写盘；
- `usage_report_every`：详细 usage 聚合和写盘。

重统计开关：

- `heavy_log_every`；
- `enable_heavy_runtime_stats`；
- `enable_grad_param_norm`；
- `enable_cuda_snapshot`。

纯吞吐 benchmark：

```json
{
  "train": {
    "benchmark_train_only": true
  }
}
```

该模式会关闭部分研究观测，不适合作为 routing/usage 实验配置。

参考配置：

- [fitmotn_config.usage_light_report.example.json](./fitmotn_config.usage_light_report.example.json)
- [fitmotn_config.benchmark_train_only.example.json](./fitmotn_config.benchmark_train_only.example.json)

## 二十三、RL 性能分析

```bash
python scripts/analyze_rl_timing.py /path/to/rl_train.jsonl --last-n 100
```

评估 vLLM 是否带来收益时，必须查看完整 update wall time，而不是只看生成 tokens/s。至少记录：

- tokenize 时间；
- rollout 时间；
- reward 时间；
- old/ref/new logprob 时间；
- backward 时间；
- checkpoint/export 时间；
- engine rebuild 时间；
- native transfer 时间；
- CPU/GPU 内存峰值；
- policy lag。

## 二十四、常用入口汇总

```bash
# 监督训练
python -m fitmotn.cli.train --config_json ./fitmotn_config.example.json

# 强化学习
python -m fitmotn.cli.train_rl --config_json ./my_rl_config.json

# HF 评测
python -m fitmotn.cli.eval_hf --model_or_ckpt /path/to/model --tasks gsm8k

# 双后端评测
python -m fitmotn.cli.eval_auto --model_or_ckpt /path/to/model --eval_backend both

# vLLM 评测
python -m fitmotn.cli.eval_vllm --model_or_ckpt /path/to/hf-export --model_impl transformers

# Hugging Face 导出
python -m fitmotn.cli.export_hf --checkpoint_dir /path/to/checkpoint --output_dir /path/to/export

# 额外数据集检查
python -m fitmotn.cli.inspect_dataset --config ./my_fitmotn_config.json

# GSM8K boundary 数据
python -m fitmotn.cli.build_boundary_gsm8k \
  --config_json ./fitmotn_config.example.json \
  --output_jsonl boundary.jsonl \
  --verified_traces_jsonl verified.jsonl

# RL timing 分析
python scripts/analyze_rl_timing.py /path/to/rl_train.jsonl --last-n 100

# actor 控制面 smoke
python scripts/smoke_vllm_actor_control.py
```

## 二十五、常见问题

### 1. 提示缺少路径环境变量

确认已经设置：

```bash
echo "$MODEL_ROOT"
echo "$DATA_ROOT"
echo "$CACHE_ROOT"
echo "$OUTPUT_ROOT"
```

### 2. 直接把 `final_model/` 交给 vLLM 报错

`final_model/` 可能仍是 raw patched checkpoint。应先执行完整 HF export，再把 export 目录交给 vLLM。

### 3. metadata-only export 无法推理

这是预期行为。metadata-only 只用于扫描和布局检查。

### 4. vLLM rollout 出现 OOM

可尝试：

- 降低 `vllm_gpu_memory_utilization`；
- 降低 `vllm_max_model_len`；
- 降低 `vllm_max_num_seqs`；
- 使用较小 batch/group；
- 使用独立 rollout GPU；
- 使用 subprocess actor；
- 检查训练模型和 vLLM 是否意外位于同一 GPU。

### 5. policy stale 错误

正式正确性实验建议保持：

```json
{
  "rl": {
    "vllm_sync_every_updates": 1,
    "allow_stale_vllm_policy": false
  }
}
```

允许 stale policy 会改变训练分布，必须显式记录并作为实验变量。

### 6. 本机有多个 fitmotn checkout

开发时优先使用：

```bash
pip install -e ".[dev]"
python -c "import fitmotn; print(list(fitmotn.__path__))"
```

自动测试会强制绑定当前 checkout，但普通 Python 运行仍应检查导入路径。

## 二十六、推荐实验流程

1. 使用 Qwen3-0.6B 做 5～20 step smoke；
2. 验证 patch、loss、usage、保存和恢复；
3. 比较 raw checkpoint 与 HF export；
4. 在 NVIDIA 环境比较 HF export 与 vLLM；
5. 运行两步 vLLM RL smoke；
6. 比较 HF rollout、进程内 vLLM、subprocess actor；
7. 将 `export_reload` 作为正确性基线；
8. 仅在完整 dryrun coverage 后测试 update-only NCCL；
9. 扩大到目标模型和正式数据；
10. 归档配置、版本、manifest、日志和验收 JSON。

## 二十七、研究结果解释边界

为了避免错误结论，报告中应区分：

- 代码实现完成；
- CPU/mock 自动测试通过；
- CUDA engine 加载通过；
- HF/vLLM 语义一致性通过；
- RL 两步 smoke 通过；
- NCCL 同步通过；
- 端到端性能优于 HF；
- 正式训练收敛和指标通过。

只有生成 tokens/s 提升，并不能证明 RL 总体变快。只有 adapter mock 测试通过，也不能证明 NCCL 实机可用。

### Stage 4F/4G 训练正确性与稳定导出

推荐的 export_reload 路径现已提供逐次 rollout 的请求、提示词、采样
参数和策略指纹，训练进程及子进程 actor 都会校验实际加载的策略身份。
同步产物采用临时目录、校验、清单和原子重命名流程；失败不会发布半成品，
相同策略快照可复用，旧临时目录和历史产物可按配置清理。正式实验应保持
vllm_verify_engine_policy=true、allow_stale_vllm_policy=false 和
vllm_fallback_to_hf=false。

完整的 20-update 严格 smoke、断点恢复/失败注入和 200-update soak
方案见 docs/stage4_vllm_validation.md。

## 二十八、维护注意事项

- `ADTN.py`、`gate.py`、`losses.py` 是核心实现镜像，需要保持语义一致；
- 不要在训练外壳中静默改变 MoTN 方法定义；
- 新增模型架构时应使用显式 architecture adapter；
- 新增 vLLM 版本支持时应扩展 capability adapter，而不是覆盖旧路径；
- fallback 必须显式配置并记录；
- 大模型权重和实验输出不要提交到 Git；
- 修改配置字段时同步更新 schema、loader、示例配置、README 和测试。

## 二十九、进一步阅读

- [主 README](./README.md)
- [Stage 4 vLLM 验收方案](./docs/stage4_vllm_validation.md)
- [额外数据集说明](./docs/extra_datasets.md)
- [完整示例配置](./fitmotn_config.example.json)
- [Stage 4 两步 smoke 配置](./fitmotn_config.stage4_vllm_smoke.example.json)
- [双评测后端配置](./fitmotn_config.eval_dual_backend.example.json)
- [模型对齐评测配置](./fitmotn_config.model_aligned_eval.example.json)

---

建议在每次正式实验前先运行：

```bash
pytest -q
python -m fitmotn.cli.inspect_dataset --config ./your_config.json --max-samples 3
```

在 NVIDIA 环境进行 Stage 4 实验时，还应完成 [Stage 4 vLLM 验收方案](./docs/stage4_vllm_validation.md) 中对应的 gate，并保留全部证据文件。
