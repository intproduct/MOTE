# FitMoTN

FitMoTN 是一个独立可运行的 MOTN 训练与评测仓库，用来在不改写原始方法语义的前提下，把数据、patch、训练、checkpoint、恢复和评测整理成一套更标准的工程化流程。

它的设计目标不是“重新发明一套 MOTN”，而是把旧实验脚本里的关键逻辑拆出来，并用 Hugging Face Trainer 承接通用训练外壳，方便：

- 直接从 GitHub 获取并跑通
- 用 JSON 管理实验配置
- 做 patch 后模型的恢复性训练和评测
- 保持现有 MOTN 核心语义不被训练外壳悄悄改掉

当前版本是偏研究实验用途的 MVP，已经重点覆盖：

- patch 后模型训练
- 两阶段 mixed data training
- baseline / mid / final eval
- 保存与恢复 patched FitMoTN checkpoint
- HF / lm-eval 评测
- 独立 vLLM 评测入口

限制也很明确：

- 目前主要面向当前 Qwen 风格 MLP patch 场景
- patched FitMoTN checkpoint 还不能直接用 vLLM 执行

## 仓库获取

如果你是第一次使用这个项目，最推荐的获取方式是直接从 GitHub clone：

```bash
git clone https://github.com/intproduct/MOTE.git fitmotn
cd fitmotn
```

如果你使用 SSH：

```bash
git clone git@github.com:intproduct/MOTE.git fitmotn
cd fitmotn
```

这个仓库本身就是 `fitmotn` 项目仓库，所以 clone 完后当前目录就是项目根目录，不需要再进入额外子目录。

推荐把本地目录名也命名为 `fitmotn`。这样可以直接沿用仓库当前的 Python 包名与模块启动方式。

## 快速开始

最常见的使用流程可以概括成 4 步：

1. clone 仓库
2. 安装依赖
3. 准备 base model 路径、数据路径、缓存路径和输出路径
4. 用 `config_json` 启动训练

先设置统一路径环境变量：

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export CACHE_ROOT=/path/to/cache
export OUTPUT_ROOT=/path/to/outputs
# 可选；相对路径会基于 PROJECT_ROOT 解析，未设置时基于项目根目录解析
export PROJECT_ROOT="$(pwd)"
```

一个最小示例：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train \
  --model_path '${MODEL_ROOT}/Qwen3-0.6B' \
  --output_root '${OUTPUT_ROOT}/fitmotn_runs' \
  --run_name demo_run \
  --batch_size 4 \
  --grad_accum 1 \
  --steps 1000 \
  --seq_len_run 1024 \
  --lr 3e-5 \
  --eval_every_updates 500 \
  --save_every_updates 500 \
  --layers_to_patch last_quarter \
  --run_baseline_eval 1 \
  --device cuda:0
```

更推荐的正式实验方式是直接使用 JSON 配置：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train --config_json ./fitmotn_config.example.json
```

如果你要做“恢复 patch 后小模型推理能力”的两阶段实验，优先参考：

- [`fitmotn_reasoning_recovery_min_b.json`](./fitmotn_reasoning_recovery_min_b.json)
- [`fitmotn_reasoning_recovery_conservative_b.json`](./fitmotn_reasoning_recovery_conservative_b.json)

## Main entrypoints

After `pip install -e .`, the main entrypoints are:

```bash
python -m fitmotn.cli.train --config_json ./fitmotn_config.example.json
python -m fitmotn.cli.train_rl --config_json ./fitmotn_config.example.json
python -m fitmotn.cli.eval_hf --model_or_ckpt /path/to/model_or_ckpt --tasks gsm8k
python -m fitmotn.cli.build_boundary_gsm8k --config_json ./fitmotn_config.example.json --output_jsonl boundary.jsonl --verified_traces_jsonl verified.jsonl
python scripts/analyze_rl_timing.py path/to/rl_train.jsonl --last-n 100
```

Current vLLM status: patched FitMoTN checkpoints are not supported by vLLM in
this stage. `cli/eval_vllm.py` and `eval/vllm_runner.py` are only for baseline
or explicitly vLLM-compatible models, and patched FitMoTN checkpoints should
continue to fail with an explicit unsupported-path error until the later
export/vLLM integration stage.

## 安装与运行环境

For development installs:

```bash
pip install -e ".[dev]"
```

Large checkpoints and runtime outputs should not be committed to the source
repository. Keep model weights, `wandb/`, `logs/`, `outputs/`, and
`checkpoints/` outside git. If a tiny weight-like file is needed as a test
fixture, add an explicit allowlist entry for it instead of relying on broad
weight-file tracking.

## 目录说明

核心目录如下：

```text
fitmotn/
├── README.md
├── cli/
├── config/
├── data/
├── eval/
├── tasks/
├── train/
├── checkpointing.py
├── model.py
└── patching.py
```

重点文件：

- `model.py`
  只做桥接与封装，复用仓库内的 `ADTN.py` 和 `gate.py`
- `patching.py`
  负责 Qwen MLP patch、只训练 patched 参数、gate freeze/warmup routing、temperature 注入
- `train/trainer.py`
  `Trainer` 子类
- `train/callbacks.py`
  step 级调度逻辑
- `train/controller.py`
  训练总控
- `eval/restore.py`
  从保存的 checkpoint 恢复 patched 模型
- `eval/lm_eval_hf.py`
  HF/lm-eval 评测
- `eval/vllm_runner.py`
  vLLM MVP 入口与边界

## 核心原则

FitMoTN 明确保留了以下 MOTN 核心语义，不把训练外壳变成方法定义层：

- 直接复用 `ADTN.py` 的 `MoTNLayer`
- 直接复用 `gate.py` 的 `GateConfig` 与 gate 行为
- Qwen MLP patch 仍然是 `gate_proj(MoTN) + up_proj(MoTN) + act + down_proj(MoTN)`
- `down_proj.k_in` 仍由前面投影的 `k_out` 推导
- base model 全冻结，只训练 patched MoTN 层
- 保留 `gate_freeze_steps`
- 保留 warmup routing
- 保留 temperature schedule
- 保留 stage A / stage B 两阶段混采思想

## 环境要求

至少需要：

- Python 3
- `torch`
- `transformers`
- `datasets`
- `tqdm`

HF/lm-eval 评测额外需要：

```bash
pip install -U "lm_eval[hf]"
```

EvalScope 参考评测额外需要：

```bash
pip install -U evalscope
```

vLLM 评测额外需要：

```bash
pip install -U vllm
```

一个最小依赖检查可以这样做：

```bash
python3 -c "import torch, transformers, datasets; print('ok')"
```

## 获取后如何组织本地资源

仓库 clone 下来以后，通常还需要你自己准备三类外部资源：

1. base model
2. 训练数据缓存目录
3. 输出目录

最常改的配置通常都在 `data` 和 `model` 两个 section 里，例如：

- `model.model_path`
- `data.tok_shard_dir`
- `data.fineweb_cache_path`
- `data.code_cache_path`
- `data.gsm8k_cache_path`
- `data.math_cache_root`

推荐做法：

1. 先复制一份配置文件
2. 只把本机路径改成你自己的
3. 再开始正式训练

例如：

```bash
cp fitmotn_config.example.json my_fitmotn_config.json
```

然后把 `my_fitmotn_config.json` 里的路径换成你的实际目录，再执行：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train --config_json ./my_fitmotn_config.json
```

## 训练入口

训练 CLI：

- [`cli/train.py`](./cli/train.py)

重要说明：

- 命令行参数只暴露了一小部分高频入口，方便快速试跑
- `config_json` 才是当前推荐的全量实验参数注入接口
- 不仅训练参数能改，MOTN 方法相关参数也能从 `config_json` 外部注入
- 默认值与旧脚本 `run_train_motn_mixed_qwen3.py` 保持一致，但不是写死不可改
- 当前 CLI 和 `config_json` 都走统一配置加载与字段校验；未知 section/key 会直接报错，不再静默忽略

最小示例：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train \
  --model_path /path/to/Qwen3-0.6B \
  --output_root ./fitmotn_runs \
  --run_name demo_run \
  --batch_size 4 \
  --grad_accum 1 \
  --steps 1000 \
  --seq_len_run 1024 \
  --lr 3e-5 \
  --eval_every_updates 500 \
  --save_every_updates 500 \
  --layers_to_patch last_quarter \
  --run_baseline_eval 1 \
  --device cuda:0
```

支持的常用参数：

- `--model_path`
- `--output_root`
- `--run_name`
- `--batch_size`
- `--grad_accum`
- `--steps`
- `--epochs`
- `--seq_len_run`
- `--lr`
- `--eval_every_updates`
- `--save_every_updates`
- `--layers_to_patch`
- `--run_baseline_eval`
- `--dataloader_num_workers`
- `--device`
- `--config_json`

这里要特别注意：

- 上面这组 CLI 参数不是完整参数面
- 如果你要调整 `E`、`topk`、`temperature`、`gate_type`、`capacity_factor`、`aux_coeff` 这类深度实验参数，请使用 `config_json`
- `torch_dtype`、`lr_warmup`、`lr_warmup_steps` 现在会真实参与运行时装载与训练调度，不再只是声明参数

## 用 JSON 配置启动

如果不想把参数全写在命令行，可以用 `--config_json`。

配置文件结构要对应 `FitMoTNConfig` 的分区：

```json
{
  "model": {
    "model_path": "${MODEL_ROOT}/Qwen3-0.6B",
    "layers_to_patch": "last_quarter",
    "device": "cuda:0"
  },
  "data": {
    "seq_len_run": 1024,
    "dataloader_num_workers": 0
  },
  "train": {
    "batch_size": 4,
    "grad_accum": 1,
    "steps": 1000,
    "lr": 3e-5,
    "block_lr": 2.2e-5,
    "router_lr": 1.1e-5,
    "eval_every_updates": 500,
    "save_every_updates": 500
  },
  "eval": {
    "run_baseline_eval": true
  },
  "output": {
    "root_dir": "${OUTPUT_ROOT}/fitmotn_runs",
    "run_name": "demo_run"
  }
}
```

启动方式：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train --config_json ./fitmotn_config.json
```

推荐做法：

- 日常正式实验统一使用 `config_json`
- 命令行只用于覆盖极少数临时参数

## 完整可调接口

当前版本中，实验参数的完整外部控制入口是：

- [`fitmotn_config.example.json`](./fitmotn_config.example.json)
- `PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train --config_json your_config.json`

也就是说，下面列出的字段都可以通过外部 JSON 修改，而不只是训练参数。

### `model` 可调字段

- `model_path`
- `layers_to_patch`
- `device`
- `use_amp`
- `torch_dtype`
- `trust_remote_code`
- `E`
- `d`
- `k_in`
- `topk`
- `gate_type`
- `temperature`
- `capacity_factor`
- `min_capacity`
- `drop_tokens`
- `drop_policy`
- `aux_coeff`
- `zloss_coeff`
- `jitter_eps`
- `use_ste`
- `pos_strategy`
- `init_gamma`

这些字段会直接影响 patched MoTN 层的构造与路由行为，不是只读默认值。

`layers_to_patch` 支持三类写法：

- 固定策略：`all`、`last_half`、`last_quarter`、`last_third`
- 连续范围：`range:START-END`
- 离散层：`layers:i,j,k`

层索引均为 0-based；`range` 是闭区间，即 `range:4-11` 会 patch `model.layers[4]` 到 `model.layers[11]`。具体层号应由每次实验配置决定，代码、默认配置和示例配置中不应硬编码某个实验专属范围。

### `data` 可调字段

- `tok_shard_dir`
- `datas_dir`
- `seq_len_run`
- `dataloader_num_workers`
- `fineweb_cache_path`
- `code_cache_path`
- `gsm8k_cache_path`
- `gsm8k_socratic_cache_path`
- `svamp_cache_path`
- `metamath_cache_path`
- `mmlu_cache_path`
- `math_cache_root`
- `openr1_math_cache_path`
- `numinamath_cot_cache_path`
- `openthoughts_math_cache_path`
- `bespoke_stratos_cache_path`
- `fineweb_hf_name`
- `fineweb_hf_config`
- `fineweb_text_field`
- `code_hf_name`
- `code_hf_config`
- `code_text_field`
- `gsm8k_hf_name`
- `gsm8k_hf_config`
- `gsm8k_socratic_hf_name`
- `gsm8k_socratic_hf_config`
- `svamp_hf_name`
- `svamp_hf_config`
- `metamath_hf_name`
- `metamath_hf_config`
- `metamath_max_samples`
- `mmlu_hf_name`
- `mmlu_hf_config`
- `mmlu_split`
- `math_hf_name`
- `math_split`
- `openr1_math_hf_name`
- `openr1_math_hf_config`
- `numinamath_cot_hf_name`
- `numinamath_cot_hf_config`
- `openthoughts_math_hf_name`
- `openthoughts_math_hf_config`
- `bespoke_stratos_hf_name`
- `bespoke_stratos_hf_config`
- `use_wiki_local`
- `use_fineweb`
- `use_code`
- `use_gsm8k_train`
- `use_gsm8k_socratic_train`
- `use_svamp_train`
- `use_synthetic_arithmetic_train`
- `use_metamath_train`
- `use_math_train`
- `use_mmlu_train`
- `use_openr1_math`
- `use_numinamath_cot`
- `use_openthoughts_math`
- `use_bespoke_stratos`
- `wt_wiki`
- `wt_fineweb`
- `wt_code`
- `wt_gsm8k`
- `wt_gsm8k_socratic`
- `wt_svamp`
- `wt_synthetic_arithmetic`
- `wt_metamath`
- `wt_math`
- `wt_mmlu`
- `wt_openr1_math`
- `wt_numinamath_cot`
- `wt_openthoughts_math`
- `wt_bespoke_stratos`
- `reasoning_max_chars`
- `reasoning_max_approx_tokens`
- `openthoughts_max_chars`
- `openthoughts_max_approx_tokens`
- `synthetic_arithmetic_num_samples`
- `synthetic_arithmetic_seed`
- `prefer_short_reasoning`
- `skip_overlong_reasoning_samples`
- `reasoning_supervision_mode`

这些字段已经真实接入训练数据管线，不只是 README 声明。当前训练 task pool 除了原有的 `wiki / fineweb / code / gsm8k / svamp / metamath / hendrycks_math / mmlu` 外，还支持：

- `open-r1/OpenR1-Math-220k`
- `AI-MO/NuminaMath-CoT`
- `open-r1/OpenThoughts-114k-math`
- `bespokelabs/Bespoke-Stratos-17k`

接入方式都是统一的：

- 通过 `use_*` 控制是否启用
- 通过 `wt_*` 控制混采权重
- 通过 `*_cache_path` 指向本地缓存目录
- 通过 `*_hf_name` 和 `*_hf_config` 指向 Hugging Face 数据源

新增 reasoning 数据默认会统一格式化成：

```text
Question:
...

Solution:
...

Final Answer:
...
```

现在支持两种显式的 reasoning supervision mode：

- `answer_only`
- `full_trace`

其中：

- `answer_only` 兼容旧行为，prompt 中保留 `Solution:`，target 只监督最终答案
- `full_trace` 用于真正的 reasoning recovery，prompt 只保留题目，target 监督完整 `Solution + Final Answer`

也就是说，`full_trace` 把训练目标从 `Question + Solution -> Final Answer` 改成了更符合推理恢复目标的 `Question -> Solution + Final Answer`。

统一 target 尾部格式固定为：

```text
Solution:
...

Final Answer:
...
```

`Final Answer:` 会作为稳定抽取点保留；同时答案抽取 helper 仍兼容旧的 `#### ...`、`\boxed{...}` 和简短末行答案。

同时长度过滤也已经真实生效：

- `reasoning_max_chars`
- `reasoning_max_approx_tokens`
- `prefer_short_reasoning`
- `skip_overlong_reasoning_samples`

新增的 `synthetic_arithmetic_train` 不依赖外部下载，会在本地按固定 seed 生成简短 arithmetic / money / count / multi-step word problem 样本，适合 GSM8K recovery 时做 core arithmetic repair。

## Bucketed Reasoning Recovery

这次改动把原来的“pretrain task + reasoning task 平铺混采”扩成了可选的分层 bucket 混采。

当前支持两种模式：

- `train.task_bucket_mode = "flat"`
  保持旧行为，pretrain 和 task 数据直接按 task 权重平铺混采
- `train.task_bucket_mode = "bucketed"`
  先按 bucket ratio 选 bucket，再在 bucket 内按 task weight 归一化采样

bucketed 模式的三层结构是：

- `pretrain_general`
  `wiki24_tok`、`fineweb`、`stack_code`
- `gsm8k_core`
  `gsm8k_train`、`gsm8k_socratic_train`、`svamp_train`、`synthetic_arithmetic_train`
- `aux_reasoning`
  `metamath_train`、`math_*`、`openr1_math_train`、`numinamath_cot_train`、`openthoughts_math_train`、`bespoke_stratos_train`、`mmlu_auxiliary_train`

设计目标是让 GSM8K 恢复训练优先集中吃到 `gsm8k_core`，而不是把所有 reasoning 数据继续平铺成一锅混采。

### 兼容性

- 默认仍然是 `flat`
- 旧配置不需要新增字段也能继续运行
- `stage_b_reasoning_boost` 在 `flat` 模式下保留旧语义
- `bucketed` 模式下，bucket 间比例完全由 stage ratio 字段决定，`stage_b_reasoning_boost` 不再改变 bucket 间采样比例

### 新增配置字段

`data`:

- `use_synthetic_arithmetic_train`
- `wt_synthetic_arithmetic`
- `synthetic_arithmetic_num_samples`
- `synthetic_arithmetic_seed`

`train`:

- `task_bucket_mode`
- `stage_a_core_task_ratio`
- `stage_a_aux_task_ratio`
- `stage_b_core_task_ratio`
- `stage_b_aux_task_ratio`

### bucketed 模式校验规则

- `task_bucket_mode` 只允许 `flat` 或 `bucketed`
- `bucketed` 模式下要求 `stage_a_core_task_ratio + stage_a_aux_task_ratio == stage_a_task_ratio`
- `bucketed` 模式下要求 `stage_b_core_task_ratio + stage_b_aux_task_ratio == stage_b_task_ratio`
- 若某个 bucket ratio 大于 0，但 bucket 内没有 enabled task，或该 bucket 内 task 权重全为 0，会直接报错

### 运行时观测

bucketed 改造后，以下信息现在会真实进入日志和 batch/runtime 统计：

- 每个 task 的 `bucket`
- stage A / stage B 的 `bucket_ratios`
- batch 里的 `bucket` 聚合信息
- 运行时 `bucket_sampling_counts`

`data/collate.py` 也已经同步更新，所以 `bucket` 会跟 `task/group/source_family` 一起进入 trainer batch，不会在 collate 阶段丢失。

### 推荐配置与启动方式

推荐直接参考新样例配置：

- [`fitmotn_reasoning_recovery_gsm8k_core_bucketed.json`](./fitmotn_reasoning_recovery_gsm8k_core_bucketed.json)

最小启动命令：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.train \
  --config_json ./fitmotn_reasoning_recovery_gsm8k_core_bucketed.json
```

### 调试与样本检查

如果你想快速检查某个 reasoning task 的 bucket、trace 样式和样本内容，可以使用：

```bash
PYTHONPATH="$(pwd)/.." python3 -m fitmotn.cli.debug_reasoning_sample \
  --config_json ./fitmotn_reasoning_recovery_gsm8k_core_bucketed.json \
  --task synthetic_arithmetic_train \
  --sample_index 0
```

这个调试命令现在会额外打印：

- task 的 `bucket / group / source_family`
- 当前配置解析出的 `task_bucket_mode`
- stage A / stage B 的 bucket ratio
- 归一化后的 prompt / target
- `reasoning_supervision_mode`

其中 `OpenThoughts-114k-math` 还有单独更严格的：

- `openthoughts_max_chars`
- `openthoughts_max_approx_tokens`

也就是说，如果你现在用下面这些配置文件启动训练，新的 reasoning 数据集接口已经会参与真实训练，而不是占位参数：

- [`fitmotn_reasoning_recovery_min_b.json`](./fitmotn_reasoning_recovery_min_b.json)
- [`fitmotn_reasoning_recovery_conservative_b.json`](./fitmotn_reasoning_recovery_conservative_b.json)

### `train` 可调字段

- `batch_size`
- `grad_accum`
- `steps`
- `epochs`
- `epoch_samples`
- `lr`
- `block_lr`
- `router_lr`
- `log_every`
- `save_every_updates`
- `eval_every_updates`
- `max_grad_norm`
- `stage_a_ratio`
- `stage_a_pretrain_ratio`
- `stage_a_task_ratio`
- `stage_b_mode`
- `stage_b_pretrain_ratio`
- `stage_b_task_ratio`
- `stage_b_disable_pretrain`
- `stage_b_reasoning_boost`
- `begin_t`
- `end_t`
- `gate_freeze_steps`
- `usage_dump_every`
- `usage_light_every`
- `usage_light_jsonl_every`
- `usage_report_every`
- `heavy_log_every`
- `enable_usage_runtime_tracking`
- `enable_usage_report`
- `enable_heavy_runtime_stats`
- `enable_grad_param_norm`
- `enable_cuda_snapshot`
- `train_jsonl_every`
- `benchmark_train_only`
- `lr_warmup`
- `lr_warmup_steps`
- `warmup_ratio`
- `early_stop`
- `early_stop_abs_gsm8k`
- `early_stop_abs_mmlu`
- `seed`
- `report_to`

`lr` 保留为旧配置入口；未设置 `block_lr` 时默认继承 `lr`，未设置 `router_lr` 时默认继承 `block_lr`。如果要让 router/gate 使用 block 的 0.5x 学习率，可以设置 `"block_lr": 2.2e-5, "router_lr": 1.1e-5`。

### `eval` 可调字段

- `run_baseline_eval`
- `lm_eval_batch_size`
- `lm_eval_num_fewshot_mmlu`
- `lm_eval_num_fewshot_gsm8k`
- `lm_eval_num_fewshot_math`
- `lm_eval_device`
- `baseline_small_tasks`
- `final_tasks`
- `early_limit_gsm8k`
- `early_limit_mmlu`
- `baseline_small_limit_gsm8k`
- `baseline_small_limit_mmlu`
- `final_limit_gsm8k`
- `final_limit_mmlu`
- `final_limit_math`
- `early_max_gen_toks_gsm8k`
- `final_max_gen_toks_gsm8k`
- `final_max_gen_toks_math`

兼容说明：

- 上面这些旧字段仍然可用
- 新版本另外支持 `eval_backend`、`primary_eval_backend`、`backend_defaults`、`protocols`、`runtime`、`generation`、`fewshot`、`limits`、`max_gen_toks`、`task_overrides`、`backend_extra`
- 新增评测结构的详细说明见文末“通用评测与双 Backend”

### `output` 可调字段

- `root_dir`
- `run_name`
- `overwrite_output_dir`

## 完整 `config_json` 模板

下面是当前 README 内联展示的完整配置形式。它和 [`fitmotn_config.example.json`](./fitmotn_config.example.json) 一致，可以直接复制后修改。

```json
{
  "model": {
    "model_path": "${MODEL_ROOT}/Qwen3-0.6B",
    "layers_to_patch": "last_quarter",
    "device": "cuda:0",
    "use_amp": true,
    "torch_dtype": "auto",
    "trust_remote_code": true,
    "E": 16,
    "d": 2,
    "k_in": 5,
    "topk": 4,
    "gate_type": "topk",
    "temperature": 1.0,
    "capacity_factor": 1.35,
    "min_capacity": 8,
    "drop_tokens": true,
    "drop_policy": "probs",
    "aux_coeff": 0.03,
    "zloss_coeff": 0.0,
    "jitter_eps": 0.0,
    "use_ste": true,
    "pos_strategy": "random",
    "init_gamma": 0.6
  },
  "data": {
    "tok_shard_dir": "${DATA_ROOT}/wiki24_tok",
    "datas_dir": "${DATA_ROOT}",
    "seq_len_run": 1024,
    "dataloader_num_workers": 0,
    "fineweb_cache_path": "${CACHE_ROOT}/fineweb_sample10bt",
    "code_cache_path": "${CACHE_ROOT}/the_stack_v2",
    "gsm8k_cache_path": "${CACHE_ROOT}/gsm8k_main",
    "gsm8k_socratic_cache_path": "${CACHE_ROOT}/gsm8k_socratic",
    "svamp_cache_path": "${CACHE_ROOT}/svamp",
    "metamath_cache_path": "${CACHE_ROOT}/metamathqa",
    "mmlu_cache_path": "${CACHE_ROOT}/mmlu_all",
    "math_cache_root": "${CACHE_ROOT}/hendrycks_math",
    "fineweb_hf_name": "HuggingFaceFW/fineweb",
    "fineweb_hf_config": "sample-10BT",
    "fineweb_text_field": "text",
    "code_hf_name": "bigcode/the-stack-v2",
    "code_hf_config": "",
    "code_text_field": "content",
    "gsm8k_hf_name": "openai/gsm8k",
    "gsm8k_hf_config": "main",
    "gsm8k_socratic_hf_name": "openai/gsm8k",
    "gsm8k_socratic_hf_config": "socratic",
    "svamp_hf_name": "garrethlee/svamp",
    "svamp_hf_config": "",
    "metamath_hf_name": "meta-math/MetaMathQA",
    "metamath_hf_config": "",
    "metamath_max_samples": 20000,
    "mmlu_hf_name": "cais/mmlu",
    "mmlu_hf_config": "all",
    "mmlu_split": "dev",
    "mmlu_train_split": "auxiliary_train",
    "math_hf_name": "EleutherAI/hendrycks_math",
    "math_split": "train",
    "use_wiki_local": true,
    "use_fineweb": true,
    "use_code": true,
    "use_gsm8k_train": true,
    "use_gsm8k_socratic_train": false,
    "use_svamp_train": false,
    "use_metamath_train": false,
    "use_math_train": true,
    "use_mmlu_train": true,
    "wt_wiki": 1.0,
    "wt_fineweb": 1.0,
    "wt_code": 1.0,
    "wt_gsm8k": 1.0,
    "wt_gsm8k_socratic": 1.0,
    "wt_svamp": 0.8,
    "wt_metamath": 0.3,
    "wt_math": 1.0,
    "wt_mmlu": 1.0
  },
  "train": {
    "batch_size": 4,
    "grad_accum": 1,
    "steps": 4000,
    "epochs": 1.0,
    "epoch_samples": 200000,
    "lr": 3e-05,
    "log_every": 50,
    "save_every_updates": 1000,
    "eval_every_updates": 1000,
    "max_grad_norm": 1.0,
    "stage_a_ratio": 0.6,
    "stage_a_pretrain_ratio": 0.7,
    "stage_a_task_ratio": 0.3,
    "stage_b_pretrain_ratio": 0.45,
    "stage_b_task_ratio": 0.55,
    "begin_t": 1.5,
    "end_t": 0.8,
    "gate_freeze_steps": 2000,
    "usage_dump_every": 500,
    "usage_light_every": 50,
    "usage_light_jsonl_every": 50,
    "usage_report_every": 500,
    "heavy_log_every": 500,
    "enable_usage_runtime_tracking": true,
    "enable_usage_report": true,
    "enable_heavy_runtime_stats": true,
    "enable_grad_param_norm": false,
    "enable_cuda_snapshot": false,
    "train_jsonl_every": 50,
    "benchmark_train_only": false,
    "lr_warmup": false,
    "lr_warmup_steps": 1000,
    "warmup_ratio": 0.5,
    "early_stop": true,
    "early_stop_abs_gsm8k": 0.45,
    "early_stop_abs_mmlu": 0.45,
    "seed": 0,
    "report_to": []
  },
  "eval": {
    "run_baseline_eval": true,
    "lm_eval_batch_size": 1,
    "lm_eval_num_fewshot_mmlu": 5,
    "lm_eval_num_fewshot_gsm8k": 8,
    "lm_eval_num_fewshot_math": 4,
    "lm_eval_device": "cuda:0",
    "baseline_small_tasks": [
      "gsm8k",
      "mmlu"
    ],
    "final_tasks": [
      "gsm8k",
      "mmlu"
    ],
    "early_limit_gsm8k": 32,
    "early_limit_mmlu": 64,
    "baseline_small_limit_gsm8k": 64,
    "baseline_small_limit_mmlu": 128,
    "final_limit_gsm8k": 0,
    "final_limit_mmlu": 512,
    "final_limit_math": 256,
    "early_max_gen_toks_gsm8k": 256,
    "final_max_gen_toks_gsm8k": 256,
    "final_max_gen_toks_math": 256
  },
  "output": {
    "root_dir": "${OUTPUT_ROOT}/fitmotn_runs",
    "run_name": "fitmotn_demo",
    "overwrite_output_dir": false
  }
}
```

新增一个可直接参考的 reasoning mix 配置：[`approx_35000_reasoning_mix_v1.json`](./approx_35000_reasoning_mix_v1.json)。它会并行启用 `gsm8k main + gsm8k socratic + svamp + metamath`，并把 `metamath_max_samples` 默认限制在 `20000`。

如果你要做“恢复 patch 后小模型推理能力”的两阶段实验，可以直接参考这两个新增配置：

- [`fitmotn_reasoning_recovery_min_b.json`](./fitmotn_reasoning_recovery_min_b.json)
- [`fitmotn_reasoning_recovery_conservative_b.json`](./fitmotn_reasoning_recovery_conservative_b.json)

这两份配置会保留 `wiki / fineweb / code / gsm8k / MATH / 少量 MMLU`，并额外接入 `OpenR1-Math-220k / NuminaMath-CoT / OpenThoughts-114k-math / Bespoke-Stratos-17k`。其中 `OpenThoughts` 默认启用更严格的长度过滤，新增 reasoning 数据统一整理为 `Question / Solution / Final Answer` 风格训练文本。

Stage B 现在支持显式 reasoning recovery 语义：

- `stage_b_mode = "reasoning_recovery"` 时，要求 `data.reasoning_supervision_mode = "full_trace"`
- `stage_b_reasoning_boost` 会真实提高 Stage B 中 reasoning-family task 的采样权重
- `stage_b_disable_pretrain = true` 时，Stage B 会把 pretrain ratio 实际压到 `0.0`

这使得 Stage A 更偏恢复/稳定化，Stage B 更适合做 reasoning-heavy 的 recovery run。

为了做短实验和 AB test，新增两份配置：

- [`fitmotn_reasoning_recovery_quick_full_trace.json`](./fitmotn_reasoning_recovery_quick_full_trace.json)
- [`fitmotn_reasoning_recovery_quick_answer_only.json`](./fitmotn_reasoning_recovery_quick_answer_only.json)

两份配置字段尽量一致，主要差异集中在 `reasoning_supervision_mode` 和 Stage B 语义：

- `quick_full_trace` 用于快速验证 full-trace reasoning SFT + Stage B reasoning recovery
- `quick_answer_only` 用于 answer-only 对照组，因此 Stage B 保持 `mixed`

可以直接用调试 CLI 检查某个 reasoning 样本是否被正确规范化：

```bash
python -m MOTN.fitmotn.cli.debug_reasoning_sample \
  --config_json ./MOTN/fitmotn/fitmotn_reasoning_recovery_quick_full_trace.json \
  --task gsm8k_train \
  --sample_index 0
```

它会直接打印 normalized `prompt`、`target`、抽取到的 `final_answer` 和 `label span`，便于人工检查 prompt 是否全 mask、target 是否完整参与 loss。

另外，当前还新增了轻量 `rl/` 接口层，用于 future verifier / reward / rerank / rejection sampling 的外围结构准备；本次不会把 RL 主训练接入主流程。

### 一个最关键的结论

如果你要做实验扫描，尤其是改这些参数：

- `E`
- `d`
- `k_in`
- `topk`
- `gate_type`
- `temperature`
- `capacity_factor`
- `min_capacity`
- `drop_tokens`
- `drop_policy`
- `aux_coeff`
- `zloss_coeff`
- `jitter_eps`
- `use_ste`
- `pos_strategy`
- `init_gamma`
- stage A/B 比例
- task 权重

那么当前推荐方式就是：

1. 基于 `fitmotn_config.example.json` 复制一个实验配置
2. 只改你关心的字段
3. 用 `--config_json` 启动

这就是当前 MVP 中的标准参数注入方式。

## 训练输出

每次运行会在 `output_root/run_name/` 下生成结果。

主要产物：

- `train.log`
  训练日志
- `train_light.jsonl`
  高频轻量 usage 轨迹；默认保留 `layer × proj × expert` 的完整 count 向量
- `train.jsonl`
  训练 step/update 级记录；只保留训练主记录，不再承载 usage 详细大对象
- `usage.jsonl`
  低频 detailed usage report；保留旧分析脚本兼容入口
- `mid_eval.jsonl`
  baseline / mid / final 评测轨迹记录
- `eval_summary.json`
  baseline / mid / final 评测汇总
- `run_summary.json`
  适合批量实验扫描的 run 级汇总
- `checkpoints/`
  Trainer 保存的阶段 checkpoint
- `final_model/`
  最终导出的模型目录

`final_model/` 下的重要文件：

- `config.json` / tokenizer 文件
- `pytorch_model.bin` 或 Trainer 保存的模型权重
- `fitmotn_state.pt`
- `fitmotn_state.json`
- `run_summary.json`

其中 `fitmotn_state.*` 保存了 FitMoTN 额外元数据，包括：

- `base_model_path`
- `layers_to_patch`（已解析的真实 patch 层 index 列表）
- `motn_cfg`
- `fit_cfg`
- `checkpoint_format`
- `patch_state_dict`（当前默认）
- `resolved_model_dtype`
- `amp_enabled`
- `warmup_updates`
- `trainable_params`
- `total_params`
- `trainable_ratio`
- `baseline_small_summary`
- `baseline_final_summary`
- `latest_mid_eval_summary`
- `final_full_summary`
- `compare_vs_baseline`
- `env_snapshot`

兼容说明：

- 新格式默认只保存 patched MoTN 层对应的状态，而不再把整模型 `state_dict` 再额外塞进 `fitmotn_state.pt`
- 恢复时仍然先加载 base model，再 patch，再把保存的 patched 状态灌回去
- 旧格式 `state_dict` 仍然兼容读取，不会破坏已有 checkpoint
- `fitmotn_state.json` 仍然保留，方便人读和批量扫描

## FitMoTN export and vLLM roadmap

Raw FitMoTN checkpoints are training artifacts. For post-training inference or evaluation, prefer a full Stage 4C exported Hugging Face directory instead of pointing tools at the raw checkpoint. Stage 4A can still create a metadata-only export for scanning and layout validation:

```bash
python -m fitmotn.cli.export_hf \
  --checkpoint_dir /path/to/raw-fitmotn-checkpoint \
  --output_dir /path/to/exported-fitmotn \
  --metadata_only \
  --validate
```

The metadata-only export directory contains:

- `fitmotn_export_manifest.json`
- `fitmotn_export_config.json`
- `README.md`
- optional tokenizer files only when `--copy_tokenizer` is explicitly passed

Stage 4C adds an opt-in Hugging Face roundtrip export that is also vLLM-ready through vLLM's Transformers modeling backend:

```bash
python -m fitmotn.cli.export_hf \
  --checkpoint_dir /path/to/raw-fitmotn-checkpoint \
  --output_dir /path/to/exported-fitmotn-hf \
  --no-metadata_only \
  --base_model /path/to/base-model \
  --validate_layout
```

The Stage 4C export writes `config.json`, `configuration_fitmotn.py`, `modeling_fitmotn.py`, model weights, export metadata, `README.md`, and tokenizer/generation files when available. It can be loaded with:

```python
from transformers import AutoModel, AutoModelForCausalLM
decoder = AutoModel.from_pretrained("/path/to/exported-fitmotn-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/path/to/exported-fitmotn-hf", trust_remote_code=True)
```

The exported wrapper requires the local `fitmotn` package to be installed. Use the full exported HF directory for `eval_hf.py`, `eval_auto.py`, ad-hoc `AutoModel` / `AutoModelForCausalLM` loading, and Stage 4C vLLM evaluation. Metadata-only exports and raw checkpoints remain unsupported by vLLM.

Stage 4C uses vLLM's Transformers modeling backend. FitMoTN/MoTN routing is implemented inside the custom Transformers model as a patched FFN replacement. Stage 4C does not provide native vLLM model registration, expert-parallel MoE execution, fused MoTN kernels, custom CUDA ops, or vLLM expert-parallel support.

Example vLLM command:

```bash
python -m fitmotn.cli.eval_vllm \
  --model_or_ckpt /path/to/exported-fitmotn \
  --model_impl transformers \
  --enforce_eager \
  --limit_per_task 32
```

Do not commit exported weights, raw checkpoints, tokenizer files copied from private models, or manifests containing private local paths.

## 论文级观测

当前版本会在训练中持续记录这些结构化信息：

- `train_light.jsonl`
  高频 usage 轻轨迹；GPU 侧可按 `usage_light_every` 采样，再按 `usage_light_jsonl_every` 独立落盘
- `train.jsonl`
  包含 `loss`、`lr`、`T`、`gate_trainable`、`tokens/s`、`optimizer`、`scheduler`、`batch_task_names`、`batch_groups`、`batch_source_families`
- `usage.jsonl`
  包含每层 `usage_*`、`top1_*`、`pos_*`、`entropy_*`、`load_balance_*`、`importance_*`、`drop_rate_*`、`capacity_*`、`active_expert_count_*`、`max_expert_share_*`、`expert_cv_*`
- `mid_eval.jsonl`
  包含每次评测对应的训练上下文、评测结果和相对 baseline 的对比
- `run_summary.json`
  用于批量实验扫描和论文总表汇总

## Usage 轻重分层

当前训练路径已经拆成三层：

- `usage_light_every`
  控制 GPU 侧轻量 usage snapshot 的采样频率。这里保留的是 `layer × proj × expert` 完整 count，目的是恢复 usage 随 step 的变化轨迹。
- `usage_light_jsonl_every`
  控制把这些高频 snapshot 写入 `train_light.jsonl` 的频率。它与 GPU 采样频率独立，可以实现“每步采样、每 50 步落盘”。
- `usage_report_every`
  控制 detailed `usage.jsonl` 导出频率。只有这一层才会做 `.cpu()`、entropy/load/importance/drop-rate/capacity 的重聚合与 JSON 化。

兼容说明：

- 旧字段 `usage_dump_every` 仍然支持。
- 如果没有显式给 `usage_report_every`，系统会自动把 `usage_dump_every` 映射过去。

Heavy runtime stats 也已与 usage 研究路径解耦：

- `heavy_log_every` 控制重统计触发频率
- `enable_heavy_runtime_stats=false` 时不再做重 runtime 观测
- `enable_grad_param_norm=false` 时跳过全模型 grad/param norm 扫描
- `enable_cuda_snapshot=false` 时跳过 CUDA snapshot

吞吐测速可以直接使用 benchmark 模式：

- `benchmark_train_only=true`
  默认会关闭 usage tracking、usage report 和 heavy runtime stats，适合做纯训练吞吐对照，不适合 usage 研究实验。

参考配置：

- [`fitmotn_config.usage_light_report.example.json`](./fitmotn_config.usage_light_report.example.json)
- [`fitmotn_config.benchmark_train_only.example.json`](./fitmotn_config.benchmark_train_only.example.json)

## HF 评测

HF/lm-eval 入口：

- [eval_hf.py](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/cli/eval_hf.py)
- [eval_auto.py](/Users/qixuanfang/Library/Mobile Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/cli/eval_auto.py)

评测保存后的 FitMoTN checkpoint：

```bash
python3 -m MOTN.fitmotn.cli.eval_hf \
  --model_or_ckpt ./MOTN/fitmotn_runs/demo_run/final_model \
  --device cuda:0
```

指定任务：

```bash
python3 -m MOTN.fitmotn.cli.eval_hf \
  --model_or_ckpt ./MOTN/fitmotn_runs/demo_run/final_model \
  --device cuda:0 \
  --tasks gsm8k mmlu
```

统一入口评测：

```bash
python3 -m MOTN.fitmotn.cli.eval_auto \
  --model_or_ckpt ./MOTN/fitmotn_runs/demo_run/final_model \
  --device cuda:0 \
  --eval_backend both \
  --primary_eval_backend lm_eval \
  --allow_backend_skip
```

如果 `model_or_ckpt` 目录下有 `fitmotn_state.pt`，脚本会自动按下面顺序恢复：

1. 加载 base model
2. 按 checkpoint 中保存的真实 `layers_to_patch` index 列表和 `motn_cfg` 重新 patch
3. 加载 `patch_state_dict`；如果是旧 checkpoint，则回退到 `state_dict`
4. `load_state_dict(strict=False)`

这一步是为了保证 patch 结构不会被 `save_pretrained()` 平铺掉。

## vLLM 评测

vLLM 入口：

- [eval_vllm.py](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/cli/eval_vllm.py)

示例：

```bash
python -m fitmotn.cli.eval_vllm \
  --model_or_ckpt /path/to/exported-fitmotn \
  --model_impl transformers \
  --enforce_eager \
  --limit_per_task 32
```

### 当前 vLLM 边界

Stage 4C 支持：

- baseline 模型评测
- 原生兼容 vLLM 的 checkpoint 评测
- full HF FitMoTN export directory，通过 vLLM Transformers modeling backend 加载

Stage 4C 仍不支持：

- raw patched FitMoTN checkpoint 直接用 vLLM 执行
- metadata-only FitMoTN export 直接用 vLLM 执行
- native vLLM model registration
- expert-parallel MoE execution
- fused MoTN kernels 或 custom CUDA ops

如果你把 `final_model/` 这类 raw patched FitMoTN checkpoint 直接传给 `eval_vllm.py`，当前实现会显式报错：

- `NotImplementedError`
- 同时会在输出 JSON 中写入 `capability` 字段，说明当前模型路径为什么不支持 vLLM 评测

这是有意为之，目的是避免 silently fallback，防止 MOTN 核心方法被偷偷替换或退化。

Use a full HF export produced by `python -m fitmotn.cli.export_hf --no-metadata_only ...` for Stage 4C vLLM evaluation.

## Stage A / Stage B 说明

当前训练是单次 `trainer.train()`，不会拆成两次独立 run。

内部行为：

- 只维护一个连续的 `global_step`
- callback 根据 `global_step` 更新当前 stage
- dataset 在取样时读取当前 stage，切换 pretrain/task 的缩放比例

这样可以保留旧脚本中的两阶段思想：

- stage A：恢复性训练
- stage B：更 task-aware 的训练

同时不会把阶段切换交给 Trainer 默认的 epoch 语义。

## 推荐设置

第一版推荐：

- `dataloader_num_workers=0`

原因：

- stage 切换是按全局 step 驱动的
- 多 worker 下 worker 侧迭代可能对 stage 变化有滞后
- MVP 优先保证语义保真，不优先追求 dataloader 并行度

## 已知限制

当前版本有这些明确限制：

- 只针对当前 Qwen 风格 `model.layers[*].mlp`
- 需要原始 MLP 具备 `gate_proj` / `up_proj` / `down_proj`
- 没做多机分布式专门适配
- 没做更复杂的 checkpoint 版本迁移
- 没做 patched-MOTN-on-vLLM 执行适配
- 训练中间评测仍走 HF/lm-eval，不走 vLLM
- `eval_vllm.py` 现在会额外输出能力报告，但这不等于 patched checkpoint 已被支持

## 核心镜像文件说明

`fitmotn/ADTN.py`、`fitmotn/gate.py`、`fitmotn/losses.py` 当前是为 `MOTN.fitmotn` 包内独立运行保留的镜像副本。

阶段性警告：

- 这些文件当前没有引入自动同步机制
- 维护时必须保持它们与 `./MOTN/` 中对应核心实现的语义一致
- 本轮只增加文档警告，不调整代码路径或模块结构

## 快速检查清单

如果训练前想先确认环境是否大体齐全，可以检查：

```bash
python3 -c "import torch, transformers, datasets; print('ok')"
python3 -c "from MOTN.fitmotn.config.defaults import make_default_config; print(make_default_config().model.layers_to_patch)"
```

如果要跑 HF 评测，再检查：

```bash
python3 -c "import lm_eval; print('lm_eval ok')"
```

如果要跑 vLLM，再检查：

```bash
python3 -c "import vllm; print('vllm ok')"
```

## 后续建议

如果你接下来要继续推进，建议顺序是：

1. 先用一个很小的 step 数做 smoke run，确认 patch、保存、恢复链路可用
2. 再做一轮和旧脚本参数对照的配置补齐
3. 最后再考虑 patched MOTN 的 vLLM 执行适配

## 通用评测与双 Backend

当前评测层已经重构为“统一入口 + 配置驱动 + 双 backend 并存”：

- `lm_eval` 仍然是训练前 / 中 / 后的主评测链
- `lm_eval` 仍可直接评测内存中的 patched `model + tokenizer`
- `EvalScope` 是第二 backend，主要用于 path/checkpoint 参考评测
- `summary` 和顶层 `tasks` 默认始终代表 `primary_eval_backend`
- `backend=both` 时，两套 backend 结果都会保存在同一个结果 JSON 中

### 为什么保留 lm_eval 主链

原因很直接：训练前 baseline、训练中 mid eval、训练后 final eval 都需要在不先保存 checkpoint 的情况下评测 patched model。

这条能力链只靠 `lm_eval` 的 HF `HFLM(pretrained=model, tokenizer=tokenizer)` 路径就能稳定提供，所以它仍然是训练主线评测入口，不会被 EvalScope 替代。

### EvalScope 的定位

EvalScope 的角色是“第二 backend / 参考 backend”：

- 更适合 path/checkpoint 形式的评测
- 更接近某些模型官方评测口径时可以作为对照
- 不承诺支持内存中的 patched `nn.Module`
- 在训练中 `mid eval` 阶段，如果没有可用 checkpoint/path，会明确 `skipped`，而不是静默失败

### 统一入口

训练控制和 CLI 现在都走统一入口：

- `fitmotn.eval.runner.run_eval_tasks(...)`
- `python3 -m MOTN.fitmotn.cli.eval_auto ...`

统一返回结构形如：

```json
{
  "eval_name": "mid_eval_u1000",
  "eval_mode": "mid",
  "primary_backend": "lm_eval",
  "results": {
    "lm_eval": {"tasks": {}, "summary": {}},
    "evalscope": {"tasks": {}, "summary": {}}
  },
  "tasks": {},
  "summary": {},
  "warnings": []
}
```

兼容性约定：

- 顶层 `tasks` 和 `summary` 永远代表主 backend
- 现有 early stop、`compare_vs_baseline`、`run_summary.json` 继续只读取主 backend
- `results.{backend}` 保留每个 backend 的完整原始结果

### 训练前 / 中 / 后如何连续评测 patched model

当前训练链路的评测入口已经统一：

- `baseline_small`: 训练前、未 patch 前的 base model 评测
- `baseline_final`: 训练前的完整任务集评测
- `mid eval`: 训练中直接评内存中的 patched model
- `final_full`: 训练后评测 patched model，并可同时把 `final_model/` 路径交给 EvalScope

其中最关键的是：

- 训练中 `mid eval` 不需要先保存 checkpoint
- 如果 `eval_backend=both` 且 `primary_eval_backend=lm_eval`，但 EvalScope 在当前场景下无法运行，则只会在结果中留下 warning / skipped，不会破坏训练主线

### JSON 配置驱动

旧配置字段仍然可用，例如：

- `lm_eval_num_fewshot_*`
- `baseline_small_limit_*`
- `early_limit_*`
- `final_limit_*`
- `early_max_gen_toks_gsm8k`
- `final_max_gen_toks_*`

但新代码内部已经统一收敛到新的嵌套配置结构，推荐优先写新结构：

```json
{
  "eval": {
    "eval_backend": "both",
    "primary_eval_backend": "lm_eval",
    "backend_defaults": {
      "lm_eval": {"device": "cuda:0", "batch_size": 1},
      "evalscope": {"device": "cuda:0", "batch_size": 1}
    },
    "protocols": {
      "default": "legacy",
      "task_protocols": {"gsm8k": "model_aligned"}
    },
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
      "task_overrides": {"gsm8k": 8, "mmlu": 5}
    },
    "limits": {
      "baseline_small": {"gsm8k": 64, "mmlu": 128},
      "mid": {"gsm8k": 32, "mmlu": 64},
      "final": {"gsm8k": 0, "mmlu": 512, "hendrycks_math": 256}
    },
    "max_gen_toks": {
      "baseline_small": {"gsm8k": 256},
      "mid": {"gsm8k": 256},
      "final": {"gsm8k": 256, "hendrycks_math": 256}
    },
    "task_overrides": {
      "gsm8k": {
        "protocol": "model_aligned",
        "runtime": {"apply_chat_template": true},
        "generation": {"temperature": 0.0, "top_p": 1.0}
      }
    }
  }
}
```

### 最小示例

仅 `lm_eval`：

```json
{
  "eval": {
    "eval_backend": "lm_eval",
    "primary_eval_backend": "lm_eval"
  }
}
```

仅 `evalscope`：

```json
{
  "eval": {
    "eval_backend": "evalscope",
    "primary_eval_backend": "evalscope"
  }
}
```

`both`：

```json
{
  "eval": {
    "eval_backend": "both",
    "primary_eval_backend": "lm_eval"
  }
}
```

### 模型原生评测参数写在 JSON 中的示例

下面这类“模型原生评测口径”现在应当优先写在 JSON，而不是再写死到代码里：

```json
{
  "eval": {
    "protocols": {
      "default": "legacy",
      "task_protocols": {
        "gsm8k": "model_aligned",
        "mmlu": "model_aligned"
      }
    },
    "runtime": {
      "apply_chat_template": true,
      "enable_thinking": true,
      "think_end_token": "</think>"
    },
    "task_overrides": {
      "gsm8k": {
        "fewshot": 8,
        "generation": {
          "do_sample": false,
          "temperature": 0.0,
          "top_p": 1.0,
          "top_k": 20
        }
      }
    }
  }
}
```

这套写法的设计目标是：同一个 patch 方法只需要改 JSON，就能切换不同模型的评测协议，而不是在 Python 代码里继续堆特例。

### 当前 backend 生效边界

`lm_eval` 当前版本中，下面这些参数会真实透传：

- `apply_chat_template`
- `enable_thinking`
- `think_end_token`
- `fewshot`
- `limit`
- `gen_kwargs.temperature`
- `gen_kwargs.top_p`
- `gen_kwargs.top_k`
- `gen_kwargs.do_sample`
- `gen_kwargs.max_gen_toks`

如果你在 JSON 中写入了当前 backend 不识别的 runtime / generation 字段：

- 不会静默吞掉
- 会在 task 结果里留下 `warnings`
- 并记录 `ignored_runtime_args` / `unsupported_*`

EvalScope 当前实现边界：

- 优先支持 path/checkpoint 评测
- `gsm8k`、`mmlu` 已做标准接入
- `hendrycks_math` 目前是 best-effort；如果当前 EvalScope 版本里的 task 映射不匹配，会显式 `skipped` 或报错记录
- 当前仓库环境若未安装 `evalscope`，`eval_backend=evalscope` 会抛清晰 ImportError；`both` 配合 `--allow_backend_skip` 或训练中 mid eval 会记录 skipped/warning

EvalScope runtime / generation 参数状态约定：

- `system_instruction`: `effective`
  映射到 `dataset_args.<task>.system_prompt`
- `fewshot`: `effective`
  映射到 `dataset_args.<task>.few_shot_num`，并同步写入顶层 `few_shot_num/num_fewshot`
- `temperature` / `top_p` / `top_k` / `do_sample`: `effective`
  映射到 `generation_config.*`
- `max_gen_toks`: `effective`
  映射为 EvalScope 的 `generation_config.max_tokens`
- `apply_chat_template`: `best_effort`
  当前会尝试透传到 `chat_template=true`
- `enable_thinking` / `think_end_token` / `chat_template_args`: `best_effort`
  当前会尝试透传到 `generation_config.chat_template_kwargs.*`
- `fewshot_as_multiturn`: `record_only`
  当前 EvalScope `TaskConfig` 没有明确的官方等价开关，因此只记录，不宣称真实生效

每个 EvalScope task 结果里现在会额外包含：

- `runtime_effects`
- `generation_effects`
- `ignored_runtime_args`
- `unsupported_runtime_args`
- `unsupported_generation_args`
- `task_config`

### 示例配置文件

仓库新增了两个示例配置：

- [`fitmotn_config.eval_dual_backend.example.json`](./fitmotn_config.eval_dual_backend.example.json)
- [`fitmotn_config.model_aligned_eval.example.json`](./fitmotn_config.model_aligned_eval.example.json)

推荐用法：

- 训练主线：`primary_eval_backend=lm_eval`
- checkpoint 参考测评：用 `eval_auto.py --eval_backend both` 或 `--eval_backend evalscope`
