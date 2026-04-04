# FitMoTN README

`fitmotn/` 是在不修改 `./MOTN/` 旧实现文件的前提下，新建的一套标准化 MOTN 训练与评测框架。它的目标不是改写 MOTN 方法本身，而是把旧入口脚本里的训练、数据、patch、checkpoint、评测逻辑拆开，并用 Hugging Face Trainer 承接通用训练外壳。

当前版本是 MVP，重点保证这几件事：

- patch 后模型可以按新框架训练
- 不改旧文件
- baseline 开关可用
- 保存/加载基本可用
- HF/lm-eval 评测可用
- vLLM 有独立入口，但 patched FitMoTN checkpoint 还不支持直接用 vLLM 执行

## 目录说明

核心目录如下：

```text
MOTN/fitmotn/
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
  只做桥接与封装，直接复用 `MOTN.ADTN.MoTNLayer` 与 `MOTN.gate.GateConfig`
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

FitMoTN 明确保留了以下 MOTN 核心语义：

- 直接复用 `ADTN.py` 的 `MoTNLayer`
- 直接复用 `gate.py` 的 `GateConfig` 与 gate 行为
- Qwen MLP patch 仍然是 `gate_proj(MoTN) + up_proj(MoTN) + act + down_proj(MoTN)`
- `down_proj.k_in` 仍由前面投影的 `k_out` 推导
- base model 全冻结，只训练 patched MoTN 层
- 保留 `gate_freeze_steps`
- 保留 warmup routing
- 保留 temperature schedule
- 保留 stage A / stage B 两阶段混采思想

换句话说，Trainer 只是“训练外壳”，不是方法定义层。

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

vLLM 评测额外需要：

```bash
pip install -U vllm
```

## 训练入口

训练 CLI：

- [train.py](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/cli/train.py)

重要说明：

- 命令行参数只暴露了一小部分高频入口，方便快速试跑
- `config_json` 才是当前推荐的全量实验参数注入接口
- 不仅训练参数能改，MOTN 方法相关参数也能从 `config_json` 外部注入
- 默认值与旧脚本 `run_train_motn_mixed_qwen3.py` 保持一致，但不是写死不可改
- 当前 CLI 和 `config_json` 都走统一配置加载与字段校验；未知 section/key 会直接报错，不再静默忽略

最小示例：

```bash
python3 -m MOTN.fitmotn.cli.train \
  --model_path /path/to/Qwen3-0.6B \
  --output_root ./MOTN/fitmotn_runs \
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
    "model_path": "/path/to/Qwen3-0.6B",
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
    "eval_every_updates": 500,
    "save_every_updates": 500
  },
  "eval": {
    "run_baseline_eval": true
  },
  "output": {
    "root_dir": "./MOTN/fitmotn_runs",
    "run_name": "demo_run"
  }
}
```

启动方式：

```bash
python3 -m MOTN.fitmotn.cli.train --config_json ./fitmotn_config.json
```

推荐做法：

- 日常正式实验统一使用 `config_json`
- 命令行只用于覆盖极少数临时参数

## 完整可调接口

当前版本中，实验参数的完整外部控制入口是：

- [fitmotn_config.example.json](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/fitmotn_config.example.json)
- `python3 -m MOTN.fitmotn.cli.train --config_json your_config.json`

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
- `use_wiki_local`
- `use_fineweb`
- `use_code`
- `use_gsm8k_train`
- `use_gsm8k_socratic_train`
- `use_svamp_train`
- `use_metamath_train`
- `use_math_train`
- `use_mmlu_train`
- `wt_wiki`
- `wt_fineweb`
- `wt_code`
- `wt_gsm8k`
- `wt_gsm8k_socratic`
- `wt_svamp`
- `wt_metamath`
- `wt_math`
- `wt_mmlu`

### `train` 可调字段

- `batch_size`
- `grad_accum`
- `steps`
- `epochs`
- `epoch_samples`
- `lr`
- `log_every`
- `save_every_updates`
- `eval_every_updates`
- `max_grad_norm`
- `stage_a_ratio`
- `stage_a_pretrain_ratio`
- `stage_a_task_ratio`
- `stage_b_pretrain_ratio`
- `stage_b_task_ratio`
- `begin_t`
- `end_t`
- `gate_freeze_steps`
- `usage_dump_every`
- `lr_warmup`
- `lr_warmup_steps`
- `warmup_ratio`
- `early_stop`
- `early_stop_abs_gsm8k`
- `early_stop_abs_mmlu`
- `seed`
- `report_to`

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

### `output` 可调字段

- `root_dir`
- `run_name`
- `overwrite_output_dir`

## 完整 `config_json` 模板

下面是当前 README 内联展示的完整配置形式。它和 [fitmotn_config.example.json](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/fitmotn_config.example.json) 一致，可以直接复制后修改。

```json
{
  "model": {
    "model_path": "/work/home/sugang2025/qxfang/models/Qwen3-0.6B",
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
    "tok_shard_dir": "/work/home/sugang2025/qxfang/wiki24_tok",
    "datas_dir": "/work/home/sugang2025/qxfang/Datas",
    "seq_len_run": 1024,
    "dataloader_num_workers": 0,
    "fineweb_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/fineweb_sample10bt",
    "code_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/the_stack_v2",
    "gsm8k_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/gsm8k_main",
    "gsm8k_socratic_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/gsm8k_socratic",
    "svamp_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/svamp",
    "metamath_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/metamathqa",
    "mmlu_cache_path": "/work/home/sugang2025/qxfang/Datas/hf_cache/mmlu_all",
    "math_cache_root": "/work/home/sugang2025/qxfang/Datas/hf_cache/hendrycks_math",
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
    "root_dir": "./MOTN/fitmotn_runs",
    "run_name": "fitmotn_demo",
    "overwrite_output_dir": false
  }
}
```

新增一个可直接参考的 reasoning mix 配置：[approx_35000_reasoning_mix_v1.json](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/approx_35000_reasoning_mix_v1.json)。它会并行启用 `gsm8k main + gsm8k socratic + svamp + metamath`，并把 `metamath_max_samples` 默认限制在 `20000`。

如果你要做“恢复 patch 后小模型推理能力”的两阶段实验，可以直接参考这两个新增配置：

- [fitmotn_reasoning_recovery_min_b.json](/Users/qixuanfang/Library/Mobile Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/fitmotn_reasoning_recovery_min_b.json)
- [fitmotn_reasoning_recovery_conservative_b.json](/Users/qixuanfang/Library/Mobile Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/fitmotn_reasoning_recovery_conservative_b.json)

这两份配置会保留 `wiki / fineweb / code / gsm8k / MATH / 少量 MMLU`，并额外接入 `OpenR1-Math-220k / NuminaMath-CoT / OpenThoughts-114k-math / Bespoke-Stratos-17k`。其中 `OpenThoughts` 默认启用更严格的长度过滤，新增 reasoning 数据统一整理为 `Question / Solution / Final Answer` 风格训练文本。

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
- `train.jsonl`
  训练 step/update 级记录
- `usage.jsonl`
  路由 / 专家 usage 结构化记录
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
- `layers_to_patch`
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

## 论文级观测

当前版本会在训练中持续记录这些结构化信息：

- `train.jsonl`
  包含 `loss`、`lr`、`T`、`gate_trainable`、`tokens/s`、`显存`、`optimizer`、`scheduler`、`batch_task_names`、`batch_groups`、`batch_source_families`
- `usage.jsonl`
  包含每层 `usage_*`、`top1_*`、`pos_*`、`entropy_*`、`load_balance_*`、`active_expert_count_*`、`max_expert_share_*`、`expert_cv_*`
- `mid_eval.jsonl`
  包含每次评测对应的训练上下文、评测结果和相对 baseline 的对比
- `run_summary.json`
  用于批量实验扫描和论文总表汇总

## HF 评测

HF/lm-eval 入口：

- [eval_hf.py](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/cli/eval_hf.py)

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

如果 `model_or_ckpt` 目录下有 `fitmotn_state.pt`，脚本会自动按下面顺序恢复：

1. 加载 base model
2. 按 `layers_to_patch + motn_cfg` 重新 patch
3. 加载 `patch_state_dict`；如果是旧 checkpoint，则回退到 `state_dict`
4. `load_state_dict(strict=False)`

这一步是为了保证 patch 结构不会被 `save_pretrained()` 平铺掉。

## vLLM 评测

vLLM 入口：

- [eval_vllm.py](/Users/admini/Library/Mobile%20Documents/com~apple~CloudDocs/document/a800/MOTN/fitmotn/cli/eval_vllm.py)

示例：

```bash
python3 -m MOTN.fitmotn.cli.eval_vllm \
  --model_or_ckpt /path/to/baseline_or_vllm_compatible_model \
  --limit_per_task 32
```

### 当前 vLLM 边界

MVP 明确只支持：

- baseline 模型评测
- 原生兼容 vLLM 的 checkpoint 评测

MVP 明确不支持：

- patched FitMoTN checkpoint 直接用 vLLM 执行

如果你把 `final_model/` 这类 patched FitMoTN checkpoint 直接传给 `eval_vllm.py`，当前实现会显式报错：

- `NotImplementedError`
- 同时会在输出 JSON 中写入 `capability` 字段，说明当前模型路径为什么不支持 vLLM 评测

这是有意为之，目的是避免 silently fallback，防止 MOTN 核心方法被偷偷替换或退化。

阶段性警告：

- 当前 `eval/vllm_runner.py` 不支持 patched FitMoTN checkpoint 的直接 vLLM 执行，这属于评测边界限制，不是 MOTN 核心训练逻辑缺陷
- 本轮只保留显式报错和 README 说明，不实现 patched checkpoint 的 vLLM 兼容桥接

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
