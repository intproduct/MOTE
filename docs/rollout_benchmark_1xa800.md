# 单卡 A800：HF 与 vLLM rollout 公平测速

该脚本只比较生成阶段，不执行反向传播、reward、logprob、checkpoint export 或
vLLM 权重同步。HF 与 vLLM 在两个独立进程中依次运行，并读取同一个预分词后的
prompt manifest，避免模型共存、显存残留和输入差异。

## Docker 环境要求

- 一张 NVIDIA A800 80GB；
- 容器能执行 `nvidia-smi`；
- PyTorch、Transformers、Datasets 与 vLLM；
- vLLM 0.19.x、Transformers 4.56～4.x；
- 一个完整的 Stage 4C HF roundtrip export，不能直接使用 raw FitMoTN checkpoint。

若没有执行 editable install，应在仓库父目录运行，并显式设置 `PYTHONPATH`。例如仓库为
`/work/home/sugang2025/qxfang/MOTE-g/fitmotn`：

```bash
cd /work/home/sugang2025/qxfang/MOTE-g
export PYTHONPATH=/work/home/sugang2025/qxfang/MOTE-g
```

先确认环境：

```bash
nvidia-smi
python -c "import torch, vllm; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), vllm.__version__)"
```

## 配置路径

复制示例配置并修改：

```bash
cp fitmotn_config.rollout_benchmark_1xa800.example.json rollout_benchmark.json
```

至少设置：

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export OUTPUT_ROOT=/path/to/outputs
export CUDA_VISIBLE_DEVICES=0
```

`model.path` 必须指向完整 HF export。`data.prompts_jsonl` 支持每行包含
`question` 或 `prompt` 的 JSONL；如果将其设为空字符串，脚本会从配置的
Hugging Face dataset 加载 GSM8K。

## 推荐执行顺序

不要直接进入吞吐 benchmark。先使用**新目录重新导出**，导出命令默认运行静态
vLLM contract gate，并生成 `vllm_export_preflight.json`。旧的
`k8_nodecay_7e_tp8_tc_hf` 若没有显式 `lm_head.weight`，不能继续复用。

然后执行单引擎 GPU acceptance：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_MODULES_CACHE=/work/home/sugang2025/qxfang/MOTE-g/hf_modules_cache/vllm_acceptance_v1 \
python fitmotn/scripts/validate_vllm_runtime_acceptance.py \
  --model /work/home/sugang2025/qxfang/MOTE-g/fitmotn_exports/NEW_EXPORT_DIR \
  --output-json /work/home/sugang2025/qxfang/MOTE-g/fitmotn_exports/benchmark/vllm_runtime_acceptance.json
```

只有 JSON 中 `ok=true` 才进入下面的 rollout benchmark。每次 remote code 修改后使用
新的 export 目录和新的 `HF_MODULES_CACHE`，避免 Transformers 动态模块缓存复用旧代码。

先做 16 题快速烟测，将复制后的配置改为：

```json
"num_prompts": 16,
"group_size": 4,
"warmup_rows": 4,
"repeats": 1
```

运行：

```bash
python scripts/benchmark_rollout_backends.py \
  --config rollout_benchmark.json \
  --output-dir "${OUTPUT_ROOT}/rollout_benchmark_smoke"
```

烟测通过后，使用示例默认值执行正式测试：128 个唯一 prompt，每题 8 条
rollout，共 1024 条生成，预热后重复 3 次。

```bash
python scripts/benchmark_rollout_backends.py \
  --config rollout_benchmark.json
```

也可以只跑单个后端定位问题：

```bash
python scripts/benchmark_rollout_backends.py --config rollout_benchmark.json --backends hf
python scripts/benchmark_rollout_backends.py --config rollout_benchmark.json --backends vllm
```

## 结果与解读

输出目录包含：

- `vllm_export_preflight.json`：vLLM engine 启动前的权重与 remote-code 静态契约；
- `effective_config.json`：实际使用的配置；
- `fixed_prompt_tokens.json`：两个后端共享的固定 token 输入；
- `hf.log`、`vllm.log`：完整日志；
- `hf_result.json`、`vllm_result.json`：分后端证据；
- `comparison.json`：最终汇总与加速比。

主要查看 `comparison.json` 中：

- `vllm_vs_hf_generated_tokens_per_second_speedup`：首要吞吐指标；
- `vllm_vs_hf_prompts_per_second_speedup`：包含生成长度差异；
- `vllm_minus_hf_load_seconds`：vLLM 额外加载代价。

由于两个采样器的随机轨迹和 EOS 位置不会完全相同，`prompts/s` 会受到平均输出
长度影响；因此应将 `generated_tokens/s` 作为首要性能指标，同时检查两边的
`median_average_generated_tokens` 是否差异过大。

`process_lifetime_peak_gpu_memory_mib` 是父进程用 `nvidia-smi` 周期采样得到的整个
worker 生命周期峰值，包含加载与生成。HF 结果还会给出 PyTorch 生成阶段的
allocated/reserved 峰值。

这组结果不能代表完整 online RL 加速比。完整 RL 还需计入 policy export、engine
rebuild/weight sync、reward、HF logprob、反向传播与优化器时间。
