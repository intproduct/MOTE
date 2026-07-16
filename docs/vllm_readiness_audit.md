# FitMoTN vLLM 0.19 readiness 审计

## 支持状态定义

`vllm_ready=true` 只表示导出采用 vLLM Transformers backend 所需的结构，不再视为
真实 GPU 验收结论。完整可用必须依次通过：

1. `validate_vllm_export_preflight.py` 静态合约；
2. `validate_vllm_runtime_acceptance.py` 单卡 TP1 engine/generate；
3. HF/vLLM 固定 prompts 吞吐对比；
4. 两次 optimizer update 的 online policy sync；
5. 20 update 稳定性与显存验收。

## 已发现并修复

- ADTN FP32 activation 与 BF16 router/block 参数不匹配；
- exported AutoModel 未发布字典型 TP/PP plan；
- HF checkpoint 与 vLLM wrapper 多一层 `model` 前缀；
- tied `lm_head.weight` 被 safetensors 省略；
- AutoModel 的 unexpected-key 规则会让 vLLM 全局过滤 `lm_head`；
- 显式 HF roundtrip validation 原先没有执行 smoke forward；
- benchmark 原先会在静态 export 错误上浪费约两分钟 engine load。

## 当前禁止外推的能力

- TP2+：Qwen3 原生参数有 TP plan，但 ADTN `U` 参数没有原生 vLLM 分片规则；
- 原生/融合 ADTN kernel：当前仍是 Transformers backend Python/PyTorch 前向；
- 单卡 online RL：trainer 与 vLLM 同卡常驻的显存尚未验收；
- 高性能在线同步：`export_reload` 日志中单次 engine load 约 120 秒；
- 跨 vLLM 大版本兼容：当前 acceptance 目标固定为 vLLM 0.19.x。

## 风险分级与当前结论

|级别|项目|当前状态|进入真实 rollout 前要求|
|---|---|---|---|
|P0|TP1 engine 严格加载|未通过真实 GPU；旧导出已确认缺少 `lm_head.weight`|新目录重导出，静态 gate 与 GPU runtime gate 均为 `ok=true`|
|P0|HF/vLLM 语义一致性|未验收|同一 token-id prompt 的 greedy 首 token 必须一致|
|P0|prefill/decode/KV cache|未验收|变长 batch 和至少四路 sampled group 通过|
|P0|online policy freshness|控制逻辑有 CPU/mock 测试，未有 GPU 两更新证据|连续两次 optimizer update 后 fingerprint/version 严格递增且无 HF fallback|
|P1|`export_reload` 性能|本次日志单次加载约 119.5 秒|计入 export、load、generate 后再判断是否真的快于 HF|
|P1|单卡训练与 rollout 共存|未验收，24B 模型有明显 OOM 风险|先分时加载 smoke；正式 online 更推荐 trainer/actor 分卡|
|P1|TP2+ 单模型并行|Qwen 原生层有 TP plan；自定义 ADTN 线性参数默认可能复制，未做 GPU correctness/memory 验收|TP1 通过后单独做 TP2 gate，不得从普通 Qwen 支持情况推断 FitMoTN 已支持|
|P1|N 个 TP1 rollout replica|调度、切分与故障传播有 mock 测试，未做多 GPU 验收|逐级执行 2 actor、4 actor、8 actor smoke|
|P1|Transformers backend 性能|无原生/融合 ADTN kernel|以实测 token/s 为准；不能预设 vLLM 一定加速|
|P2|Mistral tokenizer regex|新代码统一启用修复；旧 export 不会自动变化|必须重导出 tokenizer，并确认 warning 消失|
|P2|动态 remote-code 缓存|同一路径可能复用旧模块|每次代码修改使用新 export 目录和新 `HF_MODULES_CACHE`|
|P2|版本漂移|已将 extra 约束到 vLLM 0.19.x、Transformers 4.56～4.x|证据 JSON 必须记录实际 torch/transformers/vLLM/CUDA/GPU|

因此，“所有 vLLM 问题已排除”只能在目标 Docker/GPU 上完成。无 CUDA 环境能够排除的
静态问题已经被自动化，但 CUDA attention、KV cache、worker 生命周期、显存峰值、NCCL
以及真实性能不能由 CPU/mock 测试证明。

## 为什么此前逐层出现问题

此前测试主要覆盖 HF roundtrip、export layout、actor 控制面和 mock vLLM API，并没有启动
vLLM 0.19 的真实 `TransformersForCausalLM` strict loader。loader 按阶段执行，前一个错误
会遮蔽后一个错误：先是 `tp_plan`，再是 checkpoint prefix，最后才是 tied `lm_head`。
此外，过去的 `vllm_ready=true` 把“已经生成兼容结构”表达得过于接近“GPU 已验证”。现在
manifest 增加 `static_contract_ready_gpu_acceptance_required`，并由分层 gate 给出真实结论。

## 静态 gate

```bash
PYTHONPATH=/path/to/repo-parent \
python fitmotn/scripts/validate_vllm_export_preflight.py \
  --model /path/to/hf-export \
  --output-json /path/to/evidence/vllm_export_preflight.json
```

该 gate 不读取 tensor 数据，只读 safetensors 元数据并在 meta device 构造模型，检查：

- remote code 与当前仓库一致；
- `tp_plan`/`pp_plan` 类型；
- 显式 embedding 和 lm head；
- vLLM 0.19 风格 key mapping 后的完整覆盖；
- TP1 shape、冲突、缺失和多余键。

## 单次 GPU runtime gate

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/repo-parent \
HF_MODULES_CACHE=/path/to/new-cache \
python fitmotn/scripts/validate_vllm_runtime_acceptance.py \
  --model /path/to/hf-export \
  --output-json /path/to/evidence/vllm_runtime_acceptance.json
```

该 gate 先由 HF 生成同一输入的下一 token，释放显存后只建立一个 vLLM engine，依次
检查：加载、HF/vLLM 首 token 一致、token-id prompt、一 token greedy 可复现、变长
batch、四路随机 rollout。只有结果 `ok=true` 才允许进入吞吐 benchmark。

TP2 独立验收时，在只暴露两张 GPU 的进程中增加 `--tensor-parallel-size 2`。
