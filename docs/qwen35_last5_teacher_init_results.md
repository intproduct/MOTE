# Qwen3.5-4B 最后五层 FFN teacher-init 单层近似结果

## 实验定义

- 模型：`D:\AI\Models\Qwen3.5-4B`，Qwen3.5-4B 的语言模型共有 32 层；本实验使用 0-based 的第 27–31 层。
- 数据：从 `D:\AI\Datas\WiKI\wikien_2023.jsonl` 读取原始 Wiki 文本，再用目标 Qwen3.5-4B 自己的 tokenizer 编码。没有直接使用由其他 tokenizer 预先生成的 shard token id。
- teacher 输入：完整原始模型前向传播时，在每一层原始 MLP 的 forward pre-hook 捕获该 MLP 真正收到的 hidden state。
- teacher 目标：同一次前向传播中，在原始 MLP 的 forward hook 捕获输出。每一层独立训练，不把近似层的误差传给下一层。
- 样本：16 条 × 128 token，共 2048 token；前 1536 个用于随机 token batch 训练，后 512 个只用于验证。
- 优化：每个 `layer × backend` 独立随机初始化并训练 300 步，AdamW，学习率 `3e-4`，batch 32 token，每 50 步验证，按验证集相对 L2 保存最佳状态。
- 路由：16 experts，top-k 8，启用 global expert，router 输入为完整真实维度。
- 数值环境：RTX PRO 4000 Blackwell，PyTorch 2.10.0+cu130，BF16。Windows 上 24 个 linear-attention 层使用 Transformers 自带纯 PyTorch fallback；没有修改模型权重或 FFN 计算。

这里的 “teacher-init” 指用原始 dense FFN 的真实输入/输出监督结构化 FFN 初始化；不是把 dense 权重直接复制进结构化参数。

## 结果

`rel_L2 = ||student(x)-teacher(x)||_2 / ||teacher(x)||_2`，越低越好；cosine 越高越好。

| 层 | backend | 参数量 | 初始 rel L2 | 300 步最佳 rel L2 | 最佳 cosine | 最佳步 |
|---:|:---|---:|---:|---:|---:|---:|
| 27 | sparse_mixt | 17,825,792 | 1.3101 | **0.6723** | **0.6315** | 300 |
| 27 | mixed_mixt | 14,538,752 | 3.2927 | 1.7434 | 0.2683 | 300 |
| 28 | sparse_mixt | 17,825,792 | 1.3033 | **0.7462** | **0.6667** | 300 |
| 28 | mixed_mixt | 14,538,752 | 3.7486 | 1.8482 | 0.2819 | 300 |
| 29 | sparse_mixt | 17,825,792 | 1.2633 | **0.6966** | **0.7166** | 300 |
| 29 | mixed_mixt | 14,538,752 | 3.3231 | 1.5420 | 0.3739 | 300 |
| 30 | sparse_mixt | 17,825,792 | 1.1691 | **0.6382** | **0.7628** | 300 |
| 30 | mixed_mixt | 14,538,752 | 2.4914 | 1.2586 | 0.4682 | 300 |
| 31 | sparse_mixt | 17,825,792 | 1.0189 | **0.3603** | **0.9049** | 300 |
| 31 | mixed_mixt | 14,538,752 | 1.2691 | 0.5371 | 0.8074 | 300 |
| 平均 | sparse_mixt | 17,825,792 | 1.2129 | **0.6227** | **0.7365** | 300 |
| 平均 | mixed_mixt | 14,538,752 | 2.8250 | 1.3858 | 0.4399 | 300 |

## 计算路径核对

脚本在每个单层训练开始前，都对该层的一枚真实 teacher input 调用 `forward_with_trace`，同时打印并写入对应 run JSON 的 `computation_trace`。

Sparse 路径为：

```text
y0 = MiXT00(x0) + LR01(x1)
y1 = LR10(x0) + LR11(x1)
```

Mixed 路径为：

```text
shared_gate(full_real_input) -> probabilities/mask, calls=1
y0 = M00(x0) + M01(x1)
y1 = M10(x0) + M11(x1)
```

`gate_proj`、`up_proj` 和 `down_proj` 各自有一次共享 gate；同一 projection 内的四个 MiXT quadrant 共用该次 gate 的 probabilities/mask，但使用各自的 bond 和专家参数。三个 projection 仍按原始 Qwen MLP 逻辑组成 `down(SiLU(gate(x)) * up(x))`。

## 初步结论

1. 两种 backend 都能在真实 Qwen3.5-4B activation 上稳定反向传播，10 个运行均持续改善，且最佳验证点均在第 300 步。
2. 当前配置下 sparse_mixt 明显优于 mixed_mixt；mixed 的参数量少约 18.4%，但平均相对 L2 高约 0.763。
3. mixed 在第 27–30 层初始相对误差尤其高，而这些层的 teacher 输出 RMS 只有约 0.23–0.35；第 31 层输出 RMS 约 0.98 时，mixed 的结果显著改善。这表明当前四组 `gamma_normal` 的复合输出尺度是主要问题之一，不能仅把差距解释为 full-MiXT 结构容量不足。
4. 下一步应增加 teacher-aware scale calibration（至少分别校准 gate/up/down，最好校准每个 quadrant）后再做同协议对照；还可继续训练超过 300 步，因为所有曲线在第 300 步仍在改善。

原始机器可读结果保存在 `outputs/qwen35_teacher_init_last5/summary.json`，每层完整 history/trace 在 `outputs/qwen35_teacher_init_last5/runs/`，10 个最佳权重在 `outputs/qwen35_teacher_init_last5/states/`。`outputs/` 已由 Git 忽略，不会意外提交约 414 MiB 的 activation 和 checkpoint 文件。
