# 双 V100 执行任务：兼容性与独立复核

先读上级 README、COMMON_PROTOCOL、ENVIRONMENT 和 plans/v100_pair.json。
检查两块 GPU 的实际空闲、拓扑和 torch 的 sm70 支持；不结束其他进程，不使用子代理。
立即可执行证据哈希检查、既有 CPU 测试和两块卡各自的 GPU 数值探针。

正式使用由 PRO 4000 集成任务发布的同一个 Qwen 接口与数据版本。保留 FP16+
GradScaler、router FP32 的独立 cohort，记录 loss scale、跳过的更新及非有限值。
基模是 Qwen3-0.6B-Base，从第一步共同更新 gate 与专家。

优先两个独立单卡 worker：GPU0 依次承担 seed17、43，GPU1 承担 seed29；每个 seed
都运行四门及 Dense FFN 控制。输出中加入 worker、seed、method、release 和 run ID，
每卡单独持锁。普通 DDP 不会把两块16GB合成一块32GB。

首先确认单卡1024长度、microbatch1可运行；累积到共同有效 token 数。
若不足，保存失败日志，测激活重算与分块词表损失；修改序列长度或启用 FSDP 时建立
独立完整实验配置，对全部方法一致应用，并重新做单卡/分片更新等价验证。

读取四门梯度、专家更新、样本顺序、保存/恢复、负载和实际吞吐。正式训练前完成
全部方法的共同短测。跨 BF16/FP16 的结果作为硬件精度复核单列；同一张表中的主结论
依赖本 cohort 的完整配对矩阵。完成后输出环境、各 seed 原始结果、配对统计和中文报告。
共享代码修复在 `codex/qwen-joint-v100` 提交供整合，保持每组运行源码冻结。
