"""Build the local acceptance summary and matching Markdown/Typst reports."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))

long = read("long_export_acceptance.json")
training = read("wsl_8k_training.json")
quantum = read("wsl_quantum_training.json")
vllm = read("vllm_acceptance_native_sampler.json")
sweep = read("wsl_gpu_acceptance.json")
timeout = read("wsl_training_timeout.json")
assert long["ok"] and training["completed"] and quantum["completed"]
assert all(r["ok"] for r in training["training"] + quantum["training"])
vllm_status = "有条件通过" if vllm.get("ok") else "未通过"
summary = {
    "implementation_commit": "f6b6e9d0fd94ef48b7802b0ae55a6f8a3fca3f92",
    "publication_branch": "mote-quantum",
    "scope": "local workstation only",
    "acceptance": {
        "hf_32k_import_export_parity": "passed",
        "linear_gate_8k_last_32_supervised_update": "passed",
        "quantum_gate_512_two_updates": "passed",
        "vllm_023_single_gpu_generation": "conditional_pass" if vllm.get("ok") else "failed",
        "pinned_vllm_019": "not_tested",
        "training_32k": "incomplete_bounded_probe",
        "multi_gpu": "blocked_by_environment_and_training_checkpoint_contract",
        "task_quality_and_full_tir": "not_tested",
    },
    "hf_32k": long,
    "linear_training": training["training"],
    "quantum_training": quantum,
    "vllm": vllm,
    "vllm_environment": {
        "version": "0.23.0",
        "outside_declared_project_range": True,
        "nvml_fallback_manifest": "wsl_vllm_environment_patch.json",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "compiler": "isolated conda-forge GCC 16.2.0",
        "failed_attempts": [
            {"record": "vllm_acceptance.json", "root_cause": "C compiler missing"},
            {"record": "vllm_acceptance_retry.json", "root_cause": "FlashInfer sampler requires absent nvcc/CUDA toolkit"},
        ],
    },
    "training_32k_interruption": timeout,
    "final_integrity": read("final_integrity.json"),
    "multi_gpu_evidence": {
        "windows_ranks": [read(f"windows_two_gpu/rank_{r}.json") for r in range(2)],
        "windows_uv_ranks": [read(f"windows_two_gpu_uv/rank_{r}.json") for r in range(2)],
        "wsl_visible_gpus": ["NVIDIA RTX PRO 4000 Blackwell"],
        "code_guards": ["train/trainer.py: atomic checkpoints reject world_size > 1", "train/sequence_parallel.py: validate_model rejects qwen3_5_text"],
    },
}
(HERE / "acceptance_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

rows = "\n".join(f"| {r['tokens']} | {r['seconds']:.3f} | {r['peak_allocated_gib']:.3f} |" for r in sweep["inference"])
qrows = "\n".join(f"| {i+1} | {r['supervision']} | {r['nonzero_gradient_tensors']} | {r['updated_parameter_tensors']} | {r['peak_allocated_gib']:.3f} |" for i,r in enumerate(quantum["training"]))
vllm_result = ("两条提示词各生成 8 个 token，与合作方 HF 模型全部一致；重复调用结果一致，分别单条调用与双条批量调用一致，提示词 token 完整保留。" if vllm.get("ok") else
              "实际生成验收未通过。错误和完整阶段状态保留在 vllm_acceptance_native_sampler.json 与对应日志中。")
report = f'''<style>body,h1,h2,h3,p,table,li,pre,code {{ font-family: "Times New Roman", "KaiTi", serif; }}</style>

# MixT-2611v2 本机 GPU 验收

2026-11-25；适配代码提交 f6b6e9d；发布分支 mote-quantum。

## 验收结论

本机已通过合作方权重承接、32K 推理与导出一致性、8K 有限监督 gate 更新及 512 token QuantumGate 两步更新。vLLM 单卡生成的结果为“{vllm_status}”。32K 训练和多卡训练尚未通过完整验收。

| 项目 | 结论 | 验收范围 |
| --- | --- | --- |
| 32K HF 推理、导入、导出 | 通过 | BF16、SDPA、linear gate；末位置全词表 logits 逐位一致，后续 2 token 一致 |
| 8K gate-only 更新 | 通过 | 单卡、batch 1、最后 32 token 监督；峰值 14.66 GiB，78 个张量非零梯度 |
| QuantumGate GPU 更新 | 通过 | 512 token；两次优化更新，参数实际改变，梯度有限 |
| vLLM 0.23 单卡生成 | {vllm_status} | 隔离环境、原生适配器、TP=1、禁用图编译与前缀缓存 |
| 项目声明的 vLLM 0.19 | 未验收 | 当前本机可用 Linux 安装为 0.23；项目依赖范围保持原状 |
| 32K 训练 | 未通过 | 反向传播约 319 秒未完成，主动终止本次测试 |
| 多卡训练及恢复 | 未通过 | 本机环境受限；正式入口仍有多进程断点与混合注意力适配缺口 |

## 实测环境与长上下文

WSL Ubuntu 24.04 使用 RTX PRO 4000 Blackwell 24GB、PyTorch 2.11.0+cu130、Transformers 5.15.1。测试采用 BF16、SDPA、batch 1。导入保持四个原始窗口的均匀路由，关闭 usage 统计。以下是单次前向的观测值，峰值是 PyTorch 分配的显存；计时包含同步，不含模型加载。

| 输入 token | 前向秒数 | 峰值 GiB |
| --- | --- | --- |
{rows}

额外对同一 32768 token 输入分别运行合作方原模型、MoTE 导入模型和完整 HF 导出模型。导入与导出的末位置 logits 最大绝对误差均为 0，逐位相等；三者缓存生成均输出 token [279, 33633]。导出模型的前向和生成三次词表投影均只处理一个位置，包含生成时峰值约 13.82 GiB。

输入为重复构造的合成上下文。这一结果验证长输入的数值承接和缓存行为；检索准确率、推理能力、训练收益及配置中的更大上下文上限需要各自的数据集评测。

Windows 原生环境为 PyTorch 2.10.0+cu128。同卡 8K 推理通过，16K 前向和 8K 训练出现显存不足。两套运行栈及注意力内核不同；本轮判断是运行栈影响显存占用，不能用 Windows 的失败点推定模型长度上限。

## 训练检查

Linear gate 先完成 512 token 全监督更新，再完成 8K 上下文、末 32 token 监督的更新。第二步耗时 30.92 秒、峰值 14.66 GiB，78 个路由参数张量有非零梯度，loss 与梯度均有限。这是两步工程检查，8K 全 token 监督、长时间训练稳定性与更大 batch 尚未覆盖。

QuantumGate 在同一真实模型上验证 128 token 输入的末位置 logits，导入前后最大误差为 {quantum['gpu_import_max_abs']}，随后完成 512 token 缓存生成及两次 gate-only 更新：

| 更新 | 监督范围 | 非零梯度张量 | 实际更新张量 | 峰值 GiB |
| --- | --- | --- | --- | --- |
{qrows}

32K 训练使用末 32 token 监督、非重入梯度检查点。反向传播期间显存接近本机容量，约 319 秒仍未完成；测试被主动停止。没有得到完成的 loss backward 和优化更新，因此本项不通过。原始中间结果与终止原因分开保存在 wsl_gpu_acceptance.json 和 wsl_training_timeout.json。

## vLLM 与多卡边界

{vllm_result}

本机 vLLM 是 0.23.0，真实模型加载占用约 6.48 GiB。当前使用原生 Qwen3.5 注意力和混合缓存，加上导出的 MoTE FFN，max_model_len=4096、max_num_seqs=2、max_num_batched_tokens=512、enforce_eager=true、enable_prefix_caching=false。该检查仅覆盖所测提示词；吞吐、多卡张量并行、前缀缓存、图编译与在线权重同步保留为独立验收项。

为运行验收，建立了独立 Python 环境，并复制 vLLM 到用户缓存目录。NVML 能看到宿主机 TCC 卡但无法查询它们，复制版改为选择 vLLM 自带的 NonNvmlCudaPlatform，通过实际 CUDA 可见设备查询属性。原环境文件哈希保持一致。另在独立目录安装 C/C++ 编译工具；FlashInfer 采样器因缺少 nvcc 失败，最终使用官方开关 VLLM_USE_FLASHINFER_SAMPLER=0。此环境结果不等同于未经调整的安装或项目锁定的 0.19 环境验收。

Windows 可见两张 V100，原生 BF16 不支持；WSL 仅可见 RTX PRO 4000。双 V100 的两个 Gloo 进程均在初始化阶段报 unsupported gloo device，设置 UV transport 后仍相同。NVIDIA 的 WSL 文档也将 Tesla/TCC 排除在该支持范围之外。[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/archive/13.0.2/wsl-user-guide/index.html)

vLLM 当前官方 GPU 要求的计算能力下限为 7.5，本机 V100 为 7.0，不能作为本次 BF16 vLLM 多卡验收平台。[vLLM GPU 安装要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html)

代码层面，train/trainer.py 的原子检查点拒绝 world_size 大于 1，要求协调各 rank 状态；现有 Ulysses 只支持全注意力 Qwen3/Llama，明确拒绝 Qwen3.5 的循环注意力。因此后续需要实现并验证对应训练入口、分布式保存恢复，以及选择合适的并行方案。仅更换显卡不能消除这些代码边界。

## 代码修复与证据

export/remote_code/modeling_fitmotn.py 增加 logits_to_keep，先选择 hidden states 再计算词表投影，并保持对应标签的下一 token 对齐；生成入口传递该参数。相关回归 41 项通过，之前完整适配回归 181 项通过。GPU 训练示例显式采用 SDPA；精确恢复仍要求保留断点的注意力设置。

新导出包位于 outputs/mixt_2611v2_hf_runtime，保留此前导出包。机器总表为 research/mixt_runtime_20261125/acceptance_summary.json，包含各阶段证据和验收范围。该目录中的 gpu_acceptance.py、long_export_acceptance.py、vllm_acceptance.py、probe_distributed.py 分别用于 GPU 训练/推理、32K 三方对照、vLLM 生成与通信探测。结构化 JSON、脚本和报告纳入版本管理，完整终端日志保留在本机。

本机隔离环境根目录为 /home/frank_f/.cache/mixt-acceptance-20261125；venv、toolchain 和 overrides 分别保存 Python 环境、编译器和 vLLM 复制版。vLLM 复现时需 CUDA_VISIBLE_DEVICES=0、VLLM_WORKER_MULTIPROC_METHOD=spawn、VLLM_USE_FLASHINFER_SAMPLER=0，并将 CC/CXX 指向 toolchain/bin 下的编译器，PYTHONPATH 指向 overrides。模型来源为 /mnt/f/MixT-2611v2，完整导出位于本工作副本的 outputs/mixt_2611v2_hf_runtime。

适配代码已推送 mote-quantum 的 f6b6e9d。D:/AI/MOTE 保持冻结提交 860426ce3df30f5c2a74a030a43f22e3a76247c6，合作方源模型及原实验产物保持原状。当前机器未发现 Typst 编译器，交付同内容的 Typst 源文件。
'''
dest = ROOT / "docs" / "mixt_2611v2_runtime_acceptance_20261125"
dest.with_suffix(".md").write_text(report, encoding="utf-8")

def plain(value):
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", value).replace("`", "")
def txt(value):
    return "#text(" + json.dumps(plain(value), ensure_ascii=False) + ")"
typ = ['#set page(paper: "a4", margin: (x: 20mm, y: 20mm), numbering: "1")',
       '#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")',
       '#set par(justify: true, leading: 0.65em)', '#show heading: set text(font: ("Times New Roman", "KaiTi"))',
       '#set table(inset: 5pt, stroke: 0.4pt + rgb("b9c3ca"))']
lines = report.splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line or line.startswith("<style>"):
        i += 1
        continue
    if line.startswith("|"):
        table = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [cell.strip() for cell in lines[i].strip("|").split("|")]
            if not all(re.fullmatch(r"[-:]+", cell) for cell in cells):
                table.append(cells)
            i += 1
        typ.append("#table(columns: " + str(len(table[0])) + ", " + ",\n".join("[" + txt(cell) + "]" for row in table for cell in row) + ")")
        continue
    if line.startswith("#"):
        n = len(line)-len(line.lstrip("#"))
        typ.append("="*n + " " + txt(line[n:].strip()))
    else:
        typ.append(txt(line))
    i += 1
dest.with_suffix(".typ").write_text("\n\n".join(typ)+"\n", encoding="utf-8")
print(dest)
