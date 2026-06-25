# gate.py
# Gating functions or Routings for Mixture of Experts

import torch as tc
import torch.nn as nn
import torch.optim as opt  # 按你的要求保留
from dataclasses import dataclass

from typing import List, Optional, Union, Dict, Any
from .losses import balance_loss, router_z_loss

import sys
import os
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
target_dir = os.path.join(parent_dir, "ADQC")

# from ADQC import ADQC, ADQC_LatentGate, VQC


def softmax_with_temperature(logits: tc.Tensor, temperature: float, dim: int = -1) -> tc.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    orig_dtype = logits.dtype
    x = (logits / tc.tensor(float(temperature), device=logits.device, dtype=tc.float32)).to(tc.float32)
    return nn.functional.softmax(x, dim=dim).to(orig_dtype)


def log_softmax(x: tc.Tensor, temperature: float, dim: int = -1) -> tc.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    x_fp32 = (x / tc.tensor(float(temperature), device=x.device, dtype=tc.float32)).to(tc.float32)
    return nn.functional.log_softmax(x_fp32, dim=dim).to(x.dtype)


def topk_routing_hard(probs: tc.Tensor, k: int):
    """
    probs: (..., E)  float tensor
    return:
      norm: (..., E) 仅 top-k 非零且重新归一化
      mask: (..., E) hard 0/1 mask（不进梯度图）
    """
    probs_fp32 = probs.to(tc.float32)
    E = probs_fp32.shape[-1]
    orig_shape = probs_fp32.shape
    flat = probs_fp32.reshape(-1, E)  # (M, E)

    k_eff = int(min(k, E))
    M = flat.shape[0]

    with tc.no_grad():
        idx_sorted = tc.argsort(flat, dim=-1)
        topk_idx = idx_sorted[:, -k_eff:]  # (M, k)

        mask = tc.zeros((M, E), device=flat.device, dtype=flat.dtype)
        ones = tc.ones((M, k_eff), device=flat.device, dtype=flat.dtype)
        mask = mask.scatter(dim=1, index=topk_idx, src=ones)

        masked = flat * mask
        denom = tc.sum(masked, dim=-1, keepdim=True) + tc.tensor(1e-9, device=flat.device, dtype=flat.dtype)
        norm = masked / denom

    # 注意：norm/mask 都是 no_grad 产物，不在计算图里
    return norm.reshape(orig_shape).to(probs.dtype), mask.reshape(orig_shape).to(probs.dtype)



class _SoftGate(nn.Module):
    def __init__(self, data_dim: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts)
        self.temperature = temperature

    def forward(self, x: tc.Tensor):
        logits = self.fc(x)
        probs = softmax_with_temperature(logits, self.temperature, dim=-1)
        return probs, None, None  # 这里返回 None 为了与 topk gate 保持一致


class _TopKGate(nn.Module):
    def __init__(
        self,
        data_dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        use_ste: bool = True,
        jitter_eps: float = 0.0,
    ):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts)
        self.k = k
        self.temperature = temperature

        # 工程化增强项
        self.use_ste = use_ste          # 让 backward 用 soft 的梯度近似 hard
        self.jitter_eps = jitter_eps    # 训练时给 logits 加噪声（避免塌缩）

    def forward(self, x: tc.Tensor):
        logits = self.fc(x)

        # router jitter（只在训练态启用）
        if self.training and self.jitter_eps > 0.0:
            noise = tc.randn_like(logits.to(tc.float32)) * tc.tensor(float(self.jitter_eps), device=logits.device, dtype=tc.float32)
            logits = logits + noise.to(logits.dtype)

        # soft probs（可导）
        probs_soft = softmax_with_temperature(logits, self.temperature, dim=-1)

        # hard top-k（no_grad）
        probs_hard, mask = topk_routing_hard(probs_soft, self.k)

        if not self.use_ste:
            # 纯 hard（router 梯度会很弱/几乎断）
            return probs_hard, mask

        # STE：forward 用 hard；backward 用 soft 的梯度
        probs = probs_hard - probs_soft.detach() + probs_soft
        return probs, mask

class SoftGate(nn.Module):
    def __init__(self, data_dim: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts, bias=False)
        self.temperature = float(temperature)
        self.last_aux = None
        self.enable_usage_tracking = True
        self.runtime_expert_counts = None
        self.runtime_topk_counts = None
        self.runtime_importance = None
        self.runtime_load = None
        self.runtime_drop_rate = None
        self.runtime_capacity = None
        self.runtime_entropy_soft = None
        self.runtime_entropy_hard = None

    def set_usage_tracking_enabled(self, enabled: bool) -> None:
        self.enable_usage_tracking = bool(enabled)
        if not self.enable_usage_tracking:
            self.reset_runtime_usage_cache()

    def reset_runtime_usage_cache(self) -> None:
        self.runtime_expert_counts = None
        self.runtime_topk_counts = None
        self.runtime_importance = None
        self.runtime_load = None
        self.runtime_drop_rate = None
        self.runtime_capacity = None
        self.runtime_entropy_soft = None
        self.runtime_entropy_hard = None
        self.last_aux = None

    def collect_runtime_usage_tensors(self) -> Dict[str, Any]:
        return {
            "expert_counts": self.runtime_expert_counts,
            "topk_counts": self.runtime_topk_counts,
            "importance": self.runtime_importance,
            "load": self.runtime_load,
            "drop_rate": self.runtime_drop_rate,
            "capacity": self.runtime_capacity,
            "entropy_soft": self.runtime_entropy_soft,
            "entropy_hard": self.runtime_entropy_hard,
            "aux": self.last_aux,
        }

    def materialize_usage_report(self) -> Dict[str, Any]:
        report = self.collect_runtime_usage_tensors()
        materialized: Dict[str, Any] = {}
        for key, value in report.items():
            if isinstance(value, tc.Tensor):
                materialized[key] = value.detach().cpu()
            else:
                materialized[key] = value
        return materialized

    def forward(self, x: tc.Tensor):
        logits = self.fc(x)
        probs = tc.softmax(logits / self.temperature, dim=-1)
        aux = {"l_aux": tc.zeros((), device=probs.device, dtype=probs.dtype)}
        if self.enable_usage_tracking:
            with tc.no_grad():
                expert_counts = probs.detach().to(tc.float32).sum(dim=0)
                self.runtime_expert_counts = expert_counts
                self.runtime_topk_counts = expert_counts
                self.runtime_importance = probs.detach().to(tc.float32).mean(dim=0)
                self.runtime_load = None
                self.runtime_drop_rate = tc.zeros((), device=probs.device, dtype=tc.float32)
                self.runtime_capacity = tc.full((), int(probs.shape[0]), device=probs.device, dtype=tc.int32)
                eps = tc.tensor(1e-10, device=probs.device, dtype=tc.float32)
                p = probs.detach().to(tc.float32).clamp_min(eps)
                H_tok = -(p * p.log()).sum(dim=-1).mean()
                p_bar = p.mean(dim=0).clamp_min(eps)
                H_batch = -(p_bar * p_bar.log()).sum()
                self.runtime_entropy_soft = tc.stack((H_tok, H_batch))
                self.runtime_entropy_hard = tc.stack((H_tok, H_batch))
                self.last_aux = aux["l_aux"].detach()
        return probs, None, aux


class TopKGate(nn.Module):
    def __init__(
        self,
        data_dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        use_ste: bool = True,
        jitter_eps: float = 0.0,
        # DeepSpeed-style knobs
        capacity_factor: float = 1.0,
        min_capacity: int = 8,
        drop_tokens: bool = True,
        drop_policy: str = "probs",   # "probs" | "position"
        # aux knobs
        aux_coeff: float = 1e-2,
        zloss_coeff: float = 0.0,
        # ✅ 预留：未来你想在一个 forward 内返回多种 balance loss
        # 例如 balance_loss_v1/v2/... 由外部传策略列表
        aux_mode: str = "ds",  # "ds" or "none" or "multi"
    ):
        super().__init__()
        self.fc = nn.Linear(data_dim, num_experts, bias=False)

        self.k = int(k)
        self.temperature = float(temperature)
        self.use_ste = bool(use_ste)
        self.jitter_eps = float(jitter_eps)

        self.capacity_factor = float(capacity_factor)
        self.min_capacity = int(min_capacity)
        self.drop_tokens = bool(drop_tokens)
        self.drop_policy = str(drop_policy)

        self.aux_coeff = float(aux_coeff)
        self.zloss_coeff = float(zloss_coeff)
        self.aux_mode = str(aux_mode)

        # stats cache
        self.last_aux = None
        self.last_importance = None
        self.last_load = None
        self.last_drop_rate = None
        self.last_capacity = None

        self.last_H_tok = None
        self.last_H_batch = None
        self.last_H_tok_hard = None
        self.last_H_batch_hard = None
        self.enable_usage_tracking = True
        self.runtime_expert_counts = None
        self.runtime_topk_counts = None
        self.runtime_importance = None
        self.runtime_load = None
        self.runtime_drop_rate = None
        self.runtime_capacity = None
        self.runtime_entropy_soft = None
        self.runtime_entropy_hard = None

    def set_usage_tracking_enabled(self, enabled: bool) -> None:
        self.enable_usage_tracking = bool(enabled)
        if not self.enable_usage_tracking:
            self.reset_runtime_usage_cache()

    def reset_runtime_usage_cache(self) -> None:
        self.last_aux = None
        self.last_importance = None
        self.last_load = None
        self.last_drop_rate = None
        self.last_capacity = None
        self.last_H_tok = None
        self.last_H_batch = None
        self.last_H_tok_hard = None
        self.last_H_batch_hard = None
        self.runtime_expert_counts = None
        self.runtime_topk_counts = None
        self.runtime_importance = None
        self.runtime_load = None
        self.runtime_drop_rate = None
        self.runtime_capacity = None
        self.runtime_entropy_soft = None
        self.runtime_entropy_hard = None

    def collect_runtime_usage_tensors(self) -> Dict[str, Any]:
        return {
            "expert_counts": self.runtime_expert_counts,
            "topk_counts": self.runtime_topk_counts,
            "importance": self.runtime_importance,
            "load": self.runtime_load,
            "drop_rate": self.runtime_drop_rate,
            "capacity": self.runtime_capacity,
            "entropy_soft": self.runtime_entropy_soft,
            "entropy_hard": self.runtime_entropy_hard,
            "aux": self.last_aux,
        }

    def materialize_usage_report(self) -> Dict[str, Any]:
        report = self.collect_runtime_usage_tensors()
        materialized: Dict[str, Any] = {}
        for key, value in report.items():
            if isinstance(value, tc.Tensor):
                materialized[key] = value.detach().cpu()
            else:
                materialized[key] = value
        return materialized

    def _softmax_temp(self, logits_fp32: tc.Tensor) -> tc.Tensor:
        if self.temperature == 1.0:
            return tc.softmax(logits_fp32, dim=-1)
        return tc.softmax(logits_fp32 / self.temperature, dim=-1)

    def _capacity(self, N: int, E: int) -> int:
        cap = int(math.ceil((N / max(1, E)) * (self.capacity_factor * self.k)))
        return max(self.min_capacity, cap)

    @tc.no_grad()
    def _apply_capacity(self, logits_fp32: tc.Tensor, top_idx: tc.Tensor, cap: int):
        N, E = logits_fp32.shape
        mask = tc.zeros((N, E), device=logits_fp32.device, dtype=tc.bool)
        mask.scatter_(1, top_idx, True)

        if (not self.drop_tokens) or cap >= N:
            locations = tc.cumsum(mask, dim=0) - 1
            return mask, locations

        if self.drop_policy == "probs":
            top_gate = logits_fp32.gather(1, top_idx)          # [N,k]
            topk_masked = tc.zeros_like(logits_fp32)           # [N,E]
            topk_masked.scatter_(1, top_idx, top_gate)
            _, cap_indices = tc.topk(topk_masked, k=cap, dim=0, sorted=False)  # [cap,E]
            cap_mask = tc.zeros_like(mask).scatter_(0, cap_indices, True)
            mask = mask & cap_mask
            locations = tc.cumsum(mask, dim=0) - 1
            return mask, locations

        if self.drop_policy == "position":
            locations = tc.cumsum(mask, dim=0) - 1
            mask = mask & (locations < cap)
            return mask, locations

        raise ValueError(f"Invalid drop_policy: {self.drop_policy}")

    def forward(self, x: tc.Tensor):
        logits = self.fc(x)

        if self.training and self.jitter_eps > 0.0:
            logits = logits + tc.randn_like(logits.float()) * self.jitter_eps

        logits_fp32 = logits.float()
        N, E = logits_fp32.shape

        # 1) topk on logits
        _, top_idx = tc.topk(logits_fp32, k=self.k, dim=1)  # [N,k]

        # 2) soft gates
        probs_soft = self._softmax_temp(logits_fp32)        # [N,E]

        # 3) capacity/drop -> mask
        cap = self._capacity(N, E)
        mask, _ = self._apply_capacity(logits_fp32, top_idx, cap)  # bool [N,E]

        # 4) normalize selected experts
        probs_hard = probs_soft * mask.to(probs_soft.dtype)
        denom = probs_hard.sum(dim=-1, keepdim=True).clamp_min(tc.finfo(probs_hard.dtype).eps)
        probs_hard = probs_hard / denom

        # 5) STE
        probs = probs_hard - probs_soft.detach() + probs_soft if self.use_ste else probs_hard

        #=== Entropy ===
        eps = 1e-10
        p = probs_soft.clamp(min=eps)
        H_tok = -(p * p.log()).sum(dim=-1).mean()

        p_bar = p.mean(dim=0).clamp(min=eps)
        H_batch = -(p_bar * p_bar.log()).sum()

        p_hard = probs_hard.clamp(min=eps)
        H_tok_hard = -(p_hard * p_hard.log()).sum(dim=-1).mean()

        p_bar_hard = p_hard.mean(dim=0).clamp(min=eps)
        H_batch_hard = -(p_bar_hard * p_bar_hard.log()).sum()

        # 6) aux losses（✅ 统一用标量）
        me = probs_soft.mean(dim=0)             # importance
        ce = mask.float().mean(dim=0)           # load

        # DeepSpeed-style balance
        l_b = (me * ce).mean() * (E * E / max(1, self.k))
        l_b_aux = l_b * self.aux_coeff

        # z-loss（常见写法是 mean(logsumexp(logits))）
        l_z = tc.logsumexp(logits_fp32, dim=-1).mean()
        l_z_aux = l_z * self.zloss_coeff

        # ✅ 预留：如果你未来要“多方案 balance loss”，放进 l_aux_list
        l_aux = l_b_aux + l_z_aux
        aux: Dict[str, Any] = {
            "l_aux": l_aux,
            "l_b_aux": l_b_aux,
            "l_z_aux": l_z_aux,
            "capacity": tc.full((), int(cap), device=probs.device, dtype=tc.int32),
            "drop_rate": (1.0 - (mask.float().sum(dim=-1).mean() / float(self.k))),
            "importance": me,
            "load": ce,
            # "l_aux_list": [ ... ]   # 未来 multi 方案直接加这里
            # "viz": {...}            # 未来可视化数据也放这里
            "H_tok": H_tok,
            "H_batch": H_batch,
            "H_tok_hard": H_tok_hard,
            "H_batch_hard": H_batch_hard,
        }

        if self.enable_usage_tracking:
            with tc.no_grad():
                topk_counts = mask.detach().to(tc.float32).sum(dim=0)
                self.runtime_expert_counts = topk_counts
                self.runtime_topk_counts = topk_counts
                self.runtime_importance = me.detach()
                self.runtime_load = ce.detach()
                self.runtime_drop_rate = aux["drop_rate"].detach()
                self.runtime_capacity = aux["capacity"].detach()
                self.runtime_entropy_soft = tc.stack((H_tok.detach(), H_batch.detach()))
                self.runtime_entropy_hard = tc.stack((H_tok_hard.detach(), H_batch_hard.detach()))
                self.last_aux = l_aux.detach()
                self.last_importance = me.detach()
                self.last_load = ce.detach()
                self.last_drop_rate = aux["drop_rate"].detach()
                self.last_capacity = aux["capacity"].detach()
                self.last_H_tok = H_tok.detach()
                self.last_H_batch = H_batch.detach()
                self.last_H_tok_hard = H_tok_hard.detach()
                self.last_H_batch_hard = H_batch_hard.detach()

        return probs.to(logits.dtype), mask, aux


@dataclass
class GateConfig:
    gate_type: str = "topk"
    data_dim: int = 1024
    num_experts: int = 16
    k: int = 2
    temperature: float = 1.0
    use_ste: bool = True
    jitter_eps: float = 0.0

    capacity_factor: float = 1.0
    min_capacity: int = 8
    drop_tokens: bool = True
    drop_policy: str = "probs"

    aux_coeff: float = 1e-2
    zloss_coeff: float = 0.0
    aux_mode: str = "ds"


def gate_factory_config(config: GateConfig) -> nn.Module:
    gt = config.gate_type.lower()
    if gt == "topk":
        return TopKGate(
            config.data_dim, config.num_experts,
            k=config.k,
            temperature=config.temperature,
            use_ste=config.use_ste,
            jitter_eps=config.jitter_eps,
            capacity_factor=config.capacity_factor,
            min_capacity=config.min_capacity,
            drop_tokens=config.drop_tokens,
            drop_policy=config.drop_policy,
            aux_coeff=config.aux_coeff,
            zloss_coeff=config.zloss_coeff,
            aux_mode=config.aux_mode,
        )
    if gt == "softmax":
        return SoftGate(config.data_dim, config.num_experts, temperature=config.temperature)
    raise ValueError(f"Unsupported gate type: {config.gate_type}")

def gate_factory(gate_type: str, data_dim: int, num_experts: int, **kwargs) -> nn.Module:
    gate_type = gate_type.lower()
    if gate_type == "topk":
        return TopKGate(
            data_dim,
            num_experts,
            k=kwargs.get("k", 2),
            temperature=kwargs.get("temperature", 1.0),
            use_ste = kwargs.get("use_ste", True),
            jitter_eps=kwargs.get("jitter_eps", 0.0),
            capacity_factor = 1.0,
            min_capacity = 8,
            drio_tokens = True,
            drop_policy = "probs",
            aux_coeff = 1e-2,
        )
    if gate_type == "softmax":
        return SoftGate(
            data_dim,
            num_experts,
            temperature=kwargs.get("temperature", 1.0),
        )
    if gate_type == "quantum":
        return Quantum_layer_Gate(data_dim, num_experts, 2, "tensor", 1, True)
    raise ValueError(f"Unsupported gate type: {gate_type}")


def _gate_factory_config(config):
    gate_type = config.gate_type.lower()
    if gate_type == "topk":
        return TopKGate(
            config.data_dim,
            config.num_experts,
            k=config.k,
            temperature=config.temperature,
            use_ste = config.use_ste,
            jitter_eps=config.jitter_eps
        )
    if gate_type == "softmax":
        return SoftGate(
            config.data_dim,
            config.num_experts,
            temperature=config.temperature,
        )
    if gate_type == "quantum":
        return Quantum_layer_Gate(config.data_dim, config.num_experts,2, "tensor", 1, True)
    raise ValueError(f"Unsupported gate type: {gate_type}")



def normalize_state_general(psi: tc.Tensor, num_qubits: int) -> tc.Tensor:
    """
    psi: [..., 2,2,...,2]  (最后 num_qubits 个维度是 Hilbert 空间)
    在 Hilbert 空间上做归一化：对每个 token 的波函数单独归一化
    """
    full_shape = psi.shape
    hilbert_shape = full_shape[-num_qubits:]
    leading_shape = full_shape[:-num_qubits]

    T = 1
    for d in leading_shape:
        T *= int(d)
    D = 1
    for d in hilbert_shape:
        D *= int(d)

    psi_flat = psi.reshape(T, D)
    norm = tc.linalg.norm(psi_flat, dim=-1, keepdim=True) + tc.tensor(1e-8, device=psi.device, dtype=psi_flat.dtype)
    psi_flat = psi_flat / norm
    return psi_flat.reshape(full_shape)


def measure_sigma_z_probs_general(
    psi: tc.Tensor,
    num_qubits: int,
    eps: float = 1e-8,
) -> tc.Tensor:
    """
    psi: [..., 2,2,...,2]，最后 num_qubits 个 2 是 Hilbert 维度
    返回: [..., num_qubits]
    """
    full_shape = psi.shape
    hilbert_shape = full_shape[-num_qubits:]
    leading_shape = full_shape[:-num_qubits]

    if not all(int(d) == 2 for d in hilbert_shape):
        raise AssertionError("目前只实现 dims=2 的 qubit 情况")

    psi = normalize_state_general(psi, num_qubits)

    T = 1
    for d in leading_shape:
        T *= int(d)

    psi_flat = psi.reshape((T,) + tuple(int(d) for d in hilbert_shape))

    prob_full = tc.abs(psi_flat) ** 2

    p1_list: List[tc.Tensor] = []
    for i in range(num_qubits):
        idx = [slice(None)] * (num_qubits + 1)
        idx[i + 1] = 1
        slice_i = prob_full[tuple(idx)]
        axes_to_sum = tuple(range(1, slice_i.ndim))
        p1_i = tc.sum(slice_i, dim=axes_to_sum)  # (T,)
        p1_list.append(p1_i)

    p1 = tc.stack(p1_list, dim=-1)  # (T, num_qubits)
    Z = tc.sum(p1, dim=-1, keepdim=True) + tc.tensor(float(eps), device=p1.device, dtype=p1.dtype)
    probs_flat = p1 / Z

    probs = probs_flat.reshape(tuple(int(d) for d in leading_shape) + (num_qubits,))
    return probs


def pauli_z() -> tc.Tensor:
    return tc.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=tc.float32)


def normalize(x: Union[tc.Tensor, List[tc.Tensor]], form: str):
    if form == "tensor":
        return x / (tc.linalg.norm(x) + tc.tensor(1e-8, device=x.device, dtype=x.dtype))
    return x


def shape_init(n_qubits: int, dims: Optional[int] = None):
    if dims is not None:
        return [int(dims) for _ in range(int(n_qubits))]
    return [2 for _ in range(int(n_qubits))]


def mps_init(
    n_qubits: int,
    chi: int = 10,
    init_way: str = "standard",
    d_in: Optional[int] = None,
    gamma: Optional[int] = None,
):
    # 注意：这部分在你项目中目前主要是“提供初始化结构”，不涉及 device 搬运
    if init_way == "standard":
        return [tc.randn(1, 2, chi)] + [tc.randn(chi, 2, chi) for _ in range(n_qubits - 2)] + [tc.randn(chi, 2, 1)]
    elif init_way == "kaiming":
        if d_in is None:
            raise ValueError("d_in must be provided for kaiming init")
        std = (2.0 / float(d_in)) ** 0.5
        return [tc.randn(1, 2, chi) * std] + [tc.randn(chi, 2, chi) * std for _ in range(n_qubits - 2)] + [tc.randn(chi, 2, 1) * std]
    elif init_way == "gamma":
        if d_in is None or gamma is None:
            raise ValueError("d_in and gamma must be provided for gamma init")
        std = (1.0 / float(d_in)) ** float(gamma)
        return [tc.randn(1, 2, chi) * std] + [tc.randn(chi, 2, chi) * std for _ in range(n_qubits - 2)] + [tc.randn(chi, 2, 1) * std]
    else:
        raise ValueError(f"Unknown init_way: {init_way}")


def mpo_init(
    n_qubits: int,
    chi: int = 10,
    init_way: str = "standard",
    din: int = 5,
    gamma: Optional[int] = None,
):
    if init_way == "strandard":
        # 保持你原拼写分支（虽然看起来是 typo），以保证“路径等价”
        return [tc.randn(1, 2, chi, din)] + [tc.randn(chi, 2, chi, din) for _ in range(n_qubits - 2)] + [tc.randn(chi, 2, 1, din)]
    elif init_way == "kaiming":
        d_in = float(din ** n_qubits)
        std = (2.0 / d_in) ** 0.5
        return [tc.randn(1, 2, chi, din) * std] + [tc.randn(chi, 2, chi, din) * std for _ in range(n_qubits - 2)] + [tc.randn(chi, 2, 1, din) * std]
    elif init_way == "gamma":
        if gamma is None:
            raise ValueError("gamma must be provided for gamma init")
        d_in = float(din ** n_qubits)
        std = (1.0 / d_in) ** float(gamma)
        return [tc.randn(1, 2, chi, din) * std] + [tc.randn(chi, 2, chi, din) * std for _ in range(n_qubits - 2)] + [tc.randn(chi, 2, 1, din) * std]
    else:
        raise ValueError(f"Unknown init_way: {init_way}")


def get_quantum_layer(
    din: int,
    num_exp: int,
    method: str,
    init_way: str = "standard",
    dims: Optional[int] = None,
    gamma: Optional[int] = None,
    chi: Optional[int] = None,
):
    if method == "tensor":
        if dims is not None:
            indexs = shape_init(dims * num_exp)
        else:
            indexs = shape_init(2 * num_exp)

        if init_way == "standard":
            return tc.randn(*indexs)
        elif init_way == "kaiming":
            std = (2.0 / float(din)) ** 0.5
            return tc.randn(*indexs) * std
        elif init_way == "gamma":
            if gamma is None:
                raise ValueError("gamma must be provided for gamma init")
            std = (1.0 / float(din)) ** float(gamma)
            return tc.randn(*indexs) * std
        else:
            raise ValueError(f"Unknown init_way: {init_way}")

    if method == "mpo":
        if chi is None:
            raise ValueError("chi must be provided for mpo init")
        return mpo_init(num_exp, chi, init_way, din, gamma)

    raise ValueError(f"Unknown quantum layer method: {method}")


def evolution(tensor: tc.Tensor, evo_ope: tc.Tensor) -> tc.Tensor:
    # 你原代码未实现；保持占位，不新增逻辑
    if isinstance(tensor, list):
        _ = tensor
        _ = evo_ope
    return tensor


def measurement(tensor: tc.Tensor, ope: tc.Tensor):
    # 你原代码未实现；保持占位，不新增逻辑
    shape_t = tensor.shape
    shape_o = ope.shape
    if len(shape_t) == len(shape_o) and shape_t[0] == shape_o[0]:
        return None
    return None


class Quantum_layer_Gate(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_experts: int,
        dims: Optional[int] = None,
        form: str = "tensor",
        k: int = 1,
        sample: bool = True,
    ):
        super().__init__()
        self.form = form
        self.dims = dims
        self.num_experts = num_experts
        self.k = k
        self.sample = sample

        if dims is not None:
            self.project = nn.Linear(d_in, dims ** num_experts)
            self.gate = get_quantum_layer(dims ** num_experts, num_experts, self.form, "standard", dims, None, None)
        else:
            self.project = nn.Linear(d_in, 2 ** num_experts)
            self.gate = get_quantum_layer(2 ** num_experts, num_experts, self.form, "standard", None, None, None)

    def forward(self, x: tc.Tensor):
        psi = self.project(x)  # [B, d_in] -> [B, 2**E]（或 dims**E）

        if self.form == "tensor":
            # psi @ gate.reshape(...)
            gate_mat = self.gate.reshape(psi.shape[-1], -1).to(dtype=psi.dtype, device=psi.device)
            psi = psi @ gate_mat

            if self.dims != 2:
                raise NotImplementedError("目前只实现 dims=2 的 Pauli-Z 测量")

            full_shape = psi.shape
            leading_shape = full_shape[:-1]  # e.g. (B,) 或 (B,S)（取决于你的输入）
            psi_qubits = psi.reshape(leading_shape + (2,) * self.num_experts)

            probs = measure_sigma_z_probs_general(
                psi_qubits,
                num_qubits=self.num_experts,
            )

            if self.sample is False:
                return probs, None

            # 下面保持你原逻辑：假设 probs 是 (B, E) 的二维情况
            # 若你输入是 (B,S,...) 使 probs 变成 (B,S,E)，这里也会与原 MLX 一样不匹配（保持功能路径等价）
            B, E = probs.shape
            k = min(self.k, E)

            U = tc.rand(probs.shape, device=probs.device, dtype=tc.float32)
            g = -tc.log(-tc.log(U + 1e-8) + 1e-8)
            logits = tc.log(probs.to(tc.float32) + 1e-8) + g

            idx_sorted = tc.argsort(logits, dim=-1)
            topk_idx = idx_sorted[:, -k:].detach()

            mask = tc.zeros((B, E), device=probs.device, dtype=probs.dtype)
            ones = tc.ones((B, k), device=probs.device, dtype=probs.dtype)
            mask = mask.scatter(dim=1, index=topk_idx, src=ones).detach()

            masked = probs * mask
            denom = tc.sum(masked, dim=-1, keepdim=True) + tc.tensor(1e-9, device=probs.device, dtype=probs.dtype)
            norm = masked / denom

            return norm.to(probs.dtype), mask.to(probs.dtype)

        # 你原实现只覆盖 form == "tensor"；其它保持不扩展
        return None
