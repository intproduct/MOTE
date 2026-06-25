import os
# os.environ['HTTPS_PROXY'] = 'http://10.29.1.201:8888'
# os.environ['HTTP_PROXY'] = 'http://10.29.1.201:8888'
os.environ['HTTPS_PROXY'] = 'http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128'
os.environ['HTTP_PROXY'] = 'http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128'
os.environ['HTTP_PROXY'] = '127.0.0.1,10.254.31.0/24,10.254.128.106'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 可选，让编号更直观
os.environ["CUDA_VISIBLE_DEVICES"] = "6"        # 改成你想用的服务器物理卡号，例如只用第3号卡


# qwen3_chat_sft_ultra_alpaca_openhermes.py
from typing import List, Dict, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from dataclasses import dataclass
import numpy as np

import re, math, random, torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
)

from pathlib import Path
import importlib.util, sys

from torch.utils.data import DataLoader
from contextlib import nullcontext

import os, shutil, json
import hashlib


MMLU_BASELINE = float(os.environ.get("MMLU_BASELINE", "0.80"))
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "2000"))
WARM_START_STEPS = int(os.environ.get("WARM_START_STEPS", "1000"))
WARM_START_LR    = float(os.environ.get("WARM_START_LR", "1e-2"))


# 统一成一个变量
MODEL_DIR = Path("./model/llama/Llama-3.1-8B-Instruct-expanded-model").resolve()
# MODEL_DIR = Path("llama3_tnn_JOINT_L31_30_29_etc/checkpoint-4000_off=3").resolve()
REPO = str(MODEL_DIR)  # 既可给 HF Transformers 也可拼接本地文件

file = MODEL_DIR / "modeling_llama_tnn.py"
spec = importlib.util.spec_from_file_location("llama_modeling_tnn", file)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

ADTN_Ensemble_Projector = mod.ADTN_Ensemble_Projector

LOCAL_DATA_DIR = "./local_datasets"

# ------ 你的 config 目录 ------
# MODEL_DIR_ = Path("../model/llama/test/config.json").resolve()
# REPO_ = str(MODEL_DIR_)  # 既可给 HF Transformers 也可拼接本地文件


class ResampleSubset(TrainerCallback):
    def __init__(self, full_ds, k, replace=True):
        self.full_ds = full_ds
        self.k = int(k)
        self.replace = bool(replace)
        self.trainer = None  # ← 保存一个句柄，部分版本不会在 kwargs 里传 trainer

    def _resample(self, trainer, state):
        if trainer is None:
            return
        n = len(self.full_ds)
        if self.replace:
            idx = np.random.randint(0, n, size=self.k)  # 有放回
        else:
            k_eff = min(self.k, n)  # 无放回时截断到 n
            idx = np.random.choice(n, k_eff, replace=False)
        trainer.train_dataset = self.full_ds.select(idx.tolist())
        # 日志里打一下当前轮次的信息（state.epoch 可能是浮点）
        cur_ep = int((state.epoch or 0.0))
        print(f"[ResampleSubset] epoch={cur_ep} resampled {len(idx)}/{n} (replace={self.replace})")

    def on_train_begin(self, args, state, control, **kwargs):
        # 某些版本会在 kwargs 里给 trainer，某些不会；两种都兼容
        self.trainer = kwargs.get("trainer", self.trainer)
        self._resample(self.trainer, state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", self.trainer)
        self._resample(self.trainer, state)


# === Fix A: 保存前消毒 config（把 set 递归转成排序 list） ===
from transformers import TrainerCallback, TrainerState, TrainerControl

def _convert_sets(o):
    if isinstance(o, dict):
        return {k: _convert_sets(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_convert_sets(v) for v in o]
    if isinstance(o, set):
        return sorted(list(o))
    return o

def sanitize_config_inplace(cfg):
    # 把 cfg.to_dict() 里的 set 全部变成 list，再尽量写回 cfg 的属性
    try:
        clean = _convert_sets(cfg.to_dict())
    except Exception:
        return
    for k, v in clean.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            pass

class SanitizeConfigBeforeSave(TrainerCallback):
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model", None)
        if model is not None and hasattr(model, "config"):
            sanitize_config_inplace(model.config)
            print("✅ sanitized config (sets → lists) before save")
        return control
# === /Fix A ===


# =====================  新增：仅训练“本层新增 ADTN 参数”的开关  =====================
def set_requires_grad_only(model: nn.Module, params_to_train: list):
    """将全模型中参数置为 frozen，仅保留 params_to_train 为 True。"""
    idset = {id(p) for p in params_to_train}
    for _, p in model.named_parameters():
        p.requires_grad = (id(p) in idset)

# =====================  新增：按层构造 config.gates  =====================
def build_cfg_for_single_layer(cfg, layer_idx: int):
    """在不破坏全局超参（如 adtn_gate_offset）的前提下，临时注入当前层的 7 个 gate。"""
    c = copy.deepcopy(cfg)
    c.tnn_attn_gates = [f"{layer_idx}:q", f"{layer_idx}:k", f"{layer_idx}:v", f"{layer_idx}:o"]
    c.tnn_mlp_gates  = [f"{layer_idx}:gate", f"{layer_idx}:up", f"{layer_idx}:down"]
    return c

# =====================  新增：MMLU 5-shot 回调（达标即停训） =====================
from checkpoint_eval import evaluate_model_5shot
from transformers import TrainerCallback, TrainerState, TrainerControl

class MMLUEvalUntilBaseline(TrainerCallback):
    def __init__(self, tok, baseline: float):
        self.tok = tok
        self.baseline = float(baseline)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        was_training = model.training
        model.eval()
        try:
            score = float(evaluate_model_5shot(model, self.tok))
        except Exception as e:
            print(f"[MMLU-5shot] 评测异常: {e}")
            score = -1.0
        if was_training:
            model.train()
        print(f"[MMLU-5shot] at save step={state.global_step} score={score:.4f} baseline={self.baseline:.4f}")
        if score >= self.baseline:
            control.should_training_stop = True
            print("[MMLU-5shot] ✅ 达到基线，结束本层训练")
        return control


# ==================== 通用工具 ====================
from torch.utils.data import RandomSampler


import re
from typing import Optional, Tuple, List, Dict

_ONE_OFFSET_RE = re.compile(
    r"^\s*(\d+)\s*:\s*(q|k|v|o|gate|up|down)\s*(?::\s*(-?\d+)\s*)?$",
    re.IGNORECASE
)

def _parse_gate_spec_oneoffset(spec: str) -> Tuple[int, str, Optional[int]]:
    m = _ONE_OFFSET_RE.match(spec)
    if not m:
        raise ValueError(f"无效 gate 规格：{spec!r}，期望 '层号:门名[:offset]'")
    L = int(m.group(1))
    kind = m.group(2).lower()
    off = m.group(3)
    return L, kind, (None if off is None else int(off))

def _gates_from_config(cfg) -> List[Tuple[int, str, Optional[int]]]:
    """从 cfg.tnn_attn_gates / cfg.tnn_mlp_gates 取出 [(layer, kind, offset_or_None)]."""
    out: List[Tuple[int, str, Optional[int]]] = []
    for name in ("tnn_attn_gates", "tnn_mlp_gates"):
        lst = getattr(cfg, name, None)
        if not lst:
            continue
        for spec in lst:
            L, k, off = _parse_gate_spec_oneoffset(spec)
            if off is not None and off < 0:
                raise ValueError(f"{spec!r} 中 offset 不能为负数")
            out.append((L, k, off))
    # 去重（以最后一次为准）
    seen: Dict[Tuple[int,str], Optional[int]] = {}
    for L, k, off in out:
        seen[(L, k)] = off
    return [(L, k, seen[(L, k)]) for (L, k) in sorted(seen.keys())]


def build_cfg_for_single_layer_with_offset(base_cfg, layer_idx: int, offset: int):
    """
    基于 base_cfg 创建一个新的 config 对象。
    专门为 layer_idx 生成 tnn_attn_gates 和 tnn_mlp_gates 列表。
    关键点：将 offset 显式写入字符串，例如 "31:q:2"。
    """
    # 浅拷贝 base_cfg，避免修改原对象
    new_cfg = copy.copy(base_cfg)

    # 强制覆盖为当前层的配置
    # 格式："{layer}:{kind}:{offset}"
    # 这种格式会被 _parse_gate_spec_oneoffset 正确解析出 offset

    # Attention 部分 (q, k, v, o)
    new_cfg.tnn_attn_gates = [
        f"{layer_idx}:{kind}:{offset}" for kind in ["q", "k", "v", "o"]
    ]

    # MLP 部分 (gate, up, down)
    new_cfg.tnn_mlp_gates = [
        f"{layer_idx}:{kind}:{offset}" for kind in ["gate", "up", "down"]
    ]

    # 清理可能存在的全局 map，确保使用列表中的定义
    if hasattr(new_cfg, "_adtn_gate_offset_map"):
        delattr(new_cfg, "_adtn_gate_offset_map")

    return new_cfg


def build_cfg_for_single_layer_refined(base_cfg, layer_idx: int, strategy_map: dict):
    """
    支持 "skip" 标记的配置构建函数。
    """
    new_cfg = copy.copy(base_cfg)

    # 获取默认值
    layer_default = strategy_map.get(layer_idx, 3)

    def get_off(kind):
        # 1. 查特异性配置
        if (layer_idx, kind) in strategy_map:
            return strategy_map[(layer_idx, kind)]
        # 2. 查字符串兼容配置
        if f"{layer_idx}:{kind}" in strategy_map:
            return strategy_map[f"{layer_idx}:{kind}"]
        # 3. 回退默认
        return layer_default

    # 生成配置列表（增加 skip 判断）
    attn_specs = []
    for kind in ["q", "k", "v", "o"]:
        off = get_off(kind)
        # 【关键修改】如果设置为 "skip"，则不添加到列表
        if off == "skip":
            continue
        attn_specs.append(f"{layer_idx}:{kind}:{off}")
    new_cfg.tnn_attn_gates = attn_specs

    mlp_specs = []
    for kind in ["gate", "up", "down"]:
        off = get_off(kind)
        # 【关键修改】如果设置为 "skip"，则不添加到列表
        if off == "skip":
            continue
        mlp_specs.append(f"{layer_idx}:{kind}:{off}")
    new_cfg.tnn_mlp_gates = mlp_specs

    if hasattr(new_cfg, "_adtn_gate_offset_map"):
        delattr(new_cfg, "_adtn_gate_offset_map")

    return new_cfg

# 放在 import 之后、回调类之前
def _get_mod_and_attr(layer, kind: str):
    """按 kind 返回 (module_ref, attr_name)：
       q/k/v/o -> layer.self_attn.{q,k,v,o}_proj
       gate/up/down -> layer.mlp.{gate,up,down}_proj
    """
    if kind in ("q", "k", "v", "o"):
        return layer.self_attn, f"{kind}_proj"
    elif kind in ("gate", "up", "down"):
        name = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[kind]
        return layer.mlp, name
    else:
        raise ValueError(f"Unknown kind: {kind}")


def make_attn_projector(
    kind: str,              # "q" | "k" | "v" | "o" | "gate" | "up" | "down"
    config,
    *,
    head_dim: int,
    use_tnn: bool,
    d_dim: int = 2,
    layer_idx: Optional[int] = None,
    gate_offset: Optional[int] = None,   # ← 新增：若传 None，则回落到 cfg 或默认=1
) -> nn.Module:
    # ---- 输入维度 ----
    if kind in ("q", "k", "v", "gate", "up"):
        in_dim = config.hidden_size
    elif kind == "o":
        in_dim = config.num_attention_heads * head_dim
    elif kind == "down":
        in_dim = config.intermediate_size
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # ---- 输出维度 ----
    if kind == "q":
        out_dim = config.num_attention_heads * head_dim
    elif kind in ("k", "v"):
        out_dim = config.num_key_value_heads * head_dim
    elif kind == "o":
        out_dim = config.hidden_size
    elif kind in ("gate", "up"):     # MLP 上行
        out_dim = config.intermediate_size
    elif kind == "down":             # MLP 下行
        out_dim = config.hidden_size

    if not use_tnn:
        return nn.Linear(in_dim, out_dim, bias=getattr(config, "attention_bias", False))

    # ---- ADTN 维度（用单一 gate_offset） ----
    assert (d_dim ** int(round(math.log(in_dim, d_dim)))) == in_dim,  f"in_dim={in_dim} 必须是 {d_dim} 的幂"
    assert (d_dim ** int(round(math.log(out_dim, d_dim)))) == out_dim, f"out_dim={out_dim} 必须是 {d_dim} 的幂"

    q_in  = int(round(math.log(in_dim,  d_dim)))
    q_out = int(round(math.log(out_dim, d_dim)))

    # 解析 gate_offset：优先入参，其次全局 cfg.adtn_gate_offset，最后默认 1
    if gate_offset is None:
        gate_offset = int(getattr(config, "adtn_gate_offset", 3))
    if gate_offset < 0:
        raise ValueError("gate_offset 不能为负数")

    k_in  = max(0, q_in  - gate_offset)
    k_out = max(0, q_out - gate_offset)

    return ADTN_Ensemble_Projector(
        q_number=q_in,
        num_input_gate_dims=k_in,
        num_output_gate_dims=k_out,
        d=d_dim,
        gate_offset=gate_offset,   # ← 透传给 ADTN（forward 里 reshape 用到）
        cfg=config,
        layer_idx=layer_idx,
        kind=kind,
    )


def _unwrap_model(m):
    while hasattr(m, "module"):
        m = m.module
    return m


# ==================== 核心：参数级“全矩阵”逼近初始化 ====================
def _build_full_identity_basis(in_dim: int, q_number: int, d_dim: int, device, dtype):
    """
    构造整张单位基：I_{in} → 形状 [in_dim] + [d_dim]*q_number 以契合 ADTN 输入
    显存充足时优先使用，全矩阵一次前向可得到完整等效权重，初始化质量最佳。
    """
    I = torch.eye(in_dim, device=device, dtype=dtype)  # [in_dim, in_dim]
    basis = I.view(in_dim, *([d_dim] * q_number))  # [in_dim, d, d, ..., d]
    return basis


@torch.no_grad()
def _weight_target_T(lin: nn.Linear, device, dtype):
    """把线性层权重转置成 [in_dim, out_dim] 并拷到 device/dtype"""
    return lin.weight.detach().T.to(device=device, dtype=dtype)  # [in_dim, out_dim]


def fit_adtn_to_weight_fullmatrix(
        adtn_layer: nn.Module,
        lin: nn.Linear,
        *,
        q_number: int,
        d_dim: int,
        device,
        fit_dtype=torch.float32,  # 拟合阶段用 FP32 更稳
        steps: int = 800,  # 充裕显存+高质量初始化可以加大（如 1200~2000）
        lr: float = 1e-2,
        clip_norm: float = 1.0,
        verbose: bool = True,
):
    """
    用整张单位基做前向，直接逼近 Linear 权重矩阵（对齐为 W^T）。
    每步：W_adtn = ADTN(I)  与  W_target = W^T 做 MSE/Fro 误差，AdamW 优化 ADTN 参数。
    """
    # 解析维度
    W_target = _weight_target_T(lin, device, fit_dtype)  # [in_dim, out_dim]
    in_dim, out_dim = W_target.shape
    assert (d_dim ** q_number) == in_dim, f"in_dim {in_dim} != d_dim^{q_number}"

    # 构造整张单位基
    basis = _build_full_identity_basis(in_dim, q_number, d_dim, device, fit_dtype)  # [in_dim, d, d, ..., d]

    # 准备 ADTN（FP32）
    adtn_layer.to(device=device, dtype=fit_dtype)
    adtn_layer.train()
    for p in adtn_layer.parameters():
        p.requires_grad_(True)

    opt = optim.AdamW(adtn_layer.parameters(), lr=lr)

    for it in range(steps):
        # 前向得到整张等效权重
        W_adtn = adtn_layer(basis).squeeze(1)  # 期望 [in_dim, out_dim]
        if W_adtn.shape != W_target.shape:
            raise RuntimeError(f"ADTN output shape {tuple(W_adtn.shape)} != target {tuple(W_target.shape)}")

        # MSE/Fro 误差（MSE 等价于 Fro^2 / (in*out)，稳定）
        loss = F.mse_loss(W_adtn, W_target)
        # loss = torch.norm(W_adtn-W_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # if clip_norm and clip_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(adtn_layer.parameters(), clip_norm)
        opt.step()

        if verbose and (it % max(1, steps // 5) == 0 or it == steps - 1):
            print(f"[fit_adtn_to_weight_fullmatrix] step {it + 1}/{steps}, mse={loss.item():.6f}")

    return float(loss.detach().cpu())


# ==================== 替换：带“权重逼近暖启动” ====================
def replace_linear_with_tnn_from_config(
    model,
    cfg,
    *,
    head_dim: int,
    device,
    run_dtype,
    d_dim: int = 2,
    warm_start_weight: bool = True,
    FIT_STEPS: int = 1000,
    FIT_LR: float = 1e-2,
    verbose_fit: bool = False,
):
    """
    读取 cfg 中的 tnn_attn_gates / tnn_mlp_gates（元素形如 'L:kind[:offset]'），
    逐项将对应 Linear → ADTN，offset 缺省时用 cfg.adtn_gate_offset 或 1。
    """
    to_replace = _gates_from_config(cfg)  # [(layer, kind, offset_or_None)]
    if not to_replace:
        print("[replace_from_config] 未在 config 中找到待替换门；跳过。")
        return []

    new_params = []
    for (li, kind, off) in to_replace:
        layer = model.model.layers[li]
        mod, attr = _get_mod_and_attr(layer, kind)
        cur = getattr(mod, attr)

        # 构造 ADTN（按 gate_offset 计算 k_in/k_out，并透传 gate_offset）
        adtn = make_attn_projector(
            kind, cfg, head_dim=head_dim, use_tnn=True, d_dim=d_dim,
            layer_idx=li, gate_offset=off
        )
        adtn = adtn.to(device=device, dtype=torch.float32)

        # 暖启动（整矩阵逼近）
        if warm_start_weight:
            # 推导 in_dim 仅用于 q_number 校验
            if kind in ("q", "k", "v", "gate", "up"):
                in_dim = cfg.hidden_size
            elif kind == "o":
                in_dim = cfg.num_attention_heads * head_dim
            elif kind == "down":
                in_dim = cfg.intermediate_size
            q_number = int(round(math.log(in_dim, d_dim)))
            assert (d_dim ** q_number) == in_dim
            final_mse = fit_adtn_to_weight_fullmatrix(
                adtn, cur,
                q_number=q_number, d_dim=d_dim, device=device, fit_dtype=torch.float32,
                steps=FIT_STEPS, lr=FIT_LR, verbose=verbose_fit
            )
            if verbose_fit:
                print(f"[warm-start] layer={li} kind={kind} mse={final_mse:.6f} (offset={off if off is not None else getattr(cfg,'adtn_gate_offset',3)})")

        # 替换
        adtn = adtn.to(device=device, dtype=run_dtype)
        setattr(mod, attr, adtn)
        for p in adtn.parameters():
            new_params.append(p)

    print(f"[replace_from_config] 已按 config 替换 {len(to_replace)} 个 gates。")
    return new_params


def set_requires_grad_tnn_only(model):
    """只训练 ADTN 参数；若当前还没有 ADTN，则不改 requires_grad（避免全冻结）"""
    tnn_ids = set()
    for m in model.modules():
        if m.__class__.__name__.lower().startswith("adtn"):
            for p in m.parameters():
                tnn_ids.add(id(p))
    if not tnn_ids:
        print("[warn] no TNN modules yet; skip freezing others for now.")
        return
    for _, p in model.named_parameters():
        p.requires_grad = (id(p) in tnn_ids)


# ====== 基础规则 ======
SYSTEM_DEFAULT = "You are a helpful AI assistant."

# ==============================================================================
# 1. 数据规范化函数 (针对 Llama-3 生态的新数据集)
# ==============================================================================

# 通用清洗正则
_LEAKY_USER = re.compile(r"(^|\n)\s*(User|Human)\s*:\s*", re.IGNORECASE)


def _speaker_strip(s: str) -> str:
    """去除多余空行和人工角色标记"""
    if not isinstance(s, str): return ""
    t = re.sub(r"^\s*(User|Assistant|Human)\s*:\s*", "", s.strip(), flags=re.IGNORECASE)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def _clean_dialog(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """深度清洗对话内容"""
    cleaned = []
    for m in messages:
        txt = _speaker_strip(m["content"])
        if m["role"] == "assistant":
            leak = _LEAKY_USER.search(txt)
            if leak:
                cut = txt[:leak.start()].strip()
                if len(cut) < 5: return []  # 内容过短则丢弃
                txt = cut
        cleaned.append({"role": m["role"], "content": txt})
    return cleaned


def _norm_magpie(ex: Dict) -> Dict:
    """
    处理 Magpie-Pro 数据集。
    【关键修复】增强鲁棒性：兼容 'from/value' 格式，防止 KeyError。
    """
    raw_convs = ex.get("conversations", [])
    if not raw_convs: return {"messages": []}

    normalized_msgs = []

    # 强制遍历清洗
    for msg in raw_convs:
        # 1. 安全获取字段 (兼容 role/from 和 content/value)
        role = msg.get("role", msg.get("from", "")).lower()
        content = msg.get("content", msg.get("value", ""))

        # 2. 映射角色名
        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "assistant", "model"]:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            # 遇到未知角色或空内容，跳过
            continue

        normalized_msgs.append({"role": role, "content": content})

    if not normalized_msgs:
        return {"messages": []}

    # 3. 检查并插入 System Prompt
    if normalized_msgs[0]["role"] != "system":
        normalized_msgs.insert(0, {"role": "system", "content": SYSTEM_DEFAULT})

    return {"messages": normalized_msgs}


def _norm_openmath(ex: Dict) -> Dict:
    """
    处理 OpenMathInstruct-2 数据集。
    格式: 'problem' (str), 'generated_solution' (str)
    """
    problem = ex.get("problem", "")
    solution = ex.get("generated_solution", "")

    if not problem or not solution:
        return {"messages": []}

    # 构造成标准对话
    return {"messages": [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution}
    ]}


def _norm_openhermes(ex: Dict) -> Dict:
    """
    处理 OpenHermes-2.5 (ShareGPT 格式: conversations -> from/value)
    """
    raw_convs = ex.get("conversations", [])
    if not raw_convs: return {"messages": []}

    normalized_msgs = []
    for msg in raw_convs:
        # 映射 ShareGPT 的 from -> role, value -> content
        role = msg.get("from", "").lower()
        content = msg.get("value", "")

        # 角色标准化
        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "model", "assistant"]:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue  # 跳过未知角色

        normalized_msgs.append({"role": role, "content": content})

    if not normalized_msgs: return {"messages": []}

    # 检查并插入 System Prompt (OpenHermes 部分数据缺失 System)
    if normalized_msgs[0]["role"] != "system":
        normalized_msgs.insert(0, {"role": "system", "content": SYSTEM_DEFAULT})

    return {"messages": normalized_msgs}

def _norm_from_alpaca(ex: Dict, system_text: str = SYSTEM_DEFAULT) -> Dict:
    """Alpaca 格式规范化。"""
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if not inst or not out: return {"messages": []}

    user_prompt = f"{inst}\n{inp}" if inp else inst
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": out},
    ]
    return {"messages": msgs}

def _norm_ultrachat(ex: Dict) -> Dict:
    """
    处理 UltraChat_200k (标准 messages 格式: role/content)
    """
    raw_msgs = ex.get("messages", [])
    if not raw_msgs: return {"messages": []}

    normalized_msgs = []
    for msg in raw_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            normalized_msgs.append({"role": role, "content": content})

    if not normalized_msgs: return {"messages": []}

    # 检查 System Prompt
    if normalized_msgs[0]["role"] != "system":
        normalized_msgs.insert(0, {"role": "system", "content": SYSTEM_DEFAULT})

    return {"messages": normalized_msgs}


def _norm_fineweb(ex: Dict) -> Dict:
    """
    处理 FineWeb-Edu (纯文本知识)。
    将其包装为特殊的 'text' 角色，以便在 Tokenizer 中识别为 Pre-training 模式。
    """
    text = ex.get("text", "")
    if len(text) < 100: return {"messages": []}  # 太短的不要
    return {"messages": [{"role": "text", "content": text}]}


def _norm_cosmopedia(ex: Dict) -> Dict:
    """
    处理 Cosmopedia (合成教科书/故事)。
    字段: 'prompt', 'text'
    策略: 构造为 "User: Prompt -> Assistant: Text" 的对话，强迫模型输出知识。
    """
    prompt = ex.get("prompt", "").strip()
    content = ex.get("text", "").strip()

    if not content: return {"messages": []}

    # 如果没有 prompt，回退到纯文本模式 (Pre-training)
    if not prompt:
        return {"messages": [{"role": "text", "content": content}]}

    # 如果有 prompt，构建为 SFT 对话
    return {"messages": [
        {"role": "system", "content": SYSTEM_DEFAULT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content}
    ]}


def _norm_infinity(ex: Dict) -> Dict:
    """
    处理 Infinity-Instruct。
    通常是标准 conversations 格式 (role/content) 或 alpaca 风格，
    这里兼容其常见的 conversations 列表格式。
    """
    raw_convs = ex.get("conversations", [])
    if not raw_convs: return {"messages": []}

    normalized_msgs = []
    for msg in raw_convs:
        # 兼容不同字段名
        role = msg.get("role", msg.get("from", "")).lower()
        content = msg.get("content", msg.get("value", ""))

        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "assistant", "model"]:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue

        normalized_msgs.append({"role": role, "content": content})

    if not normalized_msgs: return {"messages": []}

    # 补全 System Prompt
    if normalized_msgs[0]["role"] != "system":
        normalized_msgs.insert(0, {"role": "system", "content": SYSTEM_DEFAULT})

    return {"messages": normalized_msgs}

def _has_valid_structure(ex: Dict) -> bool:
    """
    通用结构检查（安全版）。
    防止 KeyError: 'role' 导致整个处理进程崩溃。
    """
    msgs = ex.get("messages", [])
    if not msgs: return False

    # Case A: Pure Text (使用 .get 安全访问)
    first_role = msgs[0].get("role")
    if len(msgs) == 1 and first_role == "text":
        return True

    # Case B: Conversation
    # 必须有 assistant，且必须都有 role 字段
    try:
        has_assistant = any(m.get("role") == "assistant" for m in msgs)
        return has_assistant
    except Exception:
        return False


# ==============================================================================
# 2. Tokenization & Labeling (核心逻辑：前缀对比法)
# ==============================================================================

def build_encoder(tokenizer, block_size=2048):
    def tokenize_with_assistant_labels(messages: List[Dict[str, str]]):
        # --- 分支 A: 纯文本知识 ---
        # 使用 .get 安全访问
        if len(messages) == 1 and messages[0].get('role') == 'text':
            text = messages[0]['content']
            enc = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding=False,
                add_special_tokens=True
            )
            input_ids = enc.input_ids
            # 确保 EOS
            if len(input_ids) > 0 and input_ids[-1] != tokenizer.eos_token_id:
                if len(input_ids) < block_size:
                    input_ids.append(tokenizer.eos_token_id)
                else:
                    input_ids[-1] = tokenizer.eos_token_id  # 截断时替换最后一个

            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": list(input_ids)
            }

        # --- 分支 B: 对话数据 ---
        try:
            full_input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                truncation=True,
                max_length=block_size,
                padding=False,
                add_generation_prompt=False
            )
        except Exception:
            return None

        if not full_input_ids: return None

        labels = [-100] * len(full_input_ids)

        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant':  # 使用 .get
                prefix_ids = tokenizer.apply_chat_template(
                    messages[:i], tokenize=True, truncation=False, padding=False, add_generation_prompt=True
                )
                current_ids = tokenizer.apply_chat_template(
                    messages[:i + 1], tokenize=True, truncation=False, padding=False, add_generation_prompt=False
                )

                start_index = len(prefix_ids)
                end_index = len(current_ids)

                if start_index < len(full_input_ids):
                    valid_end = min(end_index, len(full_input_ids))
                    labels[start_index:valid_end] = full_input_ids[start_index:valid_end]

        if all(L == -100 for L in labels): return None

        return {
            "input_ids": full_input_ids,
            "attention_mask": [1] * len(full_input_ids),
            "labels": labels
        }

    def preprocess_batch(examples: Dict[str, List]):
        batch_output = {"input_ids": [], "attention_mask": [], "labels": []}

        for msgs in examples["messages"]:
            if not msgs: continue

            # 判断是否为纯文本
            # 使用 .get 增加安全性
            first_role = msgs[0].get('role')
            is_pure_text = (len(msgs) == 1 and first_role == 'text')

            # 【关键修改】如果是对话，进行清洗并检查清洗结果
            if not is_pure_text:
                msgs = _clean_dialog(msgs)

                # 1. 检查清洗后是否为空
                if not msgs:
                    continue

                # 2. 检查清洗后是否还剩 Assistant (防止只剩 User 的情况)
                has_assistant = any(m.get('role') == 'assistant' for m in msgs)
                if not has_assistant:
                    continue

            # 执行 Tokenization
            item = tokenize_with_assistant_labels(msgs)

            if item:
                batch_output["input_ids"].append(item["input_ids"])
                batch_output["attention_mask"].append(item["attention_mask"])
                batch_output["labels"].append(item["labels"])

        return batch_output

    return preprocess_batch


# ==============================================================================
# 3. Data Collator (通用)
# ==============================================================================

@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # 1. 计算当前批次的最大长度
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 0:
            max_len = int(np.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        # 2. [关键修复] 安全获取 pad_token_id
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # 如果 tokenizer 没设置 pad_token，尝试回退到 eos_token
            if self.tokenizer.eos_token_id is not None:
                pad_id = self.tokenizer.eos_token_id
                # (可选) 顺便修补 tokenizer 状态，避免下次还判断
                # self.tokenizer.pad_token_id = pad_id
            else:
                # 最后的兜底：Llama 3 的 <|end_of_text|> ID 是 128001
                pad_id = 128001
                print(f"!!! WARNING: Both pad_token_id and eos_token_id are None. Hardcoding to {pad_id} !!!")

        def pad(seq: List[int], value: int) -> List[int]:
            return seq + [value] * (max_len - len(seq))

        # 3. 构建 Tensor (使用安全的 pad_id)
        batch = {}
        batch["input_ids"] = torch.tensor(
            [pad(f["input_ids"], pad_id) for f in features],
            dtype=torch.long
        )
        batch["attention_mask"] = torch.tensor(
            [pad(f["attention_mask"], 0) for f in features],
            dtype=torch.long
        )
        batch["labels"] = torch.tensor(
            [pad(f["labels"], -100) for f in features],
            dtype=torch.long
        )
        return batch

# ==============================================================================
# 4. 去重工具 (通用)
# ==============================================================================

def _get_text_from_messages(messages: List[Dict[str, str]]) -> str:
    return "".join((m.get("content", "") or "").strip() for m in messages)


def deduplicate_dataset(dataset, num_proc: int = 4):
    print("Starting de-duplication...")
    dataset_with_hash = dataset.map(
        lambda ex: {"hash": hashlib.md5(_get_text_from_messages(ex["messages"]).encode()).hexdigest()},
        num_proc=num_proc, desc="Calculating hashes"
    )
    seen_hashes = set()

    def is_first_occurrence(example):
        h = example['hash']
        if h in seen_hashes: return False
        seen_hashes.add(h)
        return True

    deduplicated = dataset_with_hash.filter(is_first_occurrence, num_proc=1, desc="Filtering")
    return deduplicated.remove_columns("hash")


# ==============================================================================
# 5. 主函数 (针对 Llama 3 优化)
# ==============================================================================

def prepare_llama3_datasets(
        repo_id: str,
        block_size: int = 4096,
        num_proc: int = 8,
        min_token_len: int = 50,
):
    print(f"Loading tokenizer for '{repo_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    # --- Llama 3 Tokenizer 关键配置 ---
    if tokenizer.vocab_size < 100000:
        raise ValueError(f"错误：加载了错误的 Tokenizer (Vocab: {tokenizer.vocab_size})。请检查路径！")

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 128002
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(128002)
        print(f"Set pad_token to ID 128002")

    if not tokenizer.chat_template:
        print("Warning: No default chat template found. Setting Llama 3 default.")
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )

    # -------------------------------------------------------------------------
    # 加载数据集 (MMLU 救援配方：重知识、重逻辑)
    # -------------------------------------------------------------------------

    # 1. Cosmopedia (合成教科书) -> 【主力救援队】
    # 目标：恢复 MMLU/百科知识。量要大。
    print("Loading Cosmopedia (100k)...")
    ds_cosmo = load_from_disk(f"{LOCAL_DATA_DIR}/cosmopedia_stories_subset_100k")
    ds_cosmo = ds_cosmo.select(range(100000))  # 全量使用我们下载的子集
    ds_cosmo = ds_cosmo.map(
        _norm_cosmopedia, num_proc=num_proc,
        remove_columns=ds_cosmo.column_names
    )

    # 2. Infinity-Instruct (高质量指令) -> 【逻辑增强】
    # 目标：比 Alpaca 更强的指令遵循和逻辑。
    print("Loading Infinity-Instruct (50k)...")
    ds_infinity = load_from_disk(f"{LOCAL_DATA_DIR}/infinity_instruct_subset_100k")
    ds_infinity = ds_infinity.select(range(50000))  # 取 50k
    ds_infinity = ds_infinity.map(
        _norm_infinity, num_proc=num_proc,
        remove_columns=ds_infinity.column_names
    )

    # 3. FineWeb-Edu (纯文本知识) -> 【知识锚点】
    # 目标：防止语言模型熵崩塌。
    print("Loading FineWeb-Edu (50k)...")
    ds_fineweb = load_from_disk(f"{LOCAL_DATA_DIR}/fineweb_edu_subset_200k")
    ds_fineweb = ds_fineweb.select(range(50000))  # 增加到 50k
    ds_fineweb = ds_fineweb.map(
        _norm_fineweb, num_proc=num_proc,
        remove_columns=ds_fineweb.column_names
    )

    # 4. OpenHermes-2.5 (通用增强)
    print("Loading OpenHermes-2.5 (30k)...")
    ds_hermes = load_from_disk(f"{LOCAL_DATA_DIR}/openhermes_2_5")
    ds_hermes = ds_hermes.select(range(30000))
    ds_hermes = ds_hermes.map(
        _norm_openhermes, num_proc=num_proc,
        remove_columns=ds_hermes.column_names
    )

    # 5. Magpie-Llama-3 (同源对齐)
    print("Loading Magpie-Llama-3-Pro (30k)...")
    ds_magpie = load_from_disk(f"{LOCAL_DATA_DIR}/magpie_llama_3_pro")
    ds_magpie = ds_magpie.select(range(30000))
    ds_magpie = ds_magpie.map(
        _norm_magpie, num_proc=num_proc,
        remove_columns=ds_magpie.column_names
    )

    # 6. UltraChat (对话连贯性)
    print("Loading UltraChat (20k)...")
    ds_chat = load_from_disk(f"{LOCAL_DATA_DIR}/ultrachat_200k")
    ds_chat = ds_chat.select(range(20000))
    ds_chat = ds_chat.map(
        _norm_ultrachat, num_proc=num_proc,
        remove_columns=ds_chat.column_names
    )

    # 7. 理科组 (Code/Math)
    print("Loading Math/Code (20k)...")
    ds_math = load_from_disk(f"{LOCAL_DATA_DIR}/openmath_instruct_2_subset")
    ds_math = ds_math.select(range(10000))
    ds_math = ds_math.map(_norm_openmath, num_proc=num_proc, remove_columns=ds_math.column_names)

    # Alpaca (可以保留少量作为保底，或者去掉)
    print("Loading Alpaca (10k)...")
    ds_alpaca = load_from_disk(f"{LOCAL_DATA_DIR}/alpaca_cleaned")
    ds_alpaca = ds_alpaca.select(range(10000))
    ds_alpaca = ds_alpaca.map(
        lambda ex: _norm_from_alpaca(ex), num_proc=num_proc, remove_columns=["instruction", "input", "output"]
    )

    # -------------------------------------------------------------------------
    # 合并
    # -------------------------------------------------------------------------
    print("Merging datasets...")

    def safe_keep(ds):
        return ds.select_columns(["messages"])

    all_ds = [
        safe_keep(ds_cosmo.filter(_has_valid_structure, num_proc=num_proc)),  # 新增
        safe_keep(ds_infinity.filter(_has_valid_structure, num_proc=num_proc)),  # 新增
        safe_keep(ds_fineweb),
        safe_keep(ds_magpie.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_hermes.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_chat.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_math.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_alpaca.filter(_has_valid_structure, num_proc=num_proc))
    ]

    # 注意：数据量变大了（约 300k），训练时请适当增加 max_steps 或 epoch
    mixed_ds = concatenate_datasets(all_ds).shuffle(seed=42)
    mixed_ds = deduplicate_dataset(mixed_ds, num_proc=num_proc)

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------
    print("Tokenizing...")
    encode_fn = build_encoder(tokenizer, block_size=block_size)

    tokenized_ds = mixed_ds.map(
        encode_fn,
        batched=True,
        batch_size=1000,
        remove_columns=["messages"],
        num_proc=num_proc,
        desc="Tokenizing"
    )

    final_ds = tokenized_ds.filter(
        lambda ex: min_token_len <= len(ex["input_ids"]) <= block_size,
        num_proc=num_proc
    )
    print(f"Final training set size: {len(final_ds)}")

    split_ds = final_ds.train_test_split(test_size=0.01, seed=42)
    collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer, pad_to_multiple_of=8)

    return tokenizer, split_ds["train"], split_ds["test"], collator

# def debug_llama3_data(tokenizer, dataset):
#     print("\n" + "=" * 40)
#     print("      LLAMA 3 DATA DEBUGGER")
#     print("=" * 40)
#
#     # 1. 检查特殊 Token 和 模板
#     print(f"\n[1] Tokenizer Config:")
#     print(f"  - BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
#     print(f"  - EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
#     print(f"  - Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
#     print(f"  - Chat Template Snippet: {tokenizer.chat_template[:200]}...")
#
#     # 2. 取出一个样本进行解码透视
#     print(f"\n[2] Sample Inspection (Index 0):")
#     sample = dataset[0]
#     input_ids = sample['input_ids']
#     labels = sample['labels']
#
#     # 3. 对比 Input 和 Label
#     print(f"  - Sequence Length: {len(input_ids)}")
#
#     print("\n  --- [Visual Check] (Input vs Label) ---")
#     print("  Token_ID  |  Label  |  Decoded String")
#     print("  ----------|---------|----------------")
#
#     # 只打印前 150 个 token 和最后 20 个 token，避免刷屏
#     indices_to_show = list(range(150)) + list(range(len(input_ids) - 20, len(input_ids)))
#     # 去重
#     indices_to_show = sorted(list(set([i for i in indices_to_show if i < len(input_ids)])))
#
#     for i in indices_to_show:
#         tid = input_ids[i]
#         lab = labels[i]
#
#         # 解码当前 token
#         token_str = tokenizer.decode([tid]).replace('\n', '\\n')
#
#         # 状态判断
#         if lab == -100:
#             label_status = " IGNORE"  # 被 Mask 掉了，不计算 Loss
#         else:
#             if lab != tid:
#                 label_status = "!!!MISMATCH!!!"  # 严重错误：Label 和 Input 不一致
#             else:
#                 label_status = f"{lab:7d}"  # 正常训练
#
#         # 如果是断层（打印省略号）
#         if i > 0 and indices_to_show[indices_to_show.index(i) - 1] != i - 1:
#             print("      ...   |   ...   | ...")
#
#         print(f"  {tid:8d}  | {label_status} | {token_str}")
#
#     print("=" * 40 + "\n")
#
#
import itertools


def make_probe_batches(dataset, collator, *, batch_size=1, max_batches=30, device=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    probe_batches = []
    for batch in itertools.islice(loader, max_batches):
        # 强制 long，并搬到同一 device，避免 dtype/device 差异
        if "input_ids" in batch:      batch["input_ids"] = batch["input_ids"].long()
        if "labels" in batch:         batch["labels"] = batch["labels"].long()
        if "attention_mask" in batch: batch["attention_mask"] = batch["attention_mask"].long()
        if device is not None:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probe_batches.append(batch)
    return probe_batches


def eval_on_batches(model, probe_batches):
    was_training = model.training
    model.eval()
    import math
    total, n = 0.0, 0
    with torch.no_grad():
        for b in probe_batches:
            out = model(**b, use_cache=False)
            total += float(out.loss.detach().cpu())
            n += 1
    if was_training: model.train()
    mean = total / max(1, n)
    ppl = math.exp(mean) if mean < 30 else float("inf")
    return mean, ppl

def inspect_label_ratio(probe_batches):
    # 看看标签被 mask 的比例是否一致（labels != -100 的占比）
    tot_tok = 0
    eff_tok = 0
    for b in probe_batches:
        L = b.get("labels", None)
        if L is None: continue
        tot_tok += L.numel()
        eff_tok += (L != -100).sum().item()
    ratio = eff_tok / max(1, tot_tok)
    print(f"[probe] effective label ratio = {ratio:.3f}")

# =====================  新增：逐层替换+训练 主流程  =====================

# def progressive_train_from_last_layer(model, tok, base_cfg, *, device, run_dtype, head_dim,
#                                       train_ds, collator,
#                                       baseline: float,
#                                       start_layer: int = None, end_layer: int = 0):
#     """
#     从最后一层开始（默认 num_hidden_layers-1），每层：
#       (1) 7 门全部替换为 ADTN，并做整矩阵暖启动；
#       (2) 仅训练本层新增 ADTN 参数；
#       (3) 训练过程中每 2000 step 做一次 MMLU 5-shot 评测；达到 baseline 即停止并进入上一层。
#     """
#     L_total = int(getattr(base_cfg, "num_hidden_layers"))
#     start = L_total - 1 if start_layer is None else int(start_layer)
#
#     # 为每一层单独建 Trainer，便于输出目录隔离
#     for L in range(start, end_layer - 1, -1):
#         print(f"\n================  开始处理第 {L} 层（共 {L_total} 层, 0-index）  ================")
#         layer_cfg = build_cfg_for_single_layer(base_cfg, L)
#
#         # 仅对本层 7 个门做替换 + 暖启动
#         new_params = replace_linear_with_tnn_from_config(
#             model, layer_cfg,
#             head_dim=head_dim, device=device, run_dtype=run_dtype,
#             d_dim=2, warm_start_weight=True, FIT_STEPS=WARM_START_STEPS, FIT_LR=WARM_START_LR, verbose_fit=True,
#         )
#
#         # 仅训练本层新增 ADTN 参数；其余全部冻结
#         if not new_params:
#             print("[warn] 本层未产生可训练 ADTN 参数，直接跳过。")
#             continue
#         set_requires_grad_only(model, new_params)
#
#         # 将可训练参数强制转为 FP32，提高稳定性（其余保持 run_dtype，例如 bf16）
#         for p in model.parameters():
#             if p.requires_grad:
#                 p.data = p.data.to(torch.float32)
#
#         # 针对当前层构建 Trainer（save 每 3000 step），并挂载 MMLU 回调
#         args = TrainingArguments(
#             output_dir=f"qwen_tnn_L{L:02d}_ckpts_2",
#             per_device_train_batch_size=4,
#             gradient_accumulation_steps=8,
#             learning_rate=5e-5,
#             max_steps=9000,  # 很大值；回调命中基线后会提前终止
#             eval_strategy="no",
#             save_strategy="steps",
#             save_steps=EVAL_EVERY_STEPS,
#             save_safetensors=True,
#             logging_steps=20,
#             report_to="none",
#             gradient_checkpointing=False,
#             bf16=True,
#             fp16=False,
#             group_by_length=True,
#         )
#
#         trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
#         trainer.add_callback(SanitizeConfigBeforeSave())
#         trainer.add_callback(MMLUEvalUntilBaseline(tok=tok, baseline=baseline))
#         sanitize_config_inplace(model.config)
#
#         trainer.train()
#         print(f"================  第 {L} 层完成，进入下一层  ==================\n")


# def progressive_train_from_last_layer(model, tok, base_cfg, *, device, run_dtype, head_dim,
#                                       train_ds, collator,
#                                       baseline: float,
#                                       start_layer: int = None, end_layer: int = 0):
#     """
#     逐层（从末层到前）：每层替换7门+暖启动 -> 仅训本层ADTN(升FP32，其余保持run_dtype) ->
#     每 EVAL_EVERY_STEPS 做一次 MMLU 5-shot，>=baseline 早停本层；
#     每成功替换5层后，解冻“当前所有ADTN”做一次整体训练（同样按 baseline 早停，且最多 all_adtn_max_steps 步）。
#     """
#     import os, torch
#     from transformers import TrainingArguments, Trainer
#
#     # ------- 设备与混精准备（放循环外） -------
#     if hasattr(device, "type") and device.type == "cuda":
#         torch.cuda.set_device(device)
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = True
#     except Exception:
#         pass
#     prefer_bf16 = (run_dtype == torch.bfloat16)
#
#     # 整模一次性放到目标 dtype；后续只把“可训参数”升为 FP32，避免反复 cast
#     model.to(device=device, dtype=run_dtype)
#
#     # 读取评测/上限步数（优先使用全局常量；否则用环境变量兜底）
#     try:
#         eval_every = int(EVAL_EVERY_STEPS)  # 若你脚本里有同名常量，这里直接用
#     except NameError:
#         eval_every = int(os.environ.get("EVAL_EVERY_STEPS", "3000"))
#     all_adtn_max_steps = int(os.environ.get("ALL_ADTN_MAX_STEPS", "9000"))
#
#     L_total = int(getattr(base_cfg, "num_hidden_layers"))
#     start = L_total - 1 if start_layer is None else int(start_layer)
#
#     replaced_ok = 0  # 成功替换并训练过的层计数（跳过的不计）
#
#     for L in range(start, end_layer - 1, -1):
#         print(f"\n================  开始处理第 {L} 层（共 {L_total} 层, 0-index）  ================")
#         layer_cfg = build_cfg_for_single_layer(base_cfg, L)
#
#         # (1) 当前层7门替换+暖启动
#         new_params = replace_linear_with_tnn_from_config(
#             model, layer_cfg,
#             head_dim=head_dim, device=device, run_dtype=run_dtype,
#             d_dim=2, warm_start_weight=True,
#             FIT_STEPS=WARM_START_STEPS, FIT_LR=WARM_START_LR, verbose_fit=True,
#         )
#
#         # (2) 只训本层ADTN
#         if not new_params:
#             print("[warn] 本层未产生可训练 ADTN 参数，跳过且不计入 5 层计数。")
#             continue
#         set_requires_grad_only(model, new_params)
#         for p in new_params:
#             p.data = p.data.to(torch.float32)
#
#         # ——分层训练：on_save 做 MMLU，达 baseline 早停——
#         args = TrainingArguments(
#             output_dir=f"llama_tnn_L{L:02d}_ckpts_layer_2",
#             per_device_train_batch_size=4,
#             gradient_accumulation_steps=8,
#             learning_rate=5e-5,
#             max_steps=12000,        # 大值；由回调提前终止
#             eval_strategy="no",
#             save_strategy="steps",
#             save_steps=eval_every,       # 每 eval_every 步评测一次
#             save_safetensors=True,
#             logging_steps=20,
#             report_to="none",
#             gradient_checkpointing=False,
#             bf16=prefer_bf16,
#             fp16=not prefer_bf16,
#             tf32=True,
#             group_by_length=True,
#             max_grad_norm=1.0,
#             optim="adamw_torch_fused",
#             disable_tqdm=True
#         )
#         trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
#         trainer.add_callback(SanitizeConfigBeforeSave())
#         trainer.add_callback(MMLUEvalUntilBaseline(tok=tok, baseline=baseline))
#         sanitize_config_inplace(model.config)
#         print(f"[hook] per-layer baseline={baseline}, save_steps={eval_every}")
#         trainer.train()
#         print(f"================  第 {L} 层完成  ==================")
#
#         # 成功替换并训练过，才计入 5 层累计
#         replaced_ok += 1
#
#         # (3) 每成功替换 5 层后 -> 全部 ADTN 整体训练（同样按 baseline 早停；且最多 all_adtn_max_steps 步）
#         if replaced_ok % 3 == 0:
#             print(f"\n===========  已成功替换 {replaced_ok} 层，开始『全部 ADTN』整体训练（可早停）  ===========")
#             # 解冻所有 ADTN，仅它们参与训练；并全部升为 FP32
#             set_requires_grad_tnn_only(model)
#             fp32_cnt = 0
#             for p in model.parameters():
#                 if p.requires_grad:
#                     p.data = p.data.to(torch.float32)
#                     fp32_cnt += 1
#             print(f"[all-ADTN] trainable tensors（FP32）: {fp32_cnt}")
#
#             args_all = TrainingArguments(
#                 output_dir=f"llama_tnn_ALL_ADTN_after{replaced_ok:02d}_2",
#                 per_device_train_batch_size=4,
#                 gradient_accumulation_steps=8,
#                 learning_rate=5e-5,
#                 max_steps=12000,  # ← 这次严格受上限约束
#                 eval_strategy="no",
#                 save_strategy="steps",
#                 save_steps=eval_every,
#                 save_safetensors=True,
#                 logging_steps=20,
#                 report_to="none",
#                 gradient_checkpointing=False,
#                 bf16=prefer_bf16,
#                 fp16=not prefer_bf16,
#                 tf32=True,
#                 group_by_length=True,
#                 max_grad_norm=1.0,
#                 optim="adamw_torch_fused",
#                 disable_tqdm=True
#             )
#             trainer_all = Trainer(model=model, args=args_all, train_dataset=train_ds, data_collator=collator)
#             trainer_all.add_callback(SanitizeConfigBeforeSave())
#             trainer_all.add_callback(MMLUEvalUntilBaseline(tok=tok, baseline=baseline))  # 这轮也用“达基线早停”
#             sanitize_config_inplace(model.config)
#             print(f"[hook] all-ADTN baseline={baseline}, save_steps={eval_every}, max_steps={all_adtn_max_steps}")
#             trainer_all.train()
#             print(f"===========  全部 ADTN 整体训练结束（可能已因达基线提前停止）  ===========\n")
#
#
# def main():
#     # 1) 数据准备（保留你的实现）
#     tok, train_ds, eval_ds, collator = prepare_llama3_datasets(
#         repo_id=REPO)
#
#     # 2) 加载模型
#     cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         REPO,
#         config=cfg,
#         trust_remote_code=True,
#         torch_dtype=torch.float32,
#     )
#     model.to(dtype=torch.float32)
#     model.config.pad_token_id = tok.pad_token_id
#     model.config.use_cache = False
#     model.to("cuda:0")
#
#     device   = next(model.parameters()).device
#     head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
#
#     # 3) 只复制 config（不一次性替换 gates）
#     # 已把 src_config 覆盖到模型目录；然后重新加载 cfg
#     cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
#
#     # 4) 将模型转换为训练 dtype（例如 bf16 用于前向），随后进入“逐层替换+训练”
#     model.to(dtype=torch.bfloat16)
#     set_requires_grad_only(model, [])  # 初始全部冻结（随后每层只开本层门）
#
#     # 5) 启动分层训练
#     progressive_train_from_last_layer(
#         model, tok, cfg,
#         device=device, run_dtype=next(model.parameters()).dtype, head_dim=head_dim,
#         train_ds=train_ds, collator=collator,
#         baseline=MMLU_BASELINE,
#         start_layer=31,  # 例如 Llama3-8B 为 31
#         end_layer=0,
#     )
#
#
# if __name__ == "__main__":
#     main()


# def train_specific_layers_jointly(model, tok, base_cfg, *, device, run_dtype, head_dim,
#                                   train_ds, collator,
#                                   baseline: float,
#                                   target_layers: list):
#     """
#     指定层列表整体替换后统一训练：
#     1. 遍历 target_layers 中的每一层：
#        - 执行 ADTN 结构替换
#        - 执行 MSE 暖启动 (Warm Start)
#     2. 将这些层的所有 ADTN 参数设为可训练 (FP32)
#     3. 进行一次整体恢复训练 (Joint Recovery Training)，直到达到 MMLU baseline。
#     """
#
#     # ------- 设备与混精准备 -------
#     if hasattr(device, "type") and device.type == "cuda":
#         torch.cuda.set_device(device)
#     try:
#         torch.backends.cuda.matmul.allow_tf32 = True
#     except Exception:
#         pass
#     prefer_bf16 = (run_dtype == torch.bfloat16)
#
#     # 确保模型在目标 dtype
#     model.to(device=device, dtype=run_dtype)
#
#     # 读取评测设置
#     try:
#         eval_every = int(EVAL_EVERY_STEPS)
#     except NameError:
#         eval_every = int(os.environ.get("EVAL_EVERY_STEPS", "3000"))
#
#     # 收集所有被替换层产生的新参数
#     all_new_adtn_params = []
#
#     print(f"\n>>> 即将处理的层列表: {target_layers}")
#
#     # =====================================================
#     # 第一阶段：批量替换 + MSE 暖启动
#     # =====================================================
#     for L in target_layers:
#         print(f"\n[Phase 1] 正在替换并暖启动第 {L} 层...")
#         layer_cfg = build_cfg_for_single_layer(base_cfg, L)
#
#         # 执行替换 + MSE Warmup
#         # 注意：这里只做结构替换和基于输出拟合的初始化，不进行 LLM 的 Loss 训练
#         new_params = replace_linear_with_tnn_from_config(
#             model, layer_cfg,
#             head_dim=head_dim, device=device, run_dtype=run_dtype,
#             d_dim=2, warm_start_weight=True,
#             FIT_STEPS=WARM_START_STEPS, FIT_LR=WARM_START_LR, verbose_fit=True,
#         )
#
#         if new_params:
#             all_new_adtn_params.extend(new_params)
#         else:
#             print(f"[warn] 第 {L} 层未产生可训练参数。")
#
#     if not all_new_adtn_params:
#         print("未发现任何可训练的 ADTN 参数，退出训练。")
#         return
#
#     # =====================================================
#     # 第二阶段：设置梯度与精度
#     # =====================================================
#     print(f"\n[Phase 2] 准备整体微调，涉及 ADTN 参数数量: {len(all_new_adtn_params)} 个 Tensor")
#
#     # 1. 冻结全网，只开启刚才替换的所有 ADTN 参数的梯度
#     set_requires_grad_only(model, all_new_adtn_params)
#
#     # 2. 将可训练参数提升为 FP32 以保证微调稳定性
#     for p in all_new_adtn_params:
#         p.data = p.data.to(torch.float32)
#
#     # =====================================================
#     # 第三阶段：整体恢复训练 (Joint Recovery)
#     # =====================================================
#     print(f"\n[Phase 3] 开始整体恢复训练 (Target Layers: {target_layers})")
#
#     # 构造 Output Dir 名字
#     layer_str = "_".join(map(str, target_layers[:3]))
#     if len(target_layers) > 3: layer_str += "_etc"
#     output_dir_name = f"llama3_tnn_JOINT_L{layer_str}_1"
#
#     args = TrainingArguments(
#         output_dir=output_dir_name,
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=8,
#         learning_rate=5e-5,
#         max_steps=1000000,  # 给予足够的步数，主要靠 MMLU 回调早停
#         eval_strategy="no",
#         save_strategy="steps",
#         save_steps=eval_every,
#         save_safetensors=True,
#         logging_steps=20,
#         report_to="none",
#         gradient_checkpointing=False,
#         bf16=prefer_bf16,
#         fp16=not prefer_bf16,
#         tf32=True,
#         group_by_length=True,
#         max_grad_norm=1.0,
#         optim="adamw_torch_fused",
#         disable_tqdm=True
#     )
#
#     trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
#
#     # 注册回调：保存前清理 config，以及 MMLU 达标早停
#     trainer.add_callback(SanitizeConfigBeforeSave())
#     trainer.add_callback(MMLUEvalUntilBaseline(tok=tok, baseline=baseline))
#
#     sanitize_config_inplace(model.config)
#
#     print(f"[hook] Start Training... Baseline={baseline}, Check/Save Every={eval_every}")
#     trainer.train()
#     print(f"================  指定层整体训练完成  ==================")


class WSDScheduleCallback(TrainerCallback):
    def __init__(self, decay_start_ratio: float = 0.85, decay_type: str = "linear", peak_lr: float = 5e-5):
        self.decay_start_ratio = decay_start_ratio
        self.decay_type = decay_type
        self.peak_lr = peak_lr  # 需要知道峰值 LR 是多少

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        cur_step = state.global_step
        max_steps = state.max_steps
        warmup_steps = args.warmup_steps

        # 1. Warmup 阶段 (交给 Trainer 自带的逻辑，或者我们自己管)
        # 为了防止冲突，我们在 Warmup 阶段即使不管，Trainer 的 constant_with_warmup 也会工作。
        # 但进入 Stable 阶段后，我们要接管。

        # 计算 WSD 逻辑下的目标 LR
        decay_start_step = int(max_steps * self.decay_start_ratio)

        target_lr = self.peak_lr

        if cur_step < warmup_steps:
            # Warmup 阶段：线性上升
            target_lr = self.peak_lr * (cur_step / max(1, warmup_steps))
        elif cur_step < decay_start_step:
            # Stable 阶段：恒定
            target_lr = self.peak_lr
        else:
            # Decay 阶段：线性下降
            decay_steps = max_steps - decay_start_step
            progress = (cur_step - decay_start_step) / max(1, decay_steps)
            progress = min(1.0, max(0.0, progress))

            if self.decay_type == "linear":
                factor = 1.0 - progress
            else:
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            target_lr = self.peak_lr * factor

        # 【强硬手段】直接覆盖优化器的 LR
        # 无论 Trainer 的 Scheduler 做了什么，我们在 Step 开始前最后一刻把它改掉
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr

        # 打印日志 (可选)
        if cur_step % 100 == 0 and cur_step > decay_start_step:
            print(f"  [WSD] Step {cur_step} | LR forced to {target_lr:.2e}")


def train_specific_layers_jointly(model, tok, base_cfg, *, device, run_dtype, head_dim,
                                  train_ds, collator,
                                  baseline: float,
                                  target_layers: list):
    """
    指定层列表整体替换后统一训练
    """

    # ------- 设备与混精准备 -------
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.set_device(device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    prefer_bf16 = (run_dtype == torch.bfloat16)

    model.to(device=device, dtype=run_dtype)

    try:
        eval_every = int(EVAL_EVERY_STEPS)
    except NameError:
        eval_every = int(os.environ.get("EVAL_EVERY_STEPS", "3000"))

    all_new_adtn_params = []

    print(f"\n>>> 即将处理的层列表: {target_layers}")

    strategy_map = {}

    # 1. 全局基准：所有层默认 offset = 3 (高压缩)
    for L in target_layers:
        strategy_map[L] = 3
        # strategy_map[(L, 'down')] = "skip"

    # # # 2. 层级调整：第 17 层作为过渡层，整体放宽到 2
    # if 17 in target_layers:
    #     strategy_map[17] = 1
    #
    # # 3. 【关键】门级微调 (Gate-level Override)
    # # 你觉得第 17 层的 down_proj 特别重要（因为它负责将维度投影回残差流），
    # # 所以给它极低的 offset (1)，保留更多参数。
    # if 17 in target_layers:
    #     print(">>> [策略] Layer 17 'down_proj' 启用高精度模式 (Offset=1)")
    #     strategy_map[(17, 'down')] = 1  # <--- 这里实现你的需求

    # # 举例：如果你觉得所有层的 'o_proj' (Attention Output) 都很重要
    # for L in target_layers:
    #     strategy_map[(L, 'down')] = 2

    print(f"\n>>> 压缩策略详情: {strategy_map}")

    # layer_offset_map = {}
    #
    # # 默认策略：所有目标层默认为 3
    # for L in target_layers:
    #     layer_offset_map[L] = 3
    #
    # # 特殊调整：假设我们觉得 25, 26 层比较重要，降低压缩率
    # layer_offset_map[17] = 2
    # # layer_offset_map[26] = 2
    #
    # # 如果你想更精细，可以手动写死：
    # # layer_offset_map = {
    # #     35: 3, 34: 3, 33: 3, 32: 3, 31: 3,  # 深层高压缩
    # #     30: 2, 29: 2, 28: 2,                # 过渡层中压缩
    # #     27: 1, 26: 1, 25: 1                 # 敏感层低压缩
    # # }
    #
    # print(f"\n>>> 压缩策略 (Layer: Offset): {layer_offset_map}")

    # =====================================================
    # 第一阶段：智能检查 + 替换 (应用动态 Offset)
    # =====================================================
    # 假设由函数参数传入，或者在这里定义你想要训练的层列表
    # trainable_layers = [17]  # 例如：只训练最后两层
    trainable_layers = None  # None 表示默认训练 target_layers 中的所有层

    # 制作一个集合方便快速查找
    layers_to_train_set = set(target_layers) if trainable_layers is None else set(trainable_layers)

    for L in target_layers:
        print(f"\n[Phase 1] 正在检查第 {L} 层...")

        # 获取 Layer 模块
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer_module = model.model.layers[L]
        elif hasattr(model, "layers"):
            layer_module = model.layers[L]
        else:
            raise ValueError("无法定位模型层结构")

        found_adtn_in_this_layer = False
        existing_params_in_layer = []

        # 递归扫描所有子模块
        for name, submodule in layer_module.named_modules():
            if name == "": continue

            # A. 检查是否是 ADTN (看有没有 bricks)
            if hasattr(submodule, "bricks"):
                found_adtn_in_this_layer = True
                for p in submodule.parameters():
                    existing_params_in_layer.append(p)

        # --- 暂存当前层的参数 ---
        current_layer_params = []

        # 分支处理
        if found_adtn_in_this_layer:
            print(f"  ✅ [检测] 第 {L} 层已包含 ADTN 结构，复用参数。")
            current_layer_params = existing_params_in_layer

        else:
            # 获取当前层的默认值用于打印日志
            default_off = strategy_map.get(L, 3)
            print(f"  🔄 第 {L} 层为原始 Linear 层，执行替换 (LayerDefault={default_off})...")

            # 调用新的 refined 构建函数，传入整个 map
            layer_cfg = build_cfg_for_single_layer_refined(base_cfg, L, strategy_map)

            new_params = replace_linear_with_tnn_from_config(
                model, layer_cfg,
                head_dim=head_dim, device=device, run_dtype=run_dtype,
                d_dim=2, warm_start_weight=True,
                FIT_STEPS=WARM_START_STEPS, FIT_LR=WARM_START_LR, verbose_fit=True,
            )

            if new_params:
                current_layer_params = new_params

        # --- 【关键修改】决定是否将当前层参数加入训练列表 ---
        if L in layers_to_train_set:
            # 如果在训练列表中，确保开启梯度并加入总表
            for p in current_layer_params:
                p.requires_grad = True
            all_new_adtn_params.extend(current_layer_params)
        else:
            # 如果不在训练列表中，打印日志提示（参数不会加入总表，稍后会被 set_requires_grad_only 冻结）
            print(f"  🔒 第 {L} 层参数虽然存在/已替换，但不在训练列表中，将被冻结。")

    if not all_new_adtn_params:
        print("❌ 未发现任何可训练的 ADTN 参数，退出训练。")
        return

    # =====================================================
    # 第二阶段：设置梯度与精度
    # =====================================================
    print(f"\n[Phase 2] 准备整体微调，参数总量: {len(all_new_adtn_params)}")

    # 这一步会将 all_new_adtn_params 里的设为 True，其余（包括不在列表里的 ADTN 层）设为 False
    set_requires_grad_only(model, all_new_adtn_params)

    for p in all_new_adtn_params:
        p.data = p.data.to(torch.float32)

    # =====================================================
    # 第三阶段：整体恢复训练
    # =====================================================
    layer_str = "_".join(map(str, target_layers[:3]))
    if len(target_layers) > 3: layer_str += "_etc"
    output_dir_name = f"llama3_tnn_JOINT_L{layer_str}"

    # 1. 设置合理的训练总步数 (WSD 强依赖此参数)
    # 假设你的数据集大小和 Batch Size 已知，你应该计算出大概需要训练多少步。
    # 比如训练 1 个 Epoch，或者固定跑 5000 步。
    # 这里举例设为 5000，请根据实际情况修改！
    REAL_MAX_STEPS = 30000

    # 2. 【关键修改】使用 WSD Callback
    # decay_start_ratio=0.85 意味着前 4250 步保持 5e-5，最后 750 步线性降到 0
    wsd_callback = WSDScheduleCallback(decay_start_ratio=0.85, decay_type="linear", peak_lr=4e-5)

    from transformers import TrainingArguments, Trainer
    args = TrainingArguments(
        output_dir=output_dir_name,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=4e-5,

        max_steps=REAL_MAX_STEPS,

        eval_strategy="no",
        save_strategy="steps",
        save_steps=eval_every,
        save_safetensors=True,
        logging_steps=20,
        report_to="none",
        gradient_checkpointing=False,
        bf16=prefer_bf16,
        fp16=not prefer_bf16,
        tf32=True,
        group_by_length=False,
        max_grad_norm=1.0,
        disable_tqdm=True,

        warmup_ratio=0.01,  # 或 warmup_steps=...
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={
            "num_decay_steps": int(REAL_MAX_STEPS * 0.10),  # 例：最后 10% 做 cosine decay
            "decay_type": "1-sqrt",  # 默认就是 cosine，可不写
            "warmup_type": "linear",  # 默认就是 linear，可不写
            "min_lr_ratio": 0.0,  # 默认 0；如果你不想降到 0，可以改成 0.1 等
            "num_cycles": 0.5,  # cosine 默认半个周期从 max 降到 min
        },
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[SanitizeConfigBeforeSave(), MMLUEvalUntilBaseline(tok=tok, baseline=baseline)]
    )

    sanitize_config_inplace(model.config)

    print(f"[hook] Start Joint Training... Baseline={baseline}")
    print(f"[hook] LR Strategy: WSD (Warmup -> Stable until 85% -> Linear Decay)")

    trainer.train()
    print(f"================  训练完成  ==================")


def main():
    # 1) 数据准备
    tok, train_ds, eval_ds, collator = prepare_llama3_datasets(repo_id=REPO)

    # 2) 加载模型
    # 先加载 config
    base_cfg_obj = AutoConfig.from_pretrained(REPO, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        config=base_cfg_obj,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # 初始加载建议 FP32 或根据显存决定
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model.to("cuda")

    device = next(model.parameters()).device
    head_dim = getattr(base_cfg_obj, "head_dim", base_cfg_obj.hidden_size // base_cfg_obj.num_attention_heads)

    # 3) 准备配置对象 (用于传递给 ADTN 构建函数)
    # 这里重新加载一次 config 确保纯净，或者直接用上面的 base_cfg_obj
    cfg_for_tnn = AutoConfig.from_pretrained(REPO, trust_remote_code=True)

    # 4) 切换到运行精度 (BF16) 并冻结所有参数
    model.to(dtype=torch.bfloat16)
    set_requires_grad_only(model, [])

    # 5) 定义你想要压缩并联合训练的层
    # 例如：只压缩最后 3 层进行联合训练
    layers_to_compress = [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]
    # 或者全部层： layers_to_compress = list(range(31, -1, -1))

    # 6) 启动新的训练流程
    train_specific_layers_jointly(
        model, tok, cfg_for_tnn,
        device=device, run_dtype=torch.bfloat16, head_dim=head_dim,
        train_ds=train_ds, collator=collator,
        baseline=MMLU_BASELINE,
        target_layers=layers_to_compress  # <--- 传入列表
    )


if __name__ == "__main__":
    main()
