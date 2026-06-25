import os
os.environ['HTTPS_PROXY'] = 'http://10.29.1.201:8888'
os.environ['HTTP_PROXY'] = 'http://10.29.1.201:8888'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 可选，让编号更直观
os.environ["CUDA_VISIBLE_DEVICES"] = "5"        # 改成你想用的服务器物理卡号，例如只用第3号卡


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


MMLU_BASELINE = float(os.environ.get("MMLU_BASELINE", "0.75"))
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "4000"))
WARM_START_STEPS = int(os.environ.get("WARM_START_STEPS", "1000"))
WARM_START_LR    = float(os.environ.get("WARM_START_LR", "1e-2"))


# 统一成一个变量
MODEL_DIR = Path("./model/Qwen/Qwen3-8B-expanded-model").resolve()
REPO = str(MODEL_DIR)  # 既可给 HF Transformers 也可拼接本地文件

file = MODEL_DIR / "modeling_tnn.py"
spec = importlib.util.spec_from_file_location("qwen3_tnn", file)
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

    # 解析 gate_offset：优先入参，其次全局 cfg.adtn_gate_offset，最后默认
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


# ==============================================================================
# 0. 全局配置 (Qwen 风格)
# ==============================================================================

# Qwen 的官方默认 Prompt 比较简单，或者可以直接用通用的
# 保持这个 Prompt 有助于模型识别自己是助手
SYSTEM_DEFAULT = "You are a helpful and intelligent AI assistant."


# ==============================================================================
# 1. 数据规范化函数 (针对 Qwen 配方)
# ==============================================================================

def _speaker_strip(s: str) -> str:
    """通用清洗"""
    if not isinstance(s, str): return ""
    t = re.sub(r"^\s*(User|Assistant|Human)\s*:\s*", "", s.strip(), flags=re.IGNORECASE)
    return t


def _norm_magpie_qwen(ex: Dict) -> Dict:
    """
    处理 Magpie-Qwen2-Pro 数据集。
    增强鲁棒性：兼容 'from/value' 格式，处理角色映射，防止 KeyError。
    """
    raw_convs = ex.get("conversations", [])
    if not raw_convs: return {"messages": []}

    normalized_msgs = []

    # 1. 遍历清洗每一条消息
    for msg in raw_convs:
        # 兼容 role/from
        role = msg.get("role", msg.get("from", "")).lower()
        # 兼容 content/value
        content = msg.get("content", msg.get("value", ""))

        # 角色映射
        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "assistant", "model"]:
            role = "assistant"
        elif role == "system":
            role = "system"

        # 过滤无效数据
        if not role or not content:
            continue

        normalized_msgs.append({"role": role, "content": content})

    if not normalized_msgs:
        return {"messages": []}

    # 2. 检查并插入 Qwen 默认 System Prompt
    # 注意：确保 SYSTEM_DEFAULT 在此函数作用域内可见
    if normalized_msgs[0]["role"] != "system":
        normalized_msgs.insert(0, {"role": "system", "content": SYSTEM_DEFAULT})

    return {"messages": normalized_msgs}


def _norm_evol_code(ex: Dict) -> Dict:
    """
    处理 Evol-CodeAlpaca (代码数据)。
    格式: 'instruction', 'output'
    """
    inst = ex.get("instruction", "")
    out = ex.get("output", "")

    if not inst or not out: return {"messages": []}

    return {"messages": [
        {"role": "system", "content": SYSTEM_DEFAULT},
        {"role": "user", "content": inst},
        {"role": "assistant", "content": out}
    ]}


def _norm_openmath(ex: Dict) -> Dict:
    """处理 OpenMathInstruct-2"""
    problem = ex.get("problem", "")
    solution = ex.get("generated_solution", "")
    if not problem or not solution: return {"messages": []}

    return {"messages": [
        {"role": "system", "content": SYSTEM_DEFAULT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution}
    ]}

# def _norm_ultrachat(ex: Dict) -> Dict:
#     """处理 UltraChat_200k"""
#     raw = ex.get("messages", [])
#     # UltraChat 也是标准格式，直接返回，稍后统一清洗
#     return {"messages": raw}


def _norm_fineweb(ex: Dict) -> Dict:
    """处理 FineWeb-Edu (纯文本)"""
    text = ex.get("text", "")
    if len(text) < 100: return {"messages": []}
    return {"messages": [{"role": "text", "content": text}]}


def _norm_glaive_code(ex: Dict) -> Dict:
    """
    处理 Glaive-Code-Assistant-v3
    格式: 'question', 'answer'
    """
    q = ex.get("question", "")
    a = ex.get("answer", "")
    if not q or not a: return {"messages": []}

    return {"messages": [
        {"role": "system", "content": SYSTEM_DEFAULT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}
    ]}


def _norm_to_messages_from_conversations(
        ex: Dict, field: str, system_text: str = SYSTEM_DEFAULT
) -> Dict:
    """
    通用函数：将一个包含多轮对话的字段（如 'conversations', 'messages'）
    规范化为 OpenAI 格式的 messages 列表。
    """
    raw = ex.get(field, [])
    if not isinstance(raw, list): return {"messages": []}

    tmp = []
    for turn in raw:
        if not isinstance(turn, dict): continue
        # 兼容 role/from
        role_raw = str(turn.get("role", turn.get("from", ""))).strip().lower()
        # 兼容 content/value
        content = turn.get("content", turn.get("value", ""))
        if not isinstance(content, str) or not content.strip(): continue

        if role_raw in ("human", "user"):
            role = "user"
        elif role_raw in ("gpt", "assistant", "bot", "model"):
            role = "assistant"
        elif role_raw == "system":
            role = "system"
        else:
            continue
        tmp.append({"role": role, "content": content.strip()})

    if not tmp: return {"messages": []}

    # 合并处理 system prompt
    sys_prefix_content = [m["content"] for m in tmp if m["role"] == "system"]
    full_system_prompt = (
            system_text + ("\n\n" + "\n\n".join(sys_prefix_content) if sys_prefix_content else "")).strip()
    system_msg = {"role": "system", "content": full_system_prompt}

    # 清理并合并连续同角色消息
    rest = [m for m in tmp if m["role"] in ("user", "assistant")]
    if not rest: return {"messages": []}

    cleaned = []
    for m in rest:
        if cleaned and cleaned[-1]["role"] == m["role"]:
            cleaned[-1]["content"] += "\n\n" + m["content"]
        else:
            cleaned.append(m)

    # 保证对话以 user 开始，以 assistant 结束
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()

    if not cleaned or not any(m["role"] == "assistant" and m["content"] for m in cleaned):
        return {"messages": []}

    # 组合 System + 对话
    messages = [system_msg] + cleaned

    # 再次清洗以防泄露
    messages = _clean_dialog(messages)

    return {"messages": messages or []}


# 下面这个辅助函数 _norm_to_messages_from_conversations 依赖于 _clean_dialog
# 确保 _clean_dialog 也在代码中定义了（您之前的代码里应该有，这里再次确认一下）
def _clean_dialog(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """清理对话，防止 assistant 的回答中泄露了 'User:' 标记。"""
    _LEAKY_USER = re.compile(r"(^|\n)\s*(User|Human)\s*:\s*", re.IGNORECASE)
    cleaned = []
    for m in messages:
        txt = _speaker_strip(m["content"])
        if m["role"] == "assistant":
            leak = _LEAKY_USER.search(txt)
            if leak:
                cut = txt[:leak.start()].strip()
                if len(cut) < 5:
                    return []
                txt = cut
        cleaned.append({"role": m["role"], "content": txt})
    return cleaned


def _has_valid_structure(ex: Dict) -> bool:
    """通用结构检查"""
    msgs = ex.get("messages", [])
    if not msgs: return False
    if len(msgs) == 1 and msgs[0]["role"] == "text": return True  # Pretrain
    return any(m["role"] == "assistant" for m in msgs)  # Chat


def _clean_dialog(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """清理对话，防止 assistant 的回答中泄露了 'User:' 标记。"""
    _LEAKY_USER = re.compile(r"(^|\n)\s*(User|Human)\s*:\s*", re.IGNORECASE)
    cleaned = []
    for m in messages:
        txt = _speaker_strip(m["content"])
        if m["role"] == "assistant":
            leak = _LEAKY_USER.search(txt)
            if leak:
                cut = txt[:leak.start()].strip()
                if len(cut) < 10:
                    return []
                txt = cut
        cleaned.append({"role": m["role"], "content": txt})
    return cleaned

# ==============================================================================
# 2. Tokenization & Labeling (核心逻辑：前缀对比法)
# ==============================================================================

def build_encoder(tokenizer, block_size=2048):
    def tokenize_with_assistant_labels(messages: List[Dict[str, str]]):
        # =================================================================
        # 分支 A: 纯文本知识 (Pre-training 模式, FineWeb/SlimPajama)
        # =================================================================
        if len(messages) == 1 and messages[0]['role'] == 'text':
            text = messages[0]['content']

            # [Qwen Fix] Qwen 2.5 不需要 BOS
            # add_special_tokens=False，因为我们不需要它自动加任何东西，我们手动控制 EOS
            enc = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding=False,
                add_special_tokens=False
            )

            input_ids = enc.input_ids

            # [Qwen Fix] 确保结尾有 EOS
            # Qwen 的 tokenizer.eos_token_id 通常是 <|im_end|> (151645) 或 <|endoftext|> (151643)
            # 对于纯文本续写，通常使用 <|endoftext|>，但使用 tokenizer.eos_token_id 是最安全的兼容写法
            if len(input_ids) > 0 and input_ids[-1] != tokenizer.eos_token_id:
                if len(input_ids) < block_size:
                    input_ids.append(tokenizer.eos_token_id)
                else:
                    # 如果被截断，强制替换最后一个 token 为 EOS
                    input_ids[-1] = tokenizer.eos_token_id

            # Labels = Input_ids (全量 Loss)
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": list(input_ids)
            }

        # =================================================================
        # 分支 B: 指令微调数据 (Chat 模式, ChatML)
        # =================================================================
        try:
            # 1. 对整个对话进行 Tokenize
            # Qwen 的 apply_chat_template 会自动处理 <|im_start|>system...<|im_end|>
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

        if not full_input_ids:
            return None

        # 2. 初始化 Labels
        labels = [-100] * len(full_input_ids)

        # 3. 动态计算 Assistant 回复区间
        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant':
                # 获取 Assistant 之前的对话 (Prefix)
                # add_generation_prompt=True 会生成到 <|im_start|>assistant\n
                prefix_ids = tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    truncation=False,
                    padding=False,
                    add_generation_prompt=True
                )

                # 获取包含当前 Assistant 的对话 (直到 <|im_end|>)
                current_ids = tokenizer.apply_chat_template(
                    messages[:i + 1],
                    tokenize=True,
                    truncation=False,
                    padding=False,
                    add_generation_prompt=False
                )

                start_index = len(prefix_ids)
                end_index = len(current_ids)

                # 填充 Label
                if start_index < len(full_input_ids):
                    valid_end = min(end_index, len(full_input_ids))
                    labels[start_index:valid_end] = full_input_ids[start_index:valid_end]

        # 检查有效性
        if all(L == -100 for L in labels):
            return None

        return {
            "input_ids": full_input_ids,
            "attention_mask": [1] * len(full_input_ids),
            "labels": labels
        }

    def preprocess_batch(examples: Dict[str, List]):
        batch_output = {"input_ids": [], "attention_mask": [], "labels": []}

        for msgs in examples["messages"]:
            if not msgs: continue

            # 判断是否为对话数据
            is_chat = (len(msgs) > 1 or (len(msgs) == 1 and msgs[0]['role'] != 'text'))

            if is_chat:
                # 执行清洗
                msgs = _clean_dialog(msgs)

                # 清洗后有效性检查
                if not msgs: continue
                has_assistant = any(m['role'] == 'assistant' for m in msgs)
                if not has_assistant: continue

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
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 0:
            max_len = int(np.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        def pad(seq: List[int], pad_id: int) -> List[int]:
            return seq + [pad_id] * (max_len - len(seq))

        batch = {}
        batch["input_ids"] = torch.tensor(
            [pad(f["input_ids"], self.tokenizer.pad_token_id) for f in features], dtype=torch.long
        )
        batch["attention_mask"] = torch.tensor(
            [pad(f["attention_mask"], 0) for f in features], dtype=torch.long
        )
        batch["labels"] = torch.tensor(
            [pad(f["labels"], -100) for f in features], dtype=torch.long
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
# 3. 主函数 (针对 Qwen 2.5)
# ==============================================================================

def prepare_qwen_base_datasets(
        repo_id: str,  # "Qwen/Qwen2.5-7B" (Base)
        block_size: int = 4096,
        num_proc: int = 8,
        min_token_len: int = 50,
):
    print(f"Loading tokenizer for Base model '{repo_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    # 1. Pad Token 处理
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)

    # 2. 强制注入 ChatML 模板
    print("Injecting ChatML template for Base model...")
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message %}{{ '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # -------------------------------------------------------------------------
    # 加载数据集 (全能恢复配方)
    # -------------------------------------------------------------------------

    # 1. FineWeb-Edu (基座本色: 40k) -> 占比最高，稳住 PPL
    print("Loading FineWeb-Edu...")
    fw_iter = load_from_disk(f"{LOCAL_DATA_DIR}/fineweb_edu_subset_200k")
    fw_data = list(itertools.islice(fw_iter, 200000))
    ds_fineweb = Dataset.from_list(fw_data)
    ds_fineweb = ds_fineweb.map(_norm_fineweb, num_proc=num_proc, remove_columns=ds_fineweb.column_names)

    # 2. OpenHermes-2.5 (万能胶水: 20k) -> 替代 Alpaca，提升综合能力
    print("Loading OpenHermes-2.5...")
    ds_hermes = load_from_disk(f"{LOCAL_DATA_DIR}/openhermes_2_5")
    # ds_hermes = ds_hermes.select(range(50000))
    ds_hermes = ds_hermes.map(
        lambda ex: _norm_to_messages_from_conversations(ex, "conversations"),
        num_proc=num_proc, remove_columns=ds_hermes.column_names
    )

    # 3. UltraChat_200k (对话专家: 20k) -> 恢复多轮对话流
    print("Loading UltraChat 200k...")
    ds_ultra = load_from_disk(f"{LOCAL_DATA_DIR}/ultrachat_200k")
    # ds_ultra = ds_ultra.select(range(40000))
    ds_ultra = ds_ultra.map(
        lambda ex: _norm_to_messages_from_conversations(ex, "messages"),  # 复用通用处理
        num_proc=num_proc, remove_columns=ds_ultra.column_names
    )

    # 4. Magpie-Qwen (同源对齐: 10k)
    print("Loading Magpie-Qwen2.5-Pro...")
    ds_magpie = load_from_disk(f"{LOCAL_DATA_DIR}/magpie_qwen_2_5_pro")
    ds_magpie = ds_magpie.select(range(300000))
    ds_magpie = ds_magpie.map(_norm_magpie_qwen, num_proc=num_proc, remove_columns=ds_magpie.column_names)

    # 5. 理科组合 (Code + Math: 各 5k = 10k)
    print("Loading Science Mix (Code + Math)...")
    # Code (Glaive)
    ds_code = load_from_disk(f"{LOCAL_DATA_DIR}/glaive_code_v3")
    ds_code = ds_code.select(range(150000))
    ds_code = ds_code.map(_norm_glaive_code, num_proc=num_proc, remove_columns=ds_code.column_names)
    # Math (OpenMath)
    ds_math = load_from_disk(f"{LOCAL_DATA_DIR}/openmath_instruct_2_subset")
    ds_math = ds_math.select(range(150000))
    ds_math = ds_math.map(_norm_openmath, num_proc=num_proc, remove_columns=ds_math.column_names)

    # -------------------------------------------------------------------------
    # 合并
    # -------------------------------------------------------------------------
    print("Merging all datasets...")

    def safe_keep(ds):
        return ds.select_columns(["messages"])

    all_ds = [
        safe_keep(ds_fineweb),
        safe_keep(ds_hermes.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_ultra.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_magpie.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_code.filter(_has_valid_structure, num_proc=num_proc)),
        safe_keep(ds_math.filter(_has_valid_structure, num_proc=num_proc))
    ]

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
#             output_dir=f"qwen_tnn_L{L:02d}_ckpts_layer",
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
#                 output_dir=f"qwen_tnn_ALL_ADTN_after{replaced_ok:02d}",
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
#     tok, train_ds, eval_ds, collator = prepare_qwen_base_datasets(
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
#         start_layer=35,  # 例如 Llama2-7B 为 31
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
#     output_dir_name = f"qwen3_tnn_JOINT_L{layer_str}_1"
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
#
#
# def main():
#     # 1) 数据准备
#     tok, train_ds, eval_ds, collator = prepare_qwen_base_datasets(repo_id=REPO)
#
#     # 2) 加载模型
#     # 先加载 config
#     base_cfg_obj = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
#
#     # 加载模型
#     model = AutoModelForCausalLM.from_pretrained(
#         REPO,
#         config=base_cfg_obj,
#         trust_remote_code=True,
#         torch_dtype=torch.float32,  # 初始加载建议 FP32 或根据显存决定
#     )
#     model.config.pad_token_id = tok.pad_token_id
#     model.config.use_cache = False
#     model.to("cuda")
#
#     device = next(model.parameters()).device
#     head_dim = getattr(base_cfg_obj, "head_dim", base_cfg_obj.hidden_size // base_cfg_obj.num_attention_heads)
#
#     # 3) 准备配置对象 (用于传递给 ADTN 构建函数)
#     # 这里重新加载一次 config 确保纯净，或者直接用上面的 base_cfg_obj
#     cfg_for_tnn = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
#
#     # 4) 切换到运行精度 (BF16) 并冻结所有参数
#     model.to(dtype=torch.bfloat16)
#     set_requires_grad_only(model, [])
#
#     # 5) 定义你想要压缩并联合训练的层
#     # 例如：只压缩最后 3 层进行联合训练
#     layers_to_compress = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
#     # 或者全部层： layers_to_compress = list(range(31, -1, -1))
#
#     # 6) 启动新的训练流程
#     train_specific_layers_jointly(
#         model, tok, cfg_for_tnn,
#         device=device, run_dtype=torch.bfloat16, head_dim=head_dim,
#         train_ds=train_ds, collator=collator,
#         baseline=MMLU_BASELINE,
#         target_layers=layers_to_compress  # <--- 传入列表
#     )
#
#
# if __name__ == "__main__":
#     main()


# # =====================================================
# # 新增：自定义“每隔N步衰减”的回调
# # =====================================================
# class StepDecayLRCallback(TrainerCallback):
#     """
#     每隔 `decay_every_n_steps` 步，将学习率乘以 `gamma`。
#     """
#
#     def __init__(self, decay_every_n_steps: int, gamma: float = 0.5):
#         if decay_every_n_steps <= 0:
#             raise ValueError("`decay_every_n_steps` must be positive.")
#         self.decay_every_n_steps = decay_every_n_steps
#         self.gamma = gamma
#         self.last_decay_step = 0  # 记录上一次衰减的步数
#
#     def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
#         # 1. 检查是否达到下一个衰减点
#         # 例如：每3000步 -> 检查 global_step 是否是 3000, 6000, 9000...
#         # 并且确保在同一“衰减窗口”内只执行一次
#
#         # 计算当前 step 是否跨过了一个新的 milestone
#         current_milestone = (state.global_step // self.decay_every_n_steps) * self.decay_every_n_steps
#
#         # 确保 milestone 是有效的 (>= 周期)，且比上次衰减的点新
#         if current_milestone > self.last_decay_step and current_milestone >= self.decay_every_n_steps:
#             print(f"\n[StepDecayLR] Reached step {state.global_step}. Decaying LR by {self.gamma}...")
#
#             for param_group in optimizer.param_groups:
#                 if 'lr' in param_group:
#                     old_lr = param_group['lr']
#                     param_group['lr'] = old_lr * self.gamma
#                     print(f"  - Group LR: {old_lr:.2e} -> {param_group['lr']:.2e}")
#
#                 # 同时更新 initial_lr 以防被调度器重置
#                 if 'initial_lr' in param_group:
#                     param_group['initial_lr'] *= self.gamma
#
#             # 更新“上一次衰减”的记录，防止在 3001, 3002... 步重复衰减
#             self.last_decay_step = current_milestone


class WSDScheduleCallback(TrainerCallback):
    """
    WSD (Warmup-Stable-Decay) 调度策略回调。

    逻辑：
    1. Warmup: 由 Trainer 自带的 warmup_steps 处理。
    2. Stable: 在 warmup 之后，保持恒定学习率，直到达到 `decay_start_ratio` (例如总步数的 80% 或 90%)。
    3. Decay: 从 decay 点开始，线性(或指数)衰减直到训练结束。
    """

    def __init__(self, decay_start_ratio: float = 0.85, decay_type: str = "linear"):
        """
        Args:
            decay_start_ratio: 从总步数的百分之多少开始衰减 (推荐 0.8 到 0.9，即最后 10%-20% 进行衰减)。
            decay_type: 'linear' (推荐) 或 'cosine'。
        """
        self.decay_start_ratio = decay_start_ratio
        self.decay_type = decay_type
        self._has_logged_decay_start = False

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        # 获取当前步数和总步数
        cur_step = state.global_step
        max_steps = state.max_steps

        # 保护：如果 max_steps 设置不合理，不做衰减
        if max_steps <= 0:
            return

        # 计算衰减开始的步数
        decay_start_step = int(max_steps * self.decay_start_ratio)

        # 【阶段 2: Stable】如果还没到衰减点，什么都不做
        # (Trainer 的 constant_with_warmup 会负责保持 LR 恒定)
        if cur_step < decay_start_step:
            return

        # 【阶段 3: Decay】进入衰减期
        if not self._has_logged_decay_start:
            print(f"\n[WSD] Reached step {cur_step} ({self.decay_start_ratio * 100}%). Starting Decay phase...")
            self._has_logged_decay_start = True

        # 计算当前在衰减阶段的进度 (0.0 -> 1.0)
        # progress = (当前步 - 开始衰减步) / (总步 - 开始衰减步)
        decay_steps_total = max_steps - decay_start_step
        if decay_steps_total <= 0:
            return  # 防止除以0

        progress = (cur_step - decay_start_step) / decay_steps_total
        progress = min(max(progress, 0.0), 1.0)

        # 计算衰减系数 factor (从 1.0 降到 0.0)
        if self.decay_type == "linear":
            factor = 1.0 - progress
        elif self.decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0 - progress  # 默认线性

        # 应用衰减
        # 注意：这里我们假设 optimizer 中的 initial_lr 是峰值学习率
        # Trainer 的 constant schedule 会试图在每一步重置 LR 为 base_lr，
        # 所以我们需要直接修改 param_group['lr'] 覆盖它。

        for param_group in optimizer.param_groups:
            # 以此为基准（通常是 args.learning_rate）
            base_lr = param_group.get('initial_lr', args.learning_rate)

            # WSD 建议最后降到非常低，甚至接近 0
            target_lr = base_lr * factor

            # 设置当前 LR
            param_group['lr'] = target_lr

        # 调试日志 (可选，每隔一定步数打印)
        if cur_step % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [WSD Decay] Step {cur_step}/{max_steps} | Factor: {factor:.4f} | LR: {current_lr:.2e}")


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

    # =====================================================
    # 第一阶段：智能检查 + (批量替换 or 参数收集)
    # =====================================================
    for L in target_layers:
        print(f"\n[Phase 1] 正在深度扫描第 {L} 层结构...")

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

        # 分支处理
        if found_adtn_in_this_layer:
            print(f"  ✅ [检测] 第 {L} 层已包含 ADTN 结构，复用参数。")
            for p in existing_params_in_layer:
                p.requires_grad = True
            all_new_adtn_params.extend(existing_params_in_layer)

        else:
            print(f"  🚀 [检测] 第 {L} 层为纯 Linear 结构，执行替换。")
            layer_cfg = build_cfg_for_single_layer(base_cfg, L)
            new_params = replace_linear_with_tnn_from_config(
                model, layer_cfg,
                head_dim=head_dim, device=device, run_dtype=run_dtype,
                d_dim=2, warm_start_weight=True,
                FIT_STEPS=WARM_START_STEPS, FIT_LR=WARM_START_LR, verbose_fit=True,
            )

            if new_params:
                all_new_adtn_params.extend(new_params)
            else:
                print(f"[warn] 第 {L} 层替换后未发现可训练参数。")

    if not all_new_adtn_params:
        print("❌ 未发现任何可训练的 ADTN 参数，退出训练。")
        return

    # =====================================================
    # 第二阶段：设置梯度与精度
    # =====================================================
    print(f"\n[Phase 2] 准备整体微调，参数总量: {len(all_new_adtn_params)}")

    set_requires_grad_only(model, all_new_adtn_params)
    for p in all_new_adtn_params:
        p.data = p.data.to(torch.float32)

    # =====================================================
    # 第三阶段：整体恢复训练
    # =====================================================
    layer_str = "_".join(map(str, target_layers[:3]))
    if len(target_layers) > 3: layer_str += "_etc"
    output_dir_name = f"qwen3_tnn_JOINT_L{layer_str}"

    # 1. 设置合理的训练总步数 (WSD 强依赖此参数)
    # 假设你的数据集大小和 Batch Size 已知，你应该计算出大概需要训练多少步。
    # 比如训练 1 个 Epoch，或者固定跑 5000 步。
    # 这里举例设为 5000，请根据实际情况修改！
    REAL_MAX_STEPS = 20000

    # 2. 【关键修改】使用 WSD Callback
    # decay_start_ratio=0.85 意味着前 4250 步保持 5e-5，最后 750 步线性降到 0
    wsd_callback = WSDScheduleCallback(decay_start_ratio=0.85, decay_type="linear")

    from transformers import TrainingArguments, Trainer
    args = TrainingArguments(
        output_dir=output_dir_name,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=4e-5,

        # 【关键】必须设置确定的总步数，不能是随意的大数字
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
        optim="adamw_torch_fused",
        disable_tqdm=True,

        # 【关键】保持使用 constant_with_warmup
        # 配合 WSD Callback：
        # 1. Warmup 阶段 (前100步): 线性上升
        # 2. Stable 阶段 (100步 ~ 4250步): Constant 保持不变
        # 3. Decay 阶段 (4250步 ~ 结束): Callback 介入强制降低 LR
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=200
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        # 替换原来的 callback
        callbacks=[wsd_callback, SanitizeConfigBeforeSave(), MMLUEvalUntilBaseline(tok=tok, baseline=baseline)]
    )

    sanitize_config_inplace(model.config)

    print(f"[hook] Start Joint Training... Baseline={baseline}")
    print(f"[hook] LR Strategy: WSD (Warmup -> Stable until 85% -> Linear Decay)")

    trainer.train()
    print(f"================  训练完成  ==================")


def main():
    # 1) 数据准备
    tok, train_ds, eval_ds, collator = prepare_qwen_base_datasets(repo_id=REPO)

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
    layers_to_compress = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24]
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
