import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 可选，让编号更直观
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 改成你想用的服务器物理卡号，例如只用第3号卡


# qwen3_chat_sft_ultra_alpaca_openhermes.py
from typing import List, Dict, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset
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


MMLU_BASELINE = float(os.environ.get("MMLU_BASELINE", "0.46"))
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "4000"))
WARM_START_STEPS = int(os.environ.get("WARM_START_STEPS", "3000"))
WARM_START_LR    = float(os.environ.get("WARM_START_LR", "1e-2"))


# 统一成一个变量
MODEL_DIR = Path("../model/llama/llama2-7B-expanded-model_").resolve()
REPO = str(MODEL_DIR)  # 既可给 HF Transformers 也可拼接本地文件

file = MODEL_DIR / "modeling_llama_tnn.py"
spec = importlib.util.spec_from_file_location("llama_modeling_tnn", file)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

ADTN_Ensemble_Projector = mod.ADTN_Ensemble_Projector

# ------ 你的 config 目录 ------
# MODEL_DIR_ = Path("../model/llama/test/config.json").resolve()
# REPO_ = str(MODEL_DIR_)  # 既可给 HF Transformers 也可拼接本地文件

# ------ Unsloth: 设置 chat template -----


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
from typing import Tuple, Optional, List, Dict, Union

# 正则匹配： Layer:Type[:Offset]
# 例如: "0:q", "15:down:2", "31:o:4"
_ONE_OFFSET_RE = re.compile(
    r"^\s*(\d+)\s*:\s*(q|k|v|o|gate|up|down)\s*(?::\s*(-?\d+)\s*)?$",
    re.IGNORECASE
)


def _parse_gate_spec_oneoffset(spec: str) -> Tuple[int, str, Optional[int]]:
    """
    解析单个配置字符串。
    返回: (layer_index, gate_kind, offset_or_None)
    """
    m = _ONE_OFFSET_RE.match(spec)
    if not m:
        raise ValueError(f"无效的 gate 配置格式：{spec!r}，期望格式 '层号:门类型[:offset]'")

    layer_idx = int(m.group(1))
    kind = m.group(2).lower()

    # 第3组是 offset，如果没写则是 None
    offset_str = m.group(3)
    offset = int(offset_str) if offset_str is not None else None

    return layer_idx, kind, offset


def _gates_from_config(cfg) -> List[Tuple[int, str, Optional[int]]]:
    """
    从 config 中提取需要替换的门列表。
    支持格式: ["0:q", "1:k:2", "31:down:4"]
    返回: List[(layer_idx, kind, offset_or_None)]
    """
    targets = []
    # 假设 config 里有这两个列表，分别存 attn 和 mlp 的替换规则
    for list_name in ("tnn_attn_gates", "tnn_mlp_gates"):
        raw_list = getattr(cfg, list_name, None)
        if not raw_list:
            continue

        for item in raw_list:
            # item 可能是 "0:q" 或 "0:q:2"
            try:
                li, kind, off = _parse_gate_spec_oneoffset(item)
                targets.append((li, kind, off))
            except ValueError as e:
                print(f"[Warning] 解析配置出错: {e}")
                continue
    return targets


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

def _resolve_gate_offset(
    cfg,
    *,
    layer_idx: Optional[int],
    kind: Optional[str],
    explicit_offset: Optional[int] = None,
    default: int = 1
) -> int:
    # 1. 最高优先级：显式传入的参数
    if explicit_offset is not None:
        return int(explicit_offset)

    # 2. 次优先级：从 config 的映射表中查找 (如果存在)
    #    假设 config._adtn_gate_offset_map = {(0, 'q'): 2, (31, 'down'): 4}
    #    这个 map 可以通过解析字符串列表预先生成
    if hasattr(cfg, "_adtn_gate_offset_map") and layer_idx is not None and kind is not None:
        val = cfg._adtn_gate_offset_map.get((int(layer_idx), kind.lower()))
        if val is not None:
            return int(val)

    # 3. 再次优先级：全局配置 cfg.adtn_gate_offset
    if hasattr(cfg, "adtn_gate_offset"):
        return int(cfg.adtn_gate_offset)

    # 4. 兜底默认值
    return int(default)


def make_attn_projector(
        kind: str,  # "q" | "k" | "v" | "o" | "gate" | "up" | "down"
        config,
        *,
        head_dim: int,
        use_tnn: bool,
        d_dim: int = 2,
        layer_idx: Optional[int] = None,
        gate_offset: Optional[int] = None,  # 若传 None，则自动解析
) -> nn.Module:
    # ---- 1. 确定输入输出维度 (逻辑保持不变) ----
    if kind in ("q", "k", "v", "gate", "up"):
        in_dim = config.hidden_size
    elif kind == "o":
        in_dim = config.num_attention_heads * head_dim
    elif kind == "down":
        in_dim = config.intermediate_size
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if kind == "q":
        out_dim = config.num_attention_heads * head_dim
    elif kind in ("k", "v"):
        out_dim = config.num_key_value_heads * head_dim
    elif kind == "o":
        out_dim = config.hidden_size
    elif kind in ("gate", "up"):
        out_dim = config.intermediate_size
    elif kind == "down":
        out_dim = config.hidden_size

    if not use_tnn:
        return nn.Linear(in_dim, out_dim, bias=getattr(config, "attention_bias", False))

    # ---- 2. 确定 Offset ----
    # ★★★ 这里调用解析函数 ★★★
    final_offset = _resolve_gate_offset(
        config,
        layer_idx=layer_idx,
        kind=kind,
        explicit_offset=gate_offset,
        default=3  # 这里可以改默认值
    )

    if final_offset < 0:
        raise ValueError(f"Offset 不能为负数: {final_offset}")

    # ---- 3. 计算 q_in, q_out, k_in, k_out ----
    # 维度检查
    assert (d_dim ** int(round(math.log(in_dim, d_dim)))) == in_dim, f"in_dim={in_dim} 必须是 {d_dim} 的幂"
    assert (d_dim ** int(round(math.log(out_dim, d_dim)))) == out_dim, f"out_dim={out_dim} 必须是 {d_dim} 的幂"

    q_in = int(round(math.log(in_dim, d_dim)))
    q_out = int(round(math.log(out_dim, d_dim)))

    k_in = max(0, q_in - final_offset)
    k_out = max(0, q_out - final_offset)

    return ADTN_Ensemble_Projector(
        q_number=q_in,
        num_input_gate_dims=k_in,
        num_output_gate_dims=k_out,
        d=d_dim,
        gate_offset=final_offset,  # 透传最终决定的 offset
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
        target_layer_idx: int = None,
):
    """
    严格按照 config 规则替换指定层的 Linear。
    """
    if target_layer_idx is None:
        print("[Error] 必须指定 target_layer_idx")
        return []

    # 1. 获取所有规则 [(layer, kind, off), ...]
    raw_rules = _gates_from_config(cfg)

    if not raw_rules:
        return []

    new_params = []

    # 2. 遍历规则，筛选出 **仅适用于当前层** 的规则
    # 我们先收集要执行的操作，避免在循环中混乱
    tasks_to_do = []

    for (rule_layer, kind, off) in raw_rules:
        # 逻辑判断：
        # 如果规则指定了层号 (rule_layer is not None)，必须严格匹配 target_layer_idx
        # 如果规则没指定层号 (rule_layer is None)，则是通用规则，适用于所有层

        if rule_layer is not None:
            if rule_layer != target_layer_idx:
                # 规则指定了层 X，但当前是层 Y -> 跳过
                continue
            else:
                # 规则指定了层 X，当前正是层 X -> 命中！
                reason = f"Explicit Rule {rule_layer}:{kind}"
                tasks_to_do.append((kind, off, reason))
        else:
            # 通用规则 -> 命中！
            reason = f"Generic Rule {kind}"
            tasks_to_do.append((kind, off, reason))

    # 如果当前层没有匹配到任何任务，直接返回
    if not tasks_to_do:
        return []

    if verbose_fit:
        print(f"  > Layer {target_layer_idx} 命中规则: {tasks_to_do}")

    # 3. 执行替换
    for (kind, off, reason) in tasks_to_do:
        layer = model.model.layers[target_layer_idx]
        mod, attr = _get_mod_and_attr(layer, kind)
        cur_linear = getattr(mod, attr)

        # 防止重复替换 (如果已经是 ADTN 则跳过)
        if isinstance(cur_linear, ADTN_Ensemble_Projector):
            if verbose_fit:
                print(f"    [Skip] {kind} 已经是 ADTN，跳过。")
            continue

        # 构造 ADTN
        adtn = make_attn_projector(
            kind, cfg,
            head_dim=head_dim,
            use_tnn=True,
            d_dim=d_dim,
            layer_idx=target_layer_idx,
            gate_offset=off
        )
        # 放到 CPU/GPU 准备拟合
        adtn = adtn.to(device=device, dtype=torch.float32)

        # 暖启动 (MSE)
        if warm_start_weight:
            # 简单推导 dim
            if isinstance(cur_linear, nn.Linear):
                in_dim = cur_linear.in_features
            else:
                if kind in ("q", "k", "v", "gate", "up"):
                    in_dim = cfg.hidden_size
                elif kind == "o":
                    in_dim = cfg.num_attention_heads * head_dim
                elif kind == "down":
                    in_dim = cfg.intermediate_size

            q_number = int(round(math.log(in_dim, d_dim)))

            # Fit
            final_mse = fit_adtn_to_weight_fullmatrix(
                adtn, cur_linear,
                q_number=q_number, d_dim=d_dim, device=device, fit_dtype=torch.float32,
                steps=FIT_STEPS, lr=FIT_LR, verbose=False
            )

            if verbose_fit:
                print(
                    f"[warm-start] layer={target_layer_idx} kind={kind} offset={adtn.gate_offset} mse={final_mse:.6f} | Src: {reason}")

        # 替换到模型中
        adtn = adtn.to(device=device, dtype=run_dtype)
        setattr(mod, attr, adtn)
        for p in adtn.parameters():
            new_params.append(p)

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
SYSTEM_DEFAULT = "You are a helpful assistant."


# ==============================================================================
# 1. 数据清洗与规范化函数 (保持原样，逻辑通用且健壮)
# ==============================================================================

def _speaker_strip(s: str) -> str:
    """去除角色扮演标记（如 'User:'）和多余的空行。"""
    t = re.sub(r"^\s*(User|Assistant|Human)\s*:\s*", "", (s or "").strip(), flags=re.IGNORECASE)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


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


def _norm_to_messages_from_conversations(
        ex: Dict, field: str, system_text: str = SYSTEM_DEFAULT
) -> Dict:
    """通用格式规范化。"""
    raw = ex.get(field, [])
    if not isinstance(raw, list): return {"messages": []}

    tmp = []
    for turn in raw:
        if not isinstance(turn, dict): continue
        role_raw = str(turn.get("role", turn.get("from", ""))).strip().lower()
        content = turn.get("content", turn.get("value", ""))
        if not isinstance(content, str) or not content.strip(): continue

        if role_raw in ("human", "user"):
            role = "user"
        elif role_raw in ("gpt", "assistant", "bot"):
            role = "assistant"
        elif role_raw == "system":
            role = "system"
        else:
            continue
        tmp.append({"role": role, "content": content.strip()})

    if not tmp: return {"messages": []}

    # 合并 System Prompt
    sys_prefix_content = [m["content"] for m in tmp if m["role"] == "system"]
    full_system_prompt = (
            system_text + ("\n\n" + "\n\n".join(sys_prefix_content) if sys_prefix_content else "")).strip()
    system_msg = {"role": "system", "content": full_system_prompt}

    # 合并 User/Assistant
    rest = [m for m in tmp if m["role"] in ("user", "assistant")]
    if not rest: return {"messages": []}

    cleaned = []
    for m in rest:
        if cleaned and cleaned[-1]["role"] == m["role"]:
            cleaned[-1]["content"] += "\n\n" + m["content"]
        else:
            cleaned.append(m)

    # 保证对话流：User -> Assistant -> User ...
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()

    if not cleaned or not any(m["role"] == "assistant" and m["content"] for m in cleaned):
        return {"messages": []}

    messages = [system_msg] + cleaned
    messages = _clean_dialog(messages)
    return {"messages": messages or []}


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


def _has_clean_ua_pair(ex: Dict) -> bool:
    """确保数据有效性。"""
    msgs = ex.get("messages", [])
    if not msgs: return False
    first_non_sys_idx = next((i for i, m in enumerate(msgs) if m["role"] != "system"), -1)
    if first_non_sys_idx == -1 or msgs[first_non_sys_idx]["role"] != "user":
        return False
    return msgs[-1]["role"] == "assistant" and msgs[-1]["content"].strip()


# ==============================================================================
# 2. Tokenization 和 Label 构建 (关键修改)
# ==============================================================================

def build_encoder(tokenizer, block_size=2048):
    def tokenize_with_assistant_labels(messages: List[Dict[str, str]]):
        # --- 情况 A: 纯文本预训练数据 (SlimPajama) ---
        # 我们约定：如果 role 是 'text'，则视为纯文本
        if len(messages) == 1 and messages[0]['role'] == 'text':
            text = messages[0]['content']
            # 加上 BOS 和 EOS
            text = tokenizer.bos_token + text + tokenizer.eos_token

            enc = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding=False,
                add_special_tokens=False  # 我们手动加了
            )
            # 纯文本任务：Labels = Input_ids (预测下一个词)
            return {
                "input_ids": enc.input_ids,
                "attention_mask": enc.attention_mask,
                "labels": list(enc.input_ids)  # 全文计算 Loss
            }

        # --- 情况 B: 对话数据 (Alpaca/Hermes/UltraChat) ---
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
            if msg['role'] == 'assistant':
                # 计算 Assistant 内容的起止
                prefix_messages = messages[:i]
                prefix_ids = tokenizer.apply_chat_template(
                    prefix_messages, tokenize=True, truncation=False, padding=False, add_generation_prompt=True
                )
                current_messages = messages[:i + 1]
                current_ids = tokenizer.apply_chat_template(
                    current_messages, tokenize=True, truncation=False, padding=False, add_generation_prompt=False
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
            item = tokenize_with_assistant_labels(msgs)
            if item:
                batch_output["input_ids"].append(item["input_ids"])
                batch_output["attention_mask"].append(item["attention_mask"])
                batch_output["labels"].append(item["labels"])
        return batch_output

    return preprocess_batch

# ==============================================================================
# 3. Data Collator (保持原样)
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
# 4. 去重工具
# ==============================================================================

def _get_text_from_messages(messages: List[Dict[str, str]]) -> str:
    return "".join((m.get("content", "") or "").strip() for m in messages)


def deduplicate_dataset(dataset, num_proc: int = 4):
    """基于内容的哈希去重"""
    print("Starting de-duplication...")

    # 计算哈希
    dataset_with_hash = dataset.map(
        lambda ex: {"hash": hashlib.md5(_get_text_from_messages(ex["messages"]).encode()).hexdigest()},
        num_proc=num_proc,
        desc="Calculating hashes"
    )

    # 过滤
    seen_hashes = set()

    def is_first_occurrence(example):
        h = example['hash']
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    # 必须单进程执行过滤以保证全局状态正确
    deduplicated_dataset = dataset_with_hash.filter(is_first_occurrence, num_proc=1, desc="Filtering duplicates")
    final_dataset = deduplicated_dataset.remove_columns("hash")

    print(f"De-duplication complete. Original: {len(dataset)} -> New: {len(final_dataset)}")
    return final_dataset


# ==============================================================================
# 5. 主函数 (入口)
# ==============================================================================
def _norm_slim_pajama(ex: Dict) -> Dict:
    """将纯文本包装成 messages 格式，标记 role='text'"""
    return {"messages": [{"role": "text", "content": ex["text"]}]}

def prepare_chat_datasets(
        repo_id: str,
        block_size: int = 2048,
        num_proc: int = 8,
        min_token_len: int = 100,
):
    print(f"Loading tokenizer for '{repo_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 强制设置 Llama 2 Chat Template (防止 tokenizer_config.json 缺失或错误)
    # 这是一个标准的 Llama 2 模板
    llama2_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

    if tokenizer.chat_template is None or "INST" not in str(tokenizer.chat_template):
        print("Applying explicit Llama 2 chat template...")
        tokenizer.chat_template = llama2_template

    # -------------------------------------------------------------------------
    # 1) 加载 Chat 数据集
    # -------------------------------------------------------------------------
    # Alpaca
    print("Loading Alpaca...")
    ds_alpaca = load_dataset("yahma/alpaca-cleaned", split="train").map(
        lambda ex: _norm_from_alpaca(ex), num_proc=num_proc, remove_columns=["instruction", "input", "output"]
    )
    # OpenHermes (取50k)
    print("Loading OpenHermes...")
    ds_hermes = load_dataset("teknium/OpenHermes-2.5", split="train").map(
        lambda ex: _norm_to_messages_from_conversations(ex, "conversations"),
        num_proc=num_proc,
        remove_columns=["conversations", "source", "category", "custom_instruction", "hash", "avatarUrl", "model_name",
                        "title", "topic", "language", "id", "views", "idx"]
    )
    # UltraChat (取30k)
    print("Loading UltraChat...")
    ds_ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").map(
        lambda ex: _norm_to_messages_from_conversations(ex, "messages"),
        num_proc=num_proc, remove_columns=["prompt", "prompt_id", "messages"]
    )

    # -------------------------------------------------------------------------
    # 2) 加载 SlimPajama (Pre-train 数据) - 关键新增
    # -------------------------------------------------------------------------
    print("Loading SlimPajama (Streaming subset)...")
    # SlimPajama 非常大，必须使用 streaming 模式只取一小部分
    # 我们取 5000 条用于正则化
    pajama_iter = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
    pajama_head = list(itertools.islice(pajama_iter, 5000))

    # 转回 Dataset 对象以便处理
    ds_pajama = Dataset.from_list(pajama_head)

    # 规范化：Text -> Messages [{'role': 'text', ...}]
    ds_pajama = ds_pajama.map(
        _norm_slim_pajama,
        num_proc=num_proc,
        remove_columns=ds_pajama.column_names  # 移除 'text', 'meta' 等原始列
    )

    # 3) 合并所有数据
    print("Merging all datasets...")
    # 统一列名检查：确保所有 ds 只有 'messages' 列
    cols_to_keep = ["messages"]

    def safe_select_cols(ds):
        return ds.select_columns(cols_to_keep)

    all_datasets = [
        safe_select_cols(ds_alpaca.filter(_has_clean_ua_pair, num_proc=num_proc)),
        safe_select_cols(ds_hermes.filter(_has_clean_ua_pair, num_proc=num_proc)),
        safe_select_cols(ds_ultra.filter(_has_clean_ua_pair, num_proc=num_proc)),
        safe_select_cols(ds_pajama)  # Pajama 不需要 clean_ua_pair 检查，因为它只有 role='text'
    ]

    mixed_ds = concatenate_datasets(all_datasets).shuffle(seed=42)
    mixed_ds = deduplicate_dataset(mixed_ds, num_proc=num_proc)

    print("Tokenizing...")
    encode_fn = build_encoder(tokenizer, block_size=block_size)
    tokenized_ds = mixed_ds.map(encode_fn, batched=True, batch_size=1000, remove_columns=["messages"],
                                num_proc=num_proc)

    final_ds = tokenized_ds.filter(lambda ex: min_token_len <= len(ex["input_ids"]) <= block_size, num_proc=num_proc)
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

#
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
#             output_dir=f"llama_tnn_L{L:02d}_ckpts_layer_1",
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
#                 output_dir=f"llama_tnn_ALL_ADTN_after{replaced_ok:02d}_1",
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
#     tok, train_ds, eval_ds, collator = prepare_chat_datasets(
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
#     model.to("cuda")
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
#         start_layer=31,  # 例如 Llama2-7B 为 31
#         end_layer=0,
#     )
#
#
# if __name__ == "__main__":
#     main()

def _infer_layers_from_gate_config(cfg) -> List[int]:
    """
    扫描 config 中的 tnn_attn_gates 和 tnn_mlp_gates。
    提取所有显式指定的层号。

    例如:
      ["0:q", "31:v:5", "10:gate"]
      -> 提取出 {0, 31, 10}
      -> 返回 [31, 10, 0] (默认倒序，方便逐层恢复)
    """
    detected_layers = set()

    # 遍历 attn 和 mlp 的配置列表
    for list_name in ("tnn_attn_gates", "tnn_mlp_gates"):
        raw_list = getattr(cfg, list_name, None)
        if not raw_list:
            continue

        for spec in raw_list:
            # spec 类似于 "0:q" 或 "31:v:1"
            try:
                # 使用之前的解析函数
                li, kind, off = _parse_gate_spec_oneoffset(spec)

                # 如果配置中显式写了层号 (li is not None)，则加入集合
                if li is not None:
                    # 还要检查层号是否在合法范围内 (0 ~ num_layers-1)
                    num_layers = getattr(cfg, "num_hidden_layers", 32)
                    if 0 <= li < num_layers:
                        detected_layers.add(li)
                    else:
                        print(f"[Warn] 配置中的层号 {li} 超出了模型范围 (0-{num_layers - 1})，已忽略。")
                else:
                    # 如果配置写的是 "q" (没有层号)，这代表通用规则。
                    # 通用规则本身不指定“哪些层”，它依赖于其他显式指定的层，
                    # 或者如果只有通用规则，通常意味着“所有层”？
                    # 按照您的需求“只读取config字段控制”，如果只有 "q"，
                    # 这里的逻辑默认是不添加任何层的。
                    # 如果您希望写 "q" 就代表 "所有层"，需要在这里特殊处理。
                    #
                    # 现在的逻辑：只处理显式指定的层 (Sparse Specification)
                    pass

            except ValueError:
                print(f"[Warn] 无法解析配置项: {spec}")
                continue

    # 返回排序后的列表（倒序，通常对恢复训练更友好）
    return sorted(list(detected_layers), reverse=True)


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

    # 确保模型在目标 dtype
    model.to(device=device, dtype=run_dtype)

    # 读取评测设置
    try:
        eval_every = int(EVAL_EVERY_STEPS)
    except NameError:
        eval_every = int(os.environ.get("EVAL_EVERY_STEPS", "3000"))

    all_new_adtn_params = []

    print(f"\n>>> 即将处理的层列表: {target_layers}")

    # =====================================================
    # 第一阶段：批量替换 + MSE 暖启动
    # =====================================================
    for L in target_layers:
        print(f"\n[Phase 1] 正在替换并暖启动第 {L} 层...")

        # 【关键修改 1】不再需要 build_cfg_for_single_layer，直接用全局 base_cfg

        # 【关键修改 2】必须传入 target_layer_idx=L
        new_params = replace_linear_with_tnn_from_config(
            model,
            base_cfg,  # <--- 传入全局 config，因为它包含 tnn_attn_gates 列表
            head_dim=head_dim,
            device=device,
            run_dtype=run_dtype,
            d_dim=2,
            warm_start_weight=True,
            FIT_STEPS=WARM_START_STEPS,
            FIT_LR=WARM_START_LR,
            verbose_fit=True,
            target_layer_idx=L  # <--- [这里!] 必须把循环变量 L 传进去
        )

        if new_params:
            all_new_adtn_params.extend(new_params)
        else:
            # 如果这一层在 config 里没被选中（比如只选了 0 和 31，但 L 是 10），这是正常的
            # 但如果你确信 config 里有 10，那说明匹配逻辑还有问题
            pass

    if not all_new_adtn_params:
        print("未发现任何可训练的 ADTN 参数，退出训练。")
        return

    # =====================================================
    # 第二阶段：设置梯度与精度
    # =====================================================
    print(f"\n[Phase 2] 准备整体微调，涉及 ADTN 参数数量: {len(all_new_adtn_params)} 个 Tensor")

    # 1. 冻结全网，只开启刚才替换的所有 ADTN 参数的梯度
    set_requires_grad_only(model, all_new_adtn_params)

    # 2. 将可训练参数提升为 FP32 以保证微调稳定性
    for p in all_new_adtn_params:
        p.data = p.data.to(torch.float32)

    # =====================================================
    # 第三阶段：整体恢复训练 (Joint Recovery)
    # =====================================================
    print(f"\n[Phase 3] 开始整体恢复训练 (Target Layers: {target_layers})")

    # 构造 Output Dir 名字
    layer_str = "_".join(map(str, target_layers[:3]))
    if len(target_layers) > 3: layer_str += "_etc"
    output_dir_name = f"llama_tnn_JOINT_L{layer_str}"

    args = TrainingArguments(
        output_dir=output_dir_name,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        max_steps=12000,
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
        group_by_length=True,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        disable_tqdm=True
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)

    trainer.add_callback(SanitizeConfigBeforeSave())
    trainer.add_callback(MMLUEvalUntilBaseline(tok=tok, baseline=baseline))

    sanitize_config_inplace(model.config)

    print(f"[hook] Start Training... Baseline={baseline}, Check/Save Every={eval_every}")
    trainer.train()
    print(f"================  指定层整体训练完成  ==================")


def main():
    # 1) 数据准备
    tok, train_ds, eval_ds, collator = prepare_chat_datasets(repo_id=REPO)

    # 2) 加载模型配置 & 模型
    base_cfg_obj = AutoConfig.from_pretrained(REPO, trust_remote_code=True)

    print("\n[Debug Check] 当前加载的 Gate 配置:")
    print(f"Attn Gates: {getattr(base_cfg_obj, 'tnn_attn_gates', 'Not Found')}")
    print(f"MLP Gates:  {getattr(base_cfg_obj, 'tnn_mlp_gates', 'Not Found')}")
    # 如果这里打印出来包含了 'q', 'k' 等，说明文件没改对

    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        config=base_cfg_obj,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model.to("cuda")

    device = next(model.parameters()).device
    head_dim = getattr(base_cfg_obj, "head_dim", base_cfg_obj.hidden_size // base_cfg_obj.num_attention_heads)

    # 3) 准备配置对象 (用于传递给 ADTN 构建函数)
    cfg_for_tnn = AutoConfig.from_pretrained(REPO, trust_remote_code=True)

    # 4) 切换到运行精度 (BF16) 并冻结所有参数
    model.to(dtype=torch.bfloat16)
    set_requires_grad_only(model, [])

    # 5) 【修改点】自动从 config 的 gate 定义中推断要压缩的层
    layers_to_compress = _infer_layers_from_gate_config(base_cfg_obj)

    print(f"\n[Config Auto-Detect] 根据 tnn_*_gates 字段，检测到以下层需要压缩:")
    print(f"{layers_to_compress}")
    print(f"共 {len(layers_to_compress)} 层")

    if not layers_to_compress:
        print("未在 config 中检测到任何显式层号 (如 '0:q')，请检查 config.json。退出程序。")
        return

    # 6) 启动训练
    # train_specific_layers_jointly 逻辑保持不变，它会遍历 layers_to_compress
    # 对每一层调用 replace_linear_with_tnn_from_config
    train_specific_layers_jointly(
        model, tok, cfg_for_tnn,
        device=device, run_dtype=torch.bfloat16, head_dim=head_dim,
        train_ds=train_ds, collator=collator,
        baseline=MMLU_BASELINE,
        target_layers=layers_to_compress
    )


if __name__ == "__main__":
    main()
