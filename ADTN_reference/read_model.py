import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
from safetensors.numpy import safe_open as np_safe_open

# 尝试引入 torch 后端（用于统计非零参数量时的 bfloat16 兼容）
try:
    import torch
    from safetensors.torch import safe_open as torch_safe_open
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    torch_safe_open = None

# --- 可配置项 ---
# CKPT_DIR = "qwen_tnn_ALL_ADTN_after05_3_/checkpoint-5000"
CKPT_DIR = "llama3_tnn_JOINT_L31_30_29_etc/checkpoint-8000_off=4"
# CKPT_DIR = "../model/Qwen/Qwen3-8B"

COMPUTE_NONZERO = True          # 是否统计非零参数量
BACKEND = "auto"                # auto / torch / numpy
PROGRESS_EVERY = 100            # 进度打印频率；0 关闭
# ----------------


def format_int(n: int) -> str:
    return f"{n:,}"

def load_weight_map(ckpt_dir: str) -> Dict[str, str]:
    idx_path = Path(ckpt_dir) / "model.safetensors.index.json"
    with open(idx_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map")
    if not weight_map:
        raise ValueError("错误: 'weight_map' not found in model.safetensors.index.json")
    return weight_map

def group_by_shard(ckpt_dir: str, weight_map: Dict[str, str]) -> Dict[Path, List[str]]:
    groups: Dict[Path, List[str]] = {}
    base = Path(ckpt_dir)
    for tensor_name, relative_file in weight_map.items():
        shard_path = (base / relative_file).resolve()
        groups.setdefault(shard_path, []).append(tensor_name)
    for shard, names in groups.items():
        names.sort()
    return groups

def _numel_from_shape(shape) -> int:
    numel = 1
    for d in shape:
        numel *= int(d)
    return numel

def _normalize_dtype(dt: str) -> str:
    # 兼容 safetensors 元数据里可能出现的不同风格标记
    if not isinstance(dt, str):
        dt = str(dt)
    return dt.strip().lower()

def count_params_from_checkpoint(
    ckpt_dir: str,
    compute_nonzero: bool = False,
    backend: str = "auto",
    progress_every: int = 0
) -> Tuple[int, int]:
    """
    先用 numpy 的 metadata() 统计总参数量。若某张量在 metadata 中缺少 shape，
    则回退到实际读取该张量的 shape（优先 torch，其次 numpy）。
    之后按 backend 统计非零参数量（如开启）。
    """
    weight_map = load_weight_map(ckpt_dir)
    groups = group_by_shard(ckpt_dir, weight_map)

    total_params = 0
    total_nonzero = 0
    processed_shapes = 0
    dtype_set: Set[str] = set()

    numpy_bf16_ok = hasattr(np, "bfloat16")

    # ---------- 第 1 遍：统计总参数量（shape） ----------
    for shard_path, tensor_names in groups.items():
        if not shard_path.exists():
            raise FileNotFoundError(f"错误: 分片文件未找到: {shard_path}")

        # 用 numpy 后端读取 metadata（不触碰数据）
        with np_safe_open(str(shard_path), framework="np") as np_f:
            try:
                md = np_f.metadata()  # {name: {"dtype": "...","shape":[...]}, ...}
            except TypeError:
                md = getattr(np_f, "metadata", {}) or {}

            file_keys = set(np_f.keys())

            for name in tensor_names:
                if name not in file_keys:
                    raise RuntimeError(f"索引与分片不一致：{name} 不在 {shard_path.name} 中")

                info = (md or {}).get(name, {})
                dt = _normalize_dtype(info.get("dtype", ""))
                shape = tuple(info.get("shape") or [])

                if not shape:
                    # ---- 关键修复：metadata 缺 shape 时，回退读取 shape ----
                    got_shape = False

                    # 优先用 torch（兼容 bf16）
                    if TORCH_AVAILABLE:
                        try:
                            from safetensors.torch import safe_open as torch_safe_open
                            with torch_safe_open(str(shard_path), framework="pt") as t_f:
                                t = t_f.get_tensor(name)
                                shape = tuple(t.shape)
                                got_shape = True
                        except Exception as e:
                            # torch 回退失败，再尝试 numpy
                            got_shape = False

                    if not got_shape:
                        # 若 dtype 是 bf16 且 numpy 不支持，会在这里失败；抛出清晰提示
                        try:
                            arr = np_f.get_tensor(name)
                            shape = tuple(arr.shape)
                            got_shape = True
                        except Exception as e:
                            raise RuntimeError(
                                f"未在元数据中找到 shape，且无法通过当前后端获取: {name} @ {shard_path.name}\n"
                                f"建议：\n"
                                f"  1) 安装 PyTorch：pip install torch safetensors（推荐）\n"
                                f"  2) 或升级 NumPy：pip install -U numpy（以支持 bfloat16）\n"
                                f"原始错误: {e}"
                            )

                # 记录 dtype，用于后续 backend 决策
                if dt:
                    dtype_set.add(dt)

                total_params += _numel_from_shape(shape)
                processed_shapes += 1
                if progress_every and processed_shapes % progress_every == 0 and not compute_nonzero:
                    print(f"[进度] 已统计 {processed_shapes} 个张量的 shape | 当前总参数: {format_int(total_params)}")

    if not compute_nonzero:
        return total_params, 0

    # ---------- 决定用于“非零统计”的后端 ----------
    backend = (backend or "auto").lower()
    if backend == "torch":
        use_torch = True
    elif backend == "numpy":
        use_torch = False
    else:
        # auto：若包含 bf16 且 numpy 不支持，则强制用 torch
        has_bf16 = any(("bfloat16" in dt) or ("bf16" in dt) for dt in dtype_set)
        use_torch = TORCH_AVAILABLE or (has_bf16 and not numpy_bf16_ok)

    if use_torch and not TORCH_AVAILABLE:
        raise RuntimeError("后端选择为 torch，但未安装 PyTorch。请安装后重试。")

    # ---------- 第 2 遍：统计非零参数量 ----------
    processed_nonzero = 0
    if use_torch:
        from safetensors.torch import safe_open as torch_safe_open
        for shard_path, tensor_names in groups.items():
            with torch_safe_open(str(shard_path), framework="pt") as f:
                file_keys = set(f.keys())
                for name in tensor_names:
                    if name not in file_keys:
                        raise RuntimeError(f"索引与分片不一致：{name} 不在 {shard_path.name} 中")
                    t = f.get_tensor(name)
                    total_nonzero += int(torch.count_nonzero(t).item())
                    processed_nonzero += 1
                    if progress_every and processed_nonzero % progress_every == 0:
                        print(f"[进度] 非零统计 {processed_nonzero} 个张量 | 当前 non-zeros: {format_int(total_nonzero)}")
    else:
        # 使用 numpy 统计非零（需 numpy 支持 bf16）
        for shard_path, tensor_names in groups.items():
            with np_safe_open(str(shard_path), framework="np") as f:
                file_keys = set(f.keys())
                for name in tensor_names:
                    if name not in file_keys:
                        raise RuntimeError(f"索引与分片不一致：{name} 不在 {shard_path.name} 中")
                    arr = f.get_tensor(name)
                    total_nonzero += int(np.count_nonzero(arr))
                    processed_nonzero += 1
                    if progress_every and processed_nonzero % progress_every == 0:
                        print(f"[进度] 非零统计 {processed_nonzero} 个张量 | 当前 non-zeros: {format_int(total_nonzero)}")

    return total_params, total_nonzero


def list_tensor_names_from_checkpoint(ckpt_dir: str):
    print(f"--- 开始检查检查点: {ckpt_dir} ---")
    try:
        weight_map = load_weight_map(ckpt_dir)
        all_tensor_names = sorted(list(weight_map.keys()))
        print(f"\n--- 检查点中包含的 {len(all_tensor_names)} 个张量名称如下: ---")
        for name in all_tensor_names:
            print(f"  - {name}")
        print("\n--- 列表结束 ---")
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"读取 index.json 时出错: {e}")


if __name__ == "__main__":
    print(f"--- 统计开始: {CKPT_DIR} ---")
    try:
        total_params, total_nonzero = count_params_from_checkpoint(
            CKPT_DIR,
            compute_nonzero=COMPUTE_NONZERO,
            backend=BACKEND,
            progress_every=PROGRESS_EVERY
        )
        print("\n=== 统计结果 ===")
        print(f"总参数量 (elements): {format_int(total_params)}")
        if COMPUTE_NONZERO:
            print(f"总非零参数量 (non-zeros): {format_int(total_nonzero)}")
            sparsity = 1.0 - (total_nonzero / total_params) if total_params > 0 else 0.0
            density = 1.0 - sparsity
            print(f"稀疏率 (sparsity): {sparsity:.4%}")
            print(f"密度 (density):   {density:.4%}")
        else:
            print("（已关闭非零参数统计，如需开启请将 COMPUTE_NONZERO=True）")
    except Exception as e:
        print(f"统计过程中出现错误: {e}")
