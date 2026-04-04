# gated_adtn_layer.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Literal, Dict

import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .losses import balance_loss
from .gate import gate_factory_config, GateConfig


# -----------------------------
# 0) 小工具
# -----------------------------
def inverse_permu(permu: List[int]) -> List[int]:
    inv = [0] * len(permu)
    for i, p in enumerate(permu):
        inv[p] = i
    return inv

def is_pow_of(n: int, d: int) -> bool:
    if n <= 0:
        return False
    q = round(math.log(n, d))
    return d ** int(q) == n

def q_from_dim(dim: int, d: int) -> int:
    assert is_pow_of(dim, d), f"dim={dim} must be power of d={d}"
    return int(round(math.log(dim, d)))

def choose_k_out(dim_input: int, dim_output: int, d: int, k_in: int) -> tuple[int,int,int]:
    q_in  = int(round(math.log(dim_input, d)))
    q_out = int(round(math.log(dim_output, d)))
    assert d ** q_in == dim_input and d ** q_out == dim_output, "dims must be power of d"

    delta_q = q_out - q_in
    k_out = k_in + delta_q

    if k_out < 0:
        raise ValueError(
            f"Impossible single-step mapping: q_in={q_in}, q_out={q_out}, k_in={k_in} -> k_out={k_out}<0. "
            f"Either reduce k_in, change d/padding, or use multi-step ADTN."
        )

    # sanity
    assert q_out == (q_in - k_in + k_out)
    return q_in, q_out, k_out

def _score_combo(cnt: tc.Tensor, combo: List[int], target: float) -> float:
    """
    cnt: [q_in] 当前覆盖次数 (cpu int/float tensor)
    combo: positions list length=k_in
    target: 期望覆盖次数 mu
    返回：越小越好
    """
    # 选择该 combo 后的 cnt'
    # 用 L2 偏差作为目标：sum_i (cnt_i - target)^2
    # 但为了快，只计算 combo 涉及的维度的变化 + 现有 max gap 的 proxy
    # 简化：直接计算 combo 中 bond 的欠覆盖程度（越欠越优先）
    deficit = 0.0
    for i in combo:
        # 欠覆盖: target - cnt[i]
        deficit += float(target - float(cnt[i].item()))
    # deficit 越大越好，所以 score 取负
    return -deficit

def make_unique_positions(
    q_in: int,
    k_in: int,
    N: int,
    strategy: Literal["sliding", "combinations", "random", "warmup"] = "sliding",
    *,
    seed: int = 0,
    shuffle: bool = False,
    cand_per_round: int = 512,     # 每轮随机候选数
    ensure_coverage: bool = True,  # 是否保证每个 bond 至少出现一次
    warmup_ratio: float = 0.5,
    warmup_stride: int = 1,
) -> Tuple[List[List[int]], int]:
    """
    返回不重复的 positions 列表，以及实际可用的 N_effective。

    规则：
      - 先构造所有可用的不重复 position 组合（取决于 strategy）
      - 如果可用数量 < N：只返回可用的全部
      - 如果可用数量 >= N：返回 N 个（可选 shuffle）
    """
    if N <= 0:
        return [], 0
    if k_in <= 0:
        return ([[]], 1) if N > 0 else ([], 0)
    if q_in < k_in:
        return [], 0

    pool: List[List[int]] = []

    # --------------------------------------------------
    # 1) sliding
    # --------------------------------------------------
    if strategy == "sliding":
        for start in range(q_in - k_in + 1):
            pool.append(list(range(start, start + k_in)))

    # --------------------------------------------------
    # 2) combinations
    # --------------------------------------------------
    elif strategy == "combinations":
        from itertools import combinations
        pool = [list(c) for c in combinations(range(q_in), k_in)]

    # --------------------------------------------------
    # 3) random (balanced random，保留你原本的高级版本)
    # --------------------------------------------------
    elif strategy == "random":
        from itertools import combinations

        all_pos = [list(c) for c in combinations(range(q_in), k_in)]
        if len(all_pos) == 0:
            return [], 0

        g = tc.Generator(device="cpu")
        g.manual_seed(seed)

        remaining = tc.arange(len(all_pos), dtype=tc.int64)
        picked: List[List[int]] = []

        cnt = tc.zeros(q_in, dtype=tc.float32)  # 覆盖计数
        target = (N * k_in) / float(q_in)

        # ---- (a) 先保证 coverage ----
        if ensure_coverage:
            perm0 = tc.randperm(len(all_pos), generator=g).tolist()
            used = set()
            need = set(range(q_in))

            for idx in perm0:
                if not need or len(picked) >= N:
                    break
                combo = all_pos[idx]
                if need.intersection(combo) and idx not in used:
                    picked.append(combo)
                    used.add(idx)
                    for b in combo:
                        cnt[b] += 1
                    need.difference_update(combo)

            if used:
                mask = tc.ones(len(all_pos), dtype=tc.bool)
                for u in used:
                    mask[u] = False
                remaining = tc.nonzero(mask, as_tuple=False).squeeze(-1)

        # ---- (b) 主循环：按 deficit score 补足到 N ----
        while len(picked) < N and remaining.numel() > 0:
            m = remaining.numel()
            c = min(int(cand_per_round), int(m))

            perm = tc.randperm(m, generator=g)[:c]
            cand_idx = remaining.index_select(0, perm).tolist()

            best_j = None
            best_score = None
            for j in cand_idx:
                combo = all_pos[j]
                score = _score_combo(cnt, combo, target)
                if best_score is None or score < best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                break

            combo = all_pos[best_j]
            picked.append(combo)
            for b in combo:
                cnt[b] += 1

            remaining = remaining[remaining != best_j]

        pool = picked

    # --------------------------------------------------
    # 4) warmup
    # --------------------------------------------------
    
    elif strategy == "warmup":
        from itertools import combinations

        # ---- 1) sliding pool（先拿一批 sliding）----
        all_slides = [list(range(start, start + k_in)) for start in range(q_in - k_in + 1)]
        if len(all_slides) == 0:
            return [], 0

        n_slide = int(round(float(N) * float(warmup_ratio)))
        n_slide = max(1, min(n_slide, N, len(all_slides)))

        # stride 选取：i -> (i*stride) % len(all_slides)
        stride = max(1, int(warmup_stride))
        seen = set()
        slides2 = []
        for i in range(len(all_slides) * 2):  # 给点余量
            p = all_slides[(i * stride) % len(all_slides)]
            tp = tuple(p)
            if tp not in seen:
                seen.add(tp)
                slides2.append(p)
            if len(slides2) >= n_slide:
                break

        slides = slides2
        n_slide = len(slides)

        used = {tuple(p) for p in slides}

        # ---- 2) candidates = combinations \ slides ----
        all_pos = [list(c) for c in combinations(range(q_in), k_in)]
        remain_pos = [p for p in all_pos if tuple(p) not in used]

        n_rand = N - n_slide
        if n_rand <= 0:
            pool = slides
        else:
            # ---- 3) 从 remain_pos 里按你原 random(balanced) 逻辑选 n_rand ----
            if len(remain_pos) == 0:
                pool = slides
            else:
                g = tc.Generator(device="cpu")
                g.manual_seed(seed)

                remaining = tc.arange(len(remain_pos), dtype=tc.int64)
                picked: List[List[int]] = []

                cnt = tc.zeros(q_in, dtype=tc.float32)
                target = (n_rand * k_in) / float(q_in)

                used_idx: set[int] = set()
                need = set(range(q_in))

                # (a) coverage（在剩余集合里尽量 cover）
                if ensure_coverage:
                    perm0 = tc.randperm(len(remain_pos), generator=g).tolist()

                    for idx in perm0:
                        if not need or len(picked) >= n_rand:
                            break
                        combo = remain_pos[idx]
                        if need.intersection(combo) and idx not in used_idx:
                            picked.append(combo)
                            used_idx.add(idx)
                            for b in combo:
                                cnt[b] += 1
                            need.difference_update(combo)

                    if used_idx:
                        mask = tc.ones(len(remain_pos), dtype=tc.bool)
                        for u in used_idx:
                            mask[u] = False
                        remaining = tc.nonzero(mask, as_tuple=False).squeeze(-1)

                # (b) deficit score 补足
                while len(picked) < n_rand and remaining.numel() > 0:
                    m = remaining.numel()
                    c = min(int(cand_per_round), int(m))
                    perm = tc.randperm(m, generator=g)[:c]
                    cand_idx = remaining.index_select(0, perm).tolist()

                    best_j = None
                    best_score = None
                    for j in cand_idx:
                        combo = remain_pos[j]
                        score = _score_combo(cnt, combo, target)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_j = j

                    if best_j is None:
                        break

                    combo = remain_pos[best_j]
                    picked.append(combo)
                    for b in combo:
                        cnt[b] += 1
                    remaining = remaining[remaining != best_j]

                pool = slides + picked

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # --------------------------------------------------
    # 统一出口
    # --------------------------------------------------
    if shuffle and len(pool) > 1:
        g = tc.Generator(device="cpu")
        g.manual_seed(seed)
        perm = tc.randperm(len(pool), generator=g).tolist()
        pool = [pool[i] for i in perm]

    N_eff = min(N, len(pool))
    return pool[:N_eff], N_eff


# -----------------------------
# 1) Block：一个“可路由”的参数张量 + wiring(position) + id
# -----------------------------
@dataclass(frozen=True)
class BlockMeta:
    block_id: int
    pos_in: List[int]  # 作用在输入 q_in 指标的哪些位置（长度=k_in）

class TensorBlock(nn.Module):
    """
    一个 block = 参数张量 U + meta(pos/id)。
    U 的形状是 [d]*k_in + [d]*k_out（输入指标在前，输出指标在后）。
    """
    def __init__(
        self,
        *,
        meta: BlockMeta,
        d: int,
        k_in: int,
        k_out: int,
        dtype: tc.dtype = tc.float32,
        device: Optional[tc.device] = None,
        init_std: float = 1e-2,
    ):
        super().__init__()
        self.id = meta.block_id
        self.pos = meta.pos_in
        self.d = int(d)
        self.k_in = int(k_in)
        self.k_out = int(k_out)

        shape = [d] * k_in + [d] * k_out
        U = tc.randn(shape, device=device, dtype=dtype) * init_std
        self.U = nn.Parameter(U)

    def forward(self) -> tc.Tensor:
        return self.U


# -----------------------------
# 2) Apply：把 block 作用到输入的高阶张量上（核心）
# -----------------------------
def apply_block_to_sites(
    x_sites: tc.Tensor,         # [N, d, d, ..., d] (q_in site dims)
    U: tc.Tensor,               # [d]*k_in + [d]*k_out
    pos_in: List[int],          # length = k_in, 0-based over q_in
    *,
    d: int,
    q_in: int,
    k_in: int,
    k_out: int,
) -> tc.Tensor:
    # ---------- sanity ----------
    if x_sites.ndim != 1 + q_in:
        raise ValueError(f"x_sites.ndim={x_sites.ndim}, expect {1+q_in}")
    if len(pos_in) != k_in:
        raise ValueError(f"len(pos_in)={len(pos_in)} != k_in={k_in}")
    if len(set(pos_in)) != len(pos_in):
        raise ValueError(f"pos_in has duplicates: {pos_in}")
    if not all(0 <= p < q_in for p in pos_in):
        raise ValueError(f"pos_in out of range: pos_in={pos_in}, q_in={q_in}")

    contracted = sorted(pos_in)
    #print(f"contracted={contracted}")
    # dims in tensor include batch at dim 0, so site i -> dim i+1
    contract_dims = [p + 1 for p in contracted]
    #print(f"contract_dims={contract_dims}")
    unaffected_dims = [i for i in range(1, q_in + 1) if i not in contract_dims]
    #print(f"unaffected_dims={unaffected_dims}")

    # ---------- 1) move contracted dims to the end ----------
    perm_to_end = [0] + unaffected_dims + contract_dims
    #print(f"perm_to_end={perm_to_end}")
    x_perm = x_sites.permute(perm_to_end)

    # prefix dims = batch + unaffected sites
    prefix_shape = x_perm.shape[: 1 + (q_in - k_in)]
    #print(f"prefix_shape={prefix_shape}")
    prefix_prod = 1
    for s in prefix_shape:
        prefix_prod *= int(s)
    #print(f"prefix_prod={prefix_prod}")

    # ---------- 2) matmul ----------
    x_flat = x_perm.reshape(prefix_prod, d ** k_in)
    U_flat = U.reshape(d ** k_in, d ** k_out)
    y_flat = x_flat @ U_flat

    # ---------- 3) reshape back: [prefix..., new_dims(k_out)] ----------
    y = y_flat.reshape(list(prefix_shape) + [d] * k_out)
    #print(f"y.shape={y.shape}")

    # ---------- 4) permute back (supports NON-CONTIGUOUS pos_in) ----------
    out_q = (q_in - k_in) + k_out   # number of site dims after contraction
    # y currently has dims:
    #   dim0 = batch
    #   dims 1..len(unaffected_dims) = unaffected (in increasing order)
    #   dims ... last k_out dims = new output dims
    perm_back = [0] * (1 + out_q)
    #print(f"perm_back={perm_back}")

    first_contracted = contracted[0]
    #print(f"first_contracted={first_contracted}")
    contracted_set = set(contracted)
    #print(f"contracted_set={contracted_set}")

    unaffected_cursor = 1
    new_cursor = 1 + len(unaffected_dims)
    #print(f"unaffected_cursor={unaffected_cursor}, new_cursor={new_cursor}")

    target_cursor = 1
    inserted = False

    for orig_site in range(q_in):
        if orig_site in contracted_set:
            if (not inserted) and (orig_site == first_contracted):
                # insert new dims here
                for i in range(k_out):
                    perm_back[target_cursor + i] = new_cursor + i
                target_cursor += k_out
                inserted = True
            # skip all contracted sites (they disappear)
            continue
        else:
            perm_back[target_cursor] = unaffected_cursor
            unaffected_cursor += 1
            target_cursor += 1

    # ---------- safety checks ----------
    if target_cursor != 1 + out_q:
        raise RuntimeError(f"perm_back fill mismatch: target_cursor={target_cursor}, expect {1+out_q}")

    # check duplicates / range
    if len(set(perm_back)) != len(perm_back):
        raise RuntimeError(f"perm_back has duplicates: {perm_back}")
    max_dim = y.ndim - 1
    if not all(0 <= p <= max_dim for p in perm_back):
        raise RuntimeError(f"perm_back out of range: perm_back={perm_back}, y.ndim={y.ndim}")

    return y.permute(perm_back)


# -----------------------------
# 3) 主层：MoE 风格的 Gated-ADTN Layer
# -----------------------------
class GatedADTNLayer(nn.Module):
    """
      - blocks: TensorBlock 列表（参数在 blocks[i].U）
      - gate:   复用你已有 MoE gate（输出 probs/mask 或 idx/w）
      - forward: 只执行被选中的 blocks，并加权求和

    关键：BLOCKS = num_blocks（gate 的输出维度 H 必须对齐它）
    """

    def __init__(
        self,
        *,
        dim_input: int,
        dim_output: int,
        num_blocks: int,
        d: int,
        k_in: int,
        gate_config: GateConfig,
        entropy_coeff: float = 0.0,
        pos_strategy: Literal["sliding", "combinations", "random", "warmup"] = "sliding",
        dtype: tc.dtype = tc.float32,
        device: Optional[tc.device] = None,
        init_gamma: float = 0.6,   # 控制初始化幅度：std = 1/(d^k_in)^gamma
        seed: int = 0,
        warmup_ratio: float = 0.5,
        warmup_stride: int = 1,
    ):
        super().__init__()
        self.dim_input = int(dim_input)
        self.dim_output = int(dim_output)
        self.num_blocks = int(num_blocks)
        self.d = int(d)
        self.k_in = int(k_in)
        self.entropy_coeff = float(entropy_coeff)

        # 维度->qubit数（q）
        self.q_in = q_from_dim(self.dim_input, d)
        self.q_out = q_from_dim(self.dim_output, d)

        self.k_out = int(self.k_in + self.q_out - self.q_in)
        if self.k_out < 0:
            raise ValueError(
                f"[MoTN] k_out < 0: k_in={self.k_in}, q_in={self.q_in}, q_out={self.q_out}. "
                f"Choose larger k_in or change d/padding."
            )

        # 生成每个 block 的 positions（只依赖输入侧 q_in & k_in）
        positions, N_eff = make_unique_positions(
            q_in=self.q_in, k_in=self.k_in, N=num_blocks,
            strategy=pos_strategy, seed=seed,
            warmup_ratio=warmup_ratio, warmup_stride=warmup_stride
        )
        if N_eff < num_blocks or pos_strategy == "warmup":
            if pos_strategy == "warmup":
                print(f"[ADTN] warmup mode: using {pos_strategy} way, so the blocks are {N_eff}.")
            else:
                print(f"[ADTN] num_blocks requested={num_blocks}, but only {N_eff} unique positions available (q_in={self.q_in}, k_in={k_in}, strategy={pos_strategy}). Using N_eff={N_eff}.")
        self.num_blocks = N_eff
        
        self.positions: List[List[int]] = positions
        
        # 在 __init__ 里：
        stats = self._coverage_stats(self.positions, self.q_in)
        self.pos_stats = stats  # ✅ 记录下来，外部也能拿到
        assert len(positions) == self.num_blocks

        local_gate_cfg = replace(gate_config, num_experts=self.num_blocks, data_dim=self.dim_input)       
        self.gate = gate_factory_config(local_gate_cfg).to(dtype=tc.float32)

        # 初始化尺度（与你之前 gamma 初始化一致）
        fan_in = (d ** self.k_in) if self.k_in > 0 else 1
        init_std = 1.0 / (fan_in ** float(init_gamma))

        # blocks: 每个 block 持有 Parameter U + meta
        blks: List[TensorBlock] = []
        for i in range(self.num_blocks):
            meta = BlockMeta(block_id=i, pos_in=self.positions[i])
            blks.append(
                TensorBlock(
                    meta=meta, d=d, k_in=self.k_in, k_out=self.k_out,
                    dtype=dtype, device=device, init_std=init_std
                )
            )
        self.blocks = nn.ModuleList(blks)

        # （可选）你也可以给每个 block 一个 bias/scale，这里先留空，保持简单

    def _coverage_stats(self, positions: List[List[int]], q_in: int) -> dict:
            cnt = tc.zeros(q_in, dtype=tc.int32, device="cpu")
            for pos in positions:
                for i in pos:
                    cnt[i] += 1

            cnt_f = cnt.to(tc.float32)
            mean = float(cnt_f.mean().item()) if q_in > 0 else 0.0
            std  = float(cnt_f.std(unbiased=False).item()) if q_in > 0 else 0.0
            cv   = float(std / (mean + 1e-8))

            return {
                "q_in": q_in,
                "N": len(positions),
                "k_in": len(positions[0]) if len(positions) > 0 else 0,
                "mean": mean,
                "std": std,
                "cv": cv,
                "min": int(cnt.min().item()) if q_in > 0 else 0,
                "max": int(cnt.max().item()) if q_in > 0 else 0,
                "cnt": cnt.tolist(),  # 你很可能想直接看这个
            }


    def reshape_in(self, x: tc.Tensor) -> Tuple[tc.Tensor, int, int]:
        # x: [B,S,H] or [N,H] -> x_sites: [N, d, d,...] with q_in site dims
        if x.ndim == 3:
            B, S, H = x.shape
            x2 = x.reshape(B * S, H)
            N = B * S
            return x2.reshape(N, *([self.d] * self.q_in)), B, S
        elif x.ndim == 2:
            N, H = x.shape
            return x.reshape(N, *([self.d] * self.q_in)), 0, 0
        else:
            raise ValueError(f"x must be [B,S,H] or [N,H], got {tuple(x.shape)}")

    def reshape_out(self, y_sites: tc.Tensor, B: int, S: int) -> tc.Tensor:
        # y_sites: [N, d..] with q_out sites -> [B,S,out_dim] or [N,out_dim]
        y = y_sites.reshape(y_sites.shape[0], self.dim_output)
        if B > 0:
            return y.reshape(B, S, self.dim_output)
        return y

    def forward(
        self,
        x: tc.Tensor,
        return_aux: bool = False,
        *,
        probs: Optional[tc.Tensor] = None,   # 可外部注入；否则走 self.gate(x)
        mask: Optional[tc.Tensor] = None,    # 同上
        aux: Optional[Dict] = None,
        idx_w: Optional[Tuple[tc.Tensor, tc.Tensor]] = None,  # 可选保留
        dense: bool = False,
    ):
        """
        MoE 风格：
        - dense=True: 全量 blocks 平均（baseline）
        - 默认：从 self.gate(x) 拿 probs/mask
            - mask is None -> soft 全量加权求和
            - mask not None -> hard top-k dispatch（按 block 分组）
        - idx_w: 可选的显式 top-k (idx,w) 稀疏执行（如果你未来真想走这条路）
        """

        # ---- reshape input to sites ----
        x = x.to(dtype=tc.float32)
        x_sites, B, S = self.reshape_in(x)          # [N, d...q_in]
        N = x_sites.shape[0]

        # ---- get routing ----
        # 允许外部传 probs/mask 或 idx_w（用于debug/ablations），否则默认复用 gate
        if probs is not None and probs.shape[-1] != self.num_blocks:
            raise RuntimeError(f"probs last dim {probs.shape[-1]} != num_blocks {self.num_blocks}")
        if mask is not None and mask.shape[-1] != self.num_blocks:
            raise RuntimeError(f"mask last dim {mask.shape[-1]} != num_blocks {self.num_blocks}")
        if (probs is None and mask is None and idx_w is None) and (not dense):
            probs, mask, aux = self.gate(x)              # probs/mask shape: [B,S,E] or [N,E]
            self.last_aux_dict = aux
        if probs is not None and probs.ndim == 3:
            probs = probs.reshape(N, self.num_blocks)
        if mask is not None and mask.ndim == 3:
            mask = mask.reshape(N, self.num_blocks)

        with tc.no_grad():
            if probs is not None:
                dims = tuple(range(probs.ndim - 1))
                self.last_usage = probs.detach().float().mean(dim=dims)
                self.last_top1  = None if mask is None else mask.detach().float().mean(dim=dims)

        # =========================================================
        # 1) dense baseline
        # =========================================================
        if dense or (probs is None and mask is None and idx_w is None):
            acc = None
            for b in range(self.num_blocks):
                yb = apply_block_to_sites(
                    x_sites, self.blocks[b].U, self.positions[b],
                    d=self.d, q_in=self.q_in, k_in=self.k_in, k_out=self.k_out
                )
                acc = yb if acc is None else (acc + yb)
            y_sites = acc / float(self.num_blocks)
            y = self.reshape_out(y_sites, B, S)
            if return_aux:
                return y, aux
            else:
                return y

        # =========================================================
        # 2) soft routing (mask is None): y = sum_b probs_b * T_b(x)
        #    ——同构你 MoE soft 分支
        # =========================================================
        if idx_w is None and mask is None:
            if probs is None:
                raise ValueError("soft routing requires probs")
            if probs.shape != (N, self.num_blocks):
                raise ValueError(f"probs must be [N,{self.num_blocks}]")

            acc = None
            for b in range(self.num_blocks):
                yb = apply_block_to_sites(
                    x_sites, self.blocks[b].U, self.positions[b],
                    d=self.d, q_in=self.q_in, k_in=self.k_in, k_out=self.k_out
                )
                wb = probs[:, b].view(N, *([1] * (yb.ndim - 1)))
                yb = yb * wb
                acc = yb if acc is None else (acc + yb)

            y = self.reshape_out(acc, B, S)
            if return_aux:
                return y, aux
            else:
                return y

        # =========================================================
        # 3) hard routing (mask not None): MoE-style dispatch by block
        #    ——同构你 MoE hard 分支
        # =========================================================
        if idx_w is None and mask is not None:
            probs_use = probs

            if probs_use.shape != (N, self.num_blocks) or mask.shape != (N, self.num_blocks):
                raise ValueError(f"probs/mask must be [N,{self.num_blocks}]")

            # 预分配输出 sites（注意 q_out 是由 dim_output -> q_out 决定的）
            out_sites = tc.zeros(
                (N, *([self.d] * self.q_out)),
                device=x_sites.device,
                dtype=probs_use.dtype,
            )

            for b in range(self.num_blocks):
                idx = (mask[:, b] > 0).nonzero(as_tuple=False).squeeze(-1)  # (Nb,)
                if idx.numel() == 0:
                    continue

                x_sub = x_sites.index_select(0, idx)  # (Nb, d...)
                y_sub = apply_block_to_sites(
                    x_sub, self.blocks[b].U, self.positions[b],
                    d=self.d, q_in=self.q_in, k_in=self.k_in, k_out=self.k_out
                )  # (Nb, d...q_out)

                w_sub = probs_use.index_select(0, idx)[:, b].view(-1, *([1] * (y_sub.ndim - 1)))
                y_sub = y_sub * w_sub

                # 非 in-place 聚合（像你要求的那样干净）
                out_sites = tc.index_add(out_sites, 0, idx, y_sub)

            y = self.reshape_out(out_sites, B, S)
            if return_aux:
                return y, aux
            else:
                return y

def q_from_dim_pad(dim: int, d: int) -> int:
    q = 0
    p = 1
    while p < dim:
        p *= d
        q += 1
    return q

class MoTNLayer(nn.Module):
    def __init__(self, in_features, out_features, *, d, num_blocks, k_in, gate_config,
                 entropy_coeff=0.0, pos_strategy="sliding", dtype=tc.float16, device=None,
                 init_gamma=0.6, seed=0, warmup_ratio=0.5, warmup_stride=1):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.d = int(d)

        self.q_in  = q_from_dim_pad(self.in_features,  d)
        self.q_out = q_from_dim_pad(self.out_features, d)

        self.in_pad  = (d ** self.q_in)
        self.out_pad = (d ** self.q_out)

        self.core = GatedADTNLayer(
            dim_input=self.in_pad,
            dim_output=self.out_pad,
            num_blocks=num_blocks,
            d=d, k_in=k_in,
            gate_config=gate_config,
            entropy_coeff=entropy_coeff,
            pos_strategy=pos_strategy,
            dtype=dtype, device=device,
            init_gamma=init_gamma, seed=seed,
            warmup_ratio=warmup_ratio, warmup_stride=warmup_stride
        )

    def forward(self, x, **kwargs):
        # x: (..., in_features)
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # pad to in_pad
        if self.in_pad != self.in_features:
            x2 = F.pad(x2, (0, self.in_pad - self.in_features))

        out = self.core(x2, **kwargs)  # (..., out_pad) or (..., out_features) depending on core

        if isinstance(out, tuple):
            y2, aux = out
        else:
            y2, aux = out, None
        
        y2 = y2.reshape(x2.shape[0], -1)

        # slice to out_features
        if self.out_pad != self.out_features:
            y2 = y2[:, :self.out_features]

        y = y2.reshape(*orig_shape[:-1], self.out_features)
        return y, aux
