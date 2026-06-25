# ====== 新增（必须在 import torch 之前）======
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 可选，让编号更直观
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 改成你想用的服务器物理卡号，例如只用第3号卡
# 之后这张卡在程序里会变成 cuda:0
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

CKPT_DIR  = "llama3_tnn_JOINT_L31_30_29_etc/checkpoint-4000_off=3"   # 你的 checkpoint 目录
# CKPT_DIR = "./model/llama/Llama-3.1-8B-Instruct"                  # 放着 modeling_tnn.py / configuration_tnn.py 的本地目录

from accelerate import Accelerator
from datasets import load_dataset
from evaluate import load
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, disable=True)  # 全局禁用 tqdm

import json, re, shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from safetensors.torch import safe_open

# === 内部工具 ===
_ATTn_PAT = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(bricks|bias)$")
_MLP_PAT  = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.(bricks|bias)$")

def _collect_keys_by_file(weight_map: Dict[str, str]) -> Dict[str, List[str]]:
    files: Dict[str, List[str]] = {}
    for key, fname in weight_map.items():
        if _ATTn_PAT.match(key) or _MLP_PAT.match(key):
            files.setdefault(fname, []).append(key)
    return files

def _shape0_from_file(ckpt_dir: str, fname: str, keys: List[str]) -> Dict[str, int]:
    """打开单个分片，返回 {key: shape0}（仅对 bricks/bias 提供 shape0）"""
    out: Dict[str, int] = {}
    fp = str(Path(ckpt_dir) / fname)
    with safe_open(fp, framework="pt") as f:
        present = set(f.keys())
        for k in keys:
            if k in present:
                try:
                    out[k] = f.get_tensor(k).shape[0] - 1
                except Exception:
                    pass
    return out

def _g2s(layer: int, kind: str, offset: int) -> str:
    # offset==1 时省略
    return f"{layer}:{kind}" if offset == 1 else f"{layer}:{kind}:{int(offset)}"

def _infer_offsets_lists(ckpt_dir: str) -> Tuple[List[str], List[str]]:
    """生成带 offset 的 tnn_attn_gates / tnn_mlp_gates 列表"""
    idx_path = Path(ckpt_dir) / "model.safetensors.index.json"
    idx = json.load(open(idx_path, "r", encoding="utf-8"))
    weight_map = idx["weight_map"]

    # 1) 将所有 bricks/bias key 按分片归类
    file2keys = _collect_keys_by_file(weight_map)

    # 2) 扫描所有分片，取各 key 的 shape[0]
    shape0_cache: Dict[str, int] = {}
    for fname, keys in file2keys.items():
        shape0_cache.update(_shape0_from_file(ckpt_dir, fname, keys))

    # 3) 聚合到 gate 粒度（优先 bricks，其次 bias）
    attn_off: Dict[Tuple[int, str], int] = {}
    mlp_off:  Dict[Tuple[int, str], int] = {}

    # 先把 bricks 记录下来
    for key in weight_map:
        m = _ATTn_PAT.match(key)
        if m and m.group(3) == "bricks":
            li, kind = int(m.group(1)), m.group(2)
            off = shape0_cache.get(key, 1)
            attn_off[(li, kind)] = int(off)

        m2 = _MLP_PAT.match(key)
        if m2 and m2.group(3) == "bricks":
            li, kind = int(m2.group(1)), m2.group(2)
            off = shape0_cache.get(key, 1)
            mlp_off[(li, kind)] = int(off)

    # 再用 bias 补齐缺失者
    for key in weight_map:
        m = _ATTn_PAT.match(key)
        if m and m.group(3) == "bias":
            li, kind = int(m.group(1)), m.group(2)
            if (li, kind) not in attn_off:
                off = shape0_cache.get(key, 1)
                attn_off[(li, kind)] = int(off)

        m2 = _MLP_PAT.match(key)
        if m2 and m2.group(3) == "bias":
            li, kind = int(m2.group(1)), m2.group(2)
            if (li, kind) not in mlp_off:
                off = shape0_cache.get(key, 1)
                mlp_off[(li, kind)] = int(off)

    # 4) 生成排序后的列表（按层号、kind 顺序）
    attn_order = {"q": 0, "k": 1, "v": 2, "o": 3}
    mlp_order  = {"gate": 0, "up": 1, "down": 2}

    attn_list = [
        _g2s(li, kind, attn_off[(li, kind)])
        for (li, kind) in sorted(attn_off.keys(), key=lambda x: (x[0], attn_order[x[1]]))
    ]
    mlp_list = [
        _g2s(li, kind, mlp_off[(li, kind)])
        for (li, kind) in sorted(mlp_off.keys(), key=lambda x: (x[0], mlp_order[x[1]]))
    ]
    return attn_list, mlp_list

# === 公开 API：写回 config ===
def patch_config_with_gates_and_automap(ckpt_dir: str, *, default_adtn_gate_offset: Optional[int] = 1):
    """
    - 写入 tnn_attn_gates / tnn_mlp_gates，元素为 '层:门[:offset]'（offset==1 时省略）
    - 可选写入 adtn_gate_offset（当 default_adtn_gate_offset 非 1 时才写入；为 1 可不写）
    - 设置 architectures / auto_map
    - 可选复制源码到 ckpt 目录
    """
    ckpt = Path(ckpt_dir)
    cfg_path = ckpt / "config.json"
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))

    # 1) 从 safetensors 推断 gate + offset
    attn_list, mlp_list = _infer_offsets_lists(ckpt_dir)
    cfg["tnn_attn_gates"] = attn_list
    cfg["tnn_mlp_gates"]  = mlp_list

    # ====================   在这里插入新代码块   ====================
    print("正在生成 _adtn_gate_offset_map...")

    adtn_gate_offset_map = {}

    # 合并两个列表一起处理
    all_gates = attn_list + mlp_list

    for gate_string in all_gates:
        parts = gate_string.split(':')

        layer_index = int(parts[0])
        gate_type = parts[1]

        # 格式化键，例如: "(20, 'q')"
        key = f"({layer_index}, '{gate_type}')"

        # 检查是否有显式的 offset
        if len(parts) == 3:
            # 有 offset，值为数字
            offset_value = int(parts[2])
        else:
            # 没有 offset，值为 null (在 Python 中是 None)
            offset_value = None

        adtn_gate_offset_map[key] = offset_value

    # 将最终生成的 map 写入 cfg 字典
    # 请注意字段名是 "_adtn_gate_offset_map"
    cfg["_adtn_gate_offset_map"] = adtn_gate_offset_map

    # 之前那两个字段如果不需要了，可以确保它们被删除
    cfg.pop("_tnn_attn_selected_map", None)
    cfg.pop("_tnn_mlp_selected_map", None)

    print("生成完成。")
    # ========================   新代码块结束   ========================

    # 2) 全局默认 gate offset（为 1 可不写）
    if default_adtn_gate_offset is not None and int(default_adtn_gate_offset) != 1:
        cfg["adtn_gate_offset"] = int(default_adtn_gate_offset)
    else:
        # 如果已有且是 1，可以删除以保持简洁
        if "adtn_gate_offset" in cfg and int(cfg["adtn_gate_offset"]) == 1:
            cfg.pop("adtn_gate_offset", None)

    # 3) Auto* 绑定
    # cfg.setdefault("architectures", ["Qwen3ForCausalLM"])  # ← 按你的类名改
    # cfg["auto_map"] = {
    #     "AutoConfig": "configuration_tnn.Qwen3Config",       # ← 按你的配置类名改
    #     "AutoModelForCausalLM": "modeling_tnn.Qwen3ForCausalLM",  # ← 按你的模型类名改
    # }

    cfg.setdefault("architectures", ["LlamaForCausalLM"])  # ← 按你的类名改
    cfg["auto_map"] = {
        "AutoConfig": "configuration_llama_tnn.LlamaConfig",       # ← 按你的配置类名改
        "AutoModelForCausalLM": "modeling_llama_tnn.LlamaForCausalLM",  # ← 按你的模型类名改
    }

    # 4) 写回
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    #
    # # 5) （可选）把源码拷到 checkpoint 目录，方便直接 Auto* 加载
    # if base_repo is not None:
    #     for fn in ("modeling_tnn.py", "configuration_tnn.py"):
    #         shutil.copy(Path(base_repo) / fn, ckpt / fn)


def load_for_inference(ckpt_dir: str):
    tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR,
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        # load_in_4bit=True,  # ← 关键：强制把权重都 cast 到同一 dtype
        low_cpu_mem_usage=True,
        use_cache=True
    )
    # model.to(dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    # 设为评估模式
    model.eval()
    return tok, model


# MMLU 所有子集的列表
MMLU_SUBSETS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]


# ---------- 评测数据 ----------
# ---------- 5-shot 评估代码 ----------
# ---------- 评测数据 ----------
# ---------- 5-shot 评估代码 ----------

# 新增一个辅助函数，用于格式化单个样本，使其成为提示的一部分
def format_example(example, include_answer=True):
    """将一个 MMLU 样本格式化为字符串。"""
    prompt = f"Question: {example['question']}\n"
    choices = example['choices']
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer:"

    if include_answer:
        # 将数字答案（如 2）转换为字母（如 C）
        answer_char = chr(65 + int(example['answer']))
        prompt += f" {answer_char}\n\n"  # 两个换行符用于清晰地分隔样本

    return prompt


@torch.no_grad()
def evaluate_model_5shot(model, tokenizer, model_name='ADTN'):
    import os, re, torch
    from accelerate import Accelerator
    from datasets import load_from_disk
    from tqdm import tqdm

    DATASET_DIR = "./mmlu_data"  # 确保这个目录是用新版 datasets 库生成的
    assert os.path.isdir(DATASET_DIR), f"本地数据集目录不存在: {DATASET_DIR}"

    # 载入整个数据集字典，我们需要 'test' 和 'dev' 两个部分
    ds_all = load_from_disk(DATASET_DIR)
    test_all = ds_all["test"]
    dev_all = ds_all["dev"]  # <<<<<<< 新增：加载 dev 集用于 few-shot 示例

    MMLU_SUBSETS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    def simple_accuracy(preds, refs):
        n = len(refs)
        if n == 0: return 0.0
        return sum(int(p) == int(r) for p, r in zip(preds, refs)) / n

    accel = Accelerator()
    model = accel.prepare(model)
    model.eval()

    choice_pat = re.compile(r'[A-Da-d]')
    print(f"\n--- 正在进行 5-shot 评估: {model_name} ---")
    total_accuracies = []

    for subset_name in MMLU_SUBSETS:
        # <<<<<<< 核心改动：为每个子集准备 5-shot 示例 >>>>>>>
        # 1. 从 dev 集中筛选出当前子集的样本
        dev_subset = dev_all.filter(lambda ex: ex["subject"] == subset_name)

        # 2. 从 test 集中筛选出当前子集的样本 (与之前相同)
        test_subset = test_all.filter(lambda ex: ex["subject"] == subset_name)

        if len(test_subset) == 0:
            print(f"子集 {subset_name} 在 test 集中为空，跳过。")
            continue

        # 3. 构建 5-shot 的提示前缀 (prompt prefix)
        k = 5
        # 如果 dev 集样本不足5个，则使用所有可用的 dev 样本
        num_few_shot = min(k, len(dev_subset))

        few_shot_prompt = ""
        if num_few_shot > 0:
            # .select() 是从数据集中高效选择样本的方法
            few_shot_examples = dev_subset.select(range(num_few_shot))
            for example in few_shot_examples:
                few_shot_prompt += format_example(example, include_answer=True)

        predictions, references = [], []
        print(f"\n评估子集: {subset_name} ({len(test_subset)}个样本, 使用{num_few_shot}-shot)")

        for example in tqdm(test_subset):
            # 4. 构建最终的完整提示
            # 这是测试问题，不包含答案
            main_prompt = format_example(example, include_answer=False)
            # 将 5-shot 示例和测试问题拼接起来
            prompt = few_shot_prompt + main_prompt

            inputs = tokenizer(prompt, return_tensors='pt').to(accel.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.6,
                top_p=0.95
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = text.split("Answer:")[-1].strip()

            m = choice_pat.search(generated_text)
            prediction_char = m.group(0).upper() if m else None

            label_map = {chr(65 + i): i for i in range(len(example['choices']))}
            predicted_label = label_map.get(prediction_char, -1)

            predictions.append(predicted_label)
            references.append(int(example['answer']))

        acc = simple_accuracy(predictions, references)
        total_accuracies.append(acc)
        print(f"子集 {subset_name} 准确率: {acc:.4f}")

    if total_accuracies:
        avg = sum(total_accuracies) / len(total_accuracies)
        print(f"\n模型 {model_name} 在 MMLU 上的 5-shot 平均准确率: {avg:.4f}")
        return avg
    return 0.0


# @torch.no_grad()
# def evaluate_model(model, tokenizer, model_name):
#     import os, re, torch
#     from accelerate import Accelerator
#     from datasets import load_from_disk
#     from tqdm import tqdm
#
#     # 你保存 "all" 的相对路径
#     DATASET_DIR = "./mmlu_data"
#     assert os.path.isdir(DATASET_DIR), f"本地数据集目录不存在: {DATASET_DIR}"
#
#     # 载入一次 "all" 数据（包含 test 集，字段里有 'subject'）
#     ds_all = load_from_disk(DATASET_DIR)
#     test_all = ds_all["test"]
#
#     # 子集列表
#     MMLU_SUBSETS = [
#         'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
#         'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
#         'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
#         'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
#         'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
#         'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
#         'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
#         'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
#         'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
#         'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
#         'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
#         'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
#         'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
#         'sociology', 'us_foreign_policy', 'virology', 'world_religions'
#     ]
#
#     # 本地简单准确率
#     def simple_accuracy(preds, refs):
#         n = len(refs)
#         if n == 0: return 0.0
#         return sum(int(p) == int(r) for p, r in zip(preds, refs)) / n
#
#     accel = Accelerator()
#     model = accel.prepare(model)
#     model.eval()
#
#     # 解析答案时更稳：抓取首个 A-D/a-d
#     choice_pat = re.compile(r'[A-Da-d]')
#
#     print(f"\n--- 正在评估模型: {model_name} ---")
#     total_accuracies = []
#
#     # 有的导出版本字段可能不是 'subject'；这里做个兜底
#     has_subject = "subject" in test_all.features
#
#     for subset_name in MMLU_SUBSETS:
#         # 从 all 的 test 里筛出该子集
#         if has_subject:
#             dataset = test_all.filter(lambda ex: ex["subject"] == subset_name)
#         else:
#             # 极少数打包版本用 'category' 字段
#             if "category" in test_all.features:
#                 dataset = test_all.filter(lambda ex: ex["category"] == subset_name)
#             else:
#                 print(f"跳过 {subset_name}: 未找到 subject/category 字段")
#                 continue
#
#         if len(dataset) == 0:
#             print(f"子集 {subset_name} 在本地数据中为空，跳过。")
#             continue
#
#         predictions, references = [], []
#         print(f"\n评估子集: {subset_name}（样本数 {len(dataset)}）")
#
#         for example in tqdm(dataset):
#             # 构造提示
#             prompt = f"Question: {example['question']}\n"
#             for i, choice in enumerate(example['choices']):
#                 prompt += f"{chr(65 + i)}. {choice}\n"
#             prompt += "Answer: "
#
#             inputs = tokenizer(prompt, return_tensors='pt').to(accel.device)
#
#             # --- 核心修改区域 ---
#             outputs = model.generate(
#                 **inputs,
#                 # 1. 核心修复：给模型足够的空间生成答案。10个token通常很安全。
#                 max_new_tokens=10,
#                 # 2. 解决警告：对于确定性评估，只保留 do_sample=False
#                 do_sample=False
#                 # temperature 和 top_p 已被移除
#             )
#             # ---------------------
#
#             # decode的逻辑保持不变
#             text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#             # 从 "Answer:" 后面开始截取模型生成的内容
#             generated_text = text.split("Answer:")[-1].strip()
#
#             # 3. 增加调试打印：观察模型的原始输出，这是最重要的调试手段！
#             # print(f"模型原始输出: '{generated_text}'") # 调试时可以取消这行注释
#
#             # 解析首个 A-D/a-d
#             m = choice_pat.search(generated_text)
#             prediction_char = m.group(0).upper() if m else None
#
#             # 标签映射
#             label_map = {chr(65 + i): i for i in range(len(example['choices']))}
#             predicted_label = label_map.get(prediction_char, -1)
#
#             predictions.append(predicted_label)
#             references.append(int(example['answer']))
#
#         acc = simple_accuracy(predictions, references)
#         total_accuracies.append(acc)
#         print(f"子集 {subset_name} 准确率: {acc:.4f}")
#
#     if total_accuracies:
#         avg = sum(total_accuracies) / len(total_accuracies)
#         print(f"\n模型 {model_name} 在 MMLU 上的平均准确率: {avg:.4f}")
#         return avg
#     return 0.0

# pip install torch transformers datasets
from math import exp
from typing import Dict, Optional, Iterable
from contextlib import nullcontext

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from math import exp, log


def _get_amp_ctx(device: str, enable: bool):
    """Prefer torch.amp.autocast('cuda'); fallback to torch.cuda.amp.autocast()."""
    if not enable:
        return nullcontext()
    if device == "cuda":
        # 新版 PyTorch
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        # 旧版回退
        if hasattr(torch.cuda, "amp"):
            return torch.cuda.amp.autocast()
    return nullcontext()


def _tokenize_in_chunks(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    chunk_chars: int = 3000,
    add_special_tokens: bool = False,
) -> torch.Tensor:
    """
    将长文本按字符分块再分词，避免 tokenizer 报“超出最大长度”的告警。
    """
    ids = []
    for i in range(0, len(text), chunk_chars):
        piece = text[i : i + chunk_chars]
        ids.extend(
            tokenizer(
                piece,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )["input_ids"]
        )
    return torch.tensor(ids, dtype=torch.long)


def build_ids_with_chat_template(
    tokenizer,
    texts: Iterable[str],
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> torch.Tensor:
    """
    将若干段文本按你给的 Qwen3 Chat Template 封装成对话，再转成 token ids 并连接起来。
    兼容 apply_chat_template 返回 list / tensor / BatchEncoding / str 的不同实现。
    """
    def _apply_one(msgs) -> List[int]:
        # 优先尝试：直接 tokenize 成 tensor
        try:
            out = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_tensors="pt",   # 返回 torch.Tensor 或 BatchEncoding
            )
            if hasattr(out, "input_ids"):
                return out.input_ids[0].tolist()
            if isinstance(out, torch.Tensor):
                return out[0].tolist()
            if isinstance(out, dict) and "input_ids" in out:
                ids = out["input_ids"]
                if isinstance(ids, torch.Tensor):
                    return ids[0].tolist()
                return ids[0] if isinstance(ids[0], list) else list(ids[0])
        except TypeError:
            # 某些老版本会在上述参数组合时报 TypeError，继续回退
            pass

        # 次优：让它只返回字符串，再用 tokenizer 正常分词
        s = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,          # 返回字符串
        )
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    all_ids: List[int] = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        # 关键：把每条样本文本作为 assistant 内容（与你的模板一致）
        msgs.append({"role": "Human", "content": t})
        all_ids.extend(_apply_one(msgs))
    return torch.tensor(all_ids, dtype=torch.long)

def compute_wikitext2_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str = "test",
    dataset_repo: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    stride: int = 512,
    max_length: Optional[int] = None,
    add_special_tokens: bool = True,   # 与基线可比：用 False
    device: Optional[str] = None,
    dtype_autocast: bool = False,       # 与基线更可比：默认关
    chunk_chars: int = 3000,
) -> Dict[str, float]:
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("该函数面向因果语言模型 (decoder-only)。")

    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # ---- 加载并拼接原始文本 ----
    ds = load_dataset(dataset_repo, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    text_bytes = len(text.encode("utf-8"))
    text_chars = len(text)  # 注意：含空白与标点

    input_ids = _tokenize_in_chunks(
        tokenizer, text, chunk_chars=chunk_chars, add_special_tokens=False
    ).to(device)

    nll_sum = 0.0
    num_tokens = 0

    # ---- 自动推断上下文窗口 ----
    if max_length is None:
        guesses = [
            getattr(model.config, "n_positions", None),
            getattr(model.config, "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None) if isinstance(getattr(tokenizer, "model_max_length", None), int) else None,
        ]
        max_length = next((g for g in guesses if isinstance(g, int) and g > 0), 2048)

    stride = max(1, min(stride, max_length))

    nll_sum = 0.0
    num_tokens = 0

    # 用 inference_mode 比 no_grad 更快更省内存
    amp_ctx = _get_amp_ctx(device, dtype_autocast)
    embed_dev = model.get_input_embeddings().weight.device

    input_ids = input_ids.to(embed_dev, non_blocking=True)

    with torch.inference_mode():
        for i in range(0, input_ids.size(0), stride):
            begin = max(i + stride - max_length, 0)
            end = min(i + stride, input_ids.size(0))
            trg_len = end - i
            if trg_len <= 0:
                continue

            ids_slice = input_ids[begin:end]
            if ids_slice.device != embed_dev:
                ids_slice = ids_slice.to(embed_dev, non_blocking=True)

            attn = torch.ones((1, ids_slice.size(0)), dtype=torch.long, device=embed_dev)

            target_ids = ids_slice.clone()
            target_ids[:-trg_len] = -100  # 只监督新推进的 trg_len 个 token

            with amp_ctx:
                out = model(
                    input_ids=ids_slice.unsqueeze(0),
                    attention_mask=attn,
                    labels=target_ids.unsqueeze(0),
                    use_cache=False,
                )
                nll_sum += out.loss.item() * trg_len  # ← 累计“总 NLL”（nats）
                num_tokens += trg_len

    nll_avg = nll_sum / max(1, num_tokens)      # 每 token 的平均 NLL（nats/token）
    ppl = float(exp(nll_avg))

    # 同时给出跨 tokenizer 可比的指标
    bpb = (nll_sum / log(2)) / max(1, text_bytes)   # bits per byte
    bpc = (nll_sum / log(2)) / max(1, text_chars)   # bits per character

    return {
        "perplexity": ppl,
        "nll_avg": nll_avg,
        "num_tokens": int(num_tokens),
        "bpb": bpb,
        "bpc": bpc
    }


if __name__ == "__main__":
    patch_config_with_gates_and_automap(CKPT_DIR)
    tok, model = load_for_inference(CKPT_DIR)
    model.to(dtype=torch.bfloat16)
    # print("Loaded with ADTN gates in config:", AutoConfig.from_pretrained(CKPT_DIR, trust_remote_code=True).tnn_attn_gates[:10])

    # model.eval()
    # with torch.no_grad():
    #     for mname, m in model.named_modules():
    #         if "adtn" in mname:
    #             m.to(dtype=torch.bfloat16)  # 只改这个子模块


    # model.to("cuda:0")

    # ！！！参数统计！！！
    def _fmt_gib(x):
        return x / (1024 ** 3)


    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print(f"Total params: {total_params:,}  (~{total_params / 1e9:.2f}B)")
    print(f"Param memory (only weights): {_fmt_gib(param_bytes):.2f} GiB")

    # （可选）更接近真实占用
    try:
        footprint_bytes = model.get_memory_footprint()
        print(f"Runtime footprint (HF): {_fmt_gib(footprint_bytes):.2f} GiB")
    except Exception:
        pass

    # （可选）CUDA 实时显存
    if torch.cuda.is_available():
        dev = next(model.parameters()).device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            print(f"CUDA allocated: {_fmt_gib(torch.cuda.memory_allocated(dev)): .2f} GiB")
            print(f"CUDA reserved : {_fmt_gib(torch.cuda.memory_reserved(dev)) : .2f} GiB")

    # === 新增：非零参数统计（含稀疏度） ===
    with torch.no_grad():
        nonzero = 0
        checked = 0

        for p in model.parameters():
            # 跳过未初始化(meta)权重
            if getattr(p, "is_meta", False) or p.device.type == "meta":
                continue
            n = p.numel()
            checked += n
            # 直接在参数所在设备上计数，避免搬运
            try:
                nonzero += torch.count_nonzero(p).item()
            except Exception:
                # 极端量化/稀疏类型的兜底：先转到同设备的 float32 再数（耗时但稳）
                nonzero += torch.count_nonzero(p.to(dtype=torch.float32)).item()

        zeros = checked - nonzero
        sparsity = zeros / checked if checked else 0.0

    print("\n=== Sparsity Report ===")
    print(f"Non-zero params : {nonzero:,}")
    print(f"Zero params     : {zeros:,}")
    print(f"Sparsity (zero%) : {sparsity * 100:.2f}%")

    # !!!eval_mmlu！！！
    model.config.use_cache = True      # 推理时建议打开 cache

    # --- 2. (新增) 简单的单轮生成测试 ---
    print("\n" + "=" * 50)
    print("开始进行简单的生成测试...")

    # 你的提问
    my_question = "Hi! How are you today?"

    prompt = f"Human: {my_question}\nAssistant:"

    print(f"发送给模型的提示 (Prompt): \n{prompt}")

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # 给足够多的空间来生成一段话
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # 解码并打印结果
    response_text = tok.decode(outputs[0], skip_special_tokens=True)

    print("\n模型的回答:")
    print(response_text)
    print("=" * 50 + "\n")
    # --- 生成测试结束 ---

    import math
    import torch
    import torch.nn as nn

    try:
        from thop import profile, clever_format


        class HFForTHOP(nn.Module):
            """把 HuggingFace CausalLM 包成 THOP 能吃的接口（位置参数 -> 关键字参数）"""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                # input_ids: [B, T] (LongTensor)
                attention_mask = torch.ones_like(input_ids)
                # 用 use_cache=False，避免巨大的返回结构影响统计
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                return out.logits  # 返回一个 Tensor 供 THOP 挂钩


        wrapper = HFForTHOP(model).to(model.device).eval()
        prefill_ids = inputs["input_ids"]  # 直接用你上面生成测试里的 inputs
        B, T = prefill_ids.shape

        with torch.inference_mode():
            macs_prefill, params = profile(wrapper, inputs=(prefill_ids,))
        macs_prefill_str, params_str = clever_format([macs_prefill, params], "%.3f")

        print(f"[THOP] Prefill（B={B}, T={T}）≈ MACs: {macs_prefill_str}, Params: {params_str}")
        # 如需按 FLOPs 口径：
        flops_prefill = 2 * macs_prefill
        print(f"[THOP] Prefill 估算 FLOPs ≈ {flops_prefill:.3e}")
    except Exception as e:
        print(f"[THOP] 统计失败：{e}（可先 pip install thop，再试）")


    # res = compute_wikitext2_perplexity(model, tok, split="test", max_length=4096, stride=256, dtype_autocast=True)
    # print(res)  # {'perplexity': ..., 'nll_avg': ..., 'num_tokens': ...}

    # 评估模型
    accuracy = evaluate_model_5shot(model, tok, 'ADTN_1')

    # # 生成（Qwen3 的 chat 模板）
    # messages = [
    #     {"role": "system", "content": "You are a helpful and friendly AI assistant."},
    #     {"role": "user",   "content": "你好！用你当前的checkpoint来回答这一句。"}
    # ]
    # prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = tok(prompt, return_tensors="pt").to(model.device)
    #
    # with torch.no_grad():
    #     out = model.generate(
    #         **inputs,
    #         max_new_tokens=128,
    #         do_sample=True, temperature=0.7, top_p=0.9,
    #         eos_token_id=tok.eos_token_id,
    #         pad_token_id=tok.pad_token_id,
    #     )
    # print(tok.decode(out[0], skip_special_tokens=False))


