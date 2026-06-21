import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict

# =================配置区域=================
# 替换为你的模型实际路径
MODEL_PATH = "../model/Qwen/Qwen3-8B"
SAVE_FILENAME = "qwen_layer_stats.png"


# =========================================

def analyze_and_plot():
    print(f"1. 正在加载模型: {MODEL_PATH} ...")
    try:
        # 使用 float16 加载以节省显存，device_map="auto" 自动分配
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print("2. 正在提取参数统计信息（这可能需要几秒钟）...")

    # 数据存储结构: data_store[proj_name]['means'] = {layer_idx: value}
    data_store = defaultdict(lambda: {'means': {}, 'vars': {}})

    # 定义我们关心的层关键字
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

    # 正则表达式：匹配 layers.数字.后面的部分
    # 例如: model.layers.0.self_attn.q_proj.weight -> 提取出 layer_idx=0, proj_name=q_proj
    pattern = re.compile(r"layers\.(\d+)\..*?(" + "|".join(target_keywords) + r")\.weight")

    total_layers = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            match = pattern.search(name)
            if match:
                layer_idx = int(match.group(1))
                proj_name = match.group(2)

                # 更新最大层数记录
                if layer_idx > total_layers:
                    total_layers = layer_idx

                # 计算统计量 (转为 float32 保证精度)
                p_data = param.data.float()
                mean_val = p_data.mean().item()
                var_val = p_data.var().item()

                data_store[proj_name]['means'][layer_idx] = mean_val
                data_store[proj_name]['vars'][layer_idx] = var_val

    print(f"3. 数据提取完毕，共检测到 {total_layers + 1} 层 Transformer Block。正在绘图...")

    # ================= 绘图逻辑 =================
    plt.figure(figsize=(15, 10))

    # 子图1：均值 (Mean)
    plt.subplot(2, 1, 1)
    for proj_name, stats in data_store.items():
        # 将字典转换为按层号排序的列表
        sorted_indices = sorted(stats['means'].keys())
        sorted_values = [stats['means'][i] for i in sorted_indices]
        plt.plot(sorted_indices, sorted_values, label=proj_name, marker='.', markersize=4)

    plt.title(f'Weight Mean per Layer ({MODEL_PATH})', fontsize=14)
    plt.ylabel('Mean Value')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # 子图2：方差 (Variance)
    plt.subplot(2, 1, 2)
    for proj_name, stats in data_store.items():
        sorted_indices = sorted(stats['vars'].keys())
        sorted_values = [stats['vars'][i] for i in sorted_indices]
        plt.plot(sorted_indices, sorted_values, label=proj_name, marker='.', markersize=4)

    plt.title('Weight Variance per Layer', fontsize=14)
    plt.xlabel('Layer Index')
    plt.ylabel('Variance Value')
    # 如果方差差异巨大，可以开启对数坐标
    # plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(os.getcwd(), SAVE_FILENAME)
    plt.savefig(save_path, dpi=300)
    print(f"\n[成功] 图片已保存至: {save_path}")
    print("请查看该图片分析参数趋势。")


if __name__ == "__main__":
    analyze_and_plot()
