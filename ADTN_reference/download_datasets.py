import os
# os.environ['HTTPS_PROXY'] = 'http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128'
# os.environ['HTTP_PROXY'] = 'http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128'
# os.environ['HTTP_PROXY'] = '127.0.0.1,10.254.31.0/24,10.254.128.106'
os.environ['HTTPS_PROXY'] = 'http://10.29.1.201:8888'
os.environ['HTTP_PROXY'] = 'http://10.29.1.201:8888'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import itertools
from datasets import load_dataset, Dataset, concatenate_datasets

# =========================================================================
# 配置区域
# =========================================================================

import os
import itertools
from datasets import load_dataset, Dataset

# 本地存储根目录
LOCAL_DATA_DIR = "./local_datasets"

# =========================================================================
# 1. 全量下载列表 (适合数据量较小，<1G 的数据集)
# =========================================================================
FULL_DATASETS = [
    {
        "name": "yahma/alpaca-cleaned",
        "save_name": "alpaca_cleaned",
        "split": "train"
    },
    {
        "name": "teknium/OpenHermes-2.5",
        "save_name": "openhermes_2_5",
        "split": "train"
    },
    {
        "name": "HuggingFaceH4/ultrachat_200k",
        "save_name": "ultrachat_200k",
        "split": "train_sft"
    },
    {
        "name": "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1",
        "save_name": "magpie_llama_3_pro",
        "split": "train"
    },
    {
        "name": "Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered",
        "save_name": "magpie_qwen_2_5_pro",
        "split": "train"
    },
    {
        "name": "glaiveai/glaive-code-assistant-v3",
        "save_name": "glaive_code_v3",
        "split": "train"
    },
]

# =========================================================================
# 2. 流式截取列表 (适合超大数据集，只取头部高质量部分)
# =========================================================================
STREAMING_DATASETS = [
    # --- 原有数据集 ---
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "save_name": "fineweb_edu_subset_200k",
        "split": "train",
        "num_rows": 200000
    },
    {
        "name": "DKYoon/SlimPajama-6B",
        "config": None,
        "save_name": "slimpajama_6b_subset_300k",
        "split": "train",
        "num_rows": 300000
    },
    {
        "name": "nvidia/OpenMathInstruct-2",
        "config": None,
        "save_name": "openmath_instruct_2_subset",
        "split": "train_1M",
        "num_rows": 200000
    },

    # --- 新增：Cosmopedia (合成教科书/百科) ---
    # 这是一个巨大的数据集，包含 stories, stanford, wikihow 等多个子集。
    # 这里我们选择 'stories' 子集，因为它逻辑连贯性强，非常适合恢复知识。
    {
        "name": "HuggingFaceTB/cosmopedia",
        "config": "stories",  # 选择故事/教科书风格子集
        "save_name": "cosmopedia_stories_subset_100k",
        "split": "train",
        "num_rows": 300000  # 截取 10万条足够补充知识密度
    },

    # --- 新增：Infinity-Instruct (智源高质量指令) ---
    # 总量 7M+，我们截取一部分用于增强逻辑和指令遵循
    {
        "name": "BAAI/Infinity-Instruct",
        "config": "0625",  # 指定版本，通常用 '0625' 或 'default'
        "save_name": "infinity_instruct_subset_100k",
        "split": "train",
        "num_rows": 300000  # 截取 10万条
    }
]


# =========================================================================
# 执行逻辑函数
# =========================================================================

def save_full_dataset(item):
    """下载并保存全量数据集"""
    print(f"\n[正在下载] {item['name']} ...")
    save_path = os.path.join(LOCAL_DATA_DIR, item['save_name'])

    if os.path.exists(save_path):
        print(f"  -> 目录已存在，跳过: {save_path}")
        return

    try:
        # 处理 config 参数 (有的数据集需要指定 config name)
        if item.get("config"):
            ds = load_dataset(item['name'], item['config'], split=item['split'])
        else:
            ds = load_dataset(item['name'], split=item['split'])

        print(f"  -> 下载完成，正在保存到本地: {save_path} ...")
        ds.save_to_disk(save_path)
        print(f"  -> 保存成功! (Rows: {len(ds)})")
    except Exception as e:
        print(f"  -> [错误] 下载失败: {e}")


def save_streaming_subset(item):
    """从流式数据集中截取并保存"""
    print(f"\n[正在流式截取] {item['name']} (目标: {item['num_rows']} 条)...")
    save_path = os.path.join(LOCAL_DATA_DIR, item['save_name'])

    if os.path.exists(save_path):
        print(f"  -> 目录已存在，跳过: {save_path}")
        return

    try:
        # 启用 streaming=True
        if item.get("config"):
            ds_stream = load_dataset(item['name'], item['config'], split=item['split'], streaming=True)
        else:
            ds_stream = load_dataset(item['name'], split=item['split'], streaming=True)

        # 截取前 N 条
        data_list = list(itertools.islice(ds_stream, item['num_rows']))

        if len(data_list) == 0:
            print("  -> [警告] 未获取到数据，请检查网络或数据集名称。")
            return

        # 转为 Dataset 对象并保存
        ds_subset = Dataset.from_list(data_list)
        print(f"  -> 截取完成，正在保存到本地: {save_path} ...")
        ds_subset.save_to_disk(save_path)
        print(f"  -> 保存成功! (Rows: {len(ds_subset)})")

    except Exception as e:
        print(f"  -> [错误] 截取失败: {e}")


def main():
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)
        print(f"创建本地存储目录: {LOCAL_DATA_DIR}")

    print("=== 1. 开始下载全量数据集 ===")
    for item in FULL_DATASETS:
        save_full_dataset(item)

    print("\n=== 2. 开始截取大规模数据集子集 ===")
    for item in STREAMING_DATASETS:
        save_streaming_subset(item)

    print("\n========================================")
    print(f"全部任务完成！数据已保存在: {os.path.abspath(LOCAL_DATA_DIR)}")
    print("========================================")


if __name__ == "__main__":
    main()

