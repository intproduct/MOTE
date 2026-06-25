import os
os.environ['HTTPS_PROXY'] = 'http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128'
os.environ['HTTP_PROXY'] = 'http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128'
os.environ['HTTP_PROXY'] = '127.0.0.1,10.254.31.0/24,10.254.128.106'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset, concatenate_datasets, DatasetDict

# ================= 配置 =================
SAVE_DIR = "./mmlu_data"  # 保存路径，必须与测评函数中的 DATASET_DIR 一致

# MMLU 的 57 个子任务名称
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


# =======================================

def download_and_process_mmlu():
    if os.path.exists(SAVE_DIR):
        print(f"[提示] 目录 {SAVE_DIR} 已存在，如果数据损坏请删除该目录后重试。")
        # return # 如果想强制重新下载，请注释掉这行

    print(f"开始下载 MMLU 数据集 (共 {len(MMLU_SUBSETS)} 个子集)...")

    all_dev_datasets = []
    all_test_datasets = []

    # 遍历所有子集进行下载和处理
    for i, subset_name in enumerate(MMLU_SUBSETS):
        print(f"[{i + 1}/{len(MMLU_SUBSETS)}] 处理学科: {subset_name} ...")

        try:
            # 1. 下载特定子集 (cais/mmlu 是官方版本)
            ds = load_dataset("cais/mmlu", subset_name)

            # 2. 【核心步骤】添加 subject 列
            # 你的测评代码需要根据 subject 进行 filter，所以这里必须加上
            def add_subject_column(example):
                example["subject"] = subset_name
                return example

            # MMLU 包含: 'test' (用于测试), 'dev' (用于 few-shot 示例), 'validation', 'auxiliary_train'
            # 我们只需要处理 dev 和 test
            ds_dev = ds["dev"].map(add_subject_column)
            ds_test = ds["test"].map(add_subject_column)

            all_dev_datasets.append(ds_dev)
            all_test_datasets.append(ds_test)

        except Exception as e:
            print(f"❌ 下载子集 {subset_name} 失败: {e}")

    print("\n正在合并数据集...")
    # 3. 将所有子集垂直合并为一个大表
    combined_dev = concatenate_datasets(all_dev_datasets)
    combined_test = concatenate_datasets(all_test_datasets)

    print(f"合并完成: Dev集(用于FewShot)共 {len(combined_dev)} 条, Test集(用于评估)共 {len(combined_test)} 条")

    # 4. 构建最终的 DatasetDict
    final_ds = DatasetDict({
        "dev": combined_dev,
        "test": combined_test
    })

    # 5. 保存到本地磁盘
    print(f"正在保存到本地: {SAVE_DIR} ...")
    final_ds.save_to_disk(SAVE_DIR)
    print("✅ MMLU 数据集下载并处理完成！现在可以运行测评脚本了。")


if __name__ == "__main__":
    download_and_process_mmlu()
