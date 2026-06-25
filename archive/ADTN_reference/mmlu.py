import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 设置中文字体（根据您的系统环境，如果有乱码请调整这里）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def extract_latest_subset_scores(file_path):
    """
    从日志文件中提取每个子集最后一次出现的准确率。
    不依赖于特定的 step 格式，只要出现 '子集 ... 准确率' 就更新，确保取到最新值。
    """
    scores = {}
    # 匹配模式：子集 abstract_algebra 准确率: 0.2100
    subset_pattern = re.compile(r"子集\s+([\w-]+)\s+准确率:\s*(\d+\.\d+)")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = subset_pattern.search(line)
                if match:
                    subset_name = match.group(1)
                    score = float(match.group(2))
                    # 直接覆盖，这样字典里留下的就是该文件最后一次记录的该子集分数
                    scores[subset_name] = score
    except Exception as e:
        print(f"读取文件出错 {file_path}: {e}")

    return scores


def plot_subset_lines(log_files_data):
    """
    绘制 MMLU 子集得分折线图。
    X轴：57个子集名称
    Y轴：准确率
    """
    # 1. 收集所有出现过的子集名称，并排序
    all_subsets = set()
    for scores in log_files_data.values():
        all_subsets.update(scores.keys())

    if not all_subsets:
        print("未提取到任何子集数据，无法绘图。")
        return

    sorted_subsets = sorted(list(all_subsets))

    # 2. 创建画布，宽度设置大一些以容纳57个标签
    plt.figure(figsize=(20, 10))

    # 3. 循环绘制每个文件的线条
    # 定义一些样式，防止线条太多分不清
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']

    for i, (file_path, scores) in enumerate(log_files_data.items()):
        file_name = os.path.basename(file_path)

        # 准备 Y 轴数据：如果某文件没有该子集数据，记为 None（断开）或者 0.0
        # 这里建议用 0.0，这样能看出缺失，或者用 scores.get(subset, None) 让线断开
        y_values = [scores.get(subset, 0.0) for subset in sorted_subsets]

        # 循环使用样式
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]

        plt.plot(sorted_subsets, y_values,
                 label=file_name,
                 marker=marker,
                 markersize=4,
                 linestyle=ls,
                 linewidth=1.5,
                 alpha=0.8)  # 设置透明度，防止完全遮挡

    # 4. 设置图表细节
    plt.title('各模型在 MMLU 子集上的最终得分轮廓对比', fontsize=18)
    plt.ylabel('准确率 (Accuracy)', fontsize=14)
    plt.xlabel('MMLU 子集 (Subsets)', fontsize=14)

    # 设置 X 轴标签旋转，字体缩小，确保能显示全
    plt.xticks(range(len(sorted_subsets)), sorted_subsets, rotation=90, fontsize=9)

    # 设置 Y 轴网格，方便读数
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05)  # 限制Y轴在 0~1 之间

    # 图例放在最合适的位置
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)

    # 调整布局，防止下方标签被切掉
    plt.subplots_adjust(bottom=0.25)

    plt.show()


if __name__ == '__main__':
    current_directory = os.getcwd()
    log_files = [f for f in os.listdir(current_directory) if f.endswith('.log')]

    if not log_files:
        print("在当前目录下没有找到.log文件。")
    else:
        print(f"找到以下.log文件: {log_files}")

        all_data = {}
        for file in log_files:
            file_path = os.path.join(current_directory, file)
            # 提取数据
            scores = extract_latest_subset_scores(file_path)
            if scores:
                all_data[file_path] = scores
            else:
                print(f"警告: 文件 {file} 中未提取到子集分数。")

        if all_data:
            plot_subset_lines(all_data)
        else:
            print("没有有效数据可供绘图。")
