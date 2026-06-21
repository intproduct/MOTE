import os
import re
import matplotlib.pyplot as plt


def extract_data_from_file(file_path):
    """
    从单个日志文件中提取 Loss 值和 MMLU 准确率。

    返回:
        dict: 包含 'loss' (list) 和 'mmlu' (list of tuples)。
              'mmlu' 格式为 [(loss_index, score), ...]，其中 loss_index 代表该分数对应第几个 loss 点。
    """
    data = {
        'loss': [],
        'mmlu': []
    }

    # Loss 匹配规则: {'loss': 9.2562, ...
    loss_pattern = re.compile(r"'loss':\s*(\d+\.\d+)")

    # MMLU 匹配规则: 模型 ADTN 在 MMLU 上的 5-shot 平均准确率: 0.6860
    # 使用 .* 稍微放宽中间字符的限制，以防模型名变化
    mmlu_pattern = re.compile(r"模型.*在 MMLU 上的 5-shot 平均准确率:\s*(\d+\.\d+)")

    current_step_index = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 1. 尝试匹配 Loss
                loss_match = loss_pattern.search(line)
                if loss_match:
                    data['loss'].append(float(loss_match.group(1)))
                    current_step_index += 1  # 记录当前的步数进度
                    continue  # 如果这一行是loss，通常不会同时是mmlu，跳过后续检查

                # 2. 尝试匹配 MMLU
                # MMLU数据是稀疏的，我们需要记录它发生时的 X轴位置 (即 current_step_index)
                mmlu_match = mmlu_pattern.search(line)
                if mmlu_match:
                    score = float(mmlu_match.group(1))
                    # 记录 (x坐标, y数值)
                    data['mmlu'].append((current_step_index, score))

    except FileNotFoundError:
        print(f"错误：文件未找到 {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误 {file_path}: {e}")

    return data


def plot_metrics(log_files):
    """
    绘制 Loss 和 MMLU 曲线，使用上下两个子图。
    """
    # 创建两个共享 X 轴的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 设置颜色循环，确保同一个文件的 Loss 和 MMLU 颜色色系接近（可选，这里使用默认自动配色）

    has_valid_data = False

    for file_path in log_files:
        data = extract_data_from_file(file_path)
        file_label = os.path.basename(file_path)

        # 绘制 Loss (在上图)
        if data['loss']:
            has_valid_data = True
            # x 轴就是 0 到 len(loss)-1
            ax1.plot(range(len(data['loss'])), data['loss'], label=file_label, linewidth=1.5, alpha=0.8)

        # 绘制 MMLU (在下图)
        if data['mmlu']:
            # 解压 x 和 y
            x_mmlu, y_mmlu = zip(*data['mmlu'])
            # 使用带点的线 (marker='o') 因为数据点比较稀疏
            ax2.plot(x_mmlu, y_mmlu, label=file_label, marker='o', linestyle='--', linewidth=1.5)

            # 可选：在点旁边标注具体数值
            for x, y in data['mmlu']:
                ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    if not has_valid_data:
        print("未提取到有效数据，无法绘图。")
        return

    # 设置上图 (Loss) 属性
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Progress: Loss & MMLU Accuracy', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # 设置下图 (MMLU) 属性
    ax2.set_ylabel('MMLU 5-shot Accuracy', fontsize=12)
    ax2.set_xlabel('Training Steps (Count of Loss logs)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 获取当前目录下所有的.log文件
    current_directory = os.getcwd()
    log_files = [f for f in os.listdir(current_directory) if f.endswith('.log')]

    # 模拟测试（如果你没有实际log文件，取消下面这行的注释并手动创建一个 dummy.log 测试）
    # log_files = ['training.log']

    if not log_files:
        print("在当前目录下没有找到.log文件。")
    else:
        print(f"找到以下.log文件: {log_files}")
        plot_metrics(log_files)
        