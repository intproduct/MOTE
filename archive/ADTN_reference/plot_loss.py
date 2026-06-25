import os
import re
import matplotlib.pyplot as plt


def extract_loss_from_file(file_path):
    """
    从单个日志文件中提取loss值。

    参数:
        file_path (str): 日志文件的路径。

    返回:
        list: 包含所有提取到的loss值的列表（浮点数）。
    """
    loss_values = []
    # 正则表达式用于匹配 "'loss':" 后面的浮点数
    loss_pattern = re.compile(r"'loss':\s*(\d+\.\d+)")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = loss_pattern.search(line)
                if match:
                    loss_values.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"错误：文件未找到 {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误 {file_path}: {e}")

    return loss_values


def plot_loss_curves(log_files):
    """
    将从多个日志文件中提取的loss值绘制在一张图上。

    参数:
        log_files (list): 包含所有日志文件路径的列表。
    """
    plt.figure(figsize=(12, 8))

    for file_path in log_files:
        loss_data = extract_loss_from_file(file_path)
        if loss_data:
            # 横坐标为数据的索引，纵坐标为loss值
            plt.plot(range(len(loss_data)), loss_data, label=os.path.basename(file_path))

    # plt.title('多个文件的Loss曲线')
    plt.xlabel('steps/20')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 获取当前目录下所有的.log文件
    current_directory = os.getcwd()
    log_files = [f for f in os.listdir(current_directory) if f.endswith('.log')]

    if not log_files:
        print("在当前目录下没有找到.log文件。")
    else:
        print(f"找到以下.log文件: {log_files}")
        plot_loss_curves(log_files)
