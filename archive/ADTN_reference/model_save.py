import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 可选，让编号更直观
os.environ["CUDA_VISIBLE_DEVICES"] = "7"        # 改成你想用的服务器物理卡号，例如只用第3号卡
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer1 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# print(model1)

# 获取原始维度并定义新维度
config = model1.config
hidden_size = config.hidden_size
old_intermediate_size = config.intermediate_size
new_intermediate_size = 16384  # 您期望的新维度

print(f"\n--- 维度信息 ---")
print(f"Hidden Size (H): {hidden_size}")
print(f"原始 Intermediate Size: {old_intermediate_size}")
print(f"目标 Intermediate Size: {new_intermediate_size}")
print("\n--- 开始修改模型参数 ---")

# 3. 遍历模型的每一层，修改MLP块
# nn.Linear的权重形状为 (out_features, in_features)
for layer in model1.model.layers:
    # --- 扩展 gate_proj ---
    old_gate_proj = layer.mlp.gate_proj
    new_gate_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False)
    # 创建新的补零权重
    new_gate_weight = torch.zeros_like(new_gate_proj.weight)
    # 复制旧权重到新权重矩阵的对应位置 (添加了新的行)
    new_gate_weight.data[:old_intermediate_size, :] = old_gate_proj.weight.data
    new_gate_proj.weight = nn.Parameter(new_gate_weight)
    # 替换旧层
    layer.mlp.gate_proj = new_gate_proj

    # --- 扩展 up_proj (与gate_proj操作完全相同) ---
    old_up_proj = layer.mlp.up_proj
    new_up_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False)
    new_up_weight = torch.zeros_like(new_up_proj.weight)
    new_up_weight.data[:old_intermediate_size, :] = old_up_proj.weight.data
    new_up_proj.weight = nn.Parameter(new_up_weight)
    layer.mlp.up_proj = new_up_proj

    # --- 扩展 down_proj ---
    old_down_proj = layer.mlp.down_proj
    new_down_proj = nn.Linear(new_intermediate_size, hidden_size, bias=False)
    new_down_weight = torch.zeros_like(new_down_proj.weight)
    # 复制旧权重到新权重矩阵的对应位置 (添加了新的列)
    new_down_weight.data[:, :old_intermediate_size] = old_down_proj.weight.data
    new_down_proj.weight = nn.Parameter(new_down_weight)
    layer.mlp.down_proj = new_down_proj


# 4. 更新模型配置文件，使其与新结构保持一致
model1.config.intermediate_size = new_intermediate_size
print("--- 模型修改完成 ---")


# 5. 打印修改后的模型结构以供检查
print("\n--- 打印修改后的模型结构 ---")
print(model1)


# --- 6. 保存新模型路径 ---
SAVE_DIRECTORY = "model/llama/llama2-7B-expanded-model" # 定义一个新目录来保存修改后的模型


# --- 7. (新增功能) 保存修改后的模型和分词器 ---
print(f"\n--- 正在保存扩维后的模型到: {SAVE_DIRECTORY} ---")
model1.save_pretrained(SAVE_DIRECTORY)
tokenizer1.save_pretrained(SAVE_DIRECTORY)
print(f"--- 模型和分词器已成功保存！ ---")

# --- 8. (可选) 验证保存是否成功 ---
print(f"\n--- 验证：从 {SAVE_DIRECTORY} 重新加载模型 ---")
reloaded_model = AutoModelForCausalLM.from_pretrained(SAVE_DIRECTORY)
print("--- 重新加载后模型的MLP层结构 ---")
print(reloaded_model.model.layers[0].mlp)
print("\n✅ 验证成功！重新加载的模型已是扩维后的结构。")


tokenizer2 = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# print(model2)

# 获取原始维度并定义新维度
config = model2.config
hidden_size = config.hidden_size
old_intermediate_size = config.intermediate_size
new_intermediate_size = 16384  # 您期望的新维度

print(f"\n--- 维度信息 ---")
print(f"Hidden Size (H): {hidden_size}")
print(f"原始 Intermediate Size: {old_intermediate_size}")
print(f"目标 Intermediate Size: {new_intermediate_size}")
print("\n--- 开始修改模型参数 ---")

# 3. 遍历模型的每一层，修改MLP块
# nn.Linear的权重形状为 (out_features, in_features)
for layer in model2.model.layers:
    # --- 扩展 gate_proj ---
    old_gate_proj = layer.mlp.gate_proj
    new_gate_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False)
    # 创建新的补零权重
    new_gate_weight = torch.zeros_like(new_gate_proj.weight)
    # 复制旧权重到新权重矩阵的对应位置 (添加了新的行)
    new_gate_weight.data[:old_intermediate_size, :] = old_gate_proj.weight.data
    new_gate_proj.weight = nn.Parameter(new_gate_weight)
    # 替换旧层
    layer.mlp.gate_proj = new_gate_proj

    # --- 扩展 up_proj (与gate_proj操作完全相同) ---
    old_up_proj = layer.mlp.up_proj
    new_up_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False)
    new_up_weight = torch.zeros_like(new_up_proj.weight)
    new_up_weight.data[:old_intermediate_size, :] = old_up_proj.weight.data
    new_up_proj.weight = nn.Parameter(new_up_weight)
    layer.mlp.up_proj = new_up_proj

    # --- 扩展 down_proj ---
    old_down_proj = layer.mlp.down_proj
    new_down_proj = nn.Linear(new_intermediate_size, hidden_size, bias=False)
    new_down_weight = torch.zeros_like(new_down_proj.weight)
    # 复制旧权重到新权重矩阵的对应位置 (添加了新的列)
    new_down_weight.data[:, :old_intermediate_size] = old_down_proj.weight.data
    new_down_proj.weight = nn.Parameter(new_down_weight)
    layer.mlp.down_proj = new_down_proj


# 4. 更新模型配置文件，使其与新结构保持一致
model2.config.intermediate_size = new_intermediate_size
print("--- 模型修改完成 ---")


# 5. 打印修改后的模型结构以供检查
print("\n--- 打印修改后的模型结构 ---")
print(model2)


# --- 6. 保存新模型路径 ---
SAVE_DIRECTORY = "model/llama/Llama-3.1-8B-Instruct-expanded-model" # 定义一个新目录来保存修改后的模型


# --- 7. (新增功能) 保存修改后的模型和分词器 ---
print(f"\n--- 正在保存扩维后的模型到: {SAVE_DIRECTORY} ---")
model2.save_pretrained(SAVE_DIRECTORY)
tokenizer2.save_pretrained(SAVE_DIRECTORY)
print(f"--- 模型和分词器已成功保存！ ---")

# --- 8. (可选) 验证保存是否成功 ---
print(f"\n--- 验证：从 {SAVE_DIRECTORY} 重新加载模型 ---")
reloaded_model = AutoModelForCausalLM.from_pretrained(SAVE_DIRECTORY)
print("--- 重新加载后模型的MLP层结构 ---")
print(reloaded_model.model.layers[0].mlp)
print("\n✅ 验证成功！重新加载的模型已是扩维后的结构。")


# tokenizer3 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# model3 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# # print(model3)

tokenizer4 = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model4 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
# print(model4)

# 获取原始维度并定义新维度
config = model4.config
hidden_size = config.hidden_size
old_intermediate_size = config.intermediate_size
new_intermediate_size = 16384  # 您期望的新维度

print(f"\n--- 维度信息 ---")
print(f"Hidden Size (H): {hidden_size}")
print(f"原始 Intermediate Size: {old_intermediate_size}")
print(f"目标 Intermediate Size: {new_intermediate_size}")
print("\n--- 开始修改模型参数 ---")

# 3. 遍历模型的每一层，修改MLP块
# nn.Linear的权重形状为 (out_features, in_features)
for layer in model4.model.layers:
    # --- 扩展 gate_proj ---
    old_gate_proj = layer.mlp.gate_proj
    new_gate_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False)
    # 创建新的补零权重
    new_gate_weight = torch.zeros_like(new_gate_proj.weight)
    # 复制旧权重到新权重矩阵的对应位置 (添加了新的行)
    new_gate_weight.data[:old_intermediate_size, :] = old_gate_proj.weight.data
    new_gate_proj.weight = nn.Parameter(new_gate_weight)
    # 替换旧层
    layer.mlp.gate_proj = new_gate_proj

    # --- 扩展 up_proj (与gate_proj操作完全相同) ---
    old_up_proj = layer.mlp.up_proj
    new_up_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False)
    new_up_weight = torch.zeros_like(new_up_proj.weight)
    new_up_weight.data[:old_intermediate_size, :] = old_up_proj.weight.data
    new_up_proj.weight = nn.Parameter(new_up_weight)
    layer.mlp.up_proj = new_up_proj

    # --- 扩展 down_proj ---
    old_down_proj = layer.mlp.down_proj
    new_down_proj = nn.Linear(new_intermediate_size, hidden_size, bias=False)
    new_down_weight = torch.zeros_like(new_down_proj.weight)
    # 复制旧权重到新权重矩阵的对应位置 (添加了新的列)
    new_down_weight.data[:, :old_intermediate_size] = old_down_proj.weight.data
    new_down_proj.weight = nn.Parameter(new_down_weight)
    layer.mlp.down_proj = new_down_proj


# 4. 更新模型配置文件，使其与新结构保持一致
model4.config.intermediate_size = new_intermediate_size
print("--- 模型修改完成 ---")


# 5. 打印修改后的模型结构以供检查
print("\n--- 打印修改后的模型结构 ---")
print(model4)


# --- 6. 保存新模型路径 ---
SAVE_DIRECTORY = "model/llama/Qwen3-8B-expanded-model" # 定义一个新目录来保存修改后的模型


# --- 7. (新增功能) 保存修改后的模型和分词器 ---
print(f"\n--- 正在保存扩维后的模型到: {SAVE_DIRECTORY} ---")
model4.save_pretrained(SAVE_DIRECTORY)
tokenizer4.save_pretrained(SAVE_DIRECTORY)
print(f"--- 模型和分词器已成功保存！ ---")

# --- 8. (可选) 验证保存是否成功 ---
print(f"\n--- 验证：从 {SAVE_DIRECTORY} 重新加载模型 ---")
reloaded_model = AutoModelForCausalLM.from_pretrained(SAVE_DIRECTORY)
print("--- 重新加载后模型的MLP层结构 ---")
print(reloaded_model.model.layers[0].mlp)
print("\n✅ 验证成功！重新加载的模型已是扩维后的结构。")

