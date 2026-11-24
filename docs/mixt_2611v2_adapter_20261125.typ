#set page(paper: "a4", margin: (x: 20mm, y: 20mm), numbering: "1")
#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.65em)
#show heading: set text(font: ("Times New Roman", "KaiTi"))
#set table(inset: 5pt, stroke: 0.4pt + rgb("b9c3ca"))


= #text("MixT-2611v2 承接与 MoTE 后训练适配")

#text("2026-11-25；代码基线：mote-quantum，99d8876c364d22e3b1640120f1ea42fe6fc38556。")

== #text("结论")

#text("已实现可训练的原生 MixT 导入后端，可以承接合作方的后训练权重，再进行普通 gate 或 QuantumGate 后训练。 实际 3.40B 模型在均匀路由导入、原始断点恢复、完整 HF 导出重载三个环节，所测 BF16 logits 最大差异均为 0；完整导出连续生成的 3 个 token 与合作方模型一致。正式训练控制器也完成了一次实际更新、完整断点保存和恢复验证。")

#text("新增的 pretrained_mixt 复用现有 SparseMiXT、ADTN 算子、gate、训练器和断点体系。合作方的已训练算子、边界分支与骨干直接继承。初始均匀路由保持原来四算子的平均；后续训练才改变路由权重。这份实现说明接续 2026-11-24 的可行性审查，以本报告描述当前完成状态。")

== #text("承接的结构与参数")

#table(columns: (1fr, 3fr), [#text("项目")],
[#text("实际配置与处理")],
[#text("骨干")],
[#text("Qwen3.5 文本模型，32 层；24 层线性注意力、8 层全注意力；hidden 2560，FFN 9216")],
[#text("原始规模")],
[#text("3,398,405,393 个参数，777 个 BF16 权重张量")],
[#text("压缩范围")],
[#text("从 0 编号的第 19–31 层，39 个 gate/up/down 投影，全部导入")],
[#text("主区")],
[#text("gate/up：2048→8192、k=8→10；down：8192→2048、k=10→8；每投影 4 个原始位置窗口")],
[#text("边界")],
[#text("原有 rank 112 两条低秩分支、完整稠密角块、三个 alpha 全部保留")],
[#text("算子偏置")],
[#text("保留每个算子矩阵的标量偏置，并维持合作方 BF16 累加与除法顺序")],
[#text("共享权重")],
[#text("输入 embedding 与输出 head 保持共享")],
[#text("新增可训练路由")],
[#text("Linear：639,015 参数；QuantumGate：2,559,375 参数；路由保持 FP32")])

#text("合作方 w10 表示尾输入到主输出，对应框架 lr01；合作方 w01 对应框架 lr10。导入时同时转换 alpha 名称。每个投影检查所有来源权重均被消费，并保存完整几何规格，供训练恢复与独立 HF 重建使用。未知权重或不一致的结构会拒绝导入。")

#text("原有 SparseMiXT 已具备主区与边界组合能力；本次需要补齐的是已训练结构的无损接入、混合注意力生命周期以及 TIR 数据契约。配置本身能证明合作方已采用这些结构，模型能力的领先程度需要同协议评测确定。")

== #text("代码完成范围")

- #text("导入：pretrained_mixt.py、model.py、patching.py 及 cli/import_mixt.py。支持普通与量子路由；默认均匀初始化，导入命令逐条验证完整模型输出并记录来源指纹。")
- #text("训练：支持 gate_only、gate_experts、all 三种范围。这里的 all 指已导入的 39 个投影，骨干仍冻结。已适配温度调度、层选择、梯度检查点、纯 SFT 数据池、权重继续训练及精确恢复。")
- #text("稀疏化：权重继续训练时，显式设置 pretrained_mixt.uniform_init=false 后可修改 gate_type 和 topk，保留已有参数。精确恢复拒绝改变这些设置及训练范围。路由架构变更会明确报错。")
- #text("对话与 TIR：保留 reasoning_content、tool_calls、工具消息和工具定义；在不改变模板输出的条件下标注助手监督区间。用户与工具返回不计入标签，避免重复添加 EOS。冻结数据集与在线聊天数据入口均已接入。")
- #text("停止符：内存与导出配置保留原停止符 248044，并加入 tokenizer 的聊天结束符 248046。")
- #text("导出与生成：重建骨干时移除合作方代码路径依赖，传递完整混合缓存字段，修复空缓存首次生成截断提示词的问题，保持路由 FP32，并保存注意力计算设置。完整导出跳过将被覆盖的随机权重初始化。")
- #text("RL：原生 MixT 加载、HF 路由训练范围及响应 log-prob 梯度已接通；vLLM 原生 FFN 适配代码已扩展到新后端。")

== #text("验证证据")

#text("实测环境为 PyTorch 2.10.0+cu128、Transformers 5.15.1；完整模型验收使用 CPU、BF16、eager 注意力。小型混合注意力测试另外覆盖 FP32、BF16、eager 和 SDPA。")

#text("最终相关回归测试 181 项通过，1 项可选 vLLM 测试因本机运行环境未纳入执行。")

#table(columns: (1fr, 3fr), [#text("验证")],
[#text("结果")],
[#text("Linear 实模导入")],
[#text("4、12 token 两组输入，logits 最大差异 0")],
[#text("QuantumGate 实模导入")],
[#text("12 token 输入，logits 最大差异 0")],
[#text("原始包恢复")],
[#text("12 token 输入，logits 最大差异 0")],
[#text("实模反向传播")],
[#text("两步有限 loss；第二步 78 个路由参数张量得到非零梯度")],
[#text("正式训练控制器")],
[#text("冻结合成样本完成 1 次更新，梯度范数 0.3233269453，保存 SFT 完整断点")],
[#text("已训练断点恢复")],
[#text("清单与文件哈希验证通过；补丁张量逐项一致，前向有限")],
[#text("HF 完整导出")],
[#text("missing/unexpected keys 均为空；重载 logits 最大差异 0；3 token 贪心生成一致")],
[#text("训练器精确恢复")],
[#text("小型混合模型恢复后的全部模型参数与优化器状态，与连续训练逐项一致")],
[#text("路由与范围")],
[#text("Linear/QuantumGate 的三种范围、偏置与边界、可控 top-k 转换、HF RL 梯度均通过测试")])

#text("机器记录位于 research/mixt_adapter_20261125/：delivery_validation.json、export_acceptance.json、quantum_import.json、regression.xml。import_training_acceptance.json 保留实模两步反向传播记录。验收样本用于验证框架行为，不用于评价后训练收益。")

== #text("使用与产物")

#text("本工作副本已经生成以下本地产物：")

- #text("outputs/mixt_2611v2_native_import：Linear 均匀路由导入包。")
- #text("outputs/mixt_2611v2_quantum_import：QuantumGate 均匀路由导入包。")
- #text("outputs/mixt_2611v2_hf：验证通过的完整 HF 模型，来自 Linear 初始导入包。")
- #text("outputs/mixt_controller_acceptance/controller_acceptance/checkpoints/checkpoint-1：仅供验收的合成样本训练断点。")

#text("在安装当前工作副本的环境中可重新创建新的导入包；输出目录必须尚不存在：")

#block(width: 100%, inset: 8pt, fill: rgb("f4f5f7"))[#text("python -m fitmotn.cli.import_mixt --model F:/MixT-2611v2 --output outputs/mixt_import_new --dtype bfloat16 --gate-arch quantum")]

#text("复制 fitmotn_config.pretrained_mixt.example.json，填写真实模型、导入包、冻结 SFT 数据集和新输出目录。使用量子导入包时，将示例的 model.gate_arch 改为 quantum。随后运行：")

#block(width: 100%, inset: 8pt, fill: rgb("f4f5f7"))[#text("python -m fitmotn.cli.train --config_json your_mixt_training.json\npython -m fitmotn.cli.export_hf --checkpoint_dir YOUR_CHECKPOINT --output_dir YOUR_NEW_HF_DIR --no-metadata_only --torch_dtype bfloat16 --base_trust_remote_code")]

#text("示例以 gate_only 开始；扩大训练范围用 resume_weights_from 建立新的训练阶段。精确延续同一阶段使用 resume_checkpoint_from，保持路由、范围和注意力设置。进一步稀疏化可在权重继续训练配置中显式使用 gate_type=topk、topk=2、pretrained_mixt.uniform_init=false。这一转换会改变模型函数，需要独立评测稀疏化带来的能力变化。")

#text("原始导入包依赖 base_model_path 指向的合作方骨干；完整 HF 包携带所有模型权重，加载环境仍需安装当前 fitmotn 包及支持 Qwen3.5 的 Transformers。mixt 可选依赖声明了对应版本范围。示例的步数、学习率和上下文长度是可编辑起点，需要结合真实数据规模和显存调整。")

== #text("GPU 补充验收与剩余范围")

#text("本机 WSL / RTX PRO 4000 已完成 BF16、SDPA、linear gate 的 32K 实模对照：合作方原模型、导入模型、完整 HF 导出模型的末位置 logits 逐位一致，随后 2 个生成 token 一致。导出模型已支持 logits_to_keep，避免长提示词生成时计算全部位置的词表投影；本次相关回归 41 项通过。GPU 示例已显式采用 SDPA。机器证据为 research/mixt_runtime_20261125/long_export_acceptance.json。")

#text("512 token 全监督更新及 8K 输入、最后 32 token 监督的 gate-only 更新通过；8K 峰值显存 14.66 GiB，78 个路由张量有非零梯度。32K 训练在约 319 秒后停止，尚未通过。当前样本用于验证计算与梯度，真实任务得分和完整 TIR 工具执行环境仍需专项验证。")

#text("vLLM 已在独立 WSL 0.23 环境完成单卡生成：两条提示词各 8 token 与合作方 HF 模型一致，重复调用及单条/批量调用一致。测试使用 NVML 回退和官方 PyTorch 采样开关，完整条件见 mixt_2611v2_runtime_acceptance_20261125.md。目标依赖仍为 0.19 系列，吞吐、张量并行和在线权重同步均保留为待验收项。多卡还有代码边界：现有 Ulysses 会拒绝 Qwen3.5 混合注意力，常规训练器的原子断点只支持单进程，需要另行实现并验证多进程保存与恢复。")

#text("本轮修改在独立工作副本整理，发布目标为 mote-quantum。合作方源文件校验保持一致；D:/AI/MOTE 仍为冻结提交 860426ce3df30f5c2a74a030a43f22e3a76247c6，原实验源与产物保持原状。配套 Typst 使用 Times New Roman 与 KaiTi；本机未发现 Typst 编译器，因此交付可编译源文件。")
