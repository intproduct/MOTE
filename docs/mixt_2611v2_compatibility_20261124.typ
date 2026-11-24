#set page(paper: "a4", margin: (x: 22mm, y: 20mm), numbering: "1")
#set text(font: ("Times New Roman", "KaiTi"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.7em)
#set heading(numbering: none)
#show heading: set text(font: ("Times New Roman", "KaiTi"))
#set table(inset: 6pt, stroke: 0.4pt + rgb("b9c3ca"))

= MixT-2611v2 与 mote-quantum 适配审查


#text("日期：2026-11-24（上海时间）")

#text("本文保留适配前的审查结论与证据。后续实现已完成并发布，当前状态见 mixt_2611v2_adapter_20261125.md 和 mixt_2611v2_runtime_acceptance_20261125.md。")


== 结论


#text("架构可适配，参数保真 gate 化已通过数值验证；当前训练、导出和推理加速链路尚未完成原生 MixT 接入，不能直接套用现有 sparse_mixt 示例开训。应先完成原生 MixT 导入兼容层，再进行 MoTE 后训练。")


#text("判断基于合作方真实权重和代码，而不只读取配置：39 个投影在 FP32 均匀路由下输出误差全部为 0；在完整 3.40B 模型上同时替换 39 个投影后，短输入的 FP32 logits 最大绝对误差也是 0。这个结果确立了保留合作方参数的实现路线。")


#text("本轮完成可行性审查、数值原型和两项公共逻辑修正。原型位于 research/mixt_compat_20261124，尚未成为正式训练后端；没有启动后训练或发布代码。")


== 审查对象与模型结构


- #text("合作方目录：F:/MixT-2611v2。")
- #text("框架基线：从远端更新后的 origin/mote-quantum，提交 99d8876c364d22e3b1640120f1ea42fe6fc38556。")
- #text("当前 D:/AI/MOTE 实际在 codex/qwen-joint-pro4000；本轮在独立工作副本中审查和修正，保留现有实验代码。")
- #text("实测环境：PyTorch 2.10.0+cu128、Transformers 5.15.1，计算使用 CPU；checkpoint 记录的导出版本是 Transformers 5.14.1。")

#table(columns: 2,
table.header([#text("项目")], [#text("实际内容")]),
[#text("模型类")], [#text("Qwen3_5MixTForCausalLM，纯文本")],
[#text("实际参数量")], [#text("3,398,405,393；777 个 BF16 张量")],
[#text("骨干")], [#text("32 层；hidden 2560；FFN 9216")],
[#text("注意力")], [#text("24 层线性注意力、8 层全注意力，按 3:1 交替")],
[#text("压缩范围")], [#text("从 0 开始编号的第 19–31 层，13 层 × gate/up/down = 39 个投影")],
[#text("主块")], [#text("gate/up：2048→8192；down：8192→2048")],
[#text("算子")], [#text("每投影 4 个连续窗口，等权平均；d=2，gate_offset=3")],
[#text("局部维度")], [#text("gate/up：k_in=8、k_out=10；down：k_in=10、k_out=8")],
[#text("边界")], [#text("两条 rank 112 低秩分支、一块完整稠密角块、三项已训练 alpha")],
[#text("权重共享")], [#text("输入 embedding 与输出 head 共享，实测为同一参数")],
[#text("词表")], [#text("embedding 行数 248320，tokenizer 实际长度 248077")],
[#text("上下文配置")], [#text("262144；本轮短输入测试不构成长上下文验收")],
)


#text("checkpoint 的张量索引与实际文件键完整一致。配置虽有 MTP 字段，文件中没有 MTP 权重，不能据此宣称它交付了可用的 MTP 分支。")


#text("README 说明实际权重来自 math-v4 TIR SFT 的 iteration 848，承接 CoT stage-2 iteration 830；manifest 的 training_step=20000 是旧基础导出记录。应按实际文件哈希和 README 标识该检查点，不把 20000 当作最终后训练步数。manifest 中的 Qwen3.5-9B 是蒸馏教师路径，不能当作当前学生模型规模。")


== 适配程度


#table(columns: 3,
table.header([#text("链路")], [#text("结论")], [#text("证据与必须处理的内容")]),
[#text("原始 HF 加载与短前向")], [#text("已实测通过")], [#text("missing/unexpected/mismatched 均为 0；logits 有限；共享 embedding 保持")],
[#text("主块算子")], [#text("高度兼容")], [#text("4 个滑动窗口与现有 GatedADTN 收缩完全对应；FP32 数值验证通过")],
[#text("原有权重导入")], [#text("当前缺失")], [#text("现有 SparseMiXTFFNLayer 是从稠密模型新建专家，未导入 bricks、bias、alpha 和边界")],
[#text("训练入口")], [#text("当前阻断")], [#text("train/controller.py 无条件 collect_dense_ffn_targets；原生投影没有 .weight，实测 AttributeError")],
[#text("边界结构")], [#text("需扩展")], [#text("当前三条边界均为低秩；合作方角块是 full dense，且三条边界各有 alpha")],
[#text("数据监督")], [#text("单轮公共缺陷已修；TIR 待适配")], [#text("原模板没有 generation 标签，原代码在 CoT/TIR 两个探针中都只监督 1 个追加 EOS")],
[#text("保存、恢复与 HF 导出")], [#text("需原生结构契约")], [#text("当前导出按普通 AutoModel 构建，再套现有 patch；需记录原生 MixT 结构与每个已路由层")],
[#text("vLLM/RL rollout")], [#text("当前专用后端不兼容")], [#text("原生 Qwen3.5 MoTE 适配器明确仅接受 patch_backend=motn；合作方 README 要求另外安装其插件")],
[#text("Ulysses 长序列训练")], [#text("当前不支持")], [#text("validate_model 仅接受 qwen3/llama；Qwen3.5 递归线性注意力不能直接套普通 SDPA 切分")],
)


#text("Qwen3.5 的混合注意力属于骨干结构，与我们新增的专家路由 gate 是不同层面的机制。其官方说明与本地 layer_types 一致：Transformers Qwen3.5 文档（https://huggingface.co/docs/transformers/model_doc/qwen3_5）。")


== 比现有实现更需要保留的内容


#text("合作方的价值首先是已完成数学 CoT/TIR 后训练的参数、真实维度的混合块表达，以及推理导出所需的自定义模型代码。我们的优势是可训练动态路由、QuantumGate 与训练诊断基础设施。当前交付物没有动态专家选择，也没有评测成绩文件；无法从“后训练程度高”推导出相对我们已有模型的具体能力排名。")


#text("需要特别保留三个细节：")


+ #text("operator bias 加在算子矩阵上。 它不是普通输出偏置，不能直接搬成 Linear.bias。数值原型用 bricks+bias 验证算子等价；正式导入应保留两者的独立参数化，避免改变继续训练时的优化行为。")
+ #text("w01/w10 的命名方向与我们不同。 合作方 w10 是输入尾部→主输出，对应我们 lr01；合作方 w01 是主输入→尾输出，对应我们 lr10。要按实际张量方向映射。")
+ #text("稠密角块和 alpha 不能被默认初始化覆盖。 39 个投影的 alpha 范围为 0.40234375–1.1640625；统一重置为 1 会改变函数。第 19 层 gate/up/down 的稠密角块即使做最优 rank-112 截断，相对 Frobenius 误差仍分别为 71.27%、72.75%、71.43%。这不是可忽略的格式变换。")

#text("现有 sparse_mixt 示例使用 E=16、随机位置、global expert、边界 SVD 和 last_quarter；这些设置均不等于本模型的原生结构。gate_proj 是 SwiGLU 的乘法分支名称，attn_output_gate 是注意力内部 gate；两者都不表示它已经有 MoTE 动态路由。")


== 数值验证


#table(columns: 3,
table.header([#text("验证")], [#text("实测结果")], [#text("范围")]),
[#text("真实 checkpoint 加载")], [#text("0 缺失、0 额外、0 形状不匹配")], [#text("完整 3.40B 模型，BF16 CPU")],
[#text("原模型缓存预填充")], [#text("logits 最大绝对差 0")], [#text("输入“1+1=”，4 token")],
[#text("原模型单步缓存解码")], [#text("相对 L2 0.37135%，下一 token top-1 相同")], [#text("BF16；与完整前缀重算比较")],
[#text("原生投影→现有算子")], [#text("最大绝对差 0，相对 L2 0")], [#text("39 投影，每个 6 个固定种子随机向量，FP32")],
[#text("整模型接入数值原型")], [#text("logits 最大绝对差 0，相对 L2 0")], [#text("39 投影全部替换，4 token 输入，FP32")],
[#text("可训练性")], [#text("router 和专家梯度均非零")], [#text("第 19 层 gate/up/down 三个投影")],
[#text("直接复用现有 BF16 路由累加")], [#text("相对 L2 0.2189% / 0.2415% / 0.2183%")], [#text("第 19 层 gate/up/down")],
)


#text("FP32 等价说明迁移在数学结构上成立。BF16 差异来自现有路由的 FP32 权重乘法与累加路径、以及原算子的 BF16 加法顺序差异；正式兼容层应保留或明确控制累加精度，并用真实校准样本评估逐层、logits、KL 和生成差异。数值原型未验证数学题准确率、完整 TIR 工具执行、多卡训练和长上下文。")


== 数据与停止符的实质缺口


#text("合作方模板没有 {% generation %}。Transformers 5.15.1 返回全零 assistant mask，原公共逻辑将其当成有效 mask，再追加 EOS，导致两个探针只剩 1 个监督 token。修正后，单轮 CoT 样本有 12 个监督 token，多轮 TIR 明确报出缺少助手区间支持，避免伪装成有效训练。")


#text("TIR 兼容需要保留完整消息序列、reasoning_content、tool_calls 和 tool response，准确监督每轮助手内容，屏蔽用户和工具返回。应在不改变实际渲染文本与 token 序列的前提下加入助手区间标记，并测试多轮思考内容保留策略。还需检查模板已经输出结束符后额外追加 EOS 的行为；当前单轮探针会出现重复 im_end，应在正式数据契约中消除。")


#text("停止符实测为：")


- #text("<|endoftext|> = 248044，也是 tokenizer 的 pad。")
- #text("<|im_end|> = 248046，也是 tokenizer 的 eos。")
- #text("config.json 与 generation_config.json 的 eos_token_id 均为 248044。")

#text("这意味着当前生成配置没有覆盖对话模板使用的 im_end。正式加载配置应明确允许 248046 终止，并根据原模型的生成约定决定是否同时允许 248044；应通过真实生成确认停止行为，不能只把 tokenizer 与配置差异记录下来。")


== 建议的共用兼容层与实施顺序


#text("第一步：原生结构导入与保真路由。 为 pretrained MixT 增加明确后端或独立结构描述，支持已有 projection 的参数导入、配置重建和 strict state_dict 恢复。保留全部 bricks、算子 bias、低秩因子、dense corner、alpha，保持前 19 层、注意力与 embedding；不再走稠密拟合初始化。层选择读取 mixt_compress_layers，而不是硬套 last_quarter。")


#text("起始配置采用 E=4、滑动窗口、全部专家参与、无 token 丢弃、无新增 global expert。均匀权重应严格为 1/4。线性路由可通过零 logits 初始化；QuantumGate 默认输出并不保证均匀，应使用受控的均匀→可学习路由过渡，并确认过渡开始后梯度能进入 gate。先建立 soft 全专家基线，再实验 Top-2/Top-1。均匀 Top-4 不提供专家计算节省。")


#text("只在这 39 个投影的主输入上增加无 bias 线性路由，需要 638,976 个参数，约原模型的 0.0188%。全部压缩投影原有参数合计 112,779,537，占全模型 3.32%；可先只训 gate，再以较低学习率开放原有专家及必要边界参数。")


#text("第二步：训练与数据契约。 移除该后端对稠密 .weight 的依赖；提供 gate-only / gate+expert / gate+expert+boundary 冻结策略；保留原 tokenizer 与 TIR 格式，修复助手监督、EOS 和 generation_config。初期可使用同一合作方 checkpoint 作保留能力的参考教师；更大的原始 Qwen 教师可另作实验，不应替代保真基线。")


#text("第三步：保存恢复与验收。 记录源文件哈希、真实权重版本、每层窗口、分支模式、精度策略和 gate 配置。先通过 strict 保存恢复、HF 导出再加载、短 SFT 梯度及优化器恢复，再测 CoT/TIR 保持率。原交付物不含 training_state.pt，所以只能承接权重开始新的训练运行，无法恢复合作者原优化器和数据进度。")


#text("第四步：推理加速与并行。 对齐原生 MixT 的 vLLM 注册、参数映射及 gate 执行，再接 RL rollout。现有 pyproject 的 vLLM extra 限制 transformers<5，Ulysses extra 固定 4.57.6，与本轮验证环境不一致；应独立锁定兼容组合。Qwen3.5 的递归注意力序列并行单独验收，首轮保真验证可从单卡短序列开始。")


== 本轮代码与交付状态


- #text("已修正 chat_formatting.py：将全零 assistant mask 视为不可用；单轮采用前缀回退，多轮缺少区间时明确失败。")
- #text("已修正 patching.py：在任何选中层被替换前检查原生 MixT，防止现有后端静默丢弃合作方权重。")
- #text("相关回归：53 passed，含新增监督缺陷与原生 MixT 替换防护测试，以及 SparseMiXT、QuantumGate 测试。")
- #text("已保存真实权重审查结果、短前向结果和保真接入原型。正式原生 MixT 后端、TIR 区间标记、EOS 修正、导出/vLLM/并行接入是后续实施范围。")
- #text("本机未发现 Typst 编译器，附同名 Typst 源稿。")

== 证据文件


- #text("research/mixt_compat_20261124/audit.json：结构、39 投影、梯度、边界误差、修正前数据问题。")
- #text("research/mixt_compat_20261124/full_model_smoke.json：完整加载、缓存前向、39 投影接入后的整模型 FP32 对照。")
- #text("research/mixt_compat_20261124/chat_after_fix.json：修正后真实 tokenizer 监督与停止符映射。")
- #text("同目录三份 Python 程序可重跑；全部只读合作方 checkpoint，不写回原模型目录。")
