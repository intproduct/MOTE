// Generated from the matching Markdown report. Edit the Markdown and rerun.
#set page(paper: "a4", margin: (x: 23mm, y: 22mm), numbering: "1")
#set text(font: ("Noto Serif SC", "Cambria"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.75em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))
#show heading.where(level: 2): set text(size: 14pt, fill: rgb("17365d"))
#show heading.where(level: 3): set text(size: 12pt)
#show link: set text(fill: rgb("24578a"))

#heading(level: 1)[#text("MoTE 工作区与分支核对记录")]

#text("核对日期：2026-10-29。目标分支：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "mote-quantum")#text("。实验主工作区：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "D:/files/MOTE")#text("。")

#heading(level: 2)[#text("问题原因")]

#text("本聊天原工作目录为 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "D:/files/pydocs/MOTE")#text("，这是独立克隆，停留在 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "mote-sft-v2")#text("。此前提交检查只覆盖该目录，未先追溯另一个实验聊天实际使用的工作区，因而将两次新提交推送到了错误分支，也漏掉实验工作区的未提交实现。")

#text("这是工作区识别和检查范围错误，不是最近实验实际跑在了错误分支。")

#heading(level: 2)[#text("四轮实验的追溯结果")]

#enum(start: 1, [#text("首轮分类实验：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "D:/files/MOTE")#text("，")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "mote-quantum")#text("，记录的基础提交 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "807366c")#text("。证据：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "outputs/gate_benchmark/S​ESSION_STATE.md")#text("。")])
#enum(start: 2, [#text("第二轮分类实验：同一工作区与分支；源码发布从 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "807366c")#text(" 到 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "c7d6690")#text("。完成审计记录 126 项比较，其中 90 项新训练、36 项历史复用。证据：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "outputs/gate_benchmark_r​ound2/git_publication.js​on")#text(" 和 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "COMPLETION_AUDIT.json")#text("。")])
#enum(start: 3, [#text("自回归实验：同一工作区与分支，基础提交 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "c7d6690")#text("；18 项训练完成，新代码此前未提交。证据：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "outputs/autoregressive_g​ate/SESSION_STATE.md")#text(" 和 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "COMPLETION_AUDIT.json")#text("。")])
#enum(start: 4, [#text("路由稳定性实验：同一工作区与分支，基础提交 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "c7d6690")#text("；18 项训练完成，新代码此前未提交。证据：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "outputs/routing_stabilit​y_round1/COMPLETION_AUDI​T.json")#text(" 及正式组冻结源码清单。")])

#text("聊天记录亦核对了“运行 MoTE 三种 gate 基准实验”，其工作目录为 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "D:/files/MOTE")#text("；与 Git reflog 和磁盘实验记录一致。旧工作区没有这些新实验源码。")

#heading(level: 2)[#text("归并范围")]

- #text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "c726b4e")#text(" 保存实验工作区原有的 30 个变更文件，包括自回归、路由稳定性、研究脚本、测试和包注册。")
- #text("通过合并提交 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "6b49d37")#text("，将 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "mote-sft-v2")#text(" 尚未进入量子分支的四个提交整体合入：")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "c933d3d")#text("、")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "e9edf9d")#text("、")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "551735a")#text("、")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "b365d99")#text("。采用保留历史的合并，而非覆盖量子分支或直接复制旧目录。")
- #text("三处冲突分别在配置源类型、混合数据加载器和额外数据集注册。解决后同时保留 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "local_token_shards/token​_ids")#text("、拒绝样本隔离和 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "frozen_release/FrozenTok​enTask")#text(" 功能。")
- #text("另外原样归档被 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "outputs/")#text(" 忽略的 73 个 Python/PowerShell 文件，包含辅助脚本及历史源码快照，排除测试生成的临时文件。来源、字节数和 SHA-256 见 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "source_audit.json")#text("。归档不执行这些脚本，不改变原始结果目录。")
- #text("数据集、模型权重、检查点、大型训练产物和本地环境继续保留在原路径，不纳入源码提交。")

#heading(level: 2)[#text("验证结果")]

#text("合并后的路由、量子门、自回归、配置、数据平台、SFT 发布、token shard 与拒绝隔离相关测试，加上新补充的独立进程导入测试，共 ")#strong[#text("218 项通过")]#text("。")

#text("检查中暴露出数据发布模块经任务注册回到自身的循环引用。将校验器导入推迟到数据检查实际调用处，保留接口行为；新增六项独立进程测试，覆盖两个包名下的三个数据入口，避免因整个测试套件的导入顺序而掩盖问题。")

#text("首次测试因新临时目录的父目录不存在而出现 fixture 错误，修复测试目录后继续；随后发现并修复上述循环导入。最终完整重跑通过，没有把这些初次失败计为成功。")

#text("最新两轮正式实验的 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "frozen_sources.json")#text(" 均逐文件核验通过。此次归并未修改其冻结的模型、路由、训练和探针源码；原始输出和历史审计没有重写。最终测试 JUnit 记录另存为 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "integration_tests.xml")#text("。")

#heading(level: 2)[#text("后续约定与回滚提醒")]

#text("QuantumGate 实验继续使用 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "D:/files/MOTE")#text(" 与 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "mote-quantum")#text("。旧克隆保留本地数据资产，但也应同步到量子分支，避免本聊天再次误提交到旧分支。仓库 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "AGENTS.md")#text(" 已记录工作区、目标分支和文档交付约定。")

#text("面向用户的新 Markdown 报告同时提供 Typst 源文件；本地有编译器时附 PDF。本次量子调研报告、文献索引和本核对记录均提供对应版本。历史归档中的脚本和机器指令文件不作为排版报告交付。")

#strong[#text("提醒：归并确认完成后，注意将 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "mote-sft-v2")#text(" 回滚到误提交前的 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "e9edf9d")#text("。")]#text(" 该位置来自旧工作区 reflog，是 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "551735a")#text(" 与 ")#text(font: ("Consolas", "Noto Sans SC"), size: 8.5pt, "b365d99")#text(" 之前的版本。本次仅提醒，不执行 reset、revert 或强制推送；两个误提交的内容已经通过合并保留到目标分支。")

