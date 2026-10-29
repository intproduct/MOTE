"""Add reviewed interpretation and reproducibility details to the measured report."""
from pathlib import Path
import csv
import json
import statistics as st
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'formal_analysis.json').read_text())
AGG = {(a['task'], a['gate']): a for a in DATA['aggregates']}
TASKS = ['mnist', 'fashion_mnist', 'cifar10', 'sst2', 'mrpc', 'rte']
GATES = ['mlp', 'mlp_half', 'compact_born', 'mpo_c2_full', 'mpo_c1_half', 'amp_d1_full', 'amp_d1_half']
GROUP = ROOT / 'formal/20c0fefebf11'
report = ROOT / 'REPORT_zh.md'
original = report.read_text(encoding='utf-8')
backup = ROOT / 'review_sources/REPORT_generated_zh.md'
backup.parent.mkdir(exist_ok=True)
if not backup.exists():
    backup.write_text(original, encoding='utf-8')

def label(g):
    return AGG['mnist', g]['label']

def fmt(v, digits=2):
    return f'{v:.{digits}f}'

summary = '''**实测判断：半预算 MPO 是值得继续验证的经典 TN 候选；新幅度 Born 改善了原 Born 的速度，但本轮没有获得跨任务一致的准确率优势。半预算 MLP 同样保留了大部分能力，所以“参数减半后分数相近”尚不能证明 TN 或量子结构特有的收益。**

正式失败、缺失、未完成项均为 **0**；另有 **6 次因系统重启中断的训练尝试**，现场保留且同种子重跑通过。pilot **26/26**、正式审计 **126/126**、最终预测/路由重评估与串行计时 **126/126** 完成。本文的最终成绩来自内部 dev 选出的最佳 checkpoint，不是按最终评估集选出的 checkpoint。

'''
s = original.replace('## 方法与实验边界', summary + '## 方法与实验边界', 1)
s = s.replace('平均 score 不低于 full MLP 的任务数', '平均 score 高于 full MLP 超过 0.001 pp 的任务数')
s = s.replace('正式比较矩阵 **126/126** 完成', '正式比较矩阵 **126/126** 完成', 1)

protocol = [
    '## 执行与复现信息', '',
    '- 分支 `mote-quantum`；本轮源码已按用户要求推送为 `c7d6690dfb13f2bf0a00ae145020ba7b0bb222fd`。正式训练最初运行于 `807366cb8afc58b1fc37086f998ae791e8bb16a6` 加本轮未提交改动，训练中提交只改变 Git 元数据，11 个冻结训练/配置文件的 SHA256 未变。原 manifest 保留其实际 commit/dirty，代码身份以源码 hash 为准；见 `git_publication.json`。',
    '- Python 3.11.18、PyTorch 2.6.0+cu118、CUDA runtime 11.8、驱动 582.28、Transformers 4.57.6、Datasets 3.6.0；依赖和 GPU 原始记录见 `dependencies.txt`、`nvidia_smi.txt`。预训练模型实际 revision：`387825ce42dbb39b87911cdf8e383ee3b25184f8`。',
    '- 相关回归测试 146 项通过，CUDA doctor 和 26 组合成离线 smoke 通过。第一次回归有 Windows 临时目录权限造成的 4 个 setup error，改用项目内临时目录后通过；原日志保留。',
    '- 正式训练窗口为 2026-10-27 18:52 至 2026-10-28 02:32（Asia/Shanghai），约 7 小时 40 分钟，包含系统重启与恢复。全部重评估/串行计时于 03:09 结束。第二轮从 18:13 开始实现与验证，到自动分析就绪约 8 小时 57 分钟，超过原 5–6 小时规划；主要因为扩展为 90 个新增 run 并保留了全部复核与中断重跑。',
    '- 分析阶段启动前发现计时文件读取路径少了 `formal/`，修正后全量分析成功。只修改后处理路径；修改前脚本和原因保留于 `postprocessing_fixes/20261028_0242_serial_measurement_path/`，训练、配置和结果未改写。', '',
    '| 任务 | 实际 train | 内部 dev | 最终评估样本 | 最终来源 split |',
    '|---|---:|---:|---:|---|',
]
audit = json.loads((GROUP/'audit.json').read_text())
for task in TASKS:
    d = audit['data'][task]
    protocol.append(f"| {task} | {d['sizes']['train']} | {d['sizes']['dev']} | {d['sizes']['test']} | {d['source_split']} |")
protocol += ['', '缓存中的 `test` 是统一的最终评估键名：GLUE 对应官方 validation。内部 dev 与训练索引不重叠；最终评估来源也独立。训练限额和固定子集意味着这些分数不代表使用完整数据训练的公开榜单成绩。', '',
             'Pilot 只用 Fashion-MNIST/SST-2 的内部 dev，13 个变体各 1 个种子、100 步；在距最佳均值 0.5 pp 内选择更小 χ/深度。选中 MPO full χ=2、half χ=1，幅度 Born 两档深度均为 1。该选择规则、全部候选与原始结果均保留。', '']
s = s.replace('## 最终任务成绩', '\n'.join(protocol) + '\n## 最终任务成绩', 1)

interpretation = '''### 参数效率：有候选信号，尚无通用胜者

半预算 MPO 的全部路由参数为 full MLP 的 **49.45%–49.99%**，半预算幅度 Born 为 **47.47%–49.10%**；half MLP 实际为 **50.88%–52.51%**。因此这些是近似预算分组，不是完全相同的参数个数。两种新半预算方法只使整个模型减少约 **0.43%–2.18%** 参数，不能表述为“模型参数减半”。

CIFAR-10 是最值得跟进的正向结果。半预算 MPO 的 accuracy 为 **46.23±0.45%**，full MLP 为 **46.00±0.63%**，同 seed 差值为 **+0.45、+0.10、+0.15 pp**，配对均值 **+0.23±0.19 pp**。它用约一半路由参数获得了本轮略高的分数；但只有 3 个种子，描述性 95% 配对 t 区间仍跨 0，不能据此确认稳定优越性。相对 half MLP 的均值为 +0.45 pp，三个差值为 -0.35、+1.20、+0.50 pp。

半预算幅度 Born 的 CIFAR-10 accuracy 为 **46.32±0.34%**，相对 full MLP **+0.32±0.92 pp**；三个差值为 +1.35、-0.40、0.00 pp，波动比半预算 MPO 大。相对 half MLP 的三个差值均为正（+0.55、+0.70、+0.35 pp），这是值得复测的信号，仍不是跨任务优势。

MNIST 上半预算 MPO 与 full MLP 在 0.01 pp 显示精度下相同，半预算幅度 Born 高 0.03 pp。Fashion-MNIST、SST-2、MRPC、RTE 上，两种新半预算方法的均值均低于 full MLP：MPO 分别为 -0.30、-0.11、-0.51、-0.60 pp；幅度 Born 为 -0.52、-0.38、-0.60、-0.48 pp。half MLP 在六个任务上都只下降 0.11–0.42 pp，已经是强的低参数对照。

六任务等权平均 score 差值（不是混合测试集 accuracy）中，半预算 MPO 相对 full/half MLP 为 **-0.215/-0.002 pp**，半预算幅度 Born 为 **-0.272/-0.058 pp**。所以本轮支持“低预算能保留大部分能力”，未支持“低预算 TN/Born 一致优于同样缩小的 MLP”。

### 全预算与原 Born：增加容量没有稳定改善结果

全预算 MPO 仅在 SST-2 的均值略高于 full MLP（+0.08 pp），CIFAR-10 和 MRPC 分别低 0.80、0.86 pp。全预算幅度 Born 仅在 SST-2 略高 0.04 pp，Fashion-MNIST 和 RTE 分别低 1.07、1.08 pp。MRPC 全预算幅度 Born 的 accuracy/F1/综合 score 为 **73.86/83.01/78.43%**，接近 full MLP 的 **73.94/83.04/78.49%**，但没有稳定胜出。

半预算变体在若干任务上优于各自 full 变体，可能与容量、初始化、优化和有限训练预算共同有关，当前实验不能确定原因。MPO 两档同时改变编码 rank 和 χ，尤其不能把 full/half 的差异解释成单独的 bond dimension 或纠缠效应。

与原 compact_born 比较：半预算 MPO 在 MNIST/Fashion/CIFAR/MRPC 的均值较高，在 SST-2/RTE 较低；半预算幅度 Born 也呈相同的正负任务分布。全预算幅度 Born 在 SST-2/MRPC 较高，在其他四项较低。没有一个新变体在所有任务上稳定提高成绩。

### 时间效率与学习过程

新幅度 Born 两档的串行单步耗时为 full MLP 的 **1.24–1.31 倍**，相对原 compact_born 降低约 **25%–37%**；它确实缓解了第一轮 Born 的速度问题，但仍存在约四分之一到三分之一的每步成本。MPO 两档为 full MLP 的 **0.94–1.09 倍**，在本机这套短程测量中接近 MLP。CIFAR-10 的 MPO 测得略快，其余多数任务略慢；10 个测量步和桌面环境不足以建立微小速度差的普遍结论。

这里测量的是整个分类模型的一次训练步，包含数据取批、传输、前后向、裁剪与优化器，并非只测 gate 内核。每个最佳 checkpoint 使用新 AdamW 做 3 步预热和 10 步计时，续训权重丢弃；没有测量独立推理延迟，也没有用测速续训的分数替换正式成绩。峰值显存是 PyTorch 活跃分配峰值，包含预热，不是整卡 nvidia-smi 占用；模型骨干占比较大，路由参数减半并未带来接近 50% 的整体显存节省。

相同步数的内部 dev 曲线总体相近。SST-2 所有 gate/种子的最佳 checkpoint 都在 600 步；CIFAR-10 两种新半预算 gate 都选在 1000 步，尚有延长训练的验证价值。RTE 的最佳步数较早且种子差异大，例如全预算幅度 Born 对 full MLP 的三个配对差值为 +1.08、-3.61、-0.72 pp。不能把某个 seed 或某个任务的小幅胜出概括为稳定收益。

实际训练秒数曲线混合了历史对照的最多 6 并发与本轮恢复后的最多 3 并发，并且单个 run 的同时运行进程数会变化。图中曲线的横向距离不能用来宣称新 gate 比 MLP 更快；单位时间结论以同一阶段的串行诊断为依据。本轮没有完整等墙钟预算重训，因此不把“更低每步耗时”直接等同于“固定时间下最终准确率更高”。

### 路由与稳定性

90 个新训练的首步路由参数梯度均有限且非零，最佳 checkpoint 的各路由参数均有更新；126 个最终 checkpoint 的预测与路由重评估通过，未发现预测全部落入单一类别的退化。系统蓝屏中断需要与模型数值稳定性区分，具体驱动原因仍未查明。

BERT 的 soft 路由熵接近 ln(16)=2.772589，并不保证 hard Top-4 负载均匀。原 Born 在三个文本任务的负载 CV 为 **1.05–1.13**，每层/种子平均约 **1.61–1.83** 个专家未使用；新 MPO 与幅度 Born 为 **0.36–0.47**，这些任务没有未使用专家。这个路由诊断改善没有转化为一致的准确率改善。图像任务也不是单向改善：例如 MNIST 半预算 MPO 的负载 CV 为 0.60，高于 full MLP 的 0.25。

“soft 概率接近均匀”和“Top-4 选择不均衡”可以同时成立：很小的 logit 顺序差异就能改变稀疏选择。不能仅凭较高熵、较低 CV 或更多活跃专家认定表示更好。所有方法共享原 hard+soft STE，结论只适用于该规则。

### 方法解释与后续优先级

保留半预算 MPO χ=1 为下一轮首要经典 TN 候选，并继续保留 half MLP 作为强对照；如果坚持研究幺正演化方案，优先验证半预算幅度 Born，同时接受它当前更高的时间成本。全预算变体没有显示出足以优先增加预算的稳定收益。

选中的幅度 Born 深度为 1，只执行一轮数据 RY 和 0↔1、2↔3 两组双向 CRY；类名中的 reupload 不代表本次做了多轮数据重上传。对于 4 个物理二值索引，制备后的算符在 (01)|(23) 划分上分解为两个局部算符，不能跨这个切分新增纠缠。初始 16 维联合幅度由经典神经网络制备，本身可以不可分；幅度被限制为正实数，演化不含 RZ 复相位，GPU 中保留完整 16 维状态。它没有 MPS 截断 χ，也没有量子硬件或 shots。

MPO χ=1 的每个线性算符是单站点算符的直积，但整个 gate 还包含联合编码、逐坐标 tanh、残差和 E→E 读出，因此整个非线性网络不等于单个直积算符。与 MLP 的激活、读出和初始化也不同，无法将性能差异单独归因于 TN 结构。

建议后续预先固定新种子和协议，先延长 CIFAR-10/RTE 训练并增加种子，再比较编码-only、移除 CRY、相同编码 rank 下改变 χ/深度，以及固定或均匀路由对照。对 MLP 做相同规模的预算/学习率搜索；用独立验证或嵌套选择处理本轮已反复观察的最终集。2 层 BERT、64 专家、相位与跨切分纠缠消融作为独立实验组开展。本轮没有追加这些训练，也不以当前分差声称量子优势或纠缠贡献。
'''
placeholder = '此处需结合配对差值、效率、路由与各 seed 明细填写人工审阅结论；不由自动脚本按最高均值宣布胜者。'
assert placeholder in s
s = s.replace(placeholder, interpretation)

timing = ['## 完整训练日志耗时（资源竞争下的运行记录）', '',
          '下表为每个成功 run 完整训练的均值 ± 样本标准差；不含被归档中断尝试的额外消耗。训练计时与总墙钟含义沿用 runner：wall 还含 dev/最终评估与保存，但不覆盖模型构建和共享初始化。不同并发条件下不能直接排名。原始字段在 formal_runs.csv。', '',
          '| 任务 | Gate | 训练秒数 | run 墙钟秒数 | 并发吞吐 samples/s |', '|---|---|---:|---:|---:|']
for t in TASKS:
    for g in GATES:
        a = AGG[t, g]
        vals = [f"{a[k+'_mean']:.1f} ± {a[k+'_std']:.1f}" for k in ['training_seconds', 'wall_seconds', 'examples_per_second']]
        timing.append(f"| {t} | {label(g)} | {' | '.join(vals)} |")
s = s.replace('## 学习曲线', '\n'.join(timing)+'\n\n## 学习曲线', 1)

extra = '''
- `formal/runs.csv`、`formal/summary.json`：仓库自带 summarize 的原始产物。
- `formal/serial_efficiency_runs.csv`、`formal/serial_efficiency_summary.csv/.json`：全部 126 项统一串行测量的明细与汇总。
- `COMPLETION_AUDIT.json`：最终交付核验、失败/缺失/未完成清单、图表检查记录；`recovery_audit.json`：6 个中断尝试恢复后 105 条原记录精确复现及归档 hash 检查。
- `git_publication.json`、`source_snapshot_formal/`：提交过程和冻结训练源码；`review_sources/`、`postprocessing_fixes/`：自动报告原稿与后处理修正的可追溯记录。
'''
s += extra
s += '\n报告人工审阅时间：'+datetime.now().astimezone().isoformat()+'（Asia/Shanghai）。\n'
report.write_text(s, encoding='utf-8')
print(json.dumps({'report': str(report), 'characters': len(s), 'placeholder_removed': placeholder not in s}, ensure_ascii=False))
