"""Write the reviewed Chinese report from immutable measured JSON records."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import statistics as st
import shutil

R=Path(__file__).resolve().parent
S=json.loads((R/'raw_summary.json').read_text(encoding='utf-8'))
A={x['variant']:x for x in S['aggregate']}
V=S['config']['gates']
LABEL={'mlp_half':'half MLP','mlp_margin':'MLP + margin','amp_half':'原 Born','amp_margin':'原 Born + margin','born_lite':'共享 Born','born_lite_margin':'共享 Born + margin'}
NOW=dt.datetime.now().astimezone().isoformat()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pm(a,k,d=3):return f'{a[k+"_mean"]:.{d}f} ± {a[k+"_std"]:.{d}f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def stab(v,c):return [x for x in S['stability'] if x['variant']==v and x['condition']==c]
def tails(v,c):return [x for x in S['error_tails'] if x['variant']==v and x['condition']==c]
def counts(v,c):return '/'.join(str(x['changed']) for x in stab(v,c))
def pair(v,b):return next(x for x in S['paired_summary'] if x['variant']==v and x['baseline']==b)
parts=['# MoTE 路由稳定性：边界训练与共享 Born 实测报告',
f'复核完成时间：{NOW}（Asia/Shanghai）。正式组：`formal/756b04397b2d`。',
'''**18/18 新训练、18/18 最佳 checkpoint 完整重评估、18/18 串行训练计时、54/54 推理形状计时、18/18 稳定性探针及 54/54 固定提示生成均已完成。六张图逐张复核，缺失/失败/未完成清单为空。**

本轮两项设计均可完成真实训练，但尚未达到“在接近 half MLP 的质量和延迟下，将输入舍入导致的路由切换减半”的组合目标。共享 Born 的明确工程收益是相对原 Born 降低整模型推理延迟；它相对 half MLP 的 PPL、训练耗时和切换率仍有劣势。边界训练对 MLP、共享 Born 的局部切换有约 10% 和 9% 的描述性改善，但输出误差尾部并不随之普遍改善，不能视为训练—推理错位问题已解决。''',
'''## 协议与可比范围

六组 × seeds 17/29/43，共 18 个全新训练，没有复用旧正式结果。WikiText-2 raw 固定数据版本，4096 词表的 byte-level BPE 仅用训练集拟合；两层因果 LM，hidden256、FFN1024、4 heads、context128、batch32，单 TITAN V FP32。每项训练 4000 步、16,384,000 个采样 token，总计 72,000 步。这里比较的是 **MoTE 的张量块专家路由**，不能直接等同大型分布式 MoE 的整 FFN 专家或网络通信开销。

跨方法按种子共享非 router 权重、专家位置和训练采样顺序；有/无 margin 配对的完整初始权重 hash 相同。九组配对在 margin 开启前 41 个日志点的 CE 完全一致。margin 对四个 router 取平均，阈值 τ=0.05、最大 λ=0.01；前 2000 步关闭，之后线性增加，3000 步达到最大值。训练源码、配置及快照 hash 未变。

每 500 步做完整 validation，以最低 CE 选模。全部 18 项 best_step 均为 4000，尚不能判断更长训练的收敛上限。重新加载后 validation CE 与训练保存值差异全部为 0；每项评估覆盖 322,802 个 validation 预测目标与 368,043 个 test 预测目标。评估 context128、stride64，重叠上下文仅为提供条件，每个预测目标只计一次；路由负载统计包含重复的上下文计算。

下文均值 ± **三训练种子的样本 SD**，不是标准误。PPL 为每 BPE token 的 exp(CE)，表中为各 run PPL 的算术均值，不能与 word-level PPL 混用。种子是训练重复单位，相关 token、四个 router、25 次计时都不能当作新增独立训练样本。''',
'## 语言建模质量与实际参数',
table(['Gate','Test CE（nats/BPE）','Test PPL','4 个 router 总参数','模型总参数','最佳步数'],[[LABEL[v],pm(A[v],'test_ce',5),pm(A[v],'test_ppl',2),f"{int(A[v]['router_parameters_mean']):,}",f"{int(A[v]['total_parameters_mean']):,}",'4000'] for v in V]),
'''共享 Born 比 half MLP 少 74 个 router 参数（0.28%），全模型仅少 0.0044%；这不是可观的模型压缩。原 Born 的 router 参数少 9.60%，全模型少 0.1535%，预算并非严格相等。margin 不增加可训练参数。''',
'### 配对效应',
'正 ΔCE / PPL 变化表示变差；时间比和切换率比小于 1 表示下降。所有比值先按相同 seed 计算，再取三 seed 均值，因此与“均值之比”可能不同。CE 区间为 n=3、df=2 的配对 t 95% 区间，未校正多重比较。',
table(['候选 / 对照','ΔCE ± SD','ΔCE 95%区间','PPL变化% ± SD','BF16切换率比 ± SD','前向延迟比 ± SD'],[[LABEL[x['variant']]+' / '+LABEL[x['baseline']],f"{x['delta_ce_mean']:+.6f} ± {x['delta_ce_std']:.6f}",f"[{x['delta_ce_ci95_low']:+.6f}, {x['delta_ce_ci95_high']:+.6f}]",f"{x['ppl_change_percent_mean']:+.3f} ± {x['ppl_change_percent_std']:.3f}",f"{x['bf16_flip_ratio_mean']:.3f} ± {x['bf16_flip_ratio_std']:.3f}",f"{x['inference_time_ratio_mean']:.3f} ± {x['inference_time_ratio_std']:.3f}"] for x in S['paired_summary']]),
'''共享 Born 相对 half MLP 三个种子的 PPL 均较高；不加 margin 的配对相对变化为 +1.751%、+0.498%、+1.155%，平均 +1.135%。加 margin 后平均仍为 +1.119%。原 Born 相对 half MLP 则随种子反转（−1.325%、+2.419%、+0.349%）；不能从总均值接近就宣称优越或非劣。

各自 margin 对照的 PPL 变化非常小：MLP 平均 −0.0214%，原 Born +0.0118%，共享 Born −0.0152%。共享 Born seed43 加 margin 反而略变差。这些微小差别均没有排除 0 的配对 CE 95% 区间，不能作为语言建模增益的证据；“未显著变差”也不是非劣性证明。''',
'## 路由稳定性：先看事件数',
'''每 run 使用相同 128 个互不重叠 validation 窗口，共 16,384 个 token × 4 个 router = **65,536 次路由决策/条件**。比较 hard top-4 的集合，不把集合内次序变化当切换。FP16/BF16 为 router 隐状态输入舍入后转回 FP32，其余运算保留 FP32；局部测试固定该 router 的参考输入，传播测试则在完整模型的四个 router 同时舍入。

计数顺序均为 seeds 17/29/43。百分比是三种子均值 ± SD；各方法每个 seed 分母相同，三 seed 合并分母为 196,608，但统计推断仍只有三个训练重复。''',
table(['Gate','局部 BF16 计数','BF16切换% ± SD','局部 FP16 计数','batch16→1 计数','传播 BF16 计数'],[[LABEL[v],counts(v,'input_bf16_roundtrip'),pm(A[v],'flip_input_bf16_roundtrip_percent',4),counts(v,'input_fp16_roundtrip'),counts(v,'batch16_vs1'),counts(v,'propagated_input_bf16_roundtrip')] for v in V]),
'''MLP 加 margin 的局部 BF16 事件从 124/123/92 降为 103/110/89，三 seed 配对切换率比平均 0.897，即下降 **10.3%**；合并计数 339→302，下降 10.9%。共享 Born 从 151/152/170 降至 132/143/155，配对平均下降 **9.1%**，三个种子方向一致，但绝对切换率仍高于 half MLP。原 Born 为 122/122/100→132/113/111，平均比值上升 3.9%，没有一致收益。报告区分配对比值均值与合并计数，避免混淆。

三组 margin 在局部 BF16 条件下均未达到 50% 降幅。传播 BF16 的合并事件：MLP 457→416，共享 Born 680→577，原 Born 481→527；它们与单个 router 的孤立敏感度并非同一指标。

同批重复在全部 18 项均为 0 次切换。batch16→1 的事件很少：MLP 合计 3 次、MLP+margin 0 次，原 Born 0 次、原 Born+margin 1 次，共享 Born 5 次、共享 Born+margin 8 次。因此不能把 MLP 的 3→0 宣称为普适的“100%稳定”，也不能对 0 事件基线计算相对降幅。共享 Born+margin 在 batch 变化下反而出现更多切换，说明其舍入收益不具备条件间一致性。零事件仅表示这批窗口未观察到。

局部 FP16：MLP 41→36、原 Born 46→45、共享 Born 77→60。按输入逐 token RMS 缩放的均匀扰动幅值 a=0.001 时，三组为 114→111、141→114、205→150；a=0.01 时为 1106→1102、1192→1194、1642→1524。这里扰动为 a×RMS(x)×Uniform[−1,1]，不是其实际 RMS 精确等于 a；这些辅助条件同样未显示普适的大幅改进。全部条件、替换比例和每个模块的原始记录均保留。''',
'## 输出误差尾部：低切换率不等于低影响',
'''以下是完整模型四个 router 同时 BF16 输入 roundtrip 后，与原 FP32 前向相比的 **目标 token logprob 绝对差**，单位 nats。每 seed 16,384 个预测，三 seed 共 49,152；max 是三 seed 的最大值。p99 列先按每个 batch 求 p99，再取 batch 和 seed 均值，**不是全体 token 的合并 p99**。''']
tail_table=[]
tail_summaries=[]
for v in V:
    z=tails(v,'input_bf16_roundtrip')
    t={'variant':v,'mae_mean':st.mean(x['target_logprob_mae'] for x in z),'mae_std':st.stdev(x['target_logprob_mae'] for x in z),'p99_batch_mean':st.mean(x['target_logprob_p99_batch_mean'] for x in z),'max_across_seeds':max(x['target_logprob_max'] for x in z),'gt01_count':round(sum(x['target_logprob_gt01_percent']*x['predictions']/100 for x in z)),'gt01_percent':st.mean(x['target_logprob_gt01_percent'] for x in z),'argmax_percent':st.mean(x['argmax_changed_percent'] for x in z)}
    tail_summaries.append(t)
    tail_table.append([LABEL[v],f"{t['mae_mean']:.6f} ± {t['mae_std']:.6f}",f"{t['p99_batch_mean']:.5f}",f"{t['max_across_seeds']:.5f}",f"{t['gt01_count']}/49152 ({t['gt01_percent']:.4f}%)",f"{t['argmax_percent']:.4f}"])
parts += [table(['Gate','MAE ± SD','batch p99 均值','最大差','差>0.1 次数（比例）','argmax变化%'],tail_table),
'''**MLP 的局部切换减少，却伴随尾部恶化。** MAE 0.000845→0.000893，差>0.1 的事件 42→64，最大差 0.306→0.482；三个 seed 的差>0.1 事件均增加。这说明边界损失只减少部分切换，没有同时约束被替换专家的输出差异或误差传播。

共享 Born 加 margin 后 BF16 的 MAE 0.001206→0.001050、batch p99 均值 0.02348→0.01802、全样本最大差 1.241→0.465，三个 seed 的 MAE 和 batch p99 均降低；但差>0.1 事件只从 106→94，seed29 仍略增加。即使改善后，其 MAE、batch p99 和大误差事件数仍高于 half MLP。原 Born 的均值和差>0.1 事件有所下降，却有最大差 0.427→0.849 的恶化。最大值对样本敏感，只报告观察值，不作总体尾风险保证。''']
aux=[]
for v in V:
    f=tails(v,'input_fp16_roundtrip');b=tails(v,'batch16_vs1')
    aux.append([LABEL[v],f"{st.mean(x['target_logprob_mae'] for x in f):.6f}",f"{max(x['target_logprob_max'] for x in f):.5f}",round(sum(x['target_logprob_gt01_percent']*16384/100 for x in f)),f"{max(x['target_logprob_max'] for x in b):.6f}",round(sum(x['target_logprob_gt01_percent']*16384/100 for x in b))])
parts += [table(['Gate','FP16 MAE均值','FP16最大差','FP16差>0.1次数/49152','batch变化最大差','batch差>0.1次数/49152'],aux),
'''batch 变化下所有模型的 next-token argmax 均无变化，但共享 Born+margin seed43 出现 1 次目标 logprob 差>0.1；其余方法此项均为 0。FP16 条件也未复现共享 Born 在 BF16 下的全面尾部改善：其差>0.1 事件从 13 增至 17。上述结果针对相同固定权重和给定条件，**没有直接比较训练模式与部署推理引擎，也没有运行 RL rollout 对齐实验**。''',
'## 路由利用率与边界',
table(['Gate','soft概率熵（nats）±SD','hard负载CV±SD','未用专家（每模块均值）±SD','标准化gap的batch p05均值','gap<0.05比例%'],[[LABEL[v],pm(A[v],'entropy',3),pm(A[v],'load_cv',3),pm(A[v],'unused_experts',3),f"{A[v]['gap_p05_batch_mean_mean']:.6f}",f"{A[v]['gap_below_threshold_percent_mean']:.3f}"] for v in V]),
'''正式训练末期没有出现 pilot_v1 的固定 top-4 退化。完整 validation 的模块最大 CV 为 0.877；validation/test 都只有 `mlp_margin_s29 / blocks.0.up` 的 1 个专家未被使用，其余所有模块为 0。表中 0.083 是四模块和三 seed 的均值，不是“0.083 个实际专家”。共享 Born 在部分早期步骤有暂时集中：seed43 第100步最多 11 个未用专家，随后恢复，发生在 margin 开启之前；原始诊断保留于 `BORN_LITE_EARLY_ROUTING_CHECK.json`。不能隐去这个训练瞬态。

原 Born 的 hard 负载 CV 约 0.456，低于 MLP 的 0.647 和共享 Born 的 0.662；其 soft 概率熵却较低，说明软分布熵与跨 token 的 hard 专家利用率是不同量。共享 Born 的高 soft 熵不保证 hard top-k 均衡或稳定。

边界损失略提高低分位间隔，但最终低于阈值的比例仍很高：MLP 约56%、原 Born 29%、共享 Born72%。MLP 与共享 Born 的 score 分布存在接近边界或相等的情况，不能把平均 gap 变大当作最坏情况保证。gap 使用 MLP/原 Born 的 logits、共享 Born 的 log(p) 标准化，跨架构值不直接代表相同输入扰动容限。

probe 同时检查逐决策充分条件 gap > 2×实测最大中心化分数误差，所有满足条件的决策均未切换，执行中的断言全部通过。这是对**已测扰动的事后充分条件**，并非对任意未来浮点误差的认证；未满足条件也不代表必然切换。''',
'## 真实耗时、吞吐与显存',
'''训练计时包含 forward、CE/aux/margin、backward、裁剪、AdamW 和 batch 传输。串行微基准每 checkpoint 使用新 optimizer、5 步预热及 20 步测量，更新后的副本丢弃；它不是继续训练，也不是保存 optimizer/RNG 后的恢复。margin 微基准采用满强度，整次训练只有后半程启用，二者不能混用。显存为该进程 torch peak allocated，包含预热，不是全卡占用，也未单独记录推理峰值。''',
table(['Gate','实训计算秒±SD','单run壁钟秒±SD','串行step ms±SD','串行token/s±SD','训练峰值MiB±SD'],[[LABEL[v],pm(A[v],'training_seconds',1),pm(A[v],'wall_seconds',1),f"{A[v]['serial_step_seconds_mean']*1000:.2f} ± {A[v]['serial_step_seconds_std']*1000:.2f}",pm(A[v],'serial_tokens_per_second',0),pm(A[v],'peak_allocated_mb',1)] for v in V]),
'''18 项训练壁钟合计 **6.687 小时**；pipeline 23:41:20 开始、06:52:59 完成自动阶段，约 7 小时 12 分钟，另有本次报告复核。计算秒排除 validation 和 checkpoint 保存；壁钟包含这些开销。所有组训练 token 相同，**没有固定墙钟预算的独立训练**。时间曲线是在相同步数日志上汇总各 seed 的实测计算秒，不是每个方法在同一时刻重新评估。

相对各自无 margin，串行每步耗时平均增加 MLP 3.12%、原 Born 4.11%、共享 Born 4.06%。共享 Born 相对原 Born 串行每步下降约4.8%；相对 half MLP 却增加42.3%。共享 Born+margin 相对 half MLP 增加48.0%。因此推理端的缓存收益没有转化为同幅度的训练收益。''']
infer=[]
for v in V:
    a=A[v]
    throughput=st.mean(2048000/x['inference_b16_l128_ms'] for x in S['runs'] if x['variant']==v)
    infer.append([LABEL[v],pm(a,'inference_b16_l128_ms',3),pm(a,'inference_b1_l128_ms',3),pm(a,'inference_b1_l1_ms',3),f'{throughput:.0f}'])
parts += [table(['Gate','b16×128 ms±SD','b1×128 ms±SD','b1×1 ms±SD','b16×128 token/s均值'],infer),
'''推理为 GPU 驻留输入的整模型前向，每形状 10 次预热、25 次测量取中位数，再汇总三 seed。共享 U 在预热中构造并缓存。b1×1 是一个 token 的完整前向，**无 KV cache，不是连续自回归 decode 吞吐**；b16×128 的 token/s 是前向吞吐，不能当作文本生成速度。

无 margin 共享 Born 相对原 Born 的 b16×128 延迟比 0.755，即减少 **24.5%**；三个 seed 分别减少24.8%、24.5%、24.2%，这是本轮较清楚的工程收益。其平均延迟66.998ms仍比 half MLP 的62.687ms慢约6.9%。共享 Born+margin 为64.833ms，配对平均慢3.4%；但 margin 在 eval 不执行额外算子，不能把同架构不同 checkpoint 的小幅计时差全归因于边界训练“加速”。这是串行一次测量、固定执行顺序下的实测，后台桌面负载、GPU频率与专家激活分布均可能影响延迟；未做随机交错多轮计时，约几个百分点的差异需复测。''',
'## 预先定义的工程目标核对',
'目标均相对 half MLP：PPL 相对增加≤0.5%、b16×128 前向延迟增加≤5%、局部 BF16 输入舍入切换率降低≥50%。以下为描述性均值和各自达标种子数，不能替代非劣性或稳健性统计证明。']
targets=[]
for v in V[1:]:
    q=[x for x in S['paired'] if x['variant']==v and x['baseline']=='mlp_half']
    p=pair(v,'mlp_half')
    t={'variant':v,'baseline':'mlp_half','ppl_change_percent_mean':p['ppl_change_percent_mean'],'inference_ratio_mean':p['inference_time_ratio_mean'],'bf16_flip_ratio_mean':p['bf16_flip_ratio_mean'],'ppl_pass_seeds':sum(x['ppl_change_percent']<=.5 for x in q),'latency_pass_seeds':sum(x['inference_time_ratio']<=1.05 for x in q),'bf16_pass_seeds':sum(x['bf16_flip_ratio']<=.5 for x in q)}
    t['combined_mean_target_met']=t['ppl_change_percent_mean']<=.5 and t['inference_ratio_mean']<=1.05 and t['bf16_flip_ratio_mean']<=.5
    targets.append(t)
parts += [table(['候选','PPL变化%（达标seed）','延迟变化%（达标seed）','切换变化%（达标seed）','组合目标'],[[LABEL[t['variant']],f"{t['ppl_change_percent_mean']:+.3f} ({t['ppl_pass_seeds']}/3)",f"{100*(t['inference_ratio_mean']-1):+.2f} ({t['latency_pass_seeds']}/3)",f"{100*(t['bf16_flip_ratio_mean']-1):+.2f} ({t['bf16_pass_seeds']}/3)",'未达到'] for t in targets]),
'''没有候选达到组合目标，BF16 减半目标在所有候选的三个种子中均未达到。共享 Born+margin 的延迟已接近 half MLP，但质量平均约差1.12%、切换率平均高30.4%，不能据延迟一项建议直接替换。MLP+margin 的局部稳定性改善也不足以抵消输出尾部的反向观察。

本轮支持继续保留“输入无关 U + 推理缓存”的工程思路，但没有证据支持 QuantumGate 固有地更抗数值扰动。若后续推进，应把训练期边界间隔与专家输出差异、完整模型误差尾部共同作为目标，并在独立语料和真实部署精度/引擎下确认。此处只是后续研究判断，本轮没有追加训练、改阈值或按正式 test 排名再选择配置。''',
'## 生成样例与语言能力限制',
'''已读取全部 54 条固定提示、greedy 64-token 续写；每 gate/seed 三条均保留，不筛选。各组普遍出现短语重复、主题漂移、WikiText 标题符号循环，尚不具备稳定的长文本连贯性。例：seed17、提示 `The city is located`，多个方法持续重复 “city of the city”；seed43 的共享 Born 两组在 `In the early twentieth century` 后出现大量句点。PPL 的小幅变化没有对应可确认的生成质量改善，也不能据单个样例给 gate 排名。完整文本另存 `generation_samples.md`，原始 token IDs 在 `generation_samples.json`。''',
'## 失败 pilot、校准和选择过程',
'''1. v1 `pilot/5b7d1b6a5189`，6×120 步，τ=.1、λ=1、过早启用。数值检查通过，但 margin 门退化至近固定 top-4，负载 CV 约1.73、最多12个未用专家，已明确拒绝。原始 checkpoint、曲线、配置与源码快照全部保留，绝不把低切换率记作成功。
2. 训练侧 `CALIBRATION.json`、`GRADIENT_CALIBRATION.json` 发现早期标准化边界损失的梯度相对任务梯度过强。v2 `pilot/2c58817d07ec`，6×600步，τ=.05、λ=1e-4、25%启用到50%满强度；流程正常，无一致稳定性收益。
3. 成熟 checkpoint 的训练侧梯度校准 `MATURE_GRADIENT_CALIBRATION.json` 后，v3 `pilot/1361f2f058eb` 采用τ=.05、λ=.01、50%启用到75%满强度。三个 margin 门600步与probe通过，与v2同协议无margin对照比较，BF16切换为MLP85→68、原Born81→88、共享Born89→64。这是单seed的混合证据，只用于流程/梯度/负载可行性选择，不据此宣称赢家。
4. `FORMAL_SELECTION.json` 记录最终配置 hash。正式六组均重新训练，没有把不同 pilot 配置混合进三seed统计。正式过程中共享 Born 早期短暂集中也保留了诊断。''',
'## 解释边界',
'''- 共享 Born 是实数正交状态向量模拟：有界联合幅度编码→共享4个RY和3个链式CRY构成16×16矩阵→平方概率读出。相对原 Born 同时删掉输入相关角度和可学习gain，并使用直接概率读出，是重新训练的新模型；不能把效果归因于仅缓存这一变化。
- 原 Born/共享 Born 与经典 MPO 是不同结构。本轮没有 MPO、full MLP、无U、无CRY或经典正交平方读出的正式组；不拿旧轮协议/结果填补这些消融。没有量子硬件运行，也没有量子或纠缠优势证据。
- 输入FP16/BF16 roundtrip不是原生低精度矩阵运算，也不是跨设备/跨框架/分布式/RL训练与推理一致性测试。专家输入本身没有整体降精度。hard top-k的离散边界仍在，不能保证任意误差零切换。
- 单语料、小模型、3seed、固定窗口限制外推；本轮test在旧实验已被观察过，虽然本轮参数在统一test重评估前冻结，仍不应声称全新盲测。以后需要独立语料。
- 标准化gap与soft概率熵只是诊断，不是最终任务稳定性。低切换必须同时通过质量、hard负载、多条件误差尾部检查，不能用恒定路由规避问题。
- 没有随机交错多轮计时、没有单独推理峰值显存、没有KV cache解码，也没有固定墙钟预算训练。实测吞吐和时间结论仅适用于本机和此实现。''',
'## 六图复核记录']
figure_notes={
'learning_steps':'9个validation步数点完整；均值和样本SD阴影与curves.csv一致，右图显示2000–4000步。六组曲线接近，单seed差异大于margin均值变化。',
'learning_time':'横轴为各相同步数下的平均实测计算秒；MLP较快，不能把连线当作固定时刻重新验证。评估/存盘不计入横轴。',
'quality_stability':'Test PPL均值±样本SD与局部BF16集合切换率分开展示，y轴从0起。共享Born切换率高于MLP；颜色和误差棒与aggregate一致。',
'efficiency':'串行训练token/s与b16×128整模型前向ms分别展示。训练包含margin，eval移除其分支；共享Born相对原Born的推理收益明显，训练仍慢于MLP。',
'boundary_margin':'左图为batch p05均值而非全局p05；右图阈值0.05。旧Born与另两类分布不同，不能单凭跨结构gap大小断言稳定。',
'expert_load':'hard load CV和未用专家每模块均值对应test路由记录。右图对称±SD下界可小于0，这是误差棒而非负专家数；实际只发生MLP+margin seed29的一模块一个未用专家。'}
for name,note in figure_notes.items():
    parts += [f'### {name}',note,f'![{name}]({(R/"figures"/(name+".png")).as_posix()})']
parts += ['六图均已逐张打开检查：标签、图例、误差棒、坐标和数据对应，无遮挡或裁切；未改动原始图。所有柱状图误差棒均为三个训练种子的样本SD。',
'## 每 seed 配对明细',
table(['候选 / 对照','seed','ΔTest CE','PPL变化%','BF16切换率比','前向延迟比'],[[LABEL[x['variant']]+' / '+LABEL[x['baseline']],x['seed'],f"{x['delta_ce']:+.6f}",f"{x['ppl_change_percent']:+.4f}",f"{x['bf16_flip_ratio']:.4f}",f"{x['inference_time_ratio']:.4f}"] for x in S['paired']]),
'## 交付与完整性',
'''独立CPU复核 `review_audit.py` 已逐项检查18项结果/manifest/采样哈希、18个best和18个last的所有tensor有限值及strict加载、实际参数计数、冻结源码/配置、全部重评估checkpoint哈希、完整评估token数、54条生成、三种推理形状的原始1350个计时值、360个串行训练计时值、4608条路由条件比较记录及所有CSV与JSON一致性。72,000个训练步骤均有限且router梯度非零；所有router参数及共享参数有更新。九组margin配对初始化和开启前CE一致。

正式组 `audit.json` 通过；`missing_failed_unfinished.json` 的训练/后处理清单均为空。18/18评估与18/18探针指向相同的各run最佳checkpoint，validation重载差异为0。controller正常退出、流水线锁已释放、错误日志为空。这里“完成”表示授权的实验矩阵和核验已完成，**不表示研究假设或工程目标已经成立**。

原始数据与分析文件：

- `runs.csv`（18项）、`aggregate.csv`（6组）、`paired_deltas.csv`（24项配对）、`paired_summary.csv`（8组比较）。
- `curves.csv`（1620条）、`routing.csv`（72条test模块记录）、`stability.csv`（144条条件汇总）、`error_tails.csv`（54条条件汇总）、`raw_summary.json`。
- `generation_samples.json` 与 `generation_samples.md`；`figures/` 六张PNG；`FIGURE_REVIEW.json`；`REVIEW_FINDINGS.json`。
- `formal/756b04397b2d/runs/`：每run的`manifest.json/result.json/evaluation.json/stability.json/curves.jsonl/routing.jsonl/batch_starts.npy/best.pt/last.pt`。
- `REVIEW_DATA_AUDIT.json`：独立数据复核和原始产物sha256；`COMPLETION_AUDIT.json`：最终报告签核和交付hash；`SESSION_STATE.md`与`STATUS.json`：完成状态。
- 全部pilot、校准、选择记录、源码快照和日志保留。自动报告草稿及完成前状态另存`review_history/`，训练阶段审计保留其当时的未完成标志。

没有修改冻结源码/配置，没有覆盖正式run，没有重启旧实验，没有新增GPU训练。定期跟进在最终交付签核后删除。''']
history=R/'review_history';history.mkdir(exist_ok=True)
for name in ['REPORT_zh.md','SESSION_STATE.md','STATUS.json','ACTIVE_PROCESS.json','DELIVERY_CHECK.json','AUTOMATION.json']:
    if not (history/name).exists():shutil.copyfile(R/name,history/name)
(R/'REPORT_zh.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
gs=json.loads((R/'generation_samples.json').read_text(encoding='utf-8'))
gtext=['# 全部固定提示生成样例','18模型×3固定提示，greedy，64 new BPE tokens，无KV cache；完整呈现，不筛选。']
for g in gs:
    gtext += [f"## {g['variant']} / seed {g['seed']} / {g['prompt']}",g['decoding'],'```text\n'+g['prompt']+g['continuation']+'\n```']
(R/'generation_samples.md').write_text('\n\n'.join(gtext)+'\n',encoding='utf-8')
(R/'FIGURE_REVIEW.json').write_text(json.dumps({'time':NOW,'all_six_visually_reviewed':True,'figures':[{'file':name+'.png','sha256':sha(R/'figures'/(name+'.png')),'review':note,'passed':True} for name,note in figure_notes.items()]},ensure_ascii=False,indent=2),encoding='utf-8')
(R/'REVIEW_FINDINGS.json').write_text(json.dumps({'time':NOW,'tail_summary_bf16':tail_summaries,'engineering_targets':targets,'all_combined_targets_met':False,'all_54_generations_read':True,'generation_quality':'All method families show substantial repetition and topic drift; no generation-quality win inferred.','limitations':['Only input rounding in FP32, not native mixed precision or cross engine/RL','Three seeds on one small corpus/model','No no-U/no-CRY/classical orthogonal controls','Timing measured serially without randomized interleaved repeats','No KV-cache decoding or isolated inference memory peak'],'conclusion':'Feasible training; cached shared Born improves forward latency relative to original Born, but no candidate meets the combined half-MLP quality/latency/50%-flip-reduction target. Margin effects are modest and condition dependent; output error tails must be assessed separately.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Reviewed report written:',R/'REPORT_zh.md')
print('Characters:',sum(len(x) for x in parts),'generations:',len(gs),'figures:',len(figure_notes))
