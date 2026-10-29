# MoTE 路由稳定性：边界训练与共享 Born

本轮独立于已完成的 `autoregressive_gate`，不修改既有冻结 gate、数据或训练代码。
输出目录 `outputs/routing_stability_round1/`，全部产物留在被 Git 忽略的 outputs。

## 两项实现

1. **边界训练**：对中心化分数除以 `sqrt(mean(centered²)+1e-12)`，取 top-4 与第5名间隔 g，损失为 `lambda * mean(relu(0.05-g)²)`。分母参与求导。MLP 与原 Born 使用原 logits；简化 Born 仅在训练损失中取 log(p)。前50%步数关闭，50%–75%线性升至 lambda=0.01。对四个 router 的损失取平均，保留原 CE 与负载均衡损失。eval 时移除 hook，不执行 margin 的 top-5、log 或统计运算。
2. **共享 Born**：`LN(x) -> P -> tanh -> V -> 1+0.5*tanh -> L2 normalize -> U -> square -> normalize/floor -> top-4`。16幅度表示4个二值物理轴；batch/token轴不收缩。U为4个RY和链式 `CRY(0,1),CRY(1,2),CRY(2,3)`，角度可学习、与输入无关。训练时在16个基矢上构造U并通过GEMM作用于token；推理缓存U，参数版本、设备或dtype变化会重新构造。使用实数正交状态向量模拟，不使用量子硬件。

简化版删除逐token angle projection/re-upload和可学习输出gain，固定温度1，直接概率作为软权重；保留现有hard-top-k+soft STE、aux和无token dropping规则。只有gain/temperature=1时，这与原始log(p)->softmax的权重严格等价。因此它是需重新训练的新模型，不是原checkpoint的无损优化。编码中非线性使U不能简单吸收进V；没有无U/无CRY正式消融，不能归因于纠缠或量子机制。

rank按**实际half MLP router参数**匹配，包含LayerNorm、两次投影、旋转参数；所有四个模块分别预算，报告实际计数。原Born half保留原先较小的实际预算。

## 公平性和协议

六组 `mlp_half/mlp_margin/amp_half/amp_margin/born_lite/born_lite_margin`；种子17/29/43，18个全新训练。每个有/无margin对照的完整初始化hash完全相同；所有方法共享非router初始化、专家位置和batch顺序。

复用已验证的WikiText-2 raw训练集BPE4096缓存，模型2层hidden256/FFN1024、context128、batch32、FP32。正式每run4000步=16,384,000训练token。validation全量每500步，最低CE选模，全部训练完成后完整validation/test重评估。test此前实验已观察过，本轮所有参数和目标在新test重评估前冻结；仍须后续新语料确认，不能声称全新盲测。

训练侧校准：32个固定train窗口，使用旧half MLP/Born seed17 checkpoint检验FP16/BF16输入roundtrip后的标准化分数误差；不接触test。初始候选tau=.1、lambda=1的120步pilot虽数值正常却出现固定top4退化，已拒绝并完整保留。训练侧梯度校准发现，tau=.05时即使lambda=.01仍会让margin梯度达到任务router梯度的6.6–28.4倍。因此新候选采用tau=.05、lambda=1e-4，启用时间推迟，另跑600步pilot。该系数对应校准批次的梯度范数比约0.066–0.284；不是对全部训练阶段的保证。pilot按流程/有限值/更新/预算/负载核验，不按gate排名筛选。

主稳定性指标：相同128个validation窗口（16384token/run、4个router），同批重复、batch16→1、输入FP16/BF16 roundtrip、0.1%/1%逐坐标RMS均匀扰动。固定随机方向，同种子和模块跨方法一致。统计top-k**集合**切换和替换比例；不能将集合内次序变化算成切换。另对全模型四个router同时舍入，测量传播后的路由变化、目标token logprob误差尾部和argmax变化。TITAN V上的输入roundtrip不是原生BF16、跨框架、KV cache或RL rollout对齐实验。

工程目标预注册：相对half MLP的PPL增加不超过0.5%、batch16×128完整前向中位延迟增加不超过5%、BF16输入roundtrip的router集合切换率降低至少50%。同时报告相对各自无margin基线；三种子只作描述性证据，不将未显著变差当作非劣性证明。近零事件使用绝对计数；不能把相关token当独立样本扩大置信度。

效率同时报告实际训练耗时、固定checkpoint副本串行训练（包含margin）以及batch16×128/batch1×128/batch1×1完整前向。预热包含U缓存构造；length1前向不是KV缓存decode。学习曲线分别按step和实测训练秒数绘制，后者不是在相同时刻重新评估。

## 执行

Windows环境：`.venv-titan/python.exe`；`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONUTF8=1`，PATH前加`.venv-titan/Library/bin`。离线缓存准备后设置HF_HUB_OFFLINE/HF_DATASETS_OFFLINE。

```
python -m pytest tests/test_routing_stability.py tests/test_autoregressive_gate.py tests/test_advanced_routers.py -q --basetemp=outputs/routing_stability_round1/独立临时目录
python -m fitmotn.experiments.routing_stability.calibrate
python -u -m fitmotn.experiments.routing_stability.pipeline --config experiments/routing_stability/configs/pilot_v2.json
python -u -m fitmotn.experiments.routing_stability.pipeline --config experiments/routing_stability/configs/titan_v.json
```

pipeline单GPU串行运行，独占锁防止重复启动，冻结训练源码/配置哈希。完成项核验后跳过；中断项保留并拒绝覆盖。`last.pt`没有optimizer/RNG，不支持无缝resume，不得假称支持。失败先记录原因，独立归档后同协议从头重跑；不删除失败结果。

正式顺序：训练→审计→全部checkpoint重评估/生成/串行计时→稳定性探针→CSV/JSON/图表/中文报告草稿。`ready_for_review`仅表示自动阶段结束；核验解释、逐张看图、写COMPLETION_AUDIT并结束跟进后才算完成。


## 最终冻结选择（2026-10-28 23:37）

正式采用pilot_v3：tau=.05、lambda=.01，前50%关闭，50%–75%线性增至满强度。600步pilot中三种margin门数值/参数/配对/负载核验通过，无固定top4退化；只有MLP的一处router出现1个未用专家，其无margin基线也有一处1个未用专家。BF16输入舍入切换率相对各自基线：MLP降低20%，新共享Born降低约28.1%，原Born增加约8.6%。这是单seed pilot，不能作为优势结论。弱正则v2亦保留，正式不混入任何pilot数据。选择详见outputs/routing_stability_round1/FORMAL_SELECTION.json；正式配置hash 756b04397b2d8392d676c2c07833d28a9825a510da312c39c11604243f6f7296。
