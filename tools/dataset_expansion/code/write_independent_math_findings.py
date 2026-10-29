"""Independent, content-bound manual review; no original records are modified."""
import collections,datetime,hashlib,json,math
from pathlib import Path

BASE=Path('D:/files/pydocs/MOTE/Datasets/expansion_20261016/math/audit')
sample_path=BASE/'manual_sample.json'
raw=sample_path.read_bytes();samples=json.loads(raw)
# Review scope: full question, full final explanation and supplied reasoning tail.
# These are manual findings, not a proof checker and not a full-CoT approval.
notes=[
 ('quarantine','underdetermined_question_and_unjustified_single_answer',
  '一般位置并不能确定分割数；final 自己承认配置相关，却自行改成凸位置/最小值并强答 n。',
  '反例 n=2：凸四边形有 2 个可分的无序 2+2 划分；三角形及一个内部点有 3 个。可取三角形 (0,0),(4,0),(0,4)，内部点 (1,1)，均无三点共线。参考答案 n 的符号匹配不能挽救题意。'),
 ('no_change','valid_iterative_scheme_with_conditions',
  '固定点迭代和 Newton 更新式正确；题目无具体系数，给算法是合理回答。',
  'g=x-f，Newton 分母为 1-f\'，与答案一致。收敛需存在根、适当初值及局部条件；迭代差阈值本身不提供通用真误差界。此项未证明任意参数一定收敛。'),
 ('no_change','valid_under_explicit_syntactic_formula_interpretation',
  '按恰好 n 个连接符、按语法树区分公式的解释，递推及 Catalan 求和正确。',
  'F(0)=1，F(n)=F(n-1)+4 Σ F(i)F(n-1-i)。生成函数满足 G=1+xG+4xG²；两个二项系数写法因 C(n+k,2k)=C(n+k,n-k) 等价。未把公式按逻辑等价类合并，final 已说明计数解释。'),
 ('no_change','equivalent_inverse_function_differential_equation',
  '逆函数导数换元正确；最终方程在 x\' 非零条件下等价于参考 x\'\'=0。',
  '令 p=dx/dy，则最终左端为 -p p\'/(1+p²)²。原函数可微且逆函数光滑时 p 非零，故与 x\'\'=0 等价。不能只按最终文本形状判不相等。'),
 ('quarantine','dimension_error_unsupported_matrix_choice_and_field_error',
  '原问题 Bu=c 维度不成立，且未给出具体 A；回答自行改成 Bc=u 并选择系统矩阵。更严重的是把实数伪逆公式推广到 Hamming 的 GF(2) 情形。',
  '对 final 展示的 A，在 GF(2) 上 AᵀA 的首行首列全 0，其右下 3×3 全 1，秩只有 1，故 (AᵀA)^{-1} 不存在。选前四坐标的 B 只对所假设系统矩阵成立；不能作为原题的一般答案。'),
 ('no_change','valid_constructive_decomposition_not_uniqueness_claim',
  '在 final 明确设 g=x+arctan(x)、f=(2-x arctan(x))/(1+x²) 后，给出的 h 和 k 满足恒等式。',
  'h g+g\'=[-x(x+arctan x)+x²+2]/(1+x²)=f。若 h 完全不受限制则有其他表示，配系数不是唯一性证明；题目只要求找一组，因此不据此隔离。'),
 ('review','symbolic_mismatch_is_semantic_but_numeric_work_contains_error',
  '闭式 acosh(cosh3·cosh4) 及大于 5 的结论正确，符号 false 主要来自把文字和不等式当等式解析；但数值近似存在实际误算。',
  '曲率 -1 的标准归一化下，cosh3·cosh4=274.93005789501365，c=6.309660603466953。final 写乘积 274.897、c≈6.309；保留待审而不是称错误闭式或直接全放行。若取三位小数应为 6.310。'),
 ('clear_symbolic_mismatch_only','verifier_cannot_compare_natural_language_equivalent',
  'final 的正弦定理推导给出 BX/BY=R1/R2；自然语言盒内答案与公式参考一致。',
  '在通常非退化割线约定下，∠BAX 与 ∠BAY 相同或互补，因此正弦相同，两圆弦长比为半径比。l=AB 等退化情况不应解释为有定义的 0/0。仅解除符号不匹配门禁。'),
 ('clear_symbolic_mismatch_only','valid_modular_witness_not_identical_polynomial',
  '题目要求找任意一个模 1331 满足条件的整数系数多项式，不要求与参考多项式逐项相等。',
  'prediction-reference=-1331X²，故在模 1331 下相同。(1+11)^n≡1+11n+121 n(n-1)/2，且 121·666≡726 (mod1331)，证明预测式满足所有自然数 n；不靠有限抽样代替证明。'),
 ('review','equivalent_extrema_but_final_contains_derivative_inaccuracy',
  '最大最小值与参考代数等价，符号 false 属解析失败；但 final 说关于 v 的导数在上端点也消失，严格而言不成立。',
  '38/27-2√2/9=(38-6√2)/27。令 a=1+√2，P(v)=√(1+av)(v/√2-1)，其导数唯一零点 v=(2-√2)/3；在 v=1+√2 时导数为正。该端点确给最大值，只是不应声称是 v 的驻点。数学结论可接受，当前解释建议仍待审，不能泛称全推理正确。'),
 ('clear_symbolic_mismatch_only','equivalent_product_notation',
  'Gamma 递推和乘积缩放正确；解析器把省略号错误识别成额外符号。',
  'Π(n/2+k)=2^{-h}Π(n+2k)，故两式相等。按原式有定义的 n>0、h 非负整数解释，h=0 用空积 1。若 N 含 0，n=0 原 Gamma 比值无定义，此域限制在参考答案也同样存在。'),
 ('clear_symbolic_mismatch_only','defined_derangement_identity_not_resolved_by_parser',
  'final 已定义 D_m，并用 Pascal 恒等式推导 a_n=D_n+D_{n-1}，与参考容斥求和相同。',
  'n-1 个禁位处于不同行列，固定 j 个禁位留下 (n-j)! 个排列。D_n 的二项系数分拆第二项恰为 -D_{n-1}，故等价性有明确推导。仅解除符号不匹配门禁。'),
 ('no_change','geometry_result_checked_by_coordinates',
  '坐标推导支持 BC=2、面积 4，未发现明确错误。',
  'C=(0,0),D=(1,0)，小圆心为 (1/2,±1)，大圆心 (b/2,2)。相切于 C 要求两圆心和 C 共线，b>0 取正号并得到 b=2。题首残余文字是可清理的排版噪声，不足以判题无效。'),
 ('no_change','integer_solution_count_valid_display_encoding_artifact_retracted',
  'k=5 及三组正整数解正确。文件真实中文正常；此前终端乱码是显示问题，不能据此隔离。',
  '(2x-5)(2y-3)=2k+15。k>0 时右边≥17；正整数 x,y 所能给出的两个负因子乘积最多 3，故只能都正。解数等于正因子数，为奇数当且仅当 2k+15 为平方，首个可行奇平方25。UTF-8 JSON 读取后 question/final 的 U+FFFD 均为0。'),
 ('no_change','periodic_sequence_sum_checked',
  '四项周期 (-1,-1,1,1)，2017=4·504+1，答案 -1 正确。',
  '指数 n(n+1)/2 的奇偶依 n mod4 循环；每周期和为0。'),
 ('no_change','linear_counting_argument_checked',
  '四种卡片组合互斥且完备，方程相加得总人数50，混合组合40，因此同类三张者10。',
  'A+B=20，C+D=30，B+C=40，所以 A+D=10。题目是可用三张中的两张组词，final 正确按至少两张/至少各一张解释。'),
 ('quarantine','truncated_final_despite_answer_in_reasoning',
  'final 在未闭合的 \\frac{1 处结束，没有交付完成解释或最终答案；推理里曾算出答案不能代替完整 final。',
  '本轮不从 reasoning 中摘答案补成完整样本。按 AY=6、距离√15 可得 sin(A/2)=√15/6，AH=2R|cos A| 与正弦定律可支持参考 4√35；这不能消除截断。'),
 ('quarantine','answer_leakage_and_missing_game_lower_bound',
  'question 明含 Answer: 2^{50}；final 仅证明 Eva 用配对能把种类数限制到 2^{50}，未证明 Alice 面对任意 Eva 策略可保证至少这么多。',
  '某种 Eva 策略允许 2^{50} 种配置，不等于极小极大值已经证出。即便数值与参考一致，题面泄漏和缺失的下界证明均不能被答案匹配覆盖。不自行删答案后沿用原教师回答作为无泄漏样本。'),
]
findings=[]
for i,(s,note) in enumerate(zip(samples,notes)):
 action,classification,conclusion,evidence=note
 findings.append({'sample_index':i,'id':s['id'],'source':s['source'],'sample_group':s['sample_group'],
  'input_sha256':hashlib.sha256(s['question'].encode('utf-8')).hexdigest(),'output_sha256':s['output_sha256'],
  'action':action,'classification':classification,'conclusion_zh':conclusion,'evidence_zh':evidence,
  'review_scope':'full question, full final explanation, supplied reasoning-tail excerpt; full hidden/earlier reasoning not audited',
  'process_verified':False,'full_cot_correctness_claimed':False,
  'override_scope':'clear only local symbolic mismatch; preserve every other gate' if action=='clear_symbolic_mismatch_only' else 'content-bound manual disposition; do not modify source text',
  'replacement_character_counts':{'input':s['question'].count('\ufffd'),'final':s['full_final'].count('\ufffd')}})

report={'created_at':datetime.datetime.now().isoformat(),'reviewer':'independent_code_data_agent',
 'sample_file_sha256':hashlib.sha256(raw).hexdigest(),'sample_size':len(samples),
 'sampling_caveat':'Purposeful 18-record sample including six symbolic mismatches, not a random prevalence estimate.',
 'hash_scope':'input hashes computed over exact UTF-8 question strings (the sampler copies input); output hashes supplied by manual_sample.json and must be matched to target output before applying. No full 6GB rescan was performed.',
 'action_counts':dict(collections.Counter(r['action'] for r in findings)),
 'numeric_spotcheck':{'hyperbolic_c':math.acosh(math.cosh(3)*math.cosh(4)),'cosh_product':math.cosh(3)*math.cosh(4)},
 'findings':findings,
 'filter_recommendations_zh':[
  '高置信结构门禁：分别检测 final 的闭合数学括号/花括号、未完成命令和答案缺失；不要因为 reasoning 中有 boxed 就覆盖 final 截断。转义花括号、代码块和合法不配对文本必须分别处理，避免纯字符计数误杀。',
  '显式题尾 Answer/答案加参考内容可标泄漏；含 Answer 一词本身不应一律拒绝，需结合命令、位置与答案匹配。要求解释已知结论的证明题也不能仅因题干有结论就拒绝。',
  '题意修补/欠定用软筛：final 同时出现缺条件/不唯一/假设另一题等信号且给单一答案，送 review；不能把正常的条件讨论或反证假设统一隔离。人工确认后才按双 SHA 覆盖。',
  '类型与域检查：矩阵乘法维度、已知输入是否实际给出、实数/GF(p) 运算域应分开核查。伪逆等公式只能在适用域及可逆条件满足时通过；一般语义筛选先 review。',
  '符号 false 是未能证等价，不是答案错误。先识别表达式/不等式/自然语言/存在性构造/模同余等任务；文字盒、命名序列与省略号送专门等价复核。',
  '模同余题允许不同多项式见证；可用系数同余作为充分条件，不能要求严格多项式相等。任何自动验证都保留适用域与限制。',
  '数值 spotcheck 应独立于闭式等价。第6条解析答案正确但数值过程不准确，第9条极值正确但驻点描述错误；只解除 matcher 失败不能把整条 CoT 标成已验证。',
  '乱码检测必须基于 UTF-8 解码后的真实字符。终端显示乱码不是数据损坏；18条 question/final 的 U+FFFD 均为0。',
  '博弈最优性、极值、最小值类证明应检查可达性与界两个方向。自动模型只能发现风险，不能凭关键词证明完整性。',
  '所有人工覆盖都绑定 input 和 output SHA，clear_symbolic_mismatch_only 不绕过去污染、许可、完整性、泄题或其他证据门禁；process_verified 保持 false。'
 ]}
target=BASE/'independent_manual_findings.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(target),'action_counts':report['action_counts'],'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
