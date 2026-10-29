"""Independent review of twenty post-gate records, without source mutations."""
import collections,datetime,hashlib,itertools,json,math
from pathlib import Path
BASE=Path('D:/files/pydocs/MOTE/Datasets/expansion_20261016/math/audit')
raw=(BASE/'postgate_sample_20.jsonl').read_bytes()
rows=[json.loads(x) for x in raw.splitlines()]
notes=[
 ('no_change','counterexample_bounds_checked',
  '所给非负可和序列反例成立。A(s) 下界为常数乘 log(1/s)^(-0.5)，B(s) 下界为常数乘 s^-2 log(1/s)^(-1.2)，平方根积分按 log 指数0.85发散。',
  '独立核对了指数、变量替换及尾和/区间和下界；未逐字认证全部先前 CoT。'),
 ('no_change','coordinate_answer_checked',
  'A=(0,0),C=(α,0),B=(α/2,√3α/2),D=(0,-α)，距离平方为 α²(2+√3)，最终式正确。',
  '题干明确 exterior，因此这里的外侧选择有依据。E 与所问长度无关，只是多余条件。'),
 ('quarantine','explicit_false_claim_about_radical_cancellation',
  '任意小正值的构造正确，但 final 明确声称不同被开方数多重集必不抵消为0，这是错误命题：√1+√9−√4−√4=0，而 {1,9}≠{4,4}。',
  '正确结论是正值集合没有最小元、下确界0。不能从“平方自由根的线性无关”推出任意自然数根号项的多重集不同就非零；不能把答案与参考同为0视作过程正确。'),
 ('no_change','geometry_answer_not_disproved_partial_check',
  '未发现可确认的数学错误；阅读 CoT 尾部发现其给出一般代数关系，不只是 final 无解释。独立坐标实例支持结果，但不把单例当一般证明。',
  'A=(0,0),B=(6,0),C=(2,4),O=(3,1)，预测 E=(10−2√10,0)，求交后 LM/MK=1/2=AD/DB。K 在边的延长线上，按直线 AC/BC 解释可成立；未完成所有退化情形和一般证明认证。'),
 ('quarantine','insufficient_hypotheses_for_nondegenerate_pentagon',
  '线性重建公式正确，但“给定中点无三共线”不足以保证得到通常意义的五个不同顶点的五边形，final 无条件承诺存在。',
  '明确反例中点 X1=(0,0),X2=(1,0),X3=(2,1),X4=(1,3),X5=(0,2)，任取三点均不共线。唯一线性解为 A=B=(0,0),C=(2,0),D=(2,2),E=(0,4)。如果允许退化闭合五段链，则须在题目/回答明确该解释；不静默修题。'),
 ('review','demonstrably_invalid_proof_step_answer_not_disproved',
  'final Step4 对完全对称的式(3)交换 a,b 后相减，却声称得到 c(2g(b)−2g(a))=0；实际左右都不变，只能得到0=0。',
  '在 CoT 最后14,000字符也找到相同错误，并非 final 摘要漏掉一条已证明关系。完整函数分类可能正确，当前没有证明其错误；已确认的是这条关键排除混合奇偶分支的证明无效。不得标为正确推理样本。'),
 ('no_change','coloring_and_construction_checked',
  '行、列按模3染色均要求空格坐标≡1 mod3，故最多16处。充分性可明确构造：空格上、下行数都为3的倍数，用竖条铺；空格所在行左右长度均为3的倍数，用横条铺。',
  '补充了对全部16位置的独立构造核验，未运行下载程序。'),
 ('no_change','outer_measure_example_valid_minor_cover_convention',
  'Vitali 集及其在[0,1]内补集是正确构造。证明可将可测覆盖 M,N 直接选在[0,1]内，从而所用包含关系成立。',
  '原文未显式写 M,N⊂[0,1]；这是一个可无损施加的覆盖选择，不据此判答案错误。若不作此选择，M\\V⊂M∩N 字面上不成立，但外区差集为零测仍可完成证明。未完整认证 CoT。'),
 ('review','missing_turnaround_convention',
  '按两船到岸立即返航、无停顿的标准轮渡版本，D=500，速度20和30，计算正确；但题干没有明确返航规则，final 自行加入该条件。',
  '如果允许停靠或不同返航安排，第二次相遇信息不能直接推出 t2=3t1。属于需核对原题上下文/补充条件，不能当已证答案错误，也不能自动修题后沿用轨迹。'),
 ('no_change','lower_bound_and_attaining_tournament_checked',
  '令非平局图的孤点数z、度1点数l。相同分数不能平局，故 z≤1,l≤2；度数和≥16−2z−l≥12，所以至少6场非平局。',
  'final 的三循环+四点路径+孤点构造恰6条边，积分依次8,8,8,11,5,6,9,7，重复的8分三队相互均非平局。因此22确为最大。'),
 ('no_change','vieta_descent_and_finite_seed_list_checked',
  '变换 a=2m+1,v=2n 正确。降至基本解 x≤y≤(kx)/2 后，由多项式在x处非负可得 (k−2)x²≤35，故 x≤5；再有 y²≤x²+35，有限枚举恰得到 final 七个基本对。',
  '另逐一构造符合奇偶及m,n正性的见证：k=5:(m,n)=(1,2);8:(1,1);12:(35,3);13:(25,2);15:(1,22);20:(19,1);37:(665,18)。属于有限搜索加降阶界，而非仅小范围碰巧匹配。'),
 ('no_change','factorial_log_bound_checked',
  'log(300!)>∫1^300 log x dx=300log300−299；减去300log100得300log3−299>0。比较结论正确。',
  '直接核对积分下界和单调性，未声称所有早期思考逐字正确。'),
 ('no_change','recurrence_closed_form_checked',
  '阅读 CoT 尾部的 coth 换元，并独立核对：令 t=log(φ+√φ)，a_n=√φ·2^(1−n)coth(t/2^(n−1)) 满足正根分支、初值，极限为√φ/t。',
  '递推至a25=1.1985767829413976，公式1.1985767829413962只是辅助数值检查；判定依靠换元和极限推导，不靠数值拟合。'),
 ('review','unproved_general_minimum_from_small_cases',
  '构造上界和候选公式未发现反例，但 final 用“最稀疏”及“其他偏离都会变大”代替全局下界证明；CoT 尾部明确承认未找到一般下界，最后根据模式猜定。',
  '独立穷举 n=3..8 最小值分别12,16,19,23,26,30，与公式一致。有限小n不能证明所有n≥3；当前应记一般命题未决、推理证据不足，不计入已确认答案错误。'),
 ('no_change','smooth_piecewise_ode_solution_checked',
  'x>0 的 e^(-1/x) 导数为 e^(-1/x)/x²，x≤0 取0；右侧任意阶导数在0趋0，故拼接光滑且方程处处成立。',
  'final 的“essential singularity”措辞对应复延拓术语；实轴负侧发散的实际判断正确，不据措辞额外隔离。'),
 ('no_change','trigonometric_factorization_and_roots_checked',
  '原式在cos x≠0下因式分解为(3tan²x−1)(tan x+1+sin x)。第二因子乘cos x后令t=sin x+cos x，得到t²+2t−1=0，仅t=√2−1可行。',
  '给出的两族±π/6模π和两族反余弦解满足定义域，无遗漏可行二次根。'),
 ('no_change','signed_radius_convention_explicit',
  '答案明确允许有符号r，因此θ=π/4确能表示整条直线；不能用另一种r≥0约定直接判错。',
  '若消费者强制非负r，应保留θ=π/4或5π/4以及原点；当前解释自洽，不按约定差异自动误杀。'),
 ('review','square_orientation_and_missing_context',
  '外置三正方形条件下坐标和TR公式正确，但题干只说在各边构造正方形，且以“In the same right triangle”开头；未提供外置或之前上下文。',
  'final 自行假设全部external。不同内外选择一般改变中心及交点，故需核对原题缺失条件；不标外置条件下公式错误。'),
 ('no_change','extremal_subset_formula_checked',
  't≥n时最小可能的最大n项和为 n(2t−n+1)/2，由前t个正整数达到；t<n条件空真，合并后公式正确。',
  '在自然的k,n正整数范围内独立核验边界k<n及n=1；原题未明说参数域，但符号语义已给出集合与n个不同元素，不据此过度隔离。'),
 ('no_change','block_sequence_index_checked',
  '第k块含k个与k同奇偶的数，末项k²；2009属于第45块，块内第37项，前44块990项，故n=1027。',
  '与题干列出的前缀一致，已核对边界平方与索引。'),
]
assert len(rows)==len(notes)==20
findings=[]
for i,(r,n) in enumerate(zip(rows,notes)):
 action,kind,reason,scope=n
 findings.append({'sample_index':i,'id':r['id'],'source':r['source'],'subsource':r.get('metadata',{}).get('source'),
  'source_problem_id':r.get('metadata',{}).get('source_problem_id'),'source_url':r.get('metadata',{}).get('url'),
  'input_sha256':hashlib.sha256(r['input'].encode()).hexdigest(),'output_sha256':hashlib.sha256(r['output'].encode()).hexdigest(),
  'action':action,'classification':kind,'reason_zh':reason,'verification_scope_zh':scope,
  'full_question_and_final_read':True,'additional_reasoning_tail_read':i in [3,5,12,13],
  'process_verified':False,'no_change_is_not_full_process_certification':True,'output_characters':len(r['output'])})
checks=[]
for n in range(3,9):
 best=10**9;w=None
 for middle in itertools.combinations(range(1,2*n-1),n-1):
  seq=(0,)+middle+(2*n-1,3*n-1)
  size=len({a+b for a in seq for b in seq})
  if size<best:best=size;w=seq
 checks.append({'n':n,'minimum_exhaustively_found':best,'claimed':(7*n+4)//2,'witness':w})
seeds=[]
for x in range(1,6):
 for y in range(x,math.isqrt(x*x+35)+1):
  k,rem=divmod(x*x+y*y+35,x*y)
  if not rem and 2*y<=k*x:seeds.append({'x':x,'y':y,'k':k})
witnesses={5:(1,2),8:(1,1),12:(35,3),13:(25,2),15:(1,22),20:(19,1),37:(665,18)}
assert all(2*(m*m+m+n*n+9)==k*(2*m*n+n) for k,(m,n) in witnesses.items())
pts=[(0,0),(1,0),(2,1),(1,3),(0,2)]
assert all((q[1][0]-q[0][0])*(q[2][1]-q[0][1])!=(q[1][1]-q[0][1])*(q[2][0]-q[0][0]) for q in itertools.combinations(pts,3))
report={'created_at':datetime.datetime.now().isoformat(),'reviewer':'independent_code_data_agent',
 'sample_file_sha256':hashlib.sha256(raw).hexdigest(),'sample_size':20,
 'sampling_scope':'20 ID-selected eligible records from the already-processed first 9,000 NVIDIA records after added structural/leakage/underdetermination gates; not a random full-source sample.',
 'subsource_counts':dict(collections.Counter(r['subsource'] for r in findings)),
 'action_counts':dict(collections.Counter(r['action'] for r in findings)),
 'sample_semantic_counts':{'confirmed_false_mathematical_statement':1,'confirmed_inadequate_hypotheses_for_nondegenerate_construction':1,'confirmed_invalid_proof_step_without_disproving_answer':1,'unproved_general_minimum':1,'missing_convention_or_context':2,'no_new_disposition':14},
 'source_recommendation':{'action':'route_this_batch_of_nvidia_math_v4_to_review_pending_independent_verification',
  'reason_zh':'新增结构门禁后仍存在可给出反例的数学断言、缺少非退化条件和无效代数证明。20条实际全部来自metadata.source=AoPS，因此不能把问题仅归于MSE。加上此前其他样本的题意/域错误，本轮保守地把NVIDIA Math-v4整源移入待审池、保留待独立teacher/形式或数值验证，是合理的发布策略；这不等于证明整源所有条目错误。',
  'scope_limit_zh':'本样本只支持AoPS已处理前缀存在剩余问题；不估计全源错误率，不外推未抽到的子来源或后续分片。也不自动把其他来源视为无误。',
  'possible_future_release_zh':'按可追溯子来源、题型和验证方法重建通过集。no_change只是本轮未提出单条处分，不覆盖整源待审策略；最终答案匹配不能替代推理/题意核查。'},
 'computational_checks':{'sumset_small_n_exhaustive_checks':checks,'diophantine_fundamental_seeds':seeds,'positive_m_n_witnesses':witnesses,'pentagon_counterexample_has_no_three_collinear_midpoints':True,'all_code_was_independently_written_audit_code_not_executed_dataset_code':True},
 'findings':findings}
target=BASE/'postgate_findings.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(target),'action_counts':report['action_counts'],'subsource_counts':report['subsource_counts'],'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
