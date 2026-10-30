from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import hashlib
import json
import math
import subprocess
import pdfplumber

ROOT = Path('D:/files/MOTE')
R2 = ROOT / 'outputs/operator_geometry_round2'
R3 = ROOT / 'outputs/operator_geometry_round3'
REPORT = ROOT / 'research/operator_family_quantum_gate_20261030'
QA = R3 / 'reviews/pdf_functional_v1'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def finite(x):
    if isinstance(x, float):
        assert math.isfinite(x)
    elif isinstance(x, list):
        for v in x: finite(v)
    elif isinstance(x, dict):
        for v in x.values(): finite(v)

assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'mote-quantum'
assert not (R2/'PIPELINE_LOCK.json').exists()
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
audits = {}
for rel in ['reviews/functional_v1/AUDIT.json', 'reviews/functional_diagonal_v1/AUDIT.json',
            'reviews/functional_mechanism_v1/AUDIT.json', 'theory/sparse_tangent_v1/AUDIT.json',
            'timing/functional_v1/AUDIT.json']:
    p=R3/rel; a=read(p); assert a['passed']; audits[rel]=sha(p)
verified_files=0; source_checks=0; results=[]
for group,n in [('functional/50b8c7704256',36),('functional_diagonal/260b32fdf440',24)]:
    p=R3/group
    for name,h in read(p/'source_hashes.json').items():
        assert sha(ROOT/name)==h, name
        source_checks+=1
    paths=sorted((p/'runs').glob('*/result.json')); assert len(paths)==n
    for f in paths:
        r=read(f); finite(r); assert r['parameter_changed']
        for name,h in r['files'].items():
            assert sha(f.parent/name)==h, str(f.parent/name)
            verified_files+=1
        results.append(r)
    status=read(R3/group.split('/')[0]/'STATUS.json')
    assert status['completed']==n and not status['active']
q1=read(R3/'reviews/functional_v1/SUMMARY.json')
q2=read(R3/'reviews/functional_diagonal_v1/SUMMARY.json')
assert len(q1['quality'])+len(q2['quality'])==120
assert len(q1['paired'])+len(q2['paired'])==1092
timings=read(R3/'timing/functional_v1/raw_timing.json'); assert len(timings)==180
finite(timings)
for stem in ['functional_v1','functional_review_v1','functional_diagonal_v1','functional_diagonal_review_v1','functional_timing_v1']:
    assert (R3/f'logs/{stem}.err').stat().st_size==0

files={}; pages={}; fonts={}
for stem,count,prefix in [('FUNCTIONAL_GEOMETRY_REVIEW_20261030_zh',4,'main'),
                          ('FUNCTIONAL_GEOMETRY_SUPPLEMENT_20261030_zh',10,'supp')]:
    for suffix in ['md','typ','pdf']:
        p=REPORT/(stem+'.'+suffix); files[p.relative_to(ROOT).as_posix()]=sha(p)
    counters=Counter()
    with pdfplumber.open(REPORT/(stem+'.pdf')) as doc:
        assert len(doc.pages)==count
        for i,page in enumerate(doc.pages,1):
            render=QA/(f'main-{i}.png' if prefix=='main' else f'supp-{i:02}.png')
            assert render.exists()
            for c in page.chars:
                t=c['text']; f=c['fontname']
                if any('\u4e00'<=v<='\u9fff' for v in t):
                    assert 'KaiTi' in f, (i,t,f)
                    counters['Chinese_KaiTi']+=1
                elif any(v.isascii() and v.isalpha() for v in t):
                    assert 'TimesNewRoman' in f or 'NewCMMath' in f, (i,t,f)
                    counters['Latin_Times' if 'TimesNewRoman' in f else 'Math_NewCM']+=1
                assert c['x0']>=-1 and c['x1']<=page.width+1 and c['top']>=-1 and c['bottom']<=page.height+1
        pages[stem]=list(range(1,count+1)); fonts[stem]=dict(counters)
for p in sorted((REPORT/'functional_figures').glob('*.png')):
    files[p.relative_to(ROOT).as_posix()]=sha(p)
qa={'passed':True,'completed_local':now,'pages_visually_inspected':pages,
    'final_changed_pages_reinspected':{'main':[2,3,4],'supplement':[1]},
    'render_width_px':1600,'figures_visually_inspected':['functional_kl.png','cross_terms.png','cost_and_tangent.png'],
    'font_script_checks':fonts,'font_preference':'Times New Roman, KaiTi; New Computer Modern Math',
    'issues_fixed':['Product tangent bar legend colors matched','Mean-expert mathematical overline corrected'],
    'issues_remaining':[],'files':files}
write(QA/'PDF_QA.json',qa)
stage={'stage':'functional_geometry_objectives_diagonal_controls_and_sparse_tangent',
       'stage_complete':True,'completed_local':now,'counts':{'training_runs':60,'quality_rows':120,
       'input_roundtrip_records':240,'paired_statistic_rows':1092,'timed_checkpoints':60,
       'timing_groups':180,'mixture_decompositions':96,'sparse_jacobians':3072},
       'final_source_hash_checks':source_checks,'final_raw_files_verified':verified_files,
       'independent_audits':audits,'pdf_qa':'reviews/pdf_functional_v1/PDF_QA.json',
       'pdf_qa_sha256':sha(QA/'PDF_QA.json'),'files':files,'missing':[],'failed_current_formal_groups':[],
       'prior_failure_preserved':'replay/0f6a8f7407c8','next_research_not_started':'Task-driven joint gate/expert adaptation and independent task evaluation',
       'interpretation':'Teacher retention is one diagnostic axis. Different high-performing expert organization is a valid future outcome.'}
write(R3/'FUNCTIONAL_STAGE_COMPLETION_AUDIT.json',stage)
overall={'state':'authorized_research_stage_complete_for_user_review','completed_local':now,
         'authorized_phase_complete':True,'research_program_complete':False,'active':False,
         'stages':{'round2':{'path':'outputs/operator_geometry_round2/STAGE_COMPLETION_AUDIT.json','sha256':sha(R2/'STAGE_COMPLETION_AUDIT.json')},
                   'capacity_replay':{'path':'outputs/operator_geometry_round3/CAPACITY_STAGE_COMPLETION_AUDIT.json','sha256':sha(R3/'CAPACITY_STAGE_COMPLETION_AUDIT.json')},
                   'functional':{'path':'outputs/operator_geometry_round3/FUNCTIONAL_STAGE_COMPLETION_AUDIT.json','sha256':sha(R3/'FUNCTIONAL_STAGE_COMPLETION_AUDIT.json')}},
         'work_plan_outcomes':{'A':'Expanded boundary, utility cache and exact replacement diagnostics completed',
                               'B':'All prespecified matched utility, positive and classical controls completed and audited',
                               'C':'Utility extension declined on negative internal-dev evidence; independent capacity and functional controls completed instead',
                               'D':'Checkpoint replay, paired statistics, serial timing, mathematical checks and all stage reports completed'},
         'budget':{'original_target_hours':[5,6],'network_outage_excluded':True,
                   'resumed_local':'2026-10-30T11:03:00+08:00','provisional_review_target':'2026-10-30T14:45:00+08:00',
                   'accounting':'Exact active human/agent time was not metered; background jobs continued until05:31 during the outage.',
                   'decision':'Deliver the completed bounded research stages now for review. Do not repeat training to consume unused budget.'},
         'latest_report':'research/operator_family_quantum_gate_20261030/FUNCTIONAL_GEOMETRY_REVIEW_20261030_zh.pdf',
         'future_work':['Joint gate/expert optimization with task CE/PPL as primary outcomes',
                        'Matched prior quantum-router implementations and classical readout controls',
                        'Independent tasks or reserved evaluation splits and expert functional specialization'],
         'test_reserved':True,'gpu_lock_present':False,'missing':[],'failed_required_current_groups':[]}
write(R3/'COMPLETION_AUDIT.json',overall)
write(R2/'STATUS.json',{'state':'stage_complete_for_review','active':False,'authorized_phase_complete':True,
                      'research_program_complete':False,'completed_local':now,
                      'completion_audit':'outputs/operator_geometry_round3/COMPLETION_AUDIT.json',
                      'report':overall['latest_report']})
for p in [R2/'ACTIVE_PROCESS.json',R3/'ACTIVE_PROCESS.json',R3/'DIAGONAL_ACTIVE_PROCESS.json']:
    if p.exists():
        d=read(p); d.update(active=False,final_verified_local=now,completion_audit='outputs/operator_geometry_round3/COMPLETION_AUDIT.json')
        write(p,d)
ext=read(R2/'EXTENSION_STATE.json')
ext.update(state='authorized_phase_complete_for_user_review',completed_local=now,
           current_stage_report='research/operator_family_quantum_gate_20261030/FUNCTIONAL_GEOMETRY_REVIEW_20261030_zh',
           current_stage_audit='outputs/operator_geometry_round3/FUNCTIONAL_STAGE_COMPLETION_AUDIT.json',
           next_stage='Proposed task-driven joint adaptation, not started',next_config=None,next_status=None,next_logs=[])
write(R2/'EXTENSION_STATE.json',ext)
with (R2/'SESSION_STATE.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n## Final stage handoff {now}: functional reports ready for user review\n\n')
    f.write('- All60functional adaptations complete:36full objectives plus24diagonal controls;120quality,240roundtrip,1092paired rows,60checkpoint timing/180groups. Independent audits passed; final600rawfile hashes and allfrozen source hashes rechecked. Current formal failures/missing lists empty. No owned Python training process or GPUlock remains. Historical selectorv1failure and allnegative results preserved.\n')
    f.write('- Functional report FUNCTIONAL_GEOMETRY_REVIEW_20261030_zh (4pages) and FUNCTIONAL_GEOMETRY_SUPPLEMENT_20261030_zh (10pages) in research/operator_family_quantum_gate_20261030, each MD/Typst/PDF. All14pages/3figures visually checked; ChineseKaiTi, LatinTimes, dedicatedmath fonts verified. Evidence:reviews/pdf_functional_v1/PDF_QA.json; FUNCTIONAL_STAGE_COMPLETION_AUDIT.json.\n')
    f.write('- Main mechanism: CZ output objective improves teacher-retention relative to coefficient; diagonal output nearly matches full. Fisher cross-terms lack a consistent benefit. Product output fullGram yields small gains versus diagonal. Fixedsupport sparse Jacobians expose rank2product sets versus rank3CZ; QFI/state directions and model functional risk are separate.\n')
    f.write('- User clarification is explicit in report: teacher imitation alone does not decide QuantumGate viability. Different task-effective expert organizations are valid. Future task-driven joint gate/expert adaptation, priorquantum controls and independent evaluation are proposed for next stage, not launched. Test remains reserved.\n')
    f.write('- Authorized bounded research rounds and their reports are complete for review; research program remains open. COMPLETION_AUDIT.json maps originalA/B/C/D to outcomes and records budget accounting. Delivering before provisional14:45 because alluseful planned controls/audits are complete; no repetitions to consume time. Close this heartbeat after delivery is ready.\n')
print(json.dumps({'passed':True,'raw_files':verified_files,'source_hashes':source_checks,'pages':14,'completed':now},ensure_ascii=False))
