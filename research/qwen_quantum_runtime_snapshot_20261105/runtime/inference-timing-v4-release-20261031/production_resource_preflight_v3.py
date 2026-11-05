"""Read-only inventory plus explicit pending allowances before acceptance writes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

ROOT=Path('E:/AI/MOTE-quantum')
def inventory():
    totals={'data_processing_reports_diagnostics':0,'checkpoints_and_initialization':0,
            'isolated_environment':0,'other_small_records':0}
    counts={key:0 for key in totals}
    for directory, folders, files in os.walk(ROOT,followlinks=False):
        base=Path(directory)
        if base.is_symlink():
            raise RuntimeError('Unexpected linked directory: '+str(base))
        for name in files:
            path=base/name
            if path.is_symlink():raise RuntimeError('Unexpected linked file: '+str(path))
            rel=path.relative_to(ROOT)
            if rel.parts[0]=='envs':category='isolated_environment'
            elif rel.parts[0] in ('data','runtime','diagnostics','reports'):
                category='data_processing_reports_diagnostics'
            elif rel.parts[0]=='initialization' or (rel.parts[0]=='runs' and path.suffix=='.pt'
                    and not rel.parts[1].startswith('mechanism-')):
                category='checkpoints_and_initialization'
            elif rel.parts[0]=='runs':category='data_processing_reports_diagnostics'
            else:category='other_small_records'
            try:size=path.stat().st_size
            except FileNotFoundError:continue  # Only an atomic status temp may disappear.
            totals[category]+=size;counts[category]+=1
    return totals,counts

def check():
    totals,counts=inventory()
    # Conservative full allowances, including already-written partial replay
    # states; this deliberately double counts those rather than underreserving.
    pending_processing={'both_acceptance_logs_and_caches':200_000_000,
                        'second_V100_short_checks':100_000_000,
                        'remaining_replay_logs_and_caches':100_000_000,
                        'reports_and_audits':100_000_000}
    pending_outputs={'remaining_BF16_replay_states_and_failure':8_000_000_000,
                     'BF16_acceptance_states_and_failure':12_000_000_000,
                     'FP16_acceptance_states_and_failure':12_000_000_000,
                     'second_V100_smoke_states_and_failure':6_000_000_000}
    pending=sum(pending_processing.values())+sum(pending_outputs.values())
    free=shutil.disk_usage(ROOT).free
    data_total=totals['data_processing_reports_diagnostics']+sum(pending_processing.values())
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_bytes=totals,file_counts=counts,pending_processing_bytes=pending_processing,
        pending_checkpoint_failure_bytes=pending_outputs,
        processing_actual_plus_reserved_bytes=data_total,processing_cap_bytes=20_000_000_000,
        free_bytes=free,pending_all_categories_bytes=pending,
        acceptance_launch_fits=data_total<=20_000_000_000 and pending+2_000_000_000<=free,
        formal_matrix_approved=False,
        formal_note='Recompute after corrected pilots. Prior full-matrix upper 118957598128 bytes has not been reserved; acceptance allowances may leave less than this upper bound. Preserve old assets; review actual checkpoint/diagnostic sizes before formal approval.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();value=check();path=Path(args.output)
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(value,indent=2))
    if not value['acceptance_launch_fits']:raise SystemExit(2)
