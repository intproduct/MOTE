"""Compare archived interrupted attempts with same-seed reruns without GPU work."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
GROUP=ROOT/'formal/20c0fefebf11'
TIMING_KEYS={'training_seconds','examples_per_second'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def curves(path):
    result={}
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue  # a live writer may be appending
            kind='train' if 'train_ce' in row else 'dev'
            result[kind,row['step']]={k:v for k,v in row.items() if k not in TIMING_KEYS}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    issues=[]
    checks=[]
    for incident_path in sorted((ROOT/'interruptions').glob('*/incident.json')):
        incident=read(incident_path)
        if args.require_complete:
            for filename,expected in read(incident_path.parent/'completed_files_sha256.json').items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest().upper()!=expected.upper():
                    issues.append({'preserved_file_changed':filename})
        for attempt in incident['archived_incomplete']:
            old=Path(attempt['archive'])
            new=GROUP/attempt['run']
            if args.require_complete:
                for filename,expected in attempt['files_sha256'].items():
                    if hashlib.sha256((old/filename).read_bytes()).hexdigest().upper()!=expected.upper():
                        issues.append({'archived_file_changed':str(old/filename)})
            old_curves=curves(old/'curves.jsonl')
            new_curves=curves(new/'curves.jsonl')
            common=old_curves.keys() & new_curves.keys()
            mismatches=[{'kind':kind,'step':step} for kind,step in sorted(common)
                        if old_curves[kind,step]!=new_curves[kind,step]]
            for name in ['initial_dev.json']:
                if (new/name).exists() and read(old/name)!=read(new/name):
                    issues.append({'run':attempt['run'],'mismatch':name})
            if mismatches:issues.append({'run':attempt['run'],'non_timing_curve_mismatches':mismatches})
            done=(new/'result.json').exists()
            if args.require_complete and (not done or len(common)!=len(old_curves)):
                issues.append({'run':attempt['run'],'recovery_incomplete':True})
            checks.append({'run':attempt['run'],'completed':done,'archived_curve_rows':len(old_curves),
                           'compared_curve_rows':len(common),'compared_rows_exact_match':not mismatches})
    record={'time':datetime.now().astimezone().isoformat(),'require_complete':args.require_complete,
            'passed':not issues,'all_recovered':all(c['completed'] for c in checks),
            'checks':checks,'issues':issues}
    (ROOT/'recovery_audit.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
    raise SystemExit(1 if issues else 0)


if __name__=='__main__':main()
