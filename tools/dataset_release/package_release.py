"""Package standard data and audit evidence; leave all raw files unchanged."""
import json,hashlib,zipfile
from pathlib import Path
from common import ROOT,DEST

def main():
 output=ROOT/'output';output.mkdir(exist_ok=True)
 full=output/'MoTE_local_SFT_v01_full_20261016.zip';pilot=output/'MoTE_local_SFT_v01_pilot_1241_20261016.zip'
 relative=[Path('README_zh.md'),Path('schema.json'),Path('MANIFEST.json'),Path('pilot/train.jsonl')]
 for folder in ['jsonl','examples','code_review']:
  relative.extend(p.relative_to(DEST) for p in sorted((DEST/folder).glob('*.jsonl')))
 relative.extend(p.relative_to(DEST) for p in sorted((DEST/'audit').iterdir()) if p.is_file())
 with zipfile.ZipFile(full,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in relative:z.write(DEST/p,DEST.name+'/'+p.as_posix())
  for p in sorted(Path(__file__).parent.iterdir()):
   if p.suffix in ['.py','.json']:z.write(p,'tools/dataset_release/'+p.name)
 with zipfile.ZipFile(pilot,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in ['README_zh.md','schema.json','pilot/train.jsonl','examples/sample_records.jsonl','audit/summary.json','audit/export_validation.json','audit/validation_final.json']:
   z.write(DEST/p,p)
  z.writestr('PILOT_ONLY.txt','This is the 1,241-record pilot subset only. README statistics describe the full release. Full train/validation/test/review/quarantine files and their full manifest are in the full package. Pilot records are already included in full train; do not concatenate them again. No tokenizer processing has been performed.\n')
 report=[]
 for p in [full,pilot]:
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   assert all(not name.startswith('/') and '..' not in Path(name).parts for name in z.namelist())
   entries=len(z.namelist())
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  report.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'zip_crc_checked':True,'entries':entries})
 (output/'MoTE_local_SFT_v01_packages.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
