"""Package only validated, eligible splits for transfer, never the review pools."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'Datasets/processed/shared_sft_v02_20261016'

def main():
 manifest=json.loads((DEST/'MANIFEST.json').read_text(encoding='utf-8'))
 assert manifest['status']=='validated'
 files=[x for x in manifest['canonical_files'] if Path(x['path']).stem in ['train','validation','test','smoke']]
 assert len(files)==4
 inspection=json.loads((DEST/'inspection/manifest.json').read_text(encoding='utf-8'))
 files.append({k:inspection[k] for k in ['path','rows','sha256']})
 files[-1]['bytes']=(DEST/inspection['path']).stat().st_size
 folder=DEST/'transfer';folder.mkdir(exist_ok=True)
 output=folder/'shared_sft_v02_training.zip';temporary=folder/'shared_sft_v02_training.zip.partial'
 if output.exists():raise FileExistsError('Package already exists; inspect rather than overwrite: '+str(output))
 package_manifest={'release':manifest['release'],'scope':'Eligible splits and one inspection subset only; validation/test/smoke are reserved, not training. The inspection file is a subset of train, not additional records. Review, quarantine, duplicates, raw files and domain-view copies are excluded.','files':files,'inspection_subset':inspection,'evidence_tiers':manifest['eligible_evidence_tiers'],'tokenizer_processing_performed':False,'all_reasoning_semantically_certified':False}
 note='''# 训练候选便携包\n\n使用 jsonl/train.jsonl；validation、test、smoke 是项目保留集，不能合并回训练。\n\n每行以 messages 和 assistant_loss_mask 为训练语义依据；完整回答保留。final 是辅助字段，不能默认代替完整回答。读取细节见 DATA_CONTRACT_zh.md。\n\n本包没有 tokenizer、截断、packing 或训练比例配置。训练库的自然比例不是推荐配比。质量按证据分级，不保证每一步长推理都正确。逐条来源、许可和质量证据保留；没有统一替换许可证。\n\n待审、隔离、重复备选及原始下载留在本地母库，不包含在此包。inspection/stratified_train_sample.jsonl 是训练集的完整抽样，不能重复加入总训练集。PACKAGE_MANIFEST.json 给出四份划分与这份抽样的行数及 SHA256。\n'''
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  z.writestr('README_zh.md',note)
  z.writestr('PACKAGE_MANIFEST.json',json.dumps(package_manifest,ensure_ascii=False,indent=2))
  z.writestr('inspection/manifest.json',json.dumps(inspection,ensure_ascii=False,indent=2))
  for name in ['DATA_CONTRACT_zh.md','schema.json']:
   z.write(DEST/name,name)
  for name in ['source_admission_policy.json','record_dispositions.jsonl','source_question_counts.json','validation_report.json']:
   z.write(DEST/'audit'/name,'audit/'+name)
  for item in files:
   path=DEST/item['path'];digest=hashlib.sha256();size=0
   with path.open('rb',buffering=16*1024*1024) as source,z.open(item['path'].replace('\\','/'),'w',force_zip64=True) as target:
    while chunk:=source.read(8*1024*1024):target.write(chunk);digest.update(chunk);size+=len(chunk)
   assert digest.hexdigest()==item['sha256'] and size==item['bytes'],('source_changed',str(path))
   print(json.dumps({'packaged':item['path'],'bytes':size}),flush=True)
 with zipfile.ZipFile(temporary) as z:assert z.testzip() is None,'ZIP CRC validation failed'
 temporary.replace(output)
 digest=hashlib.sha256()
 with output.open('rb',buffering=16*1024*1024) as f:
  while chunk:=f.read(8*1024*1024):digest.update(chunk)
 receipt={'path':str(output.relative_to(DEST)).replace('\\','/'),'bytes':output.stat().st_size,'sha256':digest.hexdigest(),'zip_crc_checked':True,'source_file_hashes_checked':True,'files':files}
 (folder/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
