"""Build the matching Typst source, with readable landscape figure pages."""
import importlib.util
from pathlib import Path
import re
import shutil
import hashlib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('report_renderer',ROOT/'tools/render_typst_report.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
md=HERE/'EXPERIMENT_START_zh.md'
source=md.read_text(encoding='utf-8')
body,figures=source.split('## 9. 附图',1)
typ=renderer.convert(body)
typ+='\n#pagebreak()\n#set page(width: 297mm, height: 210mm, margin: 18mm)\n'
origin=ROOT/'outputs/operator_geometry_round1/router_pilot/da5325087d6b/review_v1/figures'
target=HERE/'operator_pilot_figures';target.mkdir(exist_ok=True)
for i,(caption,path) in enumerate(re.findall(r'!\[([^\]]+)\]\(([^)]+)\)',figures)):
    src=origin/Path(path).name;dst=HERE/path
    if dst.exists() and hashlib.sha256(dst.read_bytes()).digest()!=hashlib.sha256(src.read_bytes()).digest():
        raise RuntimeError(f'Existing figure differs: {dst}')
    if not dst.exists():shutil.copy2(src,dst)
    if i:typ+='\n#pagebreak()\n'
    typ+='#heading(level: 2)[附图 '+str(i+1)+']\n'
    typ+='#figure(image('+renderer.string(path)+', width: 100%), caption: ['+renderer.inline(caption)+'], numbering: none)\n'
md.with_suffix('.typ').write_text(typ,encoding='utf-8')
print(md.with_suffix('.typ'))
