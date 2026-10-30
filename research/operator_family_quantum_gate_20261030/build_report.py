"""Build a self-contained Typst companion; training files are never touched."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("render_typst_report", ROOT / "tools/render_typst_report.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
source = (HERE / "REPORT_zh.md").read_text(encoding="utf-8")

equations = [
    r"$ T_x = sum_(e in S(x)) w_e (x) T_e, quad abs(S(x)) = k, quad w_e (x) >= 0, quad sum_(e in S(x)) w_e (x) = 1. $",
    r'$ D_("KL")(P(dot.c | y) parallel P(dot.c | y + B delta w)) approx 1/2 delta w^top G_w delta w, quad G_w = B^top F_y B. $',
    r"$ angle.l U psi(x), U psi(x') angle.r = angle.l psi(x), psi(x') angle.r. $",
]
import re
blocks = re.findall(r"\$\$\n.*?\n\$\$", source, flags=re.S)
assert len(blocks) == len(equations)
for index, block in enumerate(blocks):
    source = source.replace(block, f"EQUATIONPLACEHOLDER{index}", 1)
typst = renderer.convert(source)
typst = typst.replace('#set par(justify:', '#show math.equation: set text(font: "New Computer Modern Math")\n#set par(justify:', 1)
for index, equation in enumerate(equations):
    token = f'#text("EQUATIONPLACEHOLDER{index}")'
    assert typst.count(token) == 1
    typst = typst.replace(token, equation)
typst = typst.replace('size: 10.5pt, lang:', 'size: 11pt, lang:')
typst = typst.replace('#heading(level: 2)[#text("7.', '#pagebreak()\n#heading(level: 2)[#text("7.', 1)
typst = typst.replace('#heading(level: 2)[#text("8.', '#pagebreak()\n#heading(level: 2)[#text("8.', 1)
(HERE / "REPORT_zh.typ").write_text(typst, encoding="utf-8")
print(HERE / "REPORT_zh.typ")
