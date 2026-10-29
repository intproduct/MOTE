"""Export the paper's prose, equations and citations to its Markdown companion.

This exporter targets the explicit layout of QuantumGate_review_paper.typ.
Unknown mathematical expressions fail closed so a source edit cannot silently
produce malformed Markdown equations. The paper remains the editable source.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "research" / "quantum_gate_review_20261029"

EQUATIONS = {
    "eq-born": r"p_i(x;\theta)=|\psi_{\theta,i}(x)|^2,\qquad \sum_{i=1}^{E}p_i=1.",
    "eq-simplex": r"\Delta^{E-1}=\left\{p\in\mathbb{R}^{E}:p_i\geq0,\ \sum_i p_i=1\right\}.",
    "eq-softmax": r"p=\operatorname{softmax}(\log p).",
    "eq-square": r"p_i=\frac{a_i^2}{\sum_j a_j^2}=\operatorname{softmax}(2\log|a|)_i.",
    "eq-temperature": r"\widetilde p_i=\frac{p_i+\epsilon}{1+E\epsilon},\qquad w_i=\frac{\widetilde p_i^{g/T}}{\sum_j\widetilde p_j^{g/T}}.",
    "eq-fs": r"\mathrm{d}s_{\mathrm{FS}}^2=\frac14\sum_i\frac{(\mathrm{d}p_i)^2}{p_i}+\sum_i p_i(\mathrm{d}\phi_i)^2-\left(\sum_i p_i\mathrm{d}\phi_i\right)^2.",
    "eq-tangent": r"\partial_a p_i=2\psi_i\partial_a\psi_i,\qquad\sum_i\psi_i\partial_a\psi_i=0.",
    "eq-fisher": r"F^{\mathrm C}_{ab}=\sum_i\frac{(\partial_a p_i)(\partial_b p_i)}{p_i}=4\sum_i(\partial_a\psi_i)(\partial_b\psi_i)=F^{\mathrm Q}_{ab}.",
    "eq-phase": r"\psi(\theta)=\frac1{\sqrt2}\begin{pmatrix}1\\e^{\mathrm i\theta}\end{pmatrix},\qquad F^{\mathrm C}=0,\qquad F^{\mathrm Q}=1,",
    "eq-jacobian": r"\frac{\mathrm d}{\mathrm dx}\big(U(x)a(x)\big)=\frac{\mathrm dU}{\mathrm dx}a(x)+U(x)\frac{\mathrm da}{\mathrm dx}.",
    "eq-margin": r"\|\delta z\|_\infty<\frac{m_k}{2}",
    "eq-ewc": r"\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{new}}+\frac\lambda2\sum_j F_j(\theta_j-\theta_j^\star)^2.",
    "eq-hessian": r"\frac{\mathrm d^2\mathcal L}{\mathrm d\theta^2}=p(1-p)(z')^2+(p-y)z''.",
    "eq-output": r"\delta y=\sum_i\delta w_i f_i(x).",
    "eq-forgetting": r"\overline A=\frac1K\sum_{j=1}^{K}A_{K,j},\qquad\mathcal F=\frac1{K-1}\sum_{j=1}^{K-1}\left(\max_{j\leq i\leq K}A_{i,j}-A_{K,j}\right).",
}

INLINE = {
    "x in RR^d": r"x\in\mathbb R^d",
    "psi_theta(x) in CC^E": r"\psi_\theta(x)\in\mathbb C^E",
    "psi_i = sqrt(p_i)": r"\psi_i=\sqrt{p_i}",
    "x ↦ p_theta(x)": r"x\mapsto p_\theta(x)",
    "epsilon > 0": r"\epsilon>0",
    'psi_i = sqrt(p_i) exp(upright("i") phi_i)': r"\psi_i=\sqrt{p_i}e^{\mathrm i\phi_i}",
    'F^("C") ≼ F^("Q")': r"F^{\mathrm C}\preceq F^{\mathrm Q}",
    "psi(theta)": r"\psi(\theta)",
    "p_i = psi_i^2": r"p_i=\psi_i^2",
    "m_k = z_((k)) - z_((k+1)) > 0": r"m_k=z_{(k)}-z_{(k+1)}>0",
    "norm(delta z)_infinity": r"\|\delta z\|_\infty",
    "z(theta)": r"z(\theta)",
    'p = op("sigmoid")(z)': r"p=\operatorname{sigmoid}(z)",
    "q = log_2 E": r"q=\log_2 E",
    "79.48 plus.minus 1.98": r"79.48\pm1.98",
    "80.39 plus.minus 2.47": r"80.39\pm2.47",
    "y(x) = sum_i w_i(x) f_i(x)": r"y(x)=\sum_i w_i(x)f_i(x)",
    "A_(i,j)": r"A_{i,j}",
    "3.55 times 10^(-15)": r"3.55\times10^{-15}",
    "2 times 10^(-8)": r"2\times10^{-8}",
}
SIMPLE = {"E", "a", "g", "T", "g/T", "U", "k", "k+1", "F_j", "y", "i", "j", "K > 1", "(1/2, 1/2)", r"0.1724\%", r"0.2406\%"}


def inline_math(match):
    expression = match[1].strip()
    if expression in INLINE:
        return "$" + INLINE[expression] + "$"
    if expression in SIMPLE:
        return "$" + expression + "$"
    raise ValueError(f"Unmapped inline math: {expression}")


def main():
    source = (FOLDER / "QuantumGate_review_paper.typ").read_text(encoding="utf-8")
    abstract = source.split("#align(center)[*摘要*]", 1)[1].split("\n]\n", 1)[0].strip()
    body, references = source.split("= 引言", 1)[1].split("#heading(numbering: none)[参考文献]", 1)
    body = "= 引言" + body
    body = re.sub(r"#figure\(.*?\) <fig-pipeline>", "图 1　映射链：输入 → 归一化态 → Born 概率 → 实际权重 → top-k 专家集合。编码 Jacobian、态几何、概率后处理和选择边界分别控制不同阶段的响应。", body, flags=re.S)
    body = re.sub(r"#finding\(\[(.*?)\], \[(.*?)\]\)", lambda m: "**" + m[1] + "** " + m[2], body)
    body = body.replace("#block(breakable: false)[\n", "")
    body = body.replace("#pagebreak(weak: true)", "")
    body = re.sub(r"^\]\s*$", "", body, flags=re.M)
    body = re.sub(r"#heading\(numbering: none\)\[(.*?)\]", lambda m: "## " + m[1], body)
    body = re.sub(r"^(={1,2}) (.*)$", lambda m: "#" * (len(m[1]) + 1) + " " + m[2], body, flags=re.M)
    displays = []

    def display(match):
        label = match[2]
        displays.append(EQUATIONS[label])
        return f"\n@@EQUATION_{len(displays)-1}@@\n"

    body = re.sub(r"\$([^$]+)\$\s*<(eq-[\w-]+)>", display, body)
    body = re.sub(r"\$([^$]+)\$", inline_math, body)
    for i, equation in enumerate(displays):
        body = body.replace(f"@@EQUATION_{i}@@", "$$\n" + equation + "\n$$")
    body = re.sub(r"#c\((\d+)\)", lambda m: f"[{m[1]}](#ref-{m[1]})", body)
    # Typst uses a single asterisk for strong emphasis.
    body = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"**\1**", body)
    abstract = re.sub(r"\*([^*]+)\*", r"**\1**", abstract)
    entries = []
    for number, content in re.findall(r"#reference\((\d+)\)\[(.*?)\] <ref-\d+>", references):
        content = re.sub(r'#link\("([^"]+)"\)\[([^\]]+)\]', lambda m: f"[{m[2]}]({m[1]})", content)
        entries.append(f'<a id="ref-{number}"></a>\n\n{number}. {content}')
    assert len(entries) == 26
    output = "# 量子机器学习优势向专家路由的迁移\n\nQuantumGate 的信息几何、持续学习与验证框架\n\n研究综述与机制分析 · 2026 年 10 月 29 日\n\n## 摘要\n\n" + abstract + "\n\n" + body.strip() + "\n\n## 参考文献\n\n" + "\n\n".join(entries) + "\n"
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    if re.search(r"#(?:c|heading|finding|block|figure|link)\(", output):
        raise ValueError("Unconverted Typst directive")
    (FOLDER / "REPORT_zh.md").write_text(output, encoding="utf-8")
    (FOLDER / "REPORT_zh.typ").write_text('// Companion to REPORT_zh.md; the editable paper source is shared.\n#include "QuantumGate_review_paper.typ"\n', encoding="utf-8")
    print(f"Exported {len(displays)} display equations and {len(entries)} references.")


if __name__ == "__main__":
    main()
