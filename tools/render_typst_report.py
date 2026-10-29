"""Render our prose Markdown reports to self-contained Typst sources.

Supports headings, paragraphs, flat lists, links, emphasis and inline code.
It deliberately rejects tables/fenced code rather than silently losing them.
Compile the emitted source with the local `typst compile` command.
"""

import argparse
import json
from pathlib import Path
import re


def string(value):
    return json.dumps(value, ensure_ascii=False)


def inline(value):
    pattern = r"\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`|\*([^*]+)\*"
    result, start = [], 0
    for match in re.finditer(pattern, value):
        if match.start() > start:
            result.append(f"#text({string(value[start:match.start()])})")
        label, url, bold, code, italic = match.groups()
        if label is not None:
            # Relative Markdown links remain readable labels in print. Web
            # sources keep their actual clickable URLs in the PDF.
            result.append(f"#link({string(url)})[{inline(label)}]" if url.startswith(("https://", "http://")) else inline(label))
        elif bold is not None:
            result.append(f"#strong[{inline(bold)}]")
        elif code is not None:
            # Add invisible break opportunities to long source paths/hashes.
            code = re.sub(r"(\S{24})", "\\1\u200b", code)
            result.append(f"#text(font: (\"Consolas\", \"Noto Sans SC\"), size: 8.5pt, {string(code)})")
        else:
            result.append(f"#emph[{inline(italic)}]")
        start = match.end()
    if start < len(value):
        result.append(f"#text({string(value[start:])})")
    return "".join(result)


def convert(source):
    blocks = []
    for block in re.split(r"\n\s*\n", source.strip()):
        lines = block.splitlines()
        if any(line.startswith(("```", "|")) for line in lines):
            raise ValueError("Tables/fenced code require an explicit Typst layout")
        for line in lines:
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            bullet = re.match(r"^-\s+(.+)$", line)
            numbered = re.match(r"^(\d+)\.\s+(.+)$", line)
            if heading:
                blocks.append(f"#heading(level: {len(heading[1])})[{inline(heading[2])}]")
            elif bullet:
                blocks.append(f"- {inline(bullet[1])}")
            elif numbered:
                blocks.append(f"#enum(start: {numbered[1]}, [{inline(numbered[2])}])")
            else:
                blocks.append(inline(line))
        blocks.append("")
    return '''// Generated from the matching Markdown report. Edit the Markdown and rerun.
#set page(paper: "a4", margin: (x: 23mm, y: 22mm), numbering: "1")
#set text(font: ("Noto Serif SC", "Cambria"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.75em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 19pt, fill: rgb("17365d"))
#show heading.where(level: 2): set text(size: 14pt, fill: rgb("17365d"))
#show heading.where(level: 3): set text(size: 12pt)
#show link: set text(fill: rgb("24578a"))

''' + "\n".join(blocks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.markdown:
        target = path.with_suffix(".typ")
        target.write_text(convert(path.read_text(encoding="utf-8-sig")), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
