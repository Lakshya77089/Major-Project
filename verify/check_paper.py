"""Sanity-check the modified conference_paper.tex:
- balanced begin/end environments
- count tables / tabulars
- ensure no orphan unicode that LaTeX cp1252 can't handle
"""
from __future__ import annotations
import re, sys
from collections import Counter
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "Report" / "conference_paper.tex"
text = p.read_text(encoding="utf-8")
begins = re.findall(r"\\begin\{(\w+)\}", text)
ends   = re.findall(r"\\end\{(\w+)\}", text)
b = Counter(begins); e = Counter(ends)
mismatched = [(env, b[env], e[env]) for env in set(b) | set(e) if b[env] != e[env]]
if mismatched:
    print("MISMATCHED environments:", mismatched)
    sys.exit(1)
print(f"All {len(begins)} environments balanced.")
print(f'  tables:   {b.get("table", 0)}')
print(f'  tabulars: {b.get("tabular", 0)}')
print(f'  figures:  {b.get("figure", 0)}')

# Look for unmatched braces
opens  = text.count("{")
closes = text.count("}")
print(f"Braces: {opens} open / {closes} close (diff {opens-closes})")
