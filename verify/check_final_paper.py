"""Sanity-check the final conference_paper_final.tex."""
from __future__ import annotations
import re, sys
from collections import Counter
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "Report" / "conference_paper_final.tex"
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
opens  = text.count("{")
closes = text.count("}")
print(f"Braces: {opens} open / {closes} close (diff {opens-closes})")

# Word count estimate (excluding LaTeX commands)
text_only = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", " ", text)
text_only = re.sub(r"%.*?\n", " ", text_only)
words = re.findall(r"[A-Za-z]+", text_only)
print(f"Words (rough): {len(words):,}")

# Page estimate: IEEE conference 10pt two-column ~ 750-800 words/page
print(f"Page estimate (at ~750 wpp): {len(words)/750:.1f}")
print(f"Page estimate (at ~800 wpp): {len(words)/800:.1f}")
