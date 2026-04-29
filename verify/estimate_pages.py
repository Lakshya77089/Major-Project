"""Better page estimate for IEEE conference 10pt two-column."""
from __future__ import annotations
import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "Report" / "conference_paper_final.tex"
text = p.read_text(encoding="utf-8")
text = re.sub(r"%.*?\n", "\n", text)

n_table = text.count(r"\begin{table}")
n_fig = text.count(r"\begin{figure}")
n_eq = (
    text.count(r"\begin{equation}")
    + text.count(r"\begin{align}")
    + text.count(r"\begin{equation*}")
    + text.count(r"\begin{align*}")
)
n_sec = text.count(r"\section{")
n_subsec = text.count(r"\subsection{")

print(f"Tables: {n_table}, Figures: {n_fig}, Equations/aligns: {n_eq}")
print(f"Sections: {n_sec}, Subsections: {n_subsec}")

# Strip env blocks for body word count
body = text
body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.DOTALL)
body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", body, flags=re.DOTALL)
body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.DOTALL)
body = re.sub(r"\\begin\{align\}.*?\\end\{align\}", " ", body, flags=re.DOTALL)
body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " BIB ", body, flags=re.DOTALL)
body = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})*", " ", body)

words = re.findall(r"[A-Za-z]+", body)
n_words = len(words)

# Bibliography roughly: count bibitems
n_bibitems = text.count(r"\bibitem{")
bib_pages = n_bibitems * 0.025  # ~40 bibitems per IEEE column-page

text_pg = n_words / 750  # IEEE 10pt 2-col ~ 750 wpp for body text
tab_pg = n_table * 0.18
fig_pg = n_fig * 0.30
eq_pg = n_eq * 0.05

total = text_pg + tab_pg + fig_pg + eq_pg + bib_pages
print(f"Body words: {n_words}")
print(f"Bibitems: {n_bibitems}")
print(f"Text pages : {text_pg:.2f}")
print(f"Tables     : {tab_pg:.2f}")
print(f"Figures    : {fig_pg:.2f}")
print(f"Equations  : {eq_pg:.2f}")
print(f"Bibliography: {bib_pages:.2f}")
print(f"--")
print(f"ESTIMATED TOTAL: {total:.1f} pages")
