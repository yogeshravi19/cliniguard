"""Extract the first-page text of every downloaded reference PDF so each
file can be checked against the title/author claimed in the bibliography.
Output: research_paper/references/pdf_first_page_dump.txt
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "research_papers"
OUT = ROOT / "research_paper" / "references" / "pdf_first_page_dump.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

lines = []
for pdf in sorted(PDF_DIR.glob("*.pdf")):
    try:
        doc = fitz.open(pdf)
        txt = " ".join(doc[0].get_text().split())[:420]
        lines.append(f"### {pdf.name}  (pages={len(doc)}, KB={pdf.stat().st_size//1024})\n{txt}\n")
    except Exception as e:
        lines.append(f"### {pdf.name}  ERROR {e}\n")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT} ({len(lines)} files)")
