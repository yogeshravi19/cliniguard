"""
Compile RESEARCH_PAPER_CLINIGUARD.md into a beautifully formatted Microsoft Word (.docx) document
with embedded 200 DPI figures, professional academic tables, and typography.
"""
import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[2]
MD_PATH = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD.md"
DOCX_PATH = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD.docx"
FIG_DIR = ROOT / "research_paper" / "figures"


def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="333333"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def sanitize_xml(s: str) -> str:
    """Strip XML-incompatible control characters."""
    return "".join(c for c in s if c in '\t\n\r' or ord(c) >= 32)


def format_run_math(p, text):
    """Parses basic inline markdown formatting like bold, italics, code, and math."""
    text = sanitize_xml(text)
    # Split by bold **text**
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\$.*?\$)', text)
    for tok in tokens:
        if not tok:
            continue
        clean_tok = sanitize_xml(tok)
        if clean_tok.startswith('**') and clean_tok.endswith('**'):
            r = p.add_run(clean_tok[2:-2])
            r.bold = True
        elif clean_tok.startswith('*') and clean_tok.endswith('*'):
            r = p.add_run(clean_tok[1:-1])
            r.italic = True
        elif clean_tok.startswith('`') and clean_tok.endswith('`'):
            r = p.add_run(clean_tok[1:-1])
            r.font.name = 'Consolas'
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(199, 37, 78)
        elif clean_tok.startswith('$') and clean_tok.endswith('$'):
            r = p.add_run(clean_tok[1:-1])
            r.font.name = 'Cambria Math'
            r.italic = True
        else:
            p.add_run(clean_tok)


def compile_manuscript():
    print(f"Reading {MD_PATH}...")
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()

    doc = docx.Document()

    # Page setup: Standard Letter, 1-inch margins
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    # Base styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(34, 34, 34)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    i = 0
    in_code_block = False
    table_buffer = []

    def flush_table(buffer):
        if not buffer:
            return
        # Parse rows
        rows = []
        for l in buffer:
            # Skip separator line
            if re.match(r'^\s*\|?\s*[-:]+[-| :]*$', l):
                continue
            cols = [c.strip() for c in l.strip().strip('|').split('|')]
            rows.append(cols)

        if not rows:
            return

        n_cols = max(len(r) for r in rows)
        tbl = doc.add_table(rows=len(rows), cols=n_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl)

        for r_idx, row_data in enumerate(rows):
            is_header = (r_idx == 0)
            row = tbl.rows[r_idx]
            for c_idx in range(n_cols):
                cell = row.cells[c_idx]
                val = row_data[c_idx] if c_idx < len(row_data) else ""
                cell_p = cell.paragraphs[0]
                cell_p.paragraph_format.space_after = Pt(2)
                cell_p.paragraph_format.space_before = Pt(2)
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

                if is_header:
                    set_cell_background(cell, "F0F4F8")
                    r = cell_p.add_run(val)
                    r.bold = True
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = RGBColor(24, 43, 73)
                else:
                    if r_idx % 2 == 1:
                        set_cell_background(cell, "FAFAFA")
                    cell_p.style = 'Normal'
                    format_run_math(cell_p, val)
                    for r in cell_p.runs:
                        r.font.size = Pt(9.0)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.strip().startswith('```'):
            if in_code_block:
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(line)
            r.font.name = 'Consolas'
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(51, 51, 51)
            i += 1
            continue

        # Tables
        if '|' in line and not line.strip().startswith('#'):
            table_buffer.append(line)
            i += 1
            continue
        elif table_buffer:
            flush_table(table_buffer)
            table_buffer = []

        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^-{3,}$', stripped):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            i += 1
            continue

        # Markdown Images: ![caption](path)
        img_match = re.match(r'^!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            alt, img_path = img_match.groups()
            rel_path = ROOT / "research_paper" / img_path.replace('/', '\\')
            if rel_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(2)
                # Check aspect ratio
                doc.add_picture(str(rel_path), width=Inches(5.8))
            i += 1
            continue

        # Captions like *Fig. 1. ...*
        if stripped.startswith('*Fig.') or stripped.startswith('*Table'):
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(10)
            r = p_cap.add_run(stripped.strip('*'))
            r.italic = True
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(85, 85, 85)
            i += 1
            continue

        # Headers
        if stripped.startswith('# '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(stripped[2:])
            r.bold = True
            r.font.size = Pt(20)
            r.font.color.rgb = RGBColor(16, 44, 87)
            i += 1
            continue

        if stripped.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(stripped[3:])
            r.bold = True
            r.font.size = Pt(14)
            r.font.color.rgb = RGBColor(24, 76, 120)
            i += 1
            continue

        if stripped.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(stripped[4:])
            r.bold = True
            r.font.size = Pt(11.5)
            r.font.color.rgb = RGBColor(40, 40, 40)
            i += 1
            continue

        if stripped.startswith('#### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(stripped[5:])
            r.bold = True
            r.italic = True
            r.font.size = Pt(10.5)
            r.font.color.rgb = RGBColor(60, 60, 60)
            i += 1
            continue

        # Blockquotes
        if stripped.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.right_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            format_run_math(p, stripped[2:])
            for r in p.runs:
                r.italic = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(80, 80, 80)
            i += 1
            continue

        # Bullet lists
        if stripped.startswith('* ') or stripped.startswith('- '):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(2)
            format_run_math(p, stripped[2:])
            i += 1
            continue

        # Numbered lists: 1. 2.
        num_match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if num_match:
            num, rest = num_match.groups()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after = Pt(2)
            format_run_math(p, rest)
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        format_run_math(p, stripped)
        i += 1

    if table_buffer:
        flush_table(table_buffer)

    print(f"Saving compiled document to {DOCX_PATH}...")
    try:
        doc.save(str(DOCX_PATH))
        print(f"[OK] Successfully compiled research manuscript -> {DOCX_PATH} ({DOCX_PATH.stat().st_size // 1024} KB)")
    except PermissionError:
        fallback_path = DOCX_PATH.with_name("RESEARCH_PAPER_CLINIGUARD_FINAL.docx")
        print(f"[WARNING] {DOCX_PATH.name} is currently open in Microsoft Word. Saving to {fallback_path.name} instead...")
        doc.save(str(fallback_path))
        print(f"[OK] Successfully compiled research manuscript -> {fallback_path} ({fallback_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    compile_manuscript()
