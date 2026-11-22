"""
Compile RESEARCH_PAPER_CLINIGUARD.md into an IEEE Two-Column Academic Paper (.docx)
matching the VIT Chennai IEEE Conference Paper format (SRIP_Sandhya_Final.pdf),
with zero raw LaTeX/Markdown tokens, native subscripts/superscripts, crisp spanning tables,
and embedded 200 DPI figures.
"""
import re
import shutil
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[2]
MD_PATH = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD.md"
DOCX_MAIN = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD.docx"
DOCX_FINAL = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD_FINAL.docx"
DOCX_IEEE = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD_IEEE_FINAL.docx"

FONT_BODY = "Times New Roman"
BLACK = RGBColor(0, 0, 0)

WIDE_FIGURES = {
    "fig1_architecture.png": 6.6,
    "fig4_baselines.png": 6.0,
    "fig5_ablation.png": 6.3,
}


def set_section_columns(section, num_cols: int, space_dxa: int = 340):
    """Configure a Word section for 1-column or 2-column IEEE layout."""
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    sectPr = section._sectPr
    for existing in sectPr.findall(qn("w:cols")):
        sectPr.remove(existing)
    if num_cols > 1:
        cols_xml = parse_xml(
            f'<w:cols {nsdecls("w")} w:num="{num_cols}" w:space="{space_dxa}" w:equalWidth="1"/>'
        )
    else:
        cols_xml = parse_xml(f'<w:cols {nsdecls("w")} w:num="1"/>')
    sectPr.append(cols_xml)


def set_cell_background(cell, hex_color: str):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'))


def set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(
        parse_xml(
            f'<w:tcMar {nsdecls("w")}>'
            f'<w:top w:w="{top}" w:type="dxa"/>'
            f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
            f'<w:left w:w="{left}" w:type="dxa"/>'
            f'<w:right w:w="{right}" w:type="dxa"/>'
            f"</w:tcMar>"
        )
    )


def set_borderless_table(table):
    tblPr = table._tbl.tblPr
    tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="none"/><w:bottom w:val="none"/>'
            f'<w:left w:val="none"/><w:right w:val="none"/>'
            f'<w:insideH w:val="none"/><w:insideV w:val="none"/>'
            f"</w:tblBorders>"
        )
    )


def set_ieee_table_borders(table):
    tblPr = table._tbl.tblPr
    tblPr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            f'<w:left w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
            f'<w:right w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
            f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
            f"</w:tblBorders>"
        )
    )


def prevent_row_split(row, is_header=False):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    if is_header:
        trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))


def sanitize_text(s: str) -> str:
    s = s.replace("&nbsp;", " ")
    return "".join(c for c in s if c in "\t\n\r" or ord(c) >= 32)


TOKEN_PATTERN = re.compile(
    r"(\*\*.*?\*\*|\*[^*]+?\*|`[^`]+?`|<sub>.*?</sub>|<sup>.*?</sup>)"
)


def add_formatted_runs(
    p,
    text: str,
    font_size_pt: float = 10.0,
    base_bold: bool = False,
    base_italic: bool = False,
    font_name: str = FONT_BODY,
):
    """Render inline bold, italic, code, subscript, and superscript cleanly with zero raw tokens."""
    text = sanitize_text(text)
    parts = TOKEN_PATTERN.split(text)
    for part in parts:
        if not part:
            continue
        r_bold = base_bold
        r_italic = base_italic
        r_sub = False
        r_sup = False
        content = part

        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            content = part[2:-2]
            r_bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            content = part[1:-1]
            r_italic = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            content = part[1:-1]
        elif part.startswith("<sub>") and part.endswith("</sub>"):
            content = part[5:-6]
            r_sub = True
        elif part.startswith("<sup>") and part.endswith("</sup>"):
            content = part[5:-6]
            r_sup = True

        # Strip nested * inside <sub>*m*</sub> if present
        if (r_sub or r_sup) and content.startswith("*") and content.endswith("*") and len(content) > 2:
            content = content[1:-1]
            r_italic = True

        run = p.add_run(content)
        run.font.name = font_name
        run.font.size = Pt(font_size_pt)
        run.font.color.rgb = BLACK
        run.bold = r_bold
        run.italic = r_italic
        if r_sub:
            run.font.subscript = True
        if r_sup:
            run.font.superscript = True


def add_author_header(doc, title_text: str):
    """Render IEEE title and two-column VIT Chennai author block in single-column top section."""
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(10)
    p_title.paragraph_format.line_spacing = 1.1
    r_t = p_title.add_run(title_text)
    r_t.font.name = FONT_BODY
    r_t.font.size = Pt(18)
    r_t.bold = True
    r_t.font.color.rgb = BLACK

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borderless_table(tbl)

    authors = [
        [
            ("Yogesh Ravi M", "", False),
            ("School of Computer Science and Engineering", "", False),
            ("Vellore Institute of Technology, Chennai, India", "", False),
            ("yogeshravi.m2022@vitstudent.ac.in", "", False),
        ],
    ]

    for col_idx, lines_spec in enumerate(authors):
        cell = tbl.rows[0].cells[col_idx]
        cell.width = Inches(3.55)
        set_cell_margins(cell, top=20, bottom=40, left=60, right=60)
        for line_i, (line_txt, sup_txt, is_italic) in enumerate(lines_spec):
            cp = cell.paragraphs[0] if line_i == 0 else cell.add_paragraph()
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.space_before = Pt(0)
            cp.paragraph_format.space_after = Pt(1.5)
            cp.paragraph_format.line_spacing = 1.05
            r = cp.add_run(line_txt)
            r.font.name = FONT_BODY
            r.font.size = Pt(10 if line_i == 0 else 9.5)
            r.font.color.rgb = BLACK
            r.italic = is_italic
            if sup_txt:
                rs = cp.add_run(sup_txt)
                rs.font.name = FONT_BODY
                rs.font.size = Pt(10)
                rs.font.superscript = True
                rs.font.color.rgb = BLACK

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(6)


def add_display_equation(doc, raw_eq_line: str):
    """Render a display equation inside a borderless 2-cell row with right-aligned number."""
    cleaned = sanitize_text(raw_eq_line).strip()
    eq_num = ""
    m = re.search(r"\((\d+)\)\s*$", cleaned)
    if m:
        eq_num = f"({m.group(1)})"
        cleaned = cleaned[: m.start()].strip()

    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borderless_table(tbl)
    prevent_row_split(tbl.rows[0])

    c_eq = tbl.rows[0].cells[0]
    c_num = tbl.rows[0].cells[1]
    c_eq.width = Inches(3.05)
    c_num.width = Inches(0.40)
    c_eq.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    c_num.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_margins(c_eq, top=40, bottom=40, left=20, right=20)
    set_cell_margins(c_num, top=40, bottom=40, left=10, right=10)

    p_eq = c_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(2)
    p_eq.paragraph_format.space_after = Pt(2)
    add_formatted_runs(p_eq, cleaned, font_size_pt=9.5)

    p_num = c_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(2)
    p_num.paragraph_format.space_after = Pt(2)
    if eq_num:
        rn = p_num.add_run(eq_num)
        rn.font.name = FONT_BODY
        rn.font.size = Pt(9.5)
        rn.font.color.rgb = BLACK


def render_markdown_table(doc, table_caption: str, buffer_lines: list):
    """Render an academic table across a clean full-width IEEE span and return to 2 columns."""
    rows = []
    alignments = []
    for l in buffer_lines:
        stripped = l.strip()
        if re.match(r"^\|?\s*[-:]+[-| :]*$", stripped):
            specs = [c.strip() for c in stripped.strip("|").split("|")]
            for sp in specs:
                if sp.startswith(":") and sp.endswith(":"):
                    alignments.append(WD_ALIGN_PARAGRAPH.CENTER)
                elif sp.endswith(":"):
                    alignments.append(WD_ALIGN_PARAGRAPH.RIGHT)
                else:
                    alignments.append(WD_ALIGN_PARAGRAPH.LEFT)
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        rows.append(cols)

    if not rows:
        return

    # Enter full-width 1-column span for crisp readability (IEEE table* style)
    sec_span = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(sec_span, 1)

    if table_caption:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(3)
        p_cap.paragraph_format.keep_with_next = True
        add_formatted_runs(p_cap, table_caption, font_size_pt=9.0, base_bold=True)

    n_cols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_ieee_table_borders(tbl)

    for r_idx, row_data in enumerate(rows):
        is_header = r_idx == 0
        row = tbl.rows[r_idx]
        prevent_row_split(row, is_header=is_header)
        for c_idx in range(n_cols):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            val = row_data[c_idx] if c_idx < len(row_data) else ""
            set_cell_margins(cell, top=55, bottom=55, left=80, right=80)
            cp = cell.paragraphs[0]
            cp.paragraph_format.space_before = Pt(1)
            cp.paragraph_format.space_after = Pt(1)
            cp.paragraph_format.line_spacing = 1.05

            if is_header:
                set_cell_background(cell, "EFEFEF")
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_formatted_runs(cp, val, font_size_pt=8.5, base_bold=True)
            else:
                if c_idx < len(alignments):
                    cp.alignment = alignments[c_idx]
                else:
                    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                add_formatted_runs(cp, val, font_size_pt=8.5)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

    # Resume 2-column IEEE layout
    sec_back = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(sec_back, 2)


def compile_manuscript():
    print(f"Reading {MD_PATH}...")
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()

    doc = docx.Document()
    set_section_columns(doc.sections[0], 1)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = FONT_BODY
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = BLACK
    normal_style.paragraph_format.line_spacing = 1.08
    normal_style.paragraph_format.space_after = Pt(3.5)

    # Extract Title
    title_text = "CliniGuard: A Fast, Interpretable Word-Level Risk Scorer for Medical AI Hallucinations and an Audit of Benchmark Datasets"
    start_idx = 0
    for idx, l in enumerate(lines):
        if l.strip().startswith("# "):
            title_text = l.strip()[2:].strip()
            start_idx = idx + 1
            break

    add_author_header(doc, title_text)

    # Switch into IEEE 2-column body right after title & authors
    body_sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(body_sec, 2)

    current_section_name = ""
    pending_table_caption = ""
    table_buffer = []

    i = start_idx
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip metadata lines at very top (Authors:, Institution:, Date:, ---)
        if stripped.startswith("**Authors:**") or stripped.startswith("**Institution:**") or stripped.startswith("**Date:**"):
            i += 1
            continue
        if re.match(r"^-{3,}$", stripped):
            i += 1
            continue

        # Strict Markdown Table detection (must start and end with |)
        if stripped.startswith("|") and stripped.endswith("|"):
            table_buffer.append(stripped)
            i += 1
            continue
        elif table_buffer:
            render_markdown_table(doc, pending_table_caption, table_buffer)
            table_buffer = []
            pending_table_caption = ""

        if not stripped:
            i += 1
            continue

        # Detect "**Table X. ...**" caption right before a markdown table
        if re.match(r"^\*\*Table\s+\d+\..*\*\*$", stripped):
            pending_table_caption = stripped[2:-2].strip()
            i += 1
            continue

        # Section Headings (## ...)
        if stripped.startswith("## "):
            sec_title = stripped[3:].strip()
            current_section_name = sec_title.upper()
            if current_section_name == "ABSTRACT":
                i += 1
                continue

            p_sec = doc.add_paragraph()
            p_sec.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_sec.paragraph_format.space_before = Pt(9)
            p_sec.paragraph_format.space_after = Pt(4)
            p_sec.paragraph_format.keep_with_next = True
            r = p_sec.add_run(sec_title.upper())
            r.font.name = FONT_BODY
            r.font.size = Pt(10)
            r.bold = True
            r.font.color.rgb = BLACK
            i += 1
            continue

        # Abstract body and Keywords in IEEE style (SRIP_Sandhya_Final.pdf)
        if current_section_name == "ABSTRACT":
            if stripped.startswith("**Keywords:**"):
                kw_body = stripped[len("**Keywords:**") :].strip()
                p_kw = doc.add_paragraph()
                p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p_kw.paragraph_format.space_before = Pt(3)
                p_kw.paragraph_format.space_after = Pt(6)
                p_kw.paragraph_format.line_spacing = 1.05
                rk1 = p_kw.add_run("Keywords— ")
                rk1.font.name = FONT_BODY
                rk1.font.size = Pt(9)
                rk1.bold = True
                rk1.italic = True
                rk1.font.color.rgb = BLACK
                rk2 = p_kw.add_run(kw_body)
                rk2.font.name = FONT_BODY
                rk2.font.size = Pt(9)
                rk2.bold = True
                rk2.italic = True
                rk2.font.color.rgb = BLACK
                current_section_name = ""
            else:
                p_abs = doc.add_paragraph()
                p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p_abs.paragraph_format.space_before = Pt(0)
                p_abs.paragraph_format.space_after = Pt(4)
                p_abs.paragraph_format.line_spacing = 1.05
                ra1 = p_abs.add_run("Abstract—")
                ra1.font.name = FONT_BODY
                ra1.font.size = Pt(9)
                ra1.bold = True
                ra1.italic = True
                ra1.font.color.rgb = BLACK
                add_formatted_runs(p_abs, stripped, font_size_pt=9.0, base_bold=True)
            i += 1
            continue

        # Subsection Headings (### A. ...)
        if stripped.startswith("### "):
            sub_title = stripped[4:].strip()
            p_sub = doc.add_paragraph()
            p_sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_sub.paragraph_format.space_before = Pt(6)
            p_sub.paragraph_format.space_after = Pt(2)
            p_sub.paragraph_format.keep_with_next = True
            r = p_sub.add_run(sub_title)
            r.font.name = FONT_BODY
            r.font.size = Pt(10)
            r.italic = True
            r.bold = False
            r.font.color.rgb = BLACK
            i += 1
            continue

        # Sub-subsection Headings (#### 1) ...)
        if stripped.startswith("#### "):
            sub2_title = stripped[5:].strip()
            p_sub2 = doc.add_paragraph()
            p_sub2.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_sub2.paragraph_format.left_indent = Inches(0.10)
            p_sub2.paragraph_format.space_before = Pt(4)
            p_sub2.paragraph_format.space_after = Pt(1.5)
            p_sub2.paragraph_format.keep_with_next = True
            r = p_sub2.add_run(sub2_title)
            r.font.name = FONT_BODY
            r.font.size = Pt(10)
            r.italic = True
            r.font.color.rgb = BLACK
            i += 1
            continue

        # Display Equations (> ...)
        if stripped.startswith("> "):
            add_display_equation(doc, stripped[2:])
            i += 1
            continue

        # Figures: ![alt](figures/figX.png) + optional next-line *Fig. X. ...*
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img_match:
            _, img_rel = img_match.groups()
            img_path = ROOT / "research_paper" / img_rel.replace("/", "\\")
            fig_caption = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*Fig."):
                fig_caption = lines[i + 1].strip().strip("*").strip()
                i += 1

            if img_path.exists():
                fname = img_path.name
                if fname in WIDE_FIGURES:
                    sec_f = doc.add_section(WD_SECTION.CONTINUOUS)
                    set_section_columns(sec_f, 1)
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.paragraph_format.space_before = Pt(6)
                    p_img.paragraph_format.space_after = Pt(2)
                    p_img.paragraph_format.keep_with_next = True
                    p_img.add_run().add_picture(str(img_path), width=Inches(WIDE_FIGURES[fname]))
                    if fig_caption:
                        p_fc = doc.add_paragraph()
                        p_fc.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        p_fc.paragraph_format.space_before = Pt(1)
                        p_fc.paragraph_format.space_after = Pt(6)
                        add_formatted_runs(p_fc, fig_caption, font_size_pt=9.0)
                    sec_b = doc.add_section(WD_SECTION.CONTINUOUS)
                    set_section_columns(sec_b, 2)
                else:
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.paragraph_format.space_before = Pt(5)
                    p_img.paragraph_format.space_after = Pt(2)
                    p_img.paragraph_format.keep_with_next = True
                    p_img.add_run().add_picture(str(img_path), width=Inches(3.25))
                    if fig_caption:
                        p_fc = doc.add_paragraph()
                        p_fc.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        p_fc.paragraph_format.space_before = Pt(1)
                        p_fc.paragraph_format.space_after = Pt(5)
                        add_formatted_runs(p_fc, fig_caption, font_size_pt=8.5)
            i += 1
            continue

        # Standalone *Fig. ...* if not consumed above
        if stripped.startswith("*Fig.") and stripped.endswith("*"):
            p_fc = doc.add_paragraph()
            p_fc.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_fc.paragraph_format.space_after = Pt(5)
            add_formatted_runs(p_fc, stripped.strip("*"), font_size_pt=8.5)
            i += 1
            continue

        # References section: format with explicit IEEE [1], [2], ... brackets
        if current_section_name == "REFERENCES":
            ref_m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if ref_m:
                ref_num, ref_body = ref_m.groups()
                p_ref = doc.add_paragraph()
                p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p_ref.paragraph_format.left_indent = Inches(0.26)
                p_ref.paragraph_format.first_line_indent = Inches(-0.26)
                p_ref.paragraph_format.space_before = Pt(0)
                p_ref.paragraph_format.space_after = Pt(2.0)
                p_ref.paragraph_format.line_spacing = 1.03
                rn = p_ref.add_run(f"[{ref_num}]\t")
                rn.font.name = FONT_BODY
                rn.font.size = Pt(8.5)
                rn.font.color.rgb = BLACK
                add_formatted_runs(p_ref, ref_body, font_size_pt=8.5)
                i += 1
                continue

        # Bullet items (* ... or - ... or indented   * ...)
        bullet_m = re.match(r"^(\s*)[*-]\s+(.*)$", line)
        if bullet_m:
            indent_spaces, b_text = bullet_m.groups()
            is_nested = len(indent_spaces) >= 2
            p_b = doc.add_paragraph()
            p_b.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_b.paragraph_format.left_indent = Inches(0.28 if is_nested else 0.16)
            p_b.paragraph_format.first_line_indent = Inches(-0.12)
            p_b.paragraph_format.space_before = Pt(0)
            p_b.paragraph_format.space_after = Pt(2.0)
            p_b.paragraph_format.line_spacing = 1.05
            rb = p_b.add_run("• " if not is_nested else "– ")
            rb.font.name = FONT_BODY
            rb.font.size = Pt(9.5)
            rb.font.color.rgb = BLACK
            add_formatted_runs(p_b, b_text.strip(), font_size_pt=9.5)
            i += 1
            continue

        # Numbered items (1. ... in body / appendices)
        num_m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if num_m:
            num_str, n_text = num_m.groups()
            p_n = doc.add_paragraph()
            p_n.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_n.paragraph_format.left_indent = Inches(0.18)
            p_n.paragraph_format.first_line_indent = Inches(-0.18)
            p_n.paragraph_format.space_before = Pt(1)
            p_n.paragraph_format.space_after = Pt(2.5)
            p_n.paragraph_format.line_spacing = 1.06
            rn = p_n.add_run(f"{num_str}. ")
            rn.font.name = FONT_BODY
            rn.font.size = Pt(10)
            rn.bold = True
            rn.font.color.rgb = BLACK
            add_formatted_runs(p_n, n_text.strip(), font_size_pt=10.0)
            i += 1
            continue

        # Indented continuation lines inside Appendix numbered items
        if line.startswith("   ") and not stripped.startswith("*"):
            p_ind = doc.add_paragraph()
            p_ind.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_ind.paragraph_format.left_indent = Inches(0.18)
            p_ind.paragraph_format.space_before = Pt(0)
            p_ind.paragraph_format.space_after = Pt(2.0)
            p_ind.paragraph_format.line_spacing = 1.05
            add_formatted_runs(p_ind, stripped, font_size_pt=9.5)
            i += 1
            continue

        # Regular IEEE justified body paragraph
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(0.14)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3.5)
        p.paragraph_format.line_spacing = 1.08
        add_formatted_runs(p, stripped, font_size_pt=10.0)
        i += 1

    if table_buffer:
        render_markdown_table(doc, pending_table_caption, table_buffer)

    # Save to RESEARCH_PAPER_CLINIGUARD_IEEE_FINAL.docx and sync other targets if unlocked
    doc.save(str(DOCX_IEEE))
    print(f"[OK] Saved IEEE final paper -> {DOCX_IEEE} ({DOCX_IEEE.stat().st_size // 1024} KB)")
    for target in (DOCX_MAIN, DOCX_FINAL):
        try:
            shutil.copy2(DOCX_IEEE, target)
            print(f"[OK] Synced copy -> {target} ({target.stat().st_size // 1024} KB)")
        except PermissionError:
            print(f"[NOTE] {target.name} is currently open in Microsoft Word (locked); open {DOCX_IEEE.name} instead.")


if __name__ == "__main__":
    compile_manuscript()
