"""ReportBuilder: holds the design + the content blocks and renders the PDF with reportlab.
Every public method that starts with set_ / add_ / apply_ is a TOOL the Report Agent can call.
Tools never raise: they return "OK ..." or "Error: ..." so the agent can fix its call."""
import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A3, A4, LEGAL, LETTER, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, ListFlowable, ListItem,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

from report.design import COLOR_KEYS, PAGE_SIZES, ROLES, Design, get_preset, PRESETS
from report.fonts import available_fonts, find_family, font_name
from report.markdown_blocks import inline_markup, md_to_blocks, plain_text
from report.rtl import has_arabic, rtl_paragraph, shape_line

SIZES = {"A4": A4, "A3": A3, "LETTER": LETTER, "LEGAL": LEGAL}
ALIGN = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT, "justify": TA_JUSTIFY}
ROLE_COLOR = {"title": "primary", "subtitle": "secondary", "h1": "primary", "h2": "primary",
              "h3": "secondary", "body": "text", "table": "text", "caption": "secondary"}


def to_color(value, default=None):
    try:
        return colors.toColor(value)
    except Exception:
        return default


def tint(color, amount=0.88):
    return colors.Color(color.red + (1 - color.red) * amount, color.green + (1 - color.green) * amount,
                        color.blue + (1 - color.blue) * amount)


def fmt(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def cite(point: dict) -> str:
    if not point:
        return "-"
    page = f", p.{point['page']}" if point.get("page") is not None else ""
    return f"{fmt(point.get('value', ''))} ({point.get('document', '?')}{page})"


class ReportBuilder:
    def __init__(self, question, answer, analysis=None, extras=None, preset="professional", title=""):
        self.question = (question or "").strip()
        self.answer = answer or ""
        self.analysis = analysis or {}
        self.extras = [e for e in (extras or []) if e.get("text")]
        self.design: Design = get_preset(preset)
        self.title = title.strip() or "Analysis Report"
        self.blocks: list[dict] = []
        self.added: set[str] = set()
        self.finished = False
        self._rtl = False

    # ================================================================= DESIGN TOOLS
    def apply_preset(self, name: str) -> str:
        if name.lower() not in PRESETS:
            return f"Error: unknown preset. Choose one of {list(PRESETS)}."
        keep = (self.design.header_text, self.design.footer_text)
        self.design = get_preset(name)
        return f"OK preset '{name}' applied (resets fonts, colors and spacing)."

    def set_page(self, size="A4", orientation="portrait", margin_cm=2.0, columns=1) -> str:
        if size.upper() not in PAGE_SIZES:
            return f"Error: size must be one of {PAGE_SIZES}."
        if orientation not in ("portrait", "landscape"):
            return "Error: orientation must be portrait or landscape."
        if not 1.0 <= margin_cm <= 5.0:
            return "Error: margin_cm must be between 1 and 5."
        if columns not in (1, 2):
            return "Error: columns must be 1 or 2."
        d = self.design
        d.page_size, d.orientation, d.margin_cm, d.columns = size.upper(), orientation, margin_cm, columns
        return f"OK page {d.page_size} {orientation}, margin {margin_cm} cm, {columns} column(s)."

    def set_font(self, role, family=None, size=None, bold=None, italic=None, color=None) -> str:
        if role not in ROLES:
            return f"Error: role must be one of {ROLES}."
        style = self.design.fonts[role]
        if family is not None:
            found = find_family(family)
            if not found:
                return f"Error: unknown font '{family}'. Available fonts: {list(available_fonts())}."
            style.family = found
        if size is not None:
            if not 6 <= size <= 48:
                return "Error: size must be between 6 and 48 points."
            style.size = float(size)
        if bold is not None:
            style.bold = bool(bold)
        if italic is not None:
            style.italic = bool(italic)
        if color is not None:
            if to_color(color) is None:
                return f"Error: '{color}' is not a valid color. Use a hex code like #1F3A5F."
            style.color = color
        return f"OK font for {role}: {style.family} {style.size}pt bold={style.bold} italic={style.italic} color={style.color}."

    def set_colors(self, **named) -> str:
        changed = []
        for key, value in named.items():
            if value is None:
                continue
            if key not in COLOR_KEYS:
                return f"Error: unknown color name '{key}'. Use {COLOR_KEYS}."
            if to_color(value) is None:
                return f"Error: '{value}' is not a valid color. Use a hex code like #1F3A5F."
            self.design.colors[key] = value
            changed.append(key)
        return f"OK colors changed: {changed}."

    def set_spacing(self, line_height=None, paragraph_spacing=None, section_spacing=None) -> str:
        d = self.design
        if line_height is not None:
            if not 1.0 <= line_height <= 2.5:
                return "Error: line_height must be between 1.0 and 2.5."
            d.line_height = float(line_height)
        if paragraph_spacing is not None:
            if not 0 <= paragraph_spacing <= 40:
                return "Error: paragraph_spacing must be between 0 and 40 points."
            d.paragraph_spacing = float(paragraph_spacing)
        if section_spacing is not None:
            if not 0 <= section_spacing <= 60:
                return "Error: section_spacing must be between 0 and 60 points."
            d.section_spacing = float(section_spacing)
        return f"OK spacing line_height={d.line_height}, paragraph={d.paragraph_spacing}pt, section={d.section_spacing}pt."

    def set_body_alignment(self, align: str) -> str:
        if align not in ALIGN:
            return f"Error: align must be one of {list(ALIGN)}."
        self.design.body_align = align
        return f"OK body text aligned {align}."

    def set_header_footer(self, header_text=None, footer_text=None, page_numbers=None) -> str:
        d = self.design
        if header_text is not None:
            d.header_text = header_text[:120]
        if footer_text is not None:
            d.footer_text = footer_text[:120]
        if page_numbers is not None:
            d.page_numbers = bool(page_numbers)
        return "OK header/footer updated."

    def set_cover_page(self, enabled=True, title=None, subtitle=None, author=None) -> str:
        d = self.design
        d.cover_page = bool(enabled)
        if title is not None:
            self.title = title[:150]
        if subtitle is not None:
            d.cover_subtitle = subtitle[:200]
        if author is not None:
            d.author = author[:100]
        return f"OK cover page {'on' if d.cover_page else 'off'}; title '{self.title}'."

    # ================================================================= CONTENT TOOLS
    def _add(self, block: dict, tag: str = "") -> None:
        self.blocks.append(block)
        if tag:
            self.added.add(tag)

    def add_heading(self, text: str, level: int = 1) -> str:
        if level not in (1, 2, 3):
            return "Error: level must be 1, 2 or 3."
        self._add({"kind": "heading", "text": text[:200], "level": level})
        return f"OK heading level {level}: {text[:40]}"

    def add_paragraph(self, text: str, bold=False, italic=False, color=None, align=None, size=None) -> str:
        if color is not None and to_color(color) is None:
            return f"Error: '{color}' is not a valid color."
        if align is not None and align not in ALIGN:
            return f"Error: align must be one of {list(ALIGN)}."
        self._add({"kind": "paragraph", "text": text, "bold": bold, "italic": italic,
                   "color": color, "align": align, "size": size})
        return "OK paragraph added."

    def add_bullets(self, items: list[str], numbered: bool = False) -> str:
        if not items:
            return "Error: items is empty."
        self._add({"kind": "bullets", "items": [str(i) for i in items], "numbered": numbered})
        return f"OK {'numbered' if numbered else 'bullet'} list with {len(items)} items."

    def add_table(self, headers: list[str], rows: list[list[str]], caption: str = "") -> str:
        if not headers:
            return "Error: headers is empty."
        self._add({"kind": "table", "headers": [str(h) for h in headers],
                   "rows": [[str(c) for c in r] for r in rows], "caption": caption})
        return f"OK table with {len(rows)} rows."

    def add_callout(self, text: str, color: str = None) -> str:
        if color is not None and to_color(color) is None:
            return f"Error: '{color}' is not a valid color."
        self._add({"kind": "callout", "text": text, "color": color})
        return "OK callout box added."

    def add_spacer(self, points: float = 12) -> str:
        if not 0 <= points <= 200:
            return "Error: points must be between 0 and 200."
        self._add({"kind": "spacer", "height": float(points)})
        return f"OK {points}pt of space."

    def add_page_break(self) -> str:
        self._add({"kind": "page_break"})
        return "OK page break."

    def add_rule(self) -> str:
        self._add({"kind": "rule"})
        return "OK horizontal line."

    def add_question(self, heading: str = "Question") -> str:
        if not self.question:
            return "Error: there is no question."
        self.add_heading(heading, 1)
        self._add({"kind": "paragraph", "text": self.question}, "question")
        return "OK question added."

    def add_answer(self, heading: str = "Answer") -> str:
        """The application's answer, word for word (with citations and sources)."""
        if not self.answer.strip():
            return "Error: there is no answer."
        if heading:
            self.add_heading(heading, 1)
        self._add({"kind": "markdown", "text": self.answer}, "answer")
        return "OK the original answer was added."

    def add_analysis(self, kind: str) -> str:
        a = self.analysis
        if kind == "summary":
            if not a:
                return "Error: there is no analysis."
            self.add_heading("Analysis summary", 2)
            if a.get("reasoning"):
                self._add({"kind": "paragraph", "text": a["reasoning"]})
            ok = a.get("enough_evidence")
            self._add({"kind": "callout", "text": "The documents contain enough evidence." if ok
                       else "The documents do not contain enough evidence for a complete answer."})
        elif kind == "calculations":
            calc = a.get("calculations") or {}
            if not calc:
                return "Error: there are no calculations in this analysis."
            rows = [[name, fmt(c.get("average")), cite(c.get("best")), cite(c.get("worst")),
                     fmt(c.get("difference")), f"{fmt(c.get('percent_improvement'))}%"] for name, c in calc.items()]
            self.add_heading("Calculations", 2)
            self.add_table(["Metric", "Average", "Best", "Worst", "Difference", "Improvement"], rows,
                           "Calculated in Python from the numbers found in the documents.")
        elif kind == "comparison":
            comp = a.get("comparison") or {}
            values = comp.get("values_by_document") or {}
            if not values and not comp.get("similarities") and not comp.get("differences"):
                return "Error: there is no comparison in this analysis."
            self.add_heading("Document comparison", 2)
            if values:
                metrics = list(dict.fromkeys(m for v in values.values() for m in v))
                rows = [[doc] + [cite({**v[m], "document": doc}) if m in v else "-" for m in metrics]
                        for doc, v in values.items()]
                self.add_table(["Document"] + metrics, rows)
            for title, key in (("Similarities", "similarities"), ("Differences", "differences")):
                if comp.get(key):
                    self.add_heading(title, 3)
                    self.add_bullets(comp[key])
        elif kind == "tables":
            tables = a.get("tables") or []
            if not tables:
                return "Error: no tables were found in the documents."
            self.add_heading("Tables found in the documents", 2)
            for t in tables:
                rows = t.get("rows") or []
                if rows:
                    page = f", page {t['page']}" if t.get("page") is not None else ""
                    self.add_table(rows[0], rows[1:], f"{t.get('document', '')}{page}")
        else:
            return "Error: kind must be summary, calculations, comparison or tables."
        self.added.add(f"analysis:{kind}")
        return f"OK analysis '{kind}' added."

    def add_extra_sections(self) -> str:
        if not self.extras:
            return "Error: there are no extra sections."
        for extra in self.extras:
            self.add_heading(extra.get("heading") or "Appendix", 2)
            self._add({"kind": "markdown", "text": extra["text"]})
        self.added.add("extras")
        return f"OK {len(self.extras)} extra section(s) added."

    def finish_report(self) -> str:
        self.finished = True
        return "OK the report is finished."

    # ================================================================= DEFAULT LAYOUT
    def default_layout(self) -> None:
        self.add_question()
        self.add_answer()
        for kind in ("calculations", "comparison", "tables"):
            self.add_analysis(kind)
        if self.extras:
            self.add_extra_sections()

    def _ensure_required(self) -> None:
        if not self.blocks:
            self.default_layout()
            return
        if "answer" not in self.added and self.answer.strip():
            self.add_answer()
        if self.extras and "extras" not in self.added:
            self.add_extra_sections()

    # ================================================================= RENDERING
    def _all_text(self) -> str:
        parts = [self.question, self.answer, self.title, self.design.header_text, self.design.footer_text]
        parts += [e.get("text", "") + e.get("heading", "") for e in self.extras]
        for b in self.blocks:
            parts += [b.get("text", "")] + list(b.get("items", [])) + list(b.get("headers", []))
            parts += [c for r in b.get("rows", []) for c in r]
        return " ".join(str(p) for p in parts)

    def _color(self, key):
        return to_color(self.design.colors.get(key), colors.black)

    def _style(self, role, **extra) -> ParagraphStyle:
        d, f = self.design, self.design.fonts[role]
        color = to_color(f.color) if f.color else self._color(ROLE_COLOR[role])
        align = ALIGN[d.body_align] if role == "body" else (TA_CENTER if role == "title" and d.cover_page else TA_LEFT)
        values = dict(name=role, fontName=font_name(f.family, f.bold, f.italic, self._rtl), fontSize=f.size,
                      leading=f.size * d.line_height, textColor=color, alignment=align,
                      spaceAfter=d.paragraph_spacing)
        if role in ("h1", "h2", "h3"):
            values.update(spaceBefore=d.section_spacing * {"h1": 1, "h2": 0.7, "h3": 0.5}[role],
                          spaceAfter=d.paragraph_spacing, keepWithNext=1)
        values.update(extra)
        return ParagraphStyle(**values)

    def _para(self, text, style, width, markup=True):
        if self._rtl and has_arabic(text):
            return rtl_paragraph(plain_text(text), style, width)
        return Paragraph(inline_markup(text) if markup else escape(text), style)

    def _table(self, block, width):
        d = self.design
        head_style = self._style("table", textColor=colors.white,
                                 fontName=font_name(d.fonts["table"].family, True, False, self._rtl))
        cell_style = self._style("table", spaceAfter=0)
        columns = max(len(block["headers"]), *(len(r) for r in block["rows"])) if block["rows"] else len(block["headers"])
        data = [block["headers"] + [""] * (columns - len(block["headers"]))]
        data += [r + [""] * (columns - len(r)) for r in block["rows"]]
        weights = [min(max(max(len(str(row[c])) for row in data), 6), 38) for c in range(columns)]
        widths = [width * w / sum(weights) for w in weights]
        table_data = [[self._para(str(cell), head_style if r == 0 else cell_style, widths[c] - 10, markup=False)
                       for c, cell in enumerate(row)] for r, row in enumerate(data)]
        table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self._color("table_header")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self._color("table_stripe")]),
            ("GRID", (0, 0), (-1, -1), 0.4, self._color("rule")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        out = [table]
        if block.get("caption"):
            out.append(self._para(block["caption"], self._style("caption", spaceBefore=3), width))
        out.append(Spacer(1, self.design.paragraph_spacing))
        return out

    def _flow(self, block, width) -> list:
        d, kind = self.design, block["kind"]
        if kind == "heading":
            return [self._para(block["text"], self._style(f"h{block['level']}"), width, markup=False)]
        if kind == "paragraph":
            extra = {}
            f = d.fonts["body"]
            if block.get("bold") or block.get("italic"):
                extra["fontName"] = font_name(f.family, bool(block.get("bold")), bool(block.get("italic")), self._rtl)
            if block.get("color"):
                extra["textColor"] = to_color(block["color"], colors.black)
            if block.get("align"):
                extra["alignment"] = ALIGN[block["align"]]
            if block.get("size"):
                size = min(max(float(block["size"]), 6), 48)
                extra.update(fontSize=size, leading=size * d.line_height)
            return [self._para(block["text"], self._style("body", **extra), width)]
        if kind == "bullets":
            style = self._style("body", spaceAfter=2)
            if self._rtl and has_arabic(" ".join(block["items"])):
                return [rtl_paragraph(("%d. " % (n + 1) if block["numbered"] else "\u2022 ") + plain_text(item), style, width)
                        for n, item in enumerate(block["items"])] + [Spacer(1, d.paragraph_spacing)]
            items = [ListItem(Paragraph(inline_markup(i), style)) for i in block["items"]]
            return [ListFlowable(items, bulletType="1" if block["numbered"] else "bullet", leftIndent=16,
                                 bulletFontName=style.fontName, bulletFontSize=style.fontSize,
                                 bulletColor=self._color("accent")), Spacer(1, d.paragraph_spacing)]
        if kind == "table":
            return self._table(block, width)
        if kind == "callout":
            color = to_color(block.get("color")) or self._color("accent")
            box = Table([[self._para(block["text"], self._style("body", spaceAfter=0), width - 24)]], colWidths=[width])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tint(color)),
                                     ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                     ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            return [box, Spacer(1, d.paragraph_spacing)]
        if kind == "spacer":
            return [Spacer(1, block["height"])]
        if kind == "page_break":
            return [PageBreak()]
        if kind == "rule":
            return [HRFlowable(width="100%", thickness=1, color=self._color("rule"), spaceBefore=4, spaceAfter=8)]
        if kind == "markdown":
            flows = []
            for sub in md_to_blocks(block["text"]):
                flows += self._flow(sub, width)
            return flows
        return []

    def _decorate(self, canvas, doc):
        d, W, H = self.design, doc.pagesize[0], doc.pagesize[1]
        margin = d.margin_cm * cm
        font = font_name(d.fonts["body"].family, False, False, self._rtl)
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(self._color("secondary"))
        canvas.setStrokeColor(self._color("rule"))

        def text(value):
            return shape_line(value) if has_arabic(value) else value

        if d.header_text:
            canvas.drawString(margin, H - margin * 0.55, text(d.header_text))
            canvas.line(margin, H - margin * 0.7, W - margin, H - margin * 0.7)
        canvas.line(margin, margin * 0.7, W - margin, margin * 0.7)
        if d.footer_text:
            canvas.drawString(margin, margin * 0.4, text(d.footer_text))
        if d.page_numbers:
            canvas.drawRightString(W - margin, margin * 0.4, f"Page {doc.page}")
        canvas.restoreState()

    def _cover_page(self, canvas, doc):
        d, W, H = self.design, doc.pagesize[0], doc.pagesize[1]
        margin = d.margin_cm * cm
        canvas.saveState()
        canvas.setFillColor(self._color("primary"))
        canvas.rect(0, H - 8 * cm, W, 8 * cm, fill=1, stroke=0)
        canvas.setFillColor(self._color("accent"))
        canvas.rect(0, H - 8 * cm - 4, W, 4, fill=1, stroke=0)
        title = self._para(self.title, self._style("title", textColor=colors.white, alignment=TA_LEFT), W - 2 * margin, markup=False)
        w, h = title.wrap(W - 2 * margin, 6 * cm)
        title.drawOn(canvas, margin, H - 4 * cm - h)
        if d.cover_subtitle:
            sub = self._para(d.cover_subtitle, self._style("subtitle", textColor=colors.white), W - 2 * margin, markup=False)
            w2, h2 = sub.wrap(W - 2 * margin, 3 * cm)
            sub.drawOn(canvas, margin, H - 4 * cm - h - h2 - 6)
        canvas.restoreState()

    def _title_block(self, width) -> list:
        out = [self._para(self.title, self._style("title"), width, markup=False)]
        if self.design.cover_subtitle:
            out.append(self._para(self.design.cover_subtitle, self._style("subtitle"), width, markup=False))
        meta = datetime.date.today().strftime("%d %B %Y") + (f"  |  {self.design.author}" if self.design.author else "")
        out.append(self._para(meta, self._style("caption"), width, markup=False))
        out.append(HRFlowable(width="100%", thickness=2, color=self._color("accent"), spaceBefore=2, spaceAfter=10))
        return out

    def render(self, path) -> str:
        self._ensure_required()
        d = self.design
        self._rtl = has_arabic(self._all_text())
        page = landscape(SIZES[d.page_size]) if d.orientation == "landscape" else SIZES[d.page_size]
        W, H = page
        margin, gap = d.margin_cm * cm, 0.8 * cm
        full_width = W - 2 * margin
        col_width = (full_width - gap * (d.columns - 1)) / d.columns

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        doc = BaseDocTemplate(str(path), pagesize=page, title=self.title, author=d.author or "Multi-Agent RAG",
                              leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin)
        body = PageTemplate(id="body", onPage=self._decorate, frames=[
            Frame(margin + i * (col_width + gap), margin, col_width, H - 2 * margin, id=f"col{i}",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0) for i in range(d.columns)])
        story = []
        if d.cover_page:
            cover = PageTemplate(id="cover", onPage=self._cover_page, frames=[
                Frame(margin, margin, full_width, H - 2 * margin, id="cover",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)])
            doc.addPageTemplates([cover, body])
            meta = datetime.date.today().strftime("%d %B %Y") + (f"  |  {d.author}" if d.author else "")
            story += [Spacer(1, 9.5 * cm), self._para(meta, self._style("body"), full_width, markup=False),
                      NextPageTemplate("body"), PageBreak()]
        else:
            doc.addPageTemplates([body])
            story += self._title_block(col_width)
        for block in self.blocks:
            story += self._flow(block, col_width)
        doc.build(story)
        return str(path)
