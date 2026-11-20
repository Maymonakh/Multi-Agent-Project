"""The tools the Report Agent can call. Each one is a thin wrapper around a ReportBuilder method.
The docstrings are what the LLM reads, so they describe exactly what each tool does."""
from typing import Literal, Optional
from langchain_core.tools import tool

from report.builder import ReportBuilder
from report.design import ROLES

Role = Literal["title", "subtitle", "h1", "h2", "h3", "body", "table", "caption"]


def make_tools(b: ReportBuilder) -> list:

    # ---------- design ----------
    @tool
    def apply_preset(name: Literal["professional", "classic", "modern"]) -> str:
        """Start from a ready-made design. professional = blue sans-serif, classic = black serif (Times),
        modern = teal with a cover page. Call this FIRST; it resets fonts, colors and spacing."""
        return b.apply_preset(name)

    @tool
    def set_page(size: Literal["A4", "A3", "LETTER", "LEGAL"] = "A4",
                 orientation: Literal["portrait", "landscape"] = "portrait",
                 margin_cm: float = 2.0, columns: int = 1) -> str:
        """Set the page size, orientation, margins in cm (1 to 5) and the number of text columns (1 or 2)."""
        return b.set_page(size, orientation, margin_cm, columns)

    @tool
    def set_font(role: Role, family: Optional[str] = None, size: Optional[float] = None,
                 bold: Optional[bool] = None, italic: Optional[bool] = None,
                 color: Optional[str] = None) -> str:
        """Change the font of one text role (title, subtitle, h1, h2, h3, body, table, caption).
        family: Helvetica, Times, Courier, 'DejaVu Sans' (needed for Arabic) or a custom font. size: 6 to 48 points.
        color: hex code such as #1F3A5F. Only the values you pass are changed."""
        return b.set_font(role, family, size, bold, italic, color)

    @tool
    def set_colors(primary: Optional[str] = None, secondary: Optional[str] = None, text: Optional[str] = None,
                   accent: Optional[str] = None, table_header: Optional[str] = None,
                   table_stripe: Optional[str] = None, rule: Optional[str] = None) -> str:
        """Change the color theme with hex codes. primary = title and headings, secondary = sub-headings and
        captions, text = body text, accent = bullets, lines and callouts, table_header = table header background,
        table_stripe = alternate table rows, rule = thin lines."""
        return b.set_colors(primary=primary, secondary=secondary, text=text, accent=accent,
                            table_header=table_header, table_stripe=table_stripe, rule=rule)

    @tool
    def set_spacing(line_height: Optional[float] = None, paragraph_spacing: Optional[float] = None,
                    section_spacing: Optional[float] = None) -> str:
        """Set spacing. line_height 1.0 to 2.5 (multiplier of font size), paragraph_spacing 0 to 40 points,
        section_spacing 0 to 60 points (space before headings)."""
        return b.set_spacing(line_height, paragraph_spacing, section_spacing)

    @tool
    def set_body_alignment(align: Literal["left", "center", "right", "justify"]) -> str:
        """Align the body text."""
        return b.set_body_alignment(align)

    @tool
    def set_header_footer(header_text: Optional[str] = None, footer_text: Optional[str] = None,
                          page_numbers: Optional[bool] = None) -> str:
        """Set the text at the top and bottom of every page and turn page numbers on or off."""
        return b.set_header_footer(header_text, footer_text, page_numbers)

    @tool
    def set_cover_page(enabled: bool = True, title: Optional[str] = None, subtitle: Optional[str] = None,
                       author: Optional[str] = None) -> str:
        """Turn the cover page on or off and set the report title, subtitle and author."""
        return b.set_cover_page(enabled, title, subtitle, author)

    # ---------- content ----------
    @tool
    def add_heading(text: str, level: int = 1) -> str:
        """Add a heading. level 1 = main section, 2 = sub-section, 3 = small heading."""
        return b.add_heading(text, level)

    @tool
    def add_paragraph(text: str, bold: bool = False, italic: bool = False, color: Optional[str] = None,
                      align: Optional[Literal["left", "center", "right", "justify"]] = None,
                      size: Optional[float] = None) -> str:
        """Add a paragraph. Use it ONLY for short text based on the given content (for example a summary).
        Never invent facts or numbers. Supports **bold** and *italic* inside the text."""
        return b.add_paragraph(text, bold, italic, color, align, size)

    @tool
    def add_bullets(items: list[str], numbered: bool = False) -> str:
        """Add a bullet list (or a numbered list when numbered=true)."""
        return b.add_bullets(items, numbered)

    @tool
    def add_table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
        """Add a table. Only use values that come from the given content."""
        return b.add_table(headers, rows, caption)

    @tool
    def add_callout(text: str, color: Optional[str] = None) -> str:
        """Add a highlighted box with a colored left bar (good for the key finding)."""
        return b.add_callout(text, color)

    @tool
    def add_spacer(points: float = 12) -> str:
        """Add vertical space (0 to 200 points)."""
        return b.add_spacer(points)

    @tool
    def add_page_break() -> str:
        """Start a new page."""
        return b.add_page_break()

    @tool
    def add_rule() -> str:
        """Add a thin horizontal line."""
        return b.add_rule()

    @tool
    def add_question(heading: str = "Question") -> str:
        """Add the user's question as a section."""
        return b.add_question(heading)

    @tool
    def add_answer(heading: str = "Answer") -> str:
        """REQUIRED. Add the application's answer exactly as written, with its citations and sources.
        Pass an empty heading to add it without a heading."""
        return b.add_answer(heading)

    @tool
    def add_analysis(kind: Literal["summary", "calculations", "comparison", "tables"]) -> str:
        """Add analysis results exactly as computed by the Analyst Agent: summary (reasoning and evidence status),
        calculations (average, best, worst, difference), comparison (documents side by side with similarities and
        differences) or tables (tables found in the documents). Returns an error if that part does not exist."""
        return b.add_analysis(kind)

    @tool
    def add_extra_sections() -> str:
        """Add the extra sections that were attached to this report (for example the voice transcript or OCR text)."""
        return b.add_extra_sections()

    @tool
    def finish_report() -> str:
        """Call this when the report is complete."""
        return b.finish_report()

    return [apply_preset, set_page, set_font, set_colors, set_spacing, set_body_alignment, set_header_footer,
            set_cover_page, add_heading, add_paragraph, add_bullets, add_table, add_callout, add_spacer,
            add_page_break, add_rule, add_question, add_answer, add_analysis, add_extra_sections, finish_report]
