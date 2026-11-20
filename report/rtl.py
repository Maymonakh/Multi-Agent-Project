"""Arabic / right-to-left text for reportlab (it does not do this by itself).
We wrap the words ourselves, then shape + reorder each line, then right-align."""
import re
from xml.sax.saxutils import escape
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph

ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def has_arabic(text: str) -> bool:
    return bool(ARABIC.search(text or ""))


def _reshape(text: str) -> str:
    import arabic_reshaper
    return arabic_reshaper.reshape(text)


LRI, PDI = "\u2066", "\u2069"                     # "left-to-right isolate" markers
LATIN_GROUP = re.compile(r"[(\[][^()\[\]\u0600-\u06FF]*[A-Za-z0-9][^()\[\]\u0600-\u06FF]*[)\]]")


def shape_line(text: str) -> str:
    """One visual line: joined letters, right-to-left order.
    A citation like (A.pdf, page 1) is isolated so its brackets stay in the right place."""
    try:
        from bidi import get_display
    except ImportError:
        from bidi.algorithm import get_display
    text = LATIN_GROUP.sub(lambda m: LRI + m.group(0) + PDI, text)
    return get_display(_reshape(text)).replace(LRI, "").replace(PDI, "")


def rtl_paragraph(text: str, style: ParagraphStyle, width: float) -> Paragraph:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and stringWidth(_reshape(trial), style.fontName, style.fontSize) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    markup = "<br/>".join(escape(shape_line(line)) for line in lines)
    return Paragraph(markup, ParagraphStyle("rtl", parent=style, alignment=TA_RIGHT))
