"""Fonts the report can use: 3 built-in PDF fonts, the bundled DejaVu Sans (supports Arabic),
and any .ttf file you drop into report/fonts/ (its file name becomes the font name)."""
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = Path(__file__).parent / "fonts"
UNICODE_FONT = "DejaVu Sans"

# order: regular, bold, italic, bold-italic
BUILTIN = {
    "Helvetica": ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"),
    "Times": ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"),
    "Courier": ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"),
}
DEJAVU = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")
_fonts = None


def _register(family: str, files: tuple) -> tuple:
    names = []
    for suffix, path in zip(("", "-Bold", "-Italic", "-BoldItalic"), files):
        name = (family + suffix).replace(" ", "")
        pdfmetrics.registerFont(TTFont(name, str(path)))
        names.append(name)
    pdfmetrics.registerFontFamily(names[0], normal=names[0], bold=names[1],
                                  italic=names[2], boldItalic=names[3])
    return tuple(names)


def available_fonts() -> dict:
    global _fonts
    if _fonts is None:
        fonts = dict(BUILTIN)
        dejavu = [FONT_DIR / name for name in DEJAVU]
        if all(path.exists() for path in dejavu):
            fonts[UNICODE_FONT] = _register(UNICODE_FONT, tuple(dejavu))
        for path in sorted(FONT_DIR.glob("*.ttf")):
            if not path.name.startswith("DejaVu"):
                fonts[path.stem] = _register(path.stem, (path,) * 4)
        _fonts = fonts
    return _fonts


def find_family(name: str) -> str | None:
    for family in available_fonts():
        if family.lower() == (name or "").strip().lower():
            return family
    return None


def font_name(family: str, bold: bool = False, italic: bool = False, unicode_needed: bool = False) -> str:
    fonts = available_fonts()
    if unicode_needed and (family in BUILTIN) and UNICODE_FONT in fonts:
        family = UNICODE_FONT                      # built-in PDF fonts have no Arabic glyphs
    variants = fonts.get(family) or BUILTIN["Helvetica"]
    return variants[(2 if italic else 0) + (1 if bold else 0)] if not (bold and italic) else variants[3]
