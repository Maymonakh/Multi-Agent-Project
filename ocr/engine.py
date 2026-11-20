from __future__ import annotations

import io
import os
import re
import threading
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 50_000_000          # decompression-bomb guard
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"}
MAX_SIDE = 3000                               # larger images are downscaled
MIN_SIDE = 1000                               # smaller images are upscaled (Tesseract likes big text)


class OCRError(Exception):
    """OCR could not produce text (e.g. no text in the image)."""


class InvalidImageError(OCRError):
    """The upload is not a supported / safe image."""


class OCRUnavailableError(OCRError):
    """Tesseract (or its language data) is not installed."""


def load_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
                image_format = probe.format
            if image_format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    f"Unsupported image format: {image_format}. Supported: {', '.join(sorted(ALLOWED_FORMATS))}."
                )
            image = Image.open(io.BytesIO(data))
            image.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise InvalidImageError("The uploaded file is not a valid or supported image.") from error

    return ImageOps.exif_transpose(image)


def preprocess(image: Image.Image) -> Image.Image:
    """Flatten transparency, grayscale, stretch contrast, normalise size."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background

    gray = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)

    longest = max(gray.size)
    if longest > MAX_SIDE:
        factor = MAX_SIDE / longest
    elif longest < MIN_SIDE:
        factor = min(3.0, MIN_SIDE / longest)
    else:
        factor = 1.0
    if factor != 1.0:
        gray = gray.resize((max(1, round(gray.width * factor)), max(1, round(gray.height * factor))),
                           Image.LANCZOS)
    return gray


def clean_ocr_text(text: str) -> str:
    text = (text or "").replace("\x0c", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TesseractEngine:
    name = "tesseract"

    def __init__(self, languages: str | None = None, cmd: str | None = None, timeout: int | None = None):
        raw = languages or os.getenv("OCR_LANGUAGES", "ara+eng")
        self.requested = [lang for lang in re.split(r"[+,\s]+", raw) if lang]
        self.cmd = cmd or os.getenv("TESSERACT_CMD")
        self.timeout = timeout if timeout is not None else int(os.getenv("OCR_TIMEOUT", "60"))
        self._lang: str | None = None
        self._lock = threading.Lock()

    def _pytesseract(self):
        import pytesseract

        if self.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cmd
        return pytesseract

    def _languages(self, pytesseract) -> str:
        with self._lock:
            if self._lang is None:
                try:
                    available = set(pytesseract.get_languages(config=""))
                except pytesseract.TesseractNotFoundError as error:
                    raise OCRUnavailableError(
                        "Tesseract is not installed or not on PATH. Install it or set TESSERACT_CMD."
                    ) from error
                chosen = [lang for lang in self.requested if lang in available]
                if not chosen:
                    raise OCRUnavailableError(
                        f"None of the requested OCR languages {self.requested} are installed "
                        f"(available: {sorted(available)})."
                    )
                self._lang = "+".join(chosen)
            return self._lang

    def extract_text(self, data: bytes) -> str:
        image = preprocess(load_image(data))
        pytesseract = self._pytesseract()
        language = self._languages(pytesseract)

        try:
            raw = pytesseract.image_to_string(image, lang=language, timeout=self.timeout)
        except pytesseract.TesseractNotFoundError as error:
            raise OCRUnavailableError("Tesseract is not installed or not on PATH.") from error
        except pytesseract.TesseractError as error:
            raise OCRError(f"Tesseract failed: {error}") from error
        except RuntimeError as error:                      # pytesseract raises RuntimeError on timeout
            raise OCRError(f"OCR timed out after {self.timeout}s.") from error

        text = clean_ocr_text(raw)
        if len(re.findall(r"\w", text)) < 3:
            raise OCRError("No readable text was detected in the image.")
        return text


_engine = None
_engine_lock = threading.Lock()


def get_ocr_engine() -> TesseractEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = TesseractEngine()
        return _engine
