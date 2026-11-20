import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

from api import services
from api.errors import ApiError
from api.schemas import AskRequest, AskResponse
from ocr.engine import (InvalidImageError, OCRError, OCRUnavailableError, TesseractEngine,
                        clean_ocr_text, load_image, preprocess)
from ocr.service import compose_question, ingest_image, ocr_document_name

def _find_font():
    """A TrueType font with Latin + Arabic glyphs, on Linux / Windows / macOS."""
    import os
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf",
                 "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(path):
            return path
    raise SystemExit("No TrueType font found for the OCR test images; edit _find_font() in test_task5.py.")


FONT = _find_font()


def render(text, lang=None, fmt="PNG", size=48, mode="RGB"):
    font = ImageFont.truetype(FONT, size)
    img = Image.new("RGB", (1100, 200), "white")
    ImageDraw.Draw(img).text((20, 40), text, font=font, fill="black", **({"language": lang} if lang else {}))
    buffer = io.BytesIO()
    img.convert(mode).save(buffer, fmt)
    return buffer.getvalue()


class FakeStore:
    def __init__(self, folder):
        self.folder, self.docs, self.fail = Path(folder), {}, False

    def incoming_path(self, name):
        (self.folder / ".incoming").mkdir(exist_ok=True)
        return self.folder / ".incoming" / Path(name).name

    def ingest(self, path):
        if self.fail:
            raise ValueError("No readable text was found.")
        self.docs[Path(path).name] = Path(path).read_bytes()
        Path(path).unlink()
        return f"Successfully ingested {Path(path).name} (1 chunks)."

    def document_names(self):
        return sorted(self.docs)

    def delete_document(self, name):
        if name not in self.docs:
            raise KeyError(name)
        del self.docs[name]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def ask(self, question, history=None):
        self.calls.append((question, history))
        return SimpleNamespace(answer=f"answer to: {question[:40]}", cached=False, analysis={"k": 1})


def raises(status, fn, *args):
    try:
        fn(*args)
    except ApiError as error:
        assert error.status_code == status, (error.status_code, error.detail)
        return error
    raise AssertionError(f"expected ApiError {status}")


def test_ocr_real():
    engine = TesseractEngine()
    assert "Longformer" in engine.extract_text(render("Longformer reaches 95% accuracy"))
    assert "الشبكات" in engine.extract_text(render("مرحبا بكم في الشبكات المحلية", lang="ar"))
    assert "supports" in engine.extract_text(render("The cable supports high speed", fmt="JPEG")).lower()
    assert "hello" in engine.extract_text(render("Hello world", mode="RGBA")).lower()        # transparency
    small = Image.new("RGB", (300, 60), "white")                                      # tiny image gets upscaled
    ImageDraw.Draw(small).text((5, 10), "Small text", font=ImageFont.truetype(FONT, 22), fill="black")
    buffer = io.BytesIO(); small.save(buffer, "PNG")
    assert "small" in engine.extract_text(buffer.getvalue()).lower()
    print("1 OCR (real tesseract: en / ar / jpeg / alpha / small) OK")


def test_ocr_errors():
    engine = TesseractEngine()
    blank = io.BytesIO(); Image.new("RGB", (400, 200), "white").save(blank, "PNG")
    for bad, kind in [(b"not an image", InvalidImageError), (b"", InvalidImageError),
                      (blank.getvalue(), OCRError)]:
        try:
            engine.extract_text(bad)
            raise AssertionError("expected error")
        except OCRError as error:
            assert isinstance(error, kind), (type(error), error)
    try:                                       # a real PDF header is not an image
        load_image(b"%PDF-1.4 fake")
        raise AssertionError
    except InvalidImageError:
        pass
    try:                                       # language not installed
        TesseractEngine(languages="xx_missing").extract_text(render("abc def"))
        raise AssertionError
    except OCRUnavailableError:
        pass
    assert TesseractEngine(languages="ara+eng+xx_missing").extract_text(render("abc def ghi"))   # partial ok
    assert clean_ocr_text("a  b \n\n\n\nc\x0c") == "a b\n\nc"
    assert max(preprocess(Image.new("RGB", (6000, 100))).size) == 3000
    assert max(preprocess(Image.new("RGB", (400, 100))).size) == 1000
    print("2 OCR errors / preprocessing OK")


def test_ocr_service():
    with tempfile.TemporaryDirectory() as tmp:
        store = FakeStore(tmp)
        result = ingest_image(store, "My Scan (1).PNG", render("Longformer reaches 95% accuracy"), TesseractEngine())
        assert result["document"] == "My Scan _1_ocr.txt" or result["document"].endswith("_ocr.txt")
        assert "Longformer" in store.docs[result["document"]].decode()
        store.fail = True
        try:
            ingest_image(store, "x.png", render("Another text here"), TesseractEngine())
            raise AssertionError
        except ValueError:
            pass
        assert not list((Path(tmp) / ".incoming").iterdir())            # failed ingest leaves no file behind
    assert ocr_document_name("../../etc/passwd.png") == "passwd_ocr.txt"
    assert ocr_document_name("صورة الشبكة.jpg") == "صورة الشبكة_ocr.txt"
    assert compose_question("", "text in image") == "text in image"
    assert compose_question("What is it?", "abc").startswith("What is it?\n\nText from the attached image:\nabc")
    assert len(compose_question("", "x" * 5000)) == 1500
    print("3 OCR -> document / question OK")


def test_api_services():
    with tempfile.TemporaryDirectory() as tmp:
        store, pipeline, engine = FakeStore(tmp), FakePipeline(), TesseractEngine()

        assert services.read_limited(io.BytesIO(b"abc"), 10) == b"abc"
        raises(413, services.read_limited, io.BytesIO(b"x" * 11), 10)
        raises(400, services.read_limited, io.BytesIO(b""), 10)

        out = services.upload_document(store, "../../notes.txt", b"hello")
        assert out["document"] == "notes.txt" and store.document_names() == ["notes.txt"]
        raises(415, services.upload_document, store, "virus.exe", b"x")
        raises(400, services.upload_document, store, ".hidden.txt", b"x")
        store.fail = True
        raises(422, services.upload_document, store, "empty.txt", b"x")
        assert not list((Path(tmp) / ".incoming").iterdir())
        store.fail = False

        img = render("Longformer reaches 95% accuracy")
        out = services.upload_image_document(store, engine, "scan.png", img)
        assert out["document"] == "scan_ocr.txt" and "Longformer" in out["ocr_text"] and out["characters"] > 10
        raises(415, services.upload_image_document, store, engine, "scan.png", b"not an image")
        blank = io.BytesIO(); Image.new("RGB", (400, 200), "white").save(blank, "PNG")
        raises(422, services.upload_image_document, store, engine, "blank.png", blank.getvalue())
        raises(503, services.upload_image_document, store, TesseractEngine(languages="xx"), "a.png", img)

        assert services.ocr_only(engine, img)["engine"] == "tesseract"
        assert "scan_ocr.txt" in services.list_documents(store)
        services.delete_document(store, "scan_ocr.txt")
        raises(404, services.delete_document, store, "scan_ocr.txt")

        reply = services.ask(pipeline, "q?", [{"role": "user", "content": "hi"}])
        assert reply["answer"].startswith("answer to") and reply["elapsed_ms"] >= 0 and reply["ocr_text"] is None
        AskResponse(**reply)

        reply = services.ask_with_image(pipeline, engine, "What does it say?", img, [])
        assert "Longformer" in pipeline.calls[-1][0] and pipeline.calls[-1][0].startswith("What does it say?")
        assert "Longformer" in reply["ocr_text"]
        services.ask_with_image(pipeline, engine, "", img)
        assert pipeline.calls[-1][0].startswith("Longformer")           # image alone = the question
        raises(415, services.ask_with_image, pipeline, engine, "q", b"junk")

        assert services.parse_history("") == []
        assert services.parse_history('[{"role":"user","content":"a"}]') == [{"role": "user", "content": "a"}]
        raises(422, services.parse_history, "not json")
        raises(422, services.parse_history, '[{"x": 1}]')
    print("4 API services OK")


def test_ask_stream():
    import json as _json
    from analyst.schemas import AnalystDecision

    class StreamingPipeline:
        def __init__(self, fail=False):
            self.fail = fail

        def stream(self, question, history=None):
            yield ("step", "rewrite", {"query": question})
            yield ("step", "retrieve", {"retrieved": [1, 2, 3]})
            yield ("step", "analyst", {"decision": AnalystDecision(enough_evidence=False), "follow_up": "more about C"})
            if self.fail:
                raise RuntimeError("OPENROUTER_API_KEY is missing")
            yield ("reply", SimpleNamespace(answer="done", cached=True, analysis={"a": 1}))

    lines = list(services.ask_stream(StreamingPipeline(), "q?", []))
    assert all(line.endswith("\n") for line in lines)
    events = [_json.loads(line) for line in lines]
    assert [e["type"] for e in events] == ["step", "step", "step", "result"]
    assert events[0]["label"] == "Understanding the question" and events[0]["detail"] == "q?"
    assert events[1]["detail"] == "3 chunks" and "more about C" in events[2]["detail"]
    assert events[3]["answer"] == "done" and events[3]["cached"] is True and events[3]["elapsed_ms"] >= 0

    events = [_json.loads(line) for line in services.ask_stream(StreamingPipeline(fail=True), "q?", [])]
    assert events[-1] == {"type": "error", "detail": "The language model is not configured."}
    assert services.step_detail("cache_lookup", {"cached": True}) == "hit"
    assert services.step_detail("finalize", {}) == ""
    print("5b streaming service OK")


def test_schemas():
    AskRequest(question="hi", history=[{"role": "user", "content": "a"}])
    for bad in [{"question": ""}, {"question": "x" * 2001}, {"question": "q", "history": [{"role": "bot", "content": "a"}]}]:
        try:
            AskRequest(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    print("5 schemas OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL OK")
