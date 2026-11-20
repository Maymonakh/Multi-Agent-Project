"""Run: python test_voice_report.py   (no API key, no internet, no microphone: fakes are used)"""
import os
import tempfile

try:
    import pymupdf as fitz  # PyMuPDF (already in the project's requirements)
except ImportError:
    import fitz
from langchain_core.messages import AIMessage

from report.service import generate_report
from voice import service as voice_service
from voice.speech_to_text import TranscriptionError

OUT = tempfile.mkdtemp()
QUESTION = "What is the average CNN accuracy?"
ANSWER = ("The average CNN accuracy is **92%** (A.pdf, page 1; B.pdf, page 2).\n\n"
          "| Paper | Accuracy |\n|---|---|\n| A | 92 |\n| B | 95 |\n\n**Sources**\n- A.pdf, page 1\n- B.pdf, page 2")
ANALYSIS = {
    "enough_evidence": True, "reasoning": "All three papers report CNN accuracy.", "error": "",
    "tables": [{"document": "m.pdf", "page": 1, "rows": [["Model", "Acc"], ["BERT", "92"], ["Longformer", "95"]]}],
    "calculations": {"CNN acc": {"average": 92.0, "best": {"document": "B.pdf", "page": 2, "value": 95.0},
                                 "worst": {"document": "C.pdf", "page": 1, "value": 89.0},
                                 "difference": 6.0, "percent_improvement": 6.74, "count": 3}},
    "comparison": {"documents": ["A.pdf", "B.pdf"], "compared_metrics": ["CNN acc"],
                   "values_by_document": {"A.pdf": {"CNN acc": {"value": 92.0, "page": 1}},
                                          "B.pdf": {"CNN acc": {"value": 95.0, "page": 2}}},
                   "similarities": ["Both papers test a CNN."], "differences": ["B reports higher accuracy."]},
}


def pdf_text(path):
    doc = fitz.open(path)
    return doc, "\n".join(page.get_text() for page in doc)


class ScriptedLLM:
    """Pretends to be a tool-calling LLM: each step is a list of (tool_name, args)."""
    def __init__(self, steps):
        self.steps = list(steps)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if not self.steps:
            return AIMessage(content="done")
        calls = [{"name": n, "args": a, "id": f"call{i}"} for i, (n, a) in enumerate(self.steps.pop(0))]
        return AIMessage(content="", tool_calls=calls)


class BrokenLLM:
    def bind_tools(self, tools):
        raise RuntimeError("model has no tool calling")


# 1) default report, no agent
r = generate_report(QUESTION, ANSWER, ANALYSIS, output_path=f"{OUT}/default.pdf")
doc, text = pdf_text(r["path"])
assert r["agent_used"] is False and doc.page_count >= 1
for needle in (QUESTION, "92%", "Sources", "A.pdf, page 1", "Calculations", "Document comparison", "Similarities"):
    assert needle in text, needle
print("1 default report OK")

# 2) agent controls the design (landscape, Times, red headings) + bad calls are handled
llm = ScriptedLLM([
    [("apply_preset", {"name": "classic"}),
     ("set_page", {"size": "A4", "orientation": "landscape", "margin_cm": 2.5, "columns": 1}),
     ("set_font", {"role": "h1", "family": "Times", "size": 22, "color": "#8B0000"}),
     ("set_font", {"role": "body", "family": "Comic Sans"}),          # unknown font -> error text
     ("set_colors", {"accent": "notacolor"}),                           # bad color -> error text
     ("set_font", {"role": "banana"})],                                 # invalid argument -> error text
    [("add_heading", {"text": "Key finding", "level": 1}),
     ("add_callout", {"text": "Average CNN accuracy is 92%."}),
     ("add_answer", {"heading": "Answer"}),
     ("add_analysis", {"kind": "calculations"}),
     ("add_analysis", {"kind": "nothing"}),
     ("finish_report", {})],
])
r = generate_report(QUESTION, ANSWER, ANALYSIS, style="classic, landscape, dark red headings",
                    output_path=f"{OUT}/agent.pdf", llm=llm)
doc, text = pdf_text(r["path"])
d = r["design"]
assert r["agent_used"] and d["orientation"] == "landscape" and d["margin_cm"] == 2.5
assert d["fonts"]["h1"]["family"] == "Times" and d["fonts"]["h1"]["size"] == 22 and d["fonts"]["h1"]["color"] == "#8B0000"
assert doc[0].rect.width > doc[0].rect.height
assert "Key finding" in text and "Calculations" in text and "A.pdf, page 1" in text
assert sum("Error" in line for line in r["log"]) == 4, r["log"]
print("2 agent design OK:", len(r["log"]), "tool calls,", sum("Error" in l for l in r["log"]), "handled errors")

# 3) agent forgets add_answer -> the answer is still added
llm = ScriptedLLM([[("add_heading", {"text": "Only a heading", "level": 1}), ("finish_report", {})]])
r = generate_report(QUESTION, ANSWER, ANALYSIS, style="minimal", output_path=f"{OUT}/forgot.pdf", llm=llm)
_, text = pdf_text(r["path"])
assert "Only a heading" in text and "92%" in text
print("3 answer is always included OK")

# 4) agent crashes -> still a PDF
r = generate_report(QUESTION, ANSWER, ANALYSIS, style="blue", output_path=f"{OUT}/crash.pdf", llm=BrokenLLM())
_, text = pdf_text(r["path"])
assert r["agent_used"] is False and any("Agent failed" in l for l in r["log"]) and "92%" in text
print("4 agent failure fallback OK")

# 5) modern preset = cover page; 2 columns; extra sections (voice transcript, OCR)
extras = [{"heading": "Voice transcript", "text": QUESTION}, {"heading": "OCR text", "text": "Table 3: CNN 92%"}]
r = generate_report(QUESTION, ANSWER * 12, ANALYSIS, extras, preset="modern", output_path=f"{OUT}/modern.pdf")
doc, text = pdf_text(r["path"])
assert doc.page_count >= 2 and "Voice transcript" in text and "OCR text" in text
llm = ScriptedLLM([[("set_page", {"columns": 2}), ("add_answer", {"heading": "Answer"}), ("finish_report", {})]])
r = generate_report(QUESTION, ANSWER * 12, ANALYSIS, style="two columns", output_path=f"{OUT}/cols.pdf", llm=llm)
assert r["design"]["columns"] == 2
print("5 cover, extras, columns OK")

# 6) Arabic answer
ar_answer = "متوسط دقة النموذج هو **92%** بحسب الورقة الأولى (A.pdf, page 1) والورقة الثانية.\n\n- الشبكة العصبية الالتفافية تعطي أفضل نتيجة\n- النموذج الثاني أقل دقة قليلاً"
r = generate_report("ما هو متوسط الدقة؟", ar_answer, {}, output_path=f"{OUT}/arabic.pdf")
doc, text = pdf_text(r["path"])
assert len(text.strip()) > 40
print("6 Arabic report OK")

# 7) voice (fake speech-to-text and fake pipeline)
class FakeReply:
    answer, cached, analysis = "Answer [1]", False, {"enough_evidence": True}

voice_service.transcribe = lambda audio, language=None: {"text": QUESTION, "language": "en"}
voice_service._default_ask = lambda question, history: FakeReply()
result = voice_service.voice_ask(b"fake-audio", [{"role": "user", "content": "hi"}])
assert result["transcript"] == QUESTION and result["answer"] == "Answer [1]" and result["language"] == "en"
print("7 voice service OK")

# 8) FastAPI routers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.report import router as report_router
from routers.voice import router as voice_router

app = FastAPI()
app.include_router(voice_router)
app.include_router(report_router)
client = TestClient(app)

response = client.post("/voice/ask", files={"file": ("q.wav", b"123", "audio/wav")}, data={"language": "en"})
assert response.status_code == 200 and response.json()["transcript"] == QUESTION

def no_speech(audio, language=None):
    raise TranscriptionError("No speech detected.")
voice_service.transcribe = no_speech
assert client.post("/voice/transcribe", files={"file": ("q.wav", b"123", "audio/wav")}).status_code == 422

os.environ["REPORTS_DIR"] = OUT
response = client.post("/report", json={"question": QUESTION, "answer": ANSWER, "analysis": ANALYSIS,
                                        "extra_sections": [{"heading": "OCR text", "text": "hello"}]})
assert response.status_code == 200 and response.content.startswith(b"%PDF")
assert response.headers["x-agent-used"] == "false"
assert client.post("/report", json={"question": "q", "answer": " "}).status_code == 422
assert "Times" in client.get("/report/options").json()["fonts"]
print("8 FastAPI routers OK")
print("ALL OK. Sample PDFs are in", OUT)
