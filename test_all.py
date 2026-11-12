"""Run: python test_all.py   (no API key, no internet: uses fake LLM and fake embeddings)"""
from shared.schemas import Evidence, EvidenceBundle
from analyst.schemas import AnalystDecision, Metric
from analyst.analyst_agent import run_analyst
from analyst.tools.table_extractor import extract_tables
from answer.answer_agent import run_answer
from cache.semantic_cache import SemanticCache
import orchestrator
import retriever_bridge


def ev(doc, page, text):
    return Evidence(document=doc, page=page, score=0.9, content=text)


class FakeLLM:
    """with_structured_output -> returns the next decision. invoke -> returns a fixed answer."""
    def __init__(self, decisions=(), answer="CNN average is 92 [1][2][3]."):
        self.decisions, self.answer = list(decisions), answer
    def with_structured_output(self, schema):
        return self
    def invoke(self, prompt):
        if self.decisions:
            return self.decisions.pop(0)
        class R: pass
        r = R(); r.content = self.answer
        return r


class BrokenLLM:
    def with_structured_output(self, schema):
        raise RuntimeError("model down")
    def invoke(self, prompt):
        raise RuntimeError("model down")


def fake_retrieve(query):
    return EvidenceBundle(evidences=[ev("C.pdf", 3, "CNN acc = 89%")])


start = EvidenceBundle(evidences=[ev("A.pdf", 1, "CNN acc = 92%"), ev("B.pdf", 2, "CNN acc = 95%")])

# 1) feedback loop + calculations + page numbers + similarities/differences (fixes 1, 2, 4)
llm = FakeLLM([
    AnalystDecision(enough_evidence=False, missing_query="CNN accuracy paper C"),
    AnalystDecision(enough_evidence=True, reasoning="3 papers found",
                    similarities=["All test a CNN"], differences=["B is higher than A"],
                    metrics=[Metric(name="CNN acc", document="A.pdf", value=92, page=1),
                             Metric(name="CNN acc", document="B.pdf", value=95),      # page missing on purpose
                             Metric(name="CNN acc", document="C.pdf", value=89, page=3)]),
])
result = run_analyst("Average CNN accuracy?", start, fake_retrieve, llm=llm)
calc = result.calculations["CNN acc"]
assert calc["average"] == 92.0 and calc["difference"] == 6.0
assert calc["best"] == {"document": "B.pdf", "page": 2, "value": 95.0}   # page was filled in
assert result.comparison["similarities"] == ["All test a CNN"]
assert result.comparison["compared_metrics"] == ["CNN acc"]
assert len(result.evidence.evidences) == 3
print("1 analyst OK")

# 2) tables in 3 styles (fix 3)
t = extract_tables([ev("p.pdf", 1, "| Model | Acc |\n|---|---|\n| CNN | 92 |\n| RNN | 88 |")])
assert t[0]["rows"] == [["Model", "Acc"], ["CNN", "92"], ["RNN", "88"]]
t = extract_tables([ev("p.pdf", 1, "Model    Acc    F1\nCNN    92    90\nRNN    88    85")])
assert len(t[0]["rows"]) == 3
t = extract_tables([ev("p.pdf", 1, "Results below\nModel Acc F1\nCNN 92.1 90.3\nRNN 88.0 85.5\nEnd.")])
assert t[0]["rows"][0] == ["Model", "Acc", "F1"] and t[0]["rows"][1] == ["CNN", "92.1", "90.3"]
assert extract_tables([ev("p.pdf", 1, "This is normal text.\nNothing here.")]) == []
print("2 tables OK")

# 3) LLM failure -> NOT enough (fix 5)
bad = run_analyst("q", start, fake_retrieve, llm=BrokenLLM())
assert bad.enough_evidence is False and bad.error
text, ok = run_answer("q", bad, llm=BrokenLLM())
assert ok is False
print("3 failure OK")

# 4) answer agent: citations + sources
text, ok = run_answer("Average CNN accuracy?", result, llm=FakeLLM())
assert ok and "(A.pdf, page 1)" in text and "**Sources**" in text and "- C.pdf, page 3" in text
print("4 answer OK")

# 5) semantic cache + orchestrator
def fake_embed(text):
    words = text.lower().replace("?", "").split()
    vocab = ["average", "cnn", "accuracy", "capital", "france", "what", "is", "the"]
    return [float(words.count(w)) for w in vocab]

cache = SemanticCache(fake_embed, threshold=0.7, redis_url="redis://nowhere:6379/0")  # no Redis -> memory
assert cache.redis is None
retriever_bridge.retrieve = lambda q, h=None: start
llm = FakeLLM([AnalystDecision(enough_evidence=True, reasoning="ok",
                               metrics=[Metric(name="acc", document="A.pdf", value=92, page=1)])],
              answer="CNN is 92 [1].")
first = orchestrator.ask("What is the average CNN accuracy?", llm=llm, cache=cache)
assert first.cached is False
second = orchestrator.ask("average CNN accuracy?", llm=BrokenLLM(), cache=cache)   # no LLM needed
assert second.cached is True and second.answer == first.answer
third = orchestrator.ask("capital of France?", llm=BrokenLLM(), cache=cache)
assert third.cached is False and "Sorry" in third.answer
assert cache.get("capital of France?") is None      # failed answers are not cached
cache.clear()
assert cache.get("average CNN accuracy?") is None
print("5 cache + orchestrator OK")
print("ALL OK")
