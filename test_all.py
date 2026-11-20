"""Run: python test_all.py
No API key, no internet, no models: fake LLM, fake vector store and fake embeddings.
Requires: langgraph, pydantic, numpy, python-dotenv (all in requirements.txt).
"""
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from analyst.schemas import AnalystDecision, Metric
from analyst.tools.search_more import next_query
from analyst.tools.table_extractor import extract_tables
from answer.answer_agent import run_answer
from cache.semantic_cache import SemanticCache
from orchestrator import Pipeline, build_conversation_context
from retriever_agent.agent import build_retriever_graph, run_retriever
from retriever_agent.document_store import DocumentStore
from retriever_agent.tools import reranker as reranker_module
from retriever_agent.tools.context_selector import select_context
from retriever_agent.tools.keyword_search import BM25Index
from retriever_agent.tools.metadata_filter import detect_filters, matches, to_chroma_where
from retriever_agent.tools.query_rewriter import rewrite_query
from shared.schemas import Chunk, Evidence

reranker_module._models[reranker_module.RERANKER_MODEL] = None   # no model download in tests


# ----------------------------------------------------------------------------- fakes
def chunk(doc, page, text):
    return Chunk(page_content=text, metadata={"document": doc, "page": page})


def words(text):
    return set(text.lower().replace("?", "").replace("=", " ").replace("%", " ").split())


class FakeVectorStore:
    """Mimics Chroma: similarity_search_with_score(query, k, filter) + get/delete."""

    def __init__(self, chunks=()):
        self.items = list(chunks)

    def similarity_search_with_score(self, query, k=4, filter=None):
        q, scored = words(query), []
        for c in self.items:
            if filter and not _chroma_match(c.metadata, filter):
                continue
            w = words(c.page_content)
            scored.append((c, 1.0 - len(q & w) / max(1, len(q | w))))
        return sorted(scored, key=lambda x: x[1])[:k]

    def get(self, include=None):
        return {
            "ids": [f"{i}" for i in range(len(self.items))],
            "documents": [c.page_content for c in self.items],
            "metadatas": [c.metadata for c in self.items],
        }

    def delete(self, ids=None, where=None):
        if ids is not None:
            keep = {int(i) for i in ids}
            self.items = [c for i, c in enumerate(self.items) if i not in keep]
        if where:
            self.items = [c for c in self.items if not _chroma_match(c.metadata, where)]


def _chroma_match(meta, where):
    if "$and" in where:
        return all(_chroma_match(meta, w) for w in where["$and"])
    (key, cond), = where.items()
    return meta.get(key) in cond["$in"] if isinstance(cond, dict) else meta.get(key) == cond


class FakeStore:
    """Same interface the Retriever/Pipeline use from DocumentStore."""

    def __init__(self, chunks):
        self.chunks, self.version = list(chunks), "v1"
        self.vs, self.index = FakeVectorStore(chunks), BM25Index(chunks)

    def snapshot(self):
        names = tuple(sorted({c.metadata["document"] for c in self.chunks}))
        return SimpleNamespace(vector_store=self.vs, keyword_index=self.index, documents=names)


class FakeLLM:
    def __init__(self, decisions=(), answer="Average is 92 [1][2][3].", rewrite="CNN accuracy of paper A"):
        self.decisions, self.answer, self.rewrite = list(decisions), answer, rewrite
        self.calls = {"structured": 0, "invoke": 0}

    def with_structured_output(self, schema):
        outer = self

        class Structured:
            def invoke(self, prompt):
                outer.calls["structured"] += 1
                return outer.decisions.pop(0)
        return Structured()

    def invoke(self, prompt):
        self.calls["invoke"] += 1
        text = self.rewrite if prompt.startswith("Rewrite") else self.answer
        return SimpleNamespace(content=text)


class BrokenLLM:
    def with_structured_output(self, schema):
        raise RuntimeError("model down")

    def invoke(self, prompt):
        raise RuntimeError("model down")


CORPUS = [
    chunk("paper_A.pdf", 1, "The CNN model reaches acc = 92% on the benchmark"),
    chunk("paper_B.pdf", 2, "Our CNN baseline gets acc = 95% on the same task"),
    chunk("paper_C.pdf", 3, "The CNN network achieves acc = 89% with fewer parameters"),
    chunk("notes.txt", None, "RAG lets a model use external knowledge without retraining"),
]


def fake_embed(text):
    vocab = ["average", "cnn", "accuracy", "paper", "rag", "capital", "france", "compare", "a", "b"]
    w = text.lower().replace("?", "").split()
    return [float(w.count(v)) for v in vocab]


def make_pipeline(llm, store=None):
    cache = SemanticCache(fake_embed, threshold=0.9, redis_url=None)
    return Pipeline(store or FakeStore(CORPUS), cache, llm=llm), cache


# ------------------------------------------------------------------------------ tests
def test_metadata_filter():
    docs = ["paper_A.pdf", "paper_B.pdf", "notes.txt"]
    f = detect_filters("According to paper A, page 3, what is the accuracy?", docs)
    assert f == {"documents": ["paper_A.pdf"], "page": 3}
    f = detect_filters("compare paper_A.pdf and paper B", docs)
    assert f["documents"] == ["paper_A.pdf", "paper_B.pdf"] and f["page"] is None
    assert detect_filters("what is RAG?", docs) == {"documents": None, "page": None}
    assert to_chroma_where({"documents": ["x.pdf"], "page": 2}) == {"$and": [{"document": "x.pdf"}, {"page": 2}]}
    assert to_chroma_where({"documents": ["a", "b"], "page": None}) == {"document": {"$in": ["a", "b"]}}
    assert to_chroma_where({"documents": None, "page": None}) is None
    assert matches({"document": "a", "page": 1}, {"documents": ["a"], "page": 1})
    assert not matches({"document": "b", "page": 1}, {"documents": ["a"], "page": None})
    print("1 metadata filter OK")


def test_bm25():
    index = BM25Index(CORPUS)
    hits = index.search("CNN accuracy", k=5)
    assert {h["document"] for h in hits} == {"paper_A.pdf", "paper_B.pdf", "paper_C.pdf"}
    assert index.search("zzzz unknown", k=5) == []
    only_b = index.search("CNN", k=5, filters={"documents": ["paper_B.pdf"], "page": None})
    assert [h["document"] for h in only_b] == ["paper_B.pdf"]
    wrong_page = index.search("CNN", k=5, filters={"documents": ["paper_B.pdf"], "page": 99})
    assert len(wrong_page) == 3        # empty filter result falls back instead of hiding everything
    assert index.search("rag retraining", k=1)[0]["page"] is None
    assert BM25Index([]).search("anything") == []
    print("2 keyword search OK")


def test_reranker_and_selector():
    cands = [
        {"document": "A", "page": 1, "content": "x y", "score": 0.9, "source": "semantic"},
        {"document": "B", "page": 1, "content": "z", "score": 0.5, "source": "semantic"},
        {"document": "B", "page": 1, "content": "z", "score": 7.0, "source": "keyword"},   # duplicate
        {"document": "C", "page": 1, "content": "  ", "score": 1.0, "source": "keyword"},  # empty
    ]
    ranked = reranker_module.rerank("q", cands)
    assert [r["document"] for r in ranked] == ["B", "A"]        # B found by both sources wins (RRF)
    assert all(0 <= r["score"] <= 1 for r in ranked)

    class FakeCE:
        def predict(self, pairs):
            return [-4.0 if "x" in p[1] else 4.0 for p in pairs]
    reranker_module._models["fake"] = FakeCE()
    assert reranker_module.rerank("q", cands, model_name="fake")[0]["document"] == "B"

    many = [{"document": "D", "page": 1, "score": 1 - i / 10, "content": f"c{i}"} for i in range(5)]
    assert len(select_context(many, max_per_page=2)) == 2
    assert len(select_context(many, top_k=1, max_per_page=9)) == 1
    print("3 reranker + context selector OK")


def test_retriever_graph():
    graph = build_retriever_graph(FakeStore(CORPUS))
    out = run_retriever(graph, "CNN accuracy of paper B")
    assert out and out[0].document == "paper_B.pdf" and {e.document for e in out} == {"paper_B.pdf"}
    out = run_retriever(graph, "What is RAG?")
    assert out[0].document == "notes.txt" and out[0].page is None
    assert run_retriever(build_retriever_graph(FakeStore([])), "anything") == []
    print("4 retriever graph OK")


def test_feedback_loop_and_cache():
    llm = FakeLLM([
        AnalystDecision(enough_evidence=False, missing_query="CNN accuracy paper B and paper C"),
        AnalystDecision(
            enough_evidence=True, reasoning="3 papers", similarities=["All test a CNN"],
            differences=["B is higher than A"],
            metrics=[Metric(name="CNN acc", document="paper_A.pdf", value=92, page=1),
                     Metric(name="CNN acc", document="paper_B.pdf", value=95),       # page missing on purpose
                     Metric(name="CNN acc", document="paper_C.pdf", value=89, page=3)]),
    ])
    pipeline, cache = make_pipeline(llm)
    steps = []
    reply = pipeline.ask("Compare CNN accuracy of paper A", on_step=lambda n, u: steps.append(n))

    # graph went through the feedback loop exactly once
    assert steps.count("retrieve") == 2 and steps.count("followup") == 1 and steps.count("analyst") == 2
    assert steps[:2] == ["rewrite", "cache_lookup"] and steps[-1] == "cache_store"
    calc = reply.analysis["calculations"]["CNN acc"]
    assert calc["average"] == 92.0 and calc["difference"] == 6.0
    assert calc["best"] == {"document": "paper_B.pdf", "page": 2, "value": 95.0}
    assert reply.analysis["comparison"]["similarities"] == ["All test a CNN"]
    assert "(paper_A.pdf, page 1)" in reply.answer and "**Sources**" in reply.answer
    assert reply.cached is False
    assert llm.calls["invoke"] == 1          # no chat history -> the rewriter made NO llm call

    pipeline.llm = BrokenLLM()               # cache hit must not need any LLM
    again = pipeline.ask("Compare CNN accuracy of paper A")
    assert again.cached is True and again.answer == reply.answer

    pipeline.store.version = "v2"            # documents changed -> cached answer is not reused
    assert pipeline.ask("Compare CNN accuracy of paper A").cached is False
    print("5 feedback loop + cache OK")


def test_stream_events():
    pipeline, _ = make_pipeline(FakeLLM([AnalystDecision(enough_evidence=True, reasoning="ok",
                                         metrics=[Metric(name="acc", document="paper_A.pdf", value=92, page=1)])],
                                        answer="CNN is 92 [1]."))
    events = list(pipeline.stream("CNN accuracy of paper A"))
    kinds = [e[0] for e in events]
    assert kinds[:-1] == ["step"] * (len(events) - 1) and kinds[-1] == "reply"
    nodes = [e[1] for e in events if e[0] == "step"]
    assert nodes == ["rewrite", "cache_lookup", "retrieve", "collect", "analyst", "finalize", "answer", "cache_store"]
    assert "(paper_A.pdf, page 1)" in events[-1][1].answer
    assert list(pipeline.stream("  "))[0][1].answer == "Please enter a question."
    print("6b stream events OK")


def test_loop_limits():
    # same missing query twice -> second loop must not repeat the retrieval
    decision = lambda: AnalystDecision(enough_evidence=False, missing_query="CNN paper C")
    llm = FakeLLM([decision(), decision(), decision(), decision()])
    pipeline, _ = make_pipeline(llm)
    steps = []
    reply = pipeline.ask("CNN accuracy of paper A", on_step=lambda n, u: steps.append(n))
    assert steps.count("followup") == 1 and steps.count("analyst") == 2
    assert reply.analysis["enough_evidence"] is False

    # max_loops respected
    llm = FakeLLM([AnalystDecision(enough_evidence=False, missing_query=f"CNN q{i} paper_{c}")
                   for i, c in enumerate("BCABC")])
    pipeline, _ = make_pipeline(llm)
    pipeline.max_loops = 1
    steps = []
    pipeline.ask("CNN accuracy of paper A", on_step=lambda n, u: steps.append(n))
    assert steps.count("followup") == 1
    print("6 loop limits OK")


def test_failures_and_empty_corpus():
    pipeline, cache = make_pipeline(BrokenLLM())
    reply = pipeline.ask("Compare CNN accuracy of paper A")
    assert "Sorry" in reply.answer and reply.analysis["error"]
    assert cache.get("Compare CNN accuracy of paper A") is None      # failures are never cached

    llm = FakeLLM()
    pipeline, _ = make_pipeline(llm, store=FakeStore([]))
    reply = pipeline.ask("What is RAG?")
    assert "could not find" in reply.answer and llm.calls == {"structured": 0, "invoke": 0}
    arabic = pipeline.ask("ما هو RAG؟")
    assert "لم أجد" in arabic.answer
    assert pipeline.ask("   ").answer == "Please enter a question."
    print("7 failures + empty corpus OK")


def test_rewriter_and_history():
    llm = FakeLLM(rewrite="What are the disadvantages of RAG?")
    history = [{"role": "user", "content": "What is RAG?"},
               {"role": "assistant", "content": "RAG is retrieval augmented generation."},
               {"role": "user", "content": "What are its disadvantages?"}]
    context = build_conversation_context(history, "What are its disadvantages?")
    assert "What are its disadvantages?" not in context and "What is RAG?" in context
    assert rewrite_query("What are its disadvantages?", context, llm=llm) == "What are the disadvantages of RAG?"
    assert rewrite_query("What is RAG?", "", llm=BrokenLLM()) == "What is RAG?"          # no context -> no call
    assert rewrite_query("its cons?", "User: x", llm=BrokenLLM()) == "its cons?"          # failure -> original

    pipeline, _ = make_pipeline(FakeLLM([AnalystDecision(enough_evidence=True, reasoning="ok")],
                                        rewrite="CNN accuracy of paper A"))
    pipeline.ask("and its accuracy?", [{"role": "user", "content": "Tell me about paper A"},
                                       {"role": "assistant", "content": "It is about a CNN."}])
    assert pipeline.llm.calls["invoke"] == 2     # rewrite + answer
    print("8 rewriter + history OK")


def test_cache_ttl_and_scope():
    cache = SemanticCache(fake_embed, threshold=0.9, ttl_seconds=1)
    cache.set("average cnn accuracy", {"answer": "x"}, scope="s1")
    assert cache.get("average cnn accuracy", scope="s1") == {"answer": "x"}
    assert cache.get("average cnn accuracy", scope="s2") is None
    cache.memory[0]["ts"] -= 5
    assert cache.get("average cnn accuracy", scope="s1") is None      # expired per item
    cache.set("average cnn accuracy", {"answer": "y"}, scope="s1")
    cache.clear()
    assert cache.get("average cnn accuracy", scope="s1") is None
    print("9 cache ttl/scope OK")


def test_document_store():
    backend = FakeVectorStore()
    calls = {"prepare_files": 0, "prepare_file": 0}

    def prepare_file(path):
        calls["prepare_file"] += 1
        text = Path(path).read_text()
        return [chunk(Path(path).name, 1, text)] if text.strip() else []

    def prepare_files(paths):
        calls["prepare_files"] += 1
        return [c for p in paths for c in prepare_file(p)]

    def create(chunks, persist_directory=None, replace_documents=True):
        for name in {c.metadata["document"] for c in chunks}:
            backend.delete(where={"document": name})
        backend.items.extend(chunks)
        return backend

    with tempfile.TemporaryDirectory() as tmp:
        docs, db = Path(tmp) / "documents", Path(tmp) / "db"
        docs.mkdir()
        (docs / "a.txt").write_text("alpha text about cnn")

        def new_store():
            return DocumentStore(docs, db, prepare_file=prepare_file, prepare_files=prepare_files,
                                 get_vector_store=lambda: backend, create_vector_store=create)

        s1 = new_store()
        assert s1.document_names() == ["a.txt"] and calls["prepare_files"] == 1
        v1 = s1.version

        s2 = new_store()                                  # "restart": nothing changed
        assert s2.document_names() == ["a.txt"] and calls["prepare_files"] == 1   # no re-parsing
        assert s2.snapshot().keyword_index.search("alpha")[0]["document"] == "a.txt"
        assert s2.version == v1

        incoming = s2.incoming_path("b.txt")              # upload flow
        incoming.write_text("beta text about rnn")
        assert "Successfully" in s2.ingest(incoming)
        assert (docs / "b.txt").exists() and not incoming.exists()
        assert s2.document_names() == ["a.txt", "b.txt"] and s2.version != v1

        incoming = s2.incoming_path("b.txt")              # replace existing file
        incoming.write_text("gamma replaced")
        s2.ingest(incoming)
        assert [h["content"] for h in s2.snapshot().keyword_index.search("gamma")] == ["gamma replaced"]
        assert s2.snapshot().keyword_index.search("beta") == []

        bad = s2.incoming_path("empty.txt")               # unreadable upload is rejected, nothing indexed
        bad.write_text("   ")
        try:
            s2.ingest(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        assert "empty.txt" not in s2.document_names() and not (docs / "empty.txt").exists()

        s2.delete_document("b.txt")                       # delete: file, vectors and keyword index
        assert s2.document_names() == ["a.txt"] and not (docs / "b.txt").exists()
        assert s2.snapshot().keyword_index.search("gamma") == []
        for bad_name, error in [("nope.txt", KeyError), ("../a.txt", ValueError)]:
            try:
                s2.delete_document(bad_name)
                raise AssertionError(bad_name)
            except error:
                pass
        incoming = s2.incoming_path("b.txt")
        incoming.write_text("gamma replaced")
        s2.ingest(incoming)                               # put it back for the restart check below

        s3 = new_store()                                  # restart after ingest: still no re-parse
        before = calls["prepare_files"]
        assert s3.document_names() == ["a.txt", "b.txt"] and calls["prepare_files"] == before

        (docs / "a.txt").write_text("changed outside the app")   # file changed -> automatic re-index
        s4 = new_store()
        assert s4.snapshot().keyword_index.search("outside")[0]["document"] == "a.txt"
        assert calls["prepare_files"] == before + 1
    print("10 document store OK")


def test_robust_analyst_parsing():
    # non-numeric "metrics" (complexity) are dropped instead of failing the whole analysis
    d = AnalystDecision.model_validate({
        "enough_evidence": True,
        "metrics": [
            {"name": "acc", "document": "m.pdf", "value": "92%"},
            {"name": "complexity", "document": "m.pdf", "value": "O(n^2)"},
            {"name": "f1", "document": "m.pdf", "value": 0.91},
        ]})
    assert [(m.name, m.value) for m in d.metrics] == [("acc", 92.0), ("f1", 0.91)]

    # JSON fallback: trailing text / fences must not break parsing
    from analyst.analyst_agent import decide

    class JsonOnlyLLM:
        def with_structured_output(self, schema):
            raise RuntimeError("no structured output")

        def invoke(self, prompt):
            return SimpleNamespace(content='```json\n{"enough_evidence": true, "reasoning": "ok", '
                                           '"metrics": [{"name": "c", "document": "m.pdf", "value": "O(n)"}]}\n```\nHope it helps {x}')
    out = decide(JsonOnlyLLM(), "q", [Evidence(document="m.pdf", page=1, score=1, content="t")])
    assert out.enough_evidence is True and out.metrics == [] and not out.failed
    print("11 robust analyst parsing OK")


def test_analyst_tools():
    assert next_query("CNN complexity", ["CNN accuracy"]) == "CNN complexity"
    assert next_query("cnn  accuracy", ["CNN accuracy"]) is None and next_query("", []) is None

    ev = lambda t: Evidence(document="p.pdf", page=1, score=0.9, content=t)
    t = extract_tables([ev("| Model | Acc |\n|---|---|\n| CNN | 92 |\n| RNN | 88 |")])
    assert t[0]["rows"] == [["Model", "Acc"], ["CNN", "92"], ["RNN", "88"]]
    t = extract_tables([ev("Results below\nModel Acc F1\nCNN 92.1 90.3\nRNN 88.0 85.5\nEnd.")])
    assert t[0]["rows"][0] == ["Model", "Acc", "F1"]
    assert extract_tables([ev("This is normal text.\nNothing here.")]) == []

    text, ok = run_answer("q", SimpleNamespace(error="x", evidence=None), llm=BrokenLLM())
    assert ok is False
    print("12 analyst/answer tools OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL OK")
