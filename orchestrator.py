"""Orchestrator: the whole multi-agent workflow as ONE LangGraph StateGraph.

    START -> rewrite -> cache_lookup -+-> (hit) -> END
                                      |
                                      +-> retrieve -> collect -> analyst -+-> finalize -> answer -> cache_store -> END
                                             ^                            |
                                             +------ followup <-----------+   (feedback loop: not enough evidence)

* rewrite     Query Rewriter (Retriever tool). Runs first because the cache is keyed
              by the standalone question. No LLM call when there is no chat history.
* retrieve    Retriever Agent (sub-graph: metadata filter -> semantic || keyword -> reranker -> selector)
* analyst     Analyst Agent decides if the evidence is enough; if not, `followup` sends a
              more specific query back to the Retriever (bounded by max_loops).
* finalize    Analyst tools in Python: table extractor, document comparison, calculator/data analysis
* answer      Answer Agent: citations + sources + formatting
"""
from __future__ import annotations

import os
import threading
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from analyst.analyst_agent import MAX_LOOPS, decide, finalize_analysis, merge
from analyst.schemas import AnalysisResult, AnalystDecision
from analyst.tools.search_more import next_query
from answer.answer_agent import run_answer
from cache.semantic_cache import SemanticCache
from retriever_agent.agent import build_retriever_graph, run_retriever
from retriever_agent.tools.query_rewriter import rewrite_query
from shared.llm_config import get_llm


class Reply(BaseModel):
    answer: str
    cached: bool = False
    analysis: dict = Field(default_factory=dict)


class PipelineState(TypedDict, total=False):
    question: str
    history: list
    query: str                    # standalone question (after rewriting)
    query_vector: list
    search_query: str             # what the Retriever is currently asked
    retrieved: list               # evidence returned by the last retrieval
    evidences: list               # all evidence collected so far (merged, de-duplicated)
    new_count: int                # how many NEW chunks the last retrieval added
    tried_queries: list
    loops: int
    follow_up: str | None
    decision: AnalystDecision
    analysis: AnalysisResult
    answer: str
    answer_ok: bool
    cache_saved: bool
    cached: bool
    analysis_data: dict


def build_conversation_context(
    chat_history: list | None,
    current_question: str = "",
    *,
    max_messages: int = 8,
    max_chars: int = 4000,
) -> str:
    """Format the previous turns (the current question is excluded if it is already in the list)."""
    messages = []
    for message in chat_history or []:
        if isinstance(message, dict):
            role, content = message.get("role", "user"), message.get("content", "")
        else:
            role = getattr(message, "type", getattr(message, "role", "user"))
            content = getattr(message, "content", "")
        role = "assistant" if str(role).casefold() in {"ai", "assistant"} else "user"
        content = str(content).strip()
        if content:
            messages.append((role, content))

    if messages and messages[-1] == ("user", (current_question or "").strip()):
        messages.pop()

    lines = [f"{'User' if r == 'user' else 'Assistant'}: {c}" for r, c in messages[-max_messages:]]
    return "\n".join(lines)[-max_chars:]


class Pipeline:
    def __init__(
        self,
        store,
        cache: SemanticCache,
        llm=None,
        max_loops: int = MAX_LOOPS,
    ):
        self.store = store
        self.cache = cache
        self.llm = llm                 
        self.max_loops = max_loops
        self.retriever = build_retriever_graph(store)
        self.graph = self._build()

    def _llm(self):
        return self.llm if self.llm is not None else get_llm()

    def _rewrite(self, state: PipelineState):
        question = state["question"].strip()
        context = build_conversation_context(state.get("history"), question)
        query = rewrite_query(question, context, llm=self.llm) if context else question
        return {"query": query, "search_query": query, "tried_queries": [query], "loops": 0}

    def _cache_lookup(self, state: PipelineState):
        query = state["query"]
        try:
            vector = self.cache.embed(query)
            hit = self.cache.get(query, scope=self.store.version, vector=vector)
        except Exception as error:
            print("Cache read failed; continuing without cache:", error)
            return {"cached": False}

        if hit:
            return {"cached": True, "answer": hit["answer"], "analysis_data": hit.get("analysis", {})}
        return {"cached": False, "query_vector": vector}

    def _retrieve(self, state: PipelineState):
        return {"retrieved": run_retriever(self.retriever, state["search_query"])}

    def _collect(self, state: PipelineState):
        old = state.get("evidences", [])
        merged = merge(old, state.get("retrieved", []))
        return {"evidences": merged, "new_count": len(merged) - len(old)}

    def _analyst(self, state: PipelineState):
        decision = decide(self._llm(), state["question"], state["evidences"])
        follow_up = None
        if not decision.failed and not decision.enough_evidence and state["loops"] < self.max_loops:
            follow_up = next_query(decision.missing_query, state["tried_queries"])
        return {"decision": decision, "follow_up": follow_up}

    def _followup(self, state: PipelineState):
        query = state["follow_up"]
        return {
            "search_query": query,
            "tried_queries": state["tried_queries"] + [query],
            "loops": state["loops"] + 1,
        }

    def _finalize(self, state: PipelineState):
        decision = state.get("decision") or AnalystDecision(
            enough_evidence=False, reasoning="No relevant evidence was found in the documents."
        )
        analysis = finalize_analysis(decision, state.get("evidences", []))
        return {"analysis": analysis, "analysis_data": analysis.model_dump(exclude={"evidence"})}

    def _answer(self, state: PipelineState):
        answer, ok = run_answer(state["question"], state["analysis"], llm=self.llm)
        return {"answer": answer, "answer_ok": ok}

    def _cache_store(self, state: PipelineState):
        analysis: AnalysisResult = state["analysis"]
        if state.get("answer_ok") and analysis.enough_evidence and not analysis.error:
            try:
                self.cache.set(
                    state["query"],
                    {"answer": state["answer"], "analysis": state["analysis_data"]},
                    scope=self.store.version,
                    vector=state.get("query_vector"),
                )
            except Exception as error:
                print("Cache write failed; the answer is still returned:", error)
                return {"cache_saved": False}
            return {"cache_saved": True}
        return {"cache_saved": False}

    @staticmethod
    def _after_cache(state: PipelineState):
        return "hit" if state.get("cached") else "miss"

    @staticmethod
    def _after_collect(state: PipelineState):
        if not state.get("evidences"):
            return "finalize"                       
        if state["loops"] > 0 and state.get("new_count", 0) == 0:
            return "finalize"                   
        return "analyst"

    @staticmethod
    def _after_analyst(state: PipelineState):
        return "followup" if state.get("follow_up") else "finalize"

    def _build(self):
        graph = StateGraph(PipelineState)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("cache_lookup", self._cache_lookup)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("collect", self._collect)
        graph.add_node("analyst", self._analyst)
        graph.add_node("followup", self._followup)
        graph.add_node("finalize", self._finalize)
        graph.add_node("answer", self._answer)
        graph.add_node("cache_store", self._cache_store)

        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "cache_lookup")
        graph.add_conditional_edges("cache_lookup", self._after_cache, {"hit": END, "miss": "retrieve"})
        graph.add_edge("retrieve", "collect")
        graph.add_conditional_edges("collect", self._after_collect, {"analyst": "analyst", "finalize": "finalize"})
        graph.add_conditional_edges("analyst", self._after_analyst, {"followup": "followup", "finalize": "finalize"})
        graph.add_edge("followup", "retrieve")
        graph.add_edge("finalize", "answer")
        graph.add_edge("answer", "cache_store")
        graph.add_edge("cache_store", END)
        return graph.compile()

    def stream(self, question: str, chat_history: list | None = None):
        """Generator. Yields ("step", node_name, update) after every node, then ("reply", Reply)."""
        if not question or not question.strip():
            yield ("reply", Reply(answer="Please enter a question."))
            return

        final: dict = {}
        for chunk in self.graph.stream(
            {"question": question, "history": chat_history or []},
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                update = update or {}
                final.update(update)
                yield ("step", node, update)

        yield ("reply", Reply(
            answer=final["answer"],
            cached=bool(final.get("cached")),
            analysis=final.get("analysis_data", {}),
        ))

    def ask(
        self,
        question: str,
        chat_history: list | None = None,
        on_step: Callable[[str, dict], None] | None = None,
    ) -> Reply:
        """Run the graph. `on_step(node_name, update)` is called after every node (progress UI)."""
        reply = None
        for event in self.stream(question, chat_history):
            if event[0] == "step":
                if on_step:
                    on_step(event[1], event[2])
            else:
                reply = event[1]
        return reply


_pipeline: Pipeline | None = None
_lock = threading.Lock()


def get_pipeline() -> Pipeline:
    """One shared pipeline per process (the embedding/reranker models load once)."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            from embedding_adapter import embed
            from retriever_agent.document_store import DocumentStore

            _pipeline = Pipeline(
                store=DocumentStore(),
                cache=SemanticCache(
                    embed=embed,
                    threshold=float(os.getenv("CACHE_THRESHOLD", "0.95")),
                    redis_url=os.getenv("REDIS_URL"),
                    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
                ),
            )
        return _pipeline


def ask(question: str, chat_history: list | None = None, on_step=None) -> Reply:
    return get_pipeline().ask(question, chat_history, on_step=on_step)


def clear_cache() -> None:
    """Clear the semantic cache after documents were added or replaced."""
    try:
        get_pipeline().cache.clear()
    except Exception as error:
        print("Could not clear semantic cache:", error) 
