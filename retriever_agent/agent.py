"""Retriever Agent as a LangGraph sub-graph.

    metadata_filter ─┬─> semantic_search ─┐
                     └─> keyword_search ──┴─> reranker ─> context_selector

The order of the tools is fixed, so it is a graph and not an LLM deciding (and
re-typing large JSON between tools). The two searches run in parallel. The Retriever
never writes an answer: it only returns evidence.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from retriever_agent.tools.context_selector import select_context
from retriever_agent.tools.metadata_filter import detect_filters
from retriever_agent.tools.reranker import rerank
from retriever_agent.tools.semantic_search import semantic_search
from shared.schemas import Evidence


class RetrieverState(TypedDict, total=False):
    query: str
    filters: dict
    candidates: Annotated[list, operator.add] 
    ranked: list
    evidences: list


def build_retriever_graph(store, *, search_k: int = 8, rerank_k: int = 20, top_k: int = 6):
    """`store` needs `.snapshot()` -> (vector_store, keyword_index, documents)."""

    def metadata_filter(state: RetrieverState):
        snap = store.snapshot()
        return {"filters": detect_filters(state["query"], snap.documents)}

    def semantic(state: RetrieverState):
        snap = store.snapshot()
        if not snap.documents:
            return {"candidates": []}
        hits = semantic_search(snap.vector_store, state["query"], k=search_k, filters=state.get("filters"))
        return {"candidates": [{**hit, "source": "semantic"} for hit in hits]}

    def keyword(state: RetrieverState):
        snap = store.snapshot()
        if not snap.documents:
            return {"candidates": []}
        hits = snap.keyword_index.search(state["query"], k=search_k, filters=state.get("filters"))
        return {"candidates": [{**hit, "source": "keyword"} for hit in hits]}

    def reranker(state: RetrieverState):
        return {"ranked": rerank(state["query"], state.get("candidates", []), top_k=rerank_k)}

    def context_selector(state: RetrieverState):
        chosen = select_context(state.get("ranked", []), top_k=top_k)
        return {"evidences": [Evidence(**item) for item in chosen]}

    graph = StateGraph(RetrieverState)
    graph.add_node("metadata_filter", metadata_filter)
    graph.add_node("semantic_search", semantic)
    graph.add_node("keyword_search", keyword)
    graph.add_node("reranker", reranker)
    graph.add_node("context_selector", context_selector)

    graph.add_edge(START, "metadata_filter")
    graph.add_edge("metadata_filter", "semantic_search")
    graph.add_edge("metadata_filter", "keyword_search")
    graph.add_edge(["semantic_search", "keyword_search"], "reranker")
    graph.add_edge("reranker", "context_selector")
    graph.add_edge("context_selector", END)
    return graph.compile()


def run_retriever(retriever_graph, query: str) -> list[Evidence]:
    result = retriever_graph.invoke({"query": query, "candidates": []})
    return result.get("evidences", [])
