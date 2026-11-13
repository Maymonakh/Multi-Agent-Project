from __future__ import annotations

import hashlib
import os

from pydantic import BaseModel, Field

import retriever_bridge
from analyst.analyst_agent import run_analyst
from answer.answer_agent import run_answer
from cache.semantic_cache import SemanticCache
from embedding_adapter import embed
from shared.llm_config import get_llm


semantic_cache = SemanticCache(
    embed=embed,
    threshold=float(os.getenv("CACHE_THRESHOLD", "0.92")),
    redis_url=os.getenv("REDIS_URL"),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
)


class Reply(BaseModel):
    answer: str
    cached: bool = False
    analysis: dict = Field(default_factory=dict)


def _cache_inputs(question: str, chat_history: list | None):
    """Scope cache entries to both the current corpus and prior conversation."""
    context = retriever_bridge.build_conversation_context(
        chat_history,
        current_question=question,
    )

    cache_question = (
        f"Conversation context:\n{context}\n\nQuestion:\n{question}"
        if context
        else question
    )

    context_hash = hashlib.sha256(
        context.encode("utf-8")
    ).hexdigest()

    scope = f"{retriever_bridge.get_corpus_version()}:{context_hash}"
    return cache_question, scope


def ask(
    question: str,
    chat_history: list | None = None,
    llm=None,
    cache: SemanticCache | None = None,
) -> Reply:
    """Run Cache -> Retriever Agent -> Analyst -> Answer."""
    if not question or not question.strip():
        return Reply(answer="Please enter a question.")

    active_cache = cache if cache is not None else semantic_cache
    cache_question, cache_scope = _cache_inputs(question, chat_history)

    # 1. Check semantic cache before invoking any agents.
    try:
        hit = active_cache.get(cache_question, scope=cache_scope)
    except Exception as error:
        print("Cache read failed; continuing without cache:", error)
        hit = None

    if hit:
        return Reply(
            answer=hit["answer"],
            analysis=hit.get("analysis", {}),
            cached=True,
        )

    # 2. Resolve one model instance and share it across all agent stages.
    active_llm = llm if llm is not None else get_llm()

    bundle = retriever_bridge.retrieve(
        question,
        chat_history,
        llm=active_llm,
    )
    def retrieve_follow_up(follow_up_query: str):
        return retriever_bridge.retrieve(
            follow_up_query,
            chat_history,
            llm=active_llm,
        )

    analysis = run_analyst(
        question,
        bundle,
        retrieve_follow_up,
        llm=active_llm,
    )

    answer, answer_ok = run_answer(
        question,
        analysis,
        llm=active_llm,
    )

    reply = Reply(
        answer=answer,
        analysis=analysis.model_dump(exclude={"evidence"}),
    )

    # 4. Cache only successful answers supported by sufficient evidence.
    if answer_ok and analysis.enough_evidence and not analysis.error:
        try:
            active_cache.set(
                cache_question,
                {
                    "answer": reply.answer,
                    "analysis": reply.analysis,
                },
                scope=cache_scope,
            )
        except Exception as error:
            print("Cache write failed; answer is still available:", error)

    return reply


def clear_cache() -> None:
    """Clear semantic cache after documents are added or replaced."""
    try:
        semantic_cache.clear()
    except Exception as error:
        # Cache failure must never prevent document ingestion.
        print("Could not clear semantic cache:", error)
