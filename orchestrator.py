import os
from pydantic import BaseModel
import retriever_bridge
from analyst.analyst_agent import run_analyst
from answer.answer_agent import run_answer
from cache.semantic_cache import SemanticCache
from embedding_adapter import embed

cache = SemanticCache(embed,
                      threshold=float(os.getenv("CACHE_THRESHOLD", "0.92")),
                      redis_url=os.getenv("REDIS_URL"))


class Reply(BaseModel):
    answer: str
    cached: bool = False
    analysis: dict = {}


def ask(question: str, chat_history: list | None = None, llm=None, cache=cache) -> Reply:
    # 1) cache
    try:
        hit = cache.get(question)
    except Exception as error:
        print("Cache read failed:", error)
        hit = None
    if hit:
        return Reply(answer=hit["answer"], analysis=hit["analysis"], cached=True)

    # 2) Retriever -> Analyst (with feedback loop) -> Answer
    bundle = retriever_bridge.retrieve(question, chat_history)
    analysis = run_analyst(question, bundle, retriever_bridge.retrieve, llm=llm)
    answer, ok = run_answer(question, analysis, llm=llm)
    reply = Reply(answer=answer, analysis=analysis.model_dump(exclude={"evidence"}))

    # 3) save only good answers
    if ok and analysis.enough_evidence:
        try:
            cache.set(question, {"answer": reply.answer, "analysis": reply.analysis})
        except Exception as error:
            print("Cache write failed:", error)
    return reply


def clear_cache() -> None:
    cache.clear()
