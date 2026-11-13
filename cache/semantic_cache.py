"""Semantic answer cache backed by Redis, with a safe in-memory fallback."""
from __future__ import annotations

import json
import math


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class SemanticCache:
    """Cache similar questions within the same document and conversation scope."""

    KEY = "multi_agent:semantic_cache:v1"

    def __init__(
        self,
        embed,
        threshold: float = 0.92,
        redis_url: str | None = None,
        max_items: int = 200,
        ttl_seconds: int = 3600,
        key_prefix: str | None = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        self.embed = embed
        self.threshold = threshold
        self.max_items = max(1, int(max_items))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.key = key_prefix or self.KEY
        self.memory: list[dict] = []
        self.redis = None

        if redis_url:
            try:
                import redis

                client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    health_check_interval=30,
                )
                client.ping()
                self.redis = client
                print("Semantic cache connected to Redis.")
            except Exception as error:
                print("Redis unavailable; using in-memory cache:", error)
                self.redis = None

    def _items(self) -> list[dict]:
        if self.redis is not None:
            try:
                raw_items = self.redis.lrange(self.key, 0, -1)
                items = []
                for raw in raw_items:
                    try:
                        items.append(json.loads(raw))
                    except (TypeError, json.JSONDecodeError):
                        continue
                return items
            except Exception as error:
                print("Redis cache read failed; switching to memory:", error)
                self.redis = None

        return list(self.memory)

    def get(self, question: str, scope: str = "default"):
        if not question or not question.strip():
            return None

        vector = self.embed(question)
        best_item = None
        best_score = 0.0

        for item in self._items():
            if item.get("scope", "default") != scope:
                continue

            score = cosine(vector, item.get("vector", []))
            if score > best_score:
                best_item = item
                best_score = score

        if best_item is not None and best_score >= self.threshold:
            return best_item.get("value")

        return None

    def set(
        self,
        question: str,
        value: dict,
        scope: str = "default",
    ) -> None:
        if not question or not question.strip():
            return

        item = {
            "question": question,
            "vector": self.embed(question),
            "value": value,
            "scope": scope,
        }

        # Keep a process-local fallback copy even when Redis is active.
        self.memory.append(item)
        del self.memory[:-self.max_items]

        if self.redis is not None:
            try:
                self.redis.rpush(self.key, json.dumps(item, ensure_ascii=False))
                self.redis.ltrim(self.key, -self.max_items, -1)
                if self.ttl_seconds:
                    self.redis.expire(self.key, self.ttl_seconds)
            except Exception as error:
                print("Redis cache write failed; switching to memory:", error)
                self.redis = None

    def clear(self) -> None:
        if self.redis is not None:
            try:
                self.redis.delete(self.key)
            except Exception as error:
                print("Redis cache clear failed:", error)
                self.redis = None

        self.memory.clear()
