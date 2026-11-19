from __future__ import annotations

import json
import time

import numpy as np


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm else 0.0


class SemanticCache:
    KEY = "multi_agent:semantic_cache:v2"

    def __init__(
        self,
        embed,
        threshold: float = 0.95,
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
                items = []
                for raw in self.redis.lrange(self.key, 0, -1):
                    try:
                        items.append(json.loads(raw))
                    except (TypeError, json.JSONDecodeError):
                        continue
                return items
            except Exception as error:
                print("Redis cache read failed; switching to memory:", error)
                self.redis = None
        return list(self.memory)

    def _fresh(self, item: dict, now: float) -> bool:
        return not self.ttl_seconds or now - item.get("ts", 0) <= self.ttl_seconds

    def get(self, question: str, scope: str = "default", vector=None):
        if not question or not question.strip():
            return None

        vector = vector if vector is not None else self.embed(question)
        now = time.time()
        best_item, best_score = None, 0.0

        for item in self._items():
            if item.get("scope", "default") != scope or not self._fresh(item, now):
                continue
            score = cosine(vector, item.get("vector", []))
            if score > best_score:
                best_item, best_score = item, score

        if best_item is not None and best_score >= self.threshold:
            return best_item.get("value")
        return None

    def set(self, question: str, value: dict, scope: str = "default", vector=None) -> None:
        if not question or not question.strip():
            return

        item = {
            "question": question,
            "vector": list(vector if vector is not None else self.embed(question)),
            "value": value,
            "scope": scope,
            "ts": time.time(),
        }

        self.memory.append(item)
        del self.memory[:-self.max_items]

        if self.redis is not None:
            try:
                self.redis.rpush(self.key, json.dumps(item, ensure_ascii=False))
                self.redis.ltrim(self.key, -self.max_items, -1)
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
