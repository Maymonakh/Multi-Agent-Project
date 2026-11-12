"""Semantic cache: if a NEW question has almost the same meaning as an old one,
return the old answer without running the agents again.
Stored in Redis if REDIS_URL works, otherwise in memory."""
import json
import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SemanticCache:
    KEY = "semantic_cache"

    def __init__(self, embed, threshold: float = 0.92, redis_url: str | None = None, max_items: int = 200):
        self.embed = embed            # function: text -> list[float]
        self.threshold = threshold
        self.max_items = max_items
        self.memory = []
        self.redis = None
        if redis_url:
            try:
                import redis
                self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self.redis.ping()
            except Exception as error:
                print("Redis not available, using memory cache:", error)
                self.redis = None

    def _items(self) -> list[dict]:
        if self.redis:
            return [json.loads(x) for x in self.redis.lrange(self.KEY, 0, -1)]
        return self.memory

    def get(self, question: str):
        vector = self.embed(question)
        best, best_score = None, 0.0
        for item in self._items():
            score = cosine(vector, item["vector"])
            if score > best_score:
                best, best_score = item, score
        return best["value"] if best and best_score >= self.threshold else None

    def set(self, question: str, value: dict) -> None:
        item = {"question": question, "vector": self.embed(question), "value": value}
        if self.redis:
            self.redis.rpush(self.KEY, json.dumps(item))
            self.redis.ltrim(self.KEY, -self.max_items, -1)
        else:
            self.memory.append(item)
            del self.memory[:-self.max_items]

    def clear(self) -> None:
        if self.redis:
            self.redis.delete(self.KEY)
        self.memory.clear()
