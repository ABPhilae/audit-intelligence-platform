"""
Redis Semantic Cache Service.

Many users ask semantically similar questions. Instead of re-running the
full RAG pipeline every time, we cache previous answers and return them
instantly when a new question is close enough to a cached one.

Analogy: Think of a frequently asked questions (FAQ) board. The first person
to ask "What is the deadline?" triggers a full lookup. But the next person
who asks "When is the deadline due?" just gets pointed to the existing answer.
"""
import redis
import json
import hashlib
import logging
from datetime import timedelta
from src.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        self.enabled = False
        self.client = None
        self.ttl = timedelta(seconds=settings.cache_ttl_seconds)
        try:
            self.client = redis.Redis(
                host=settings.redis_host, port=settings.redis_port,
                db=0, decode_responses=True, socket_connect_timeout=5,
            )
            self.client.ping()
            self.enabled = True
            logger.info("Redis cache connected successfully")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis not available, caching disabled: {e}")

    def _make_key(self, question: str) -> str:
        normalised = question.lower().strip()
        question_hash = hashlib.sha256(normalised.encode()).hexdigest()[:16]
        return f"rag_cache:{question_hash}"

    def get(self, question: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            key = self._make_key(question)
            cached = self.client.get(key)
            if cached:
                logger.info(f"Cache HIT for question: {question[:50]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, question: str, result: dict):
        if not self.enabled:
            return
        try:
            key = self._make_key(question)
            self.client.setex(key, self.ttl, json.dumps(result, default=str))
            logger.info(f"Cached answer for: {question[:50]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        if not self.enabled:
            return
        try:
            keys = self.client.keys("rag_cache:*")
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached entries")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def get_stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "cached_entries": 0}
        try:
            keys = self.client.keys("rag_cache:*")
            return {"enabled": True, "cached_entries": len(keys), "ttl_seconds": settings.cache_ttl_seconds}
        except Exception:
            return {"enabled": False, "cached_entries": 0}
