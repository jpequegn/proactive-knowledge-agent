"""Redis cache and event queue wrapper."""

from typing import Any

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger()


class Cache:
    """Redis cache manager."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis = Redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        await self.redis.ping()
        logger.info("Redis connected")

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis disconnected")

    async def set(self, key: str, value: str, expire: int | None = None) -> None:
        """Set a value in cache."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        await self.redis.set(key, value, ex=expire)

    async def get(self, key: str) -> str | None:
        """Get a value from cache."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        return await self.redis.get(key)

    async def delete(self, key: str) -> None:
        """Delete a value from cache."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        await self.redis.delete(key)

    async def publish(self, channel: str, message: str) -> None:
        """Publish a message to a channel."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        await self.redis.publish(channel, message)
