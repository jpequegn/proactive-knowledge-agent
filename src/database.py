"""Database connection and operations for PKA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

logger = structlog.get_logger()


class Database:
    """Async PostgreSQL database connection manager."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            init=self._init_connection,
        )
        logger.info("Database pool created")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize connection with pgvector extension."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        async with self.pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection with transaction."""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Fetch all rows from a query."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Fetch a single row from a query."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value from a query."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)


# SQL Schema for articles table
ARTICLES_SCHEMA = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Articles from RSS feeds
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    url_hash TEXT UNIQUE NOT NULL,
    summary TEXT,
    content TEXT,
    published TIMESTAMPTZ,
    source TEXT NOT NULL,
    category TEXT,
    author TEXT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS articles_embedding_idx ON articles
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for URL hash lookups
CREATE INDEX IF NOT EXISTS articles_url_hash_idx ON articles (url_hash);

-- Index for source filtering
CREATE INDEX IF NOT EXISTS articles_source_idx ON articles (source);

-- Index for category filtering
CREATE INDEX IF NOT EXISTS articles_category_idx ON articles (category);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS articles_published_idx ON articles (published DESC);
"""


async def init_schema(db: Database) -> None:
    """Initialize database schema."""
    async with db.transaction() as conn:
        await conn.execute(ARTICLES_SCHEMA)
    logger.info("Database schema initialized")
