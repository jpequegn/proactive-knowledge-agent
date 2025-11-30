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


ACTIVITIES_SCHEMA = """
-- Fitness activities from Strava
CREATE TABLE IF NOT EXISTS activities (
    id SERIAL PRIMARY KEY,
    external_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    sport_type TEXT,
    start_date TIMESTAMPTZ NOT NULL,
    distance_meters FLOAT DEFAULT 0,
    moving_time_seconds INT DEFAULT 0,
    elapsed_time_seconds INT DEFAULT 0,
    total_elevation_gain FLOAT DEFAULT 0,
    avg_speed FLOAT,
    max_speed FLOAT,
    avg_hr INT,
    max_hr INT,
    avg_power INT,
    max_power INT,
    calories INT,
    suffer_score INT,
    tss FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS activities_start_date_idx
ON activities (start_date DESC);

-- Index for activity type filtering
CREATE INDEX IF NOT EXISTS activities_type_idx
ON activities (activity_type);

-- Daily fitness metrics
CREATE TABLE IF NOT EXISTS fitness_metrics (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    daily_tss FLOAT DEFAULT 0,
    atl FLOAT DEFAULT 0,
    ctl FLOAT DEFAULT 0,
    tsb FLOAT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for date lookups
CREATE INDEX IF NOT EXISTS fitness_metrics_date_idx
ON fitness_metrics (date DESC);

-- Strava tokens storage
CREATE TABLE IF NOT EXISTS strava_tokens (
    id SERIAL PRIMARY KEY,
    athlete_id TEXT UNIQUE NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

MARKET_SCHEMA = """
-- Market Data
CREATE TABLE IF NOT EXISTS market_ohlcv (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, date)
);

-- Index for market queries
CREATE INDEX IF NOT EXISTS market_ohlcv_symbol_date_idx
ON market_ohlcv (symbol, date DESC);
"""

PODCAST_SCHEMA = """
-- Podcast Episodes
CREATE TABLE IF NOT EXISTS podcast_episodes (
    id SERIAL PRIMARY KEY,
    external_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    podcast_name TEXT NOT NULL,
    published TIMESTAMPTZ,
    summary TEXT,
    content TEXT,
    duration_seconds INT,
    url TEXT,
    topics TEXT[],
    entities TEXT[],
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector similarity search on podcasts
CREATE INDEX IF NOT EXISTS podcast_episodes_embedding_idx ON podcast_episodes
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for podcast date queries
CREATE INDEX IF NOT EXISTS podcast_episodes_published_idx
ON podcast_episodes (published DESC);
"""


async def init_schema(db: Database) -> None:
    """Initialize database schema."""
    async with db.transaction() as conn:
        await conn.execute(ARTICLES_SCHEMA)
        await conn.execute(ACTIVITIES_SCHEMA)
        await conn.execute(MARKET_SCHEMA)
        await conn.execute(PODCAST_SCHEMA)
    logger.info("Database schema initialized")
