"""Repository pattern for data access."""

from typing import Any

import asyncpg
import structlog

from src.database import Database
from src.models import Article, ArticleWithEmbedding

logger = structlog.get_logger()


class ArticleRepository:
    """Repository for article CRUD operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(
        self,
        article: Article,
        embedding: list[float] | None = None,
    ) -> int:
        """Create a new article. Returns the article ID."""
        query = """
        INSERT INTO articles (
            title, url, url_hash, summary, content,
            published, source, category, author, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id
        """
        async with self.db.acquire() as conn:
            article_id = await conn.fetchval(
                query,
                article.title,
                article.url,
                article.url_hash,
                article.summary,
                article.content,
                article.published,
                article.source,
                article.category,
                article.author,
                embedding,
            )
            logger.debug("Article created", id=article_id, title=article.title[:50])
            return article_id

    async def upsert(
        self,
        article: Article,
        embedding: list[float] | None = None,
    ) -> tuple[int, bool]:
        """
        Insert or update an article based on URL hash.
        Returns (article_id, is_new).
        """
        query = """
        INSERT INTO articles (
            title, url, url_hash, summary, content,
            published, source, category, author, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (url_hash) DO UPDATE SET
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            content = EXCLUDED.content,
            updated_at = NOW()
        RETURNING id, (xmax = 0) as is_new
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                query,
                article.title,
                article.url,
                article.url_hash,
                article.summary,
                article.content,
                article.published,
                article.source,
                article.category,
                article.author,
                embedding,
            )
            is_new = row["is_new"]
            logger.debug(
                "Article upserted",
                id=row["id"],
                is_new=is_new,
                title=article.title[:50],
            )
            return row["id"], is_new

    async def exists_by_url_hash(self, url_hash: str) -> bool:
        """Check if an article with given URL hash exists."""
        query = "SELECT 1 FROM articles WHERE url_hash = $1"
        result = await self.db.fetchval(query, url_hash)
        return result is not None

    async def get_by_id(self, article_id: int) -> ArticleWithEmbedding | None:
        """Get article by ID."""
        query = """
        SELECT id, title, url, summary, content, published,
               source, category, author, embedding
        FROM articles WHERE id = $1
        """
        row = await self.db.fetchrow(query, article_id)
        if row is None:
            return None
        return self._row_to_article(row)

    async def get_by_url_hash(self, url_hash: str) -> ArticleWithEmbedding | None:
        """Get article by URL hash."""
        query = """
        SELECT id, title, url, summary, content, published,
               source, category, author, embedding
        FROM articles WHERE url_hash = $1
        """
        row = await self.db.fetchrow(query, url_hash)
        if row is None:
            return None
        return self._row_to_article(row)

    async def find_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[ArticleWithEmbedding, float]]:
        """Find articles with similar embeddings."""
        query = """
        SELECT id, title, url, summary, content, published,
               source, category, author, embedding,
               1 - (embedding <=> $1) as similarity
        FROM articles
        WHERE embedding IS NOT NULL
          AND 1 - (embedding <=> $1) >= $2
        ORDER BY embedding <=> $1
        LIMIT $3
        """
        rows = await self.db.fetch(query, embedding, threshold, limit)
        return [
            (self._row_to_article(row), row["similarity"])
            for row in rows
        ]

    async def search_by_text(
        self,
        search_text: str,
        limit: int = 20,
        category: str | None = None,
        source: str | None = None,
    ) -> list[ArticleWithEmbedding]:
        """Search articles by text (title, summary)."""
        conditions = ["(title ILIKE $1 OR summary ILIKE $1)"]
        params: list[Any] = [f"%{search_text}%"]

        if category:
            conditions.append(f"category = ${len(params) + 1}")
            params.append(category)

        if source:
            conditions.append(f"source = ${len(params) + 1}")
            params.append(source)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, title, url, summary, content, published,
               source, category, author, embedding
        FROM articles
        WHERE {where_clause}
        ORDER BY published DESC NULLS LAST
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_article(row) for row in rows]

    async def get_recent(
        self,
        limit: int = 50,
        category: str | None = None,
        source: str | None = None,
    ) -> list[ArticleWithEmbedding]:
        """Get recent articles."""
        conditions = ["1=1"]
        params: list[Any] = []

        if category:
            conditions.append(f"category = ${len(params) + 1}")
            params.append(category)

        if source:
            conditions.append(f"source = ${len(params) + 1}")
            params.append(source)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, title, url, summary, content, published,
               source, category, author, embedding
        FROM articles
        WHERE {where_clause}
        ORDER BY published DESC NULLS LAST
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_article(row) for row in rows]

    async def count(
        self,
        category: str | None = None,
        source: str | None = None,
    ) -> int:
        """Count articles matching filters."""
        conditions = ["1=1"]
        params: list[Any] = []

        if category:
            conditions.append(f"category = ${len(params) + 1}")
            params.append(category)

        if source:
            conditions.append(f"source = ${len(params) + 1}")
            params.append(source)

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM articles WHERE {where_clause}"

        return await self.db.fetchval(query, *params)

    async def get_sources(self) -> list[str]:
        """Get list of unique sources."""
        query = "SELECT DISTINCT source FROM articles ORDER BY source"
        rows = await self.db.fetch(query)
        return [row["source"] for row in rows]

    async def get_categories(self) -> list[str]:
        """Get list of unique categories."""
        query = """
            SELECT DISTINCT category FROM articles
            WHERE category IS NOT NULL ORDER BY category
        """
        rows = await self.db.fetch(query)
        return [row["category"] for row in rows]

    def _row_to_article(self, row: asyncpg.Record) -> ArticleWithEmbedding:
        """Convert database row to Article model."""
        embedding = row["embedding"]
        if embedding is not None:
            embedding = list(embedding)

        return ArticleWithEmbedding(
            id=row["id"],
            title=row["title"],
            url=row["url"],
            summary=row["summary"],
            content=row["content"],
            published=row["published"],
            source=row["source"],
            category=row["category"],
            author=row["author"],
            embedding=embedding,
        )


class DeduplicationService:
    """Service for detecting and handling duplicate articles."""

    def __init__(
        self,
        article_repo: ArticleRepository,
        similarity_threshold: float = 0.95,
    ):
        self.article_repo = article_repo
        self.similarity_threshold = similarity_threshold

    async def is_duplicate_by_url(self, article: Article) -> bool:
        """Check if article URL already exists."""
        return await self.article_repo.exists_by_url_hash(article.url_hash)

    async def find_semantic_duplicates(
        self,
        embedding: list[float],
        exclude_url_hash: str | None = None,
    ) -> list[tuple[ArticleWithEmbedding, float]]:
        """Find semantically similar articles that might be duplicates."""
        similar = await self.article_repo.find_similar(
            embedding=embedding,
            limit=5,
            threshold=self.similarity_threshold,
        )

        if exclude_url_hash:
            similar = [
                (article, score)
                for article, score in similar
                if article.url_hash != exclude_url_hash
            ]

        return similar
