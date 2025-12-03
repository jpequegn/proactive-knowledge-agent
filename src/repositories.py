"""Repository pattern for data access."""

from typing import Any

import asyncpg
import structlog

from uuid import UUID
import json

from src.database import Database
from src.models import (
    Article,
    ArticleWithEmbedding,
    MarketOHLCV,
    PodcastEpisode,
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
)

logger = structlog.get_logger()


class KnowledgeGraphRepository:
    """Repository for knowledge graph operations."""

    def __init__(self, db: Database):
        self.db = db

    async def upsert_entity(
        self, entity: Entity, embedding: list[float] | None = None
    ) -> UUID:
        """
        Insert or update an entity.
        Returns the entity ID.
        """
        query = """
        INSERT INTO entities (
            id, name, type, description, aliases, attributes, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            type = EXCLUDED.type,
            description = EXCLUDED.description,
            aliases = EXCLUDED.aliases,
            attributes = EXCLUDED.attributes,
            embedding = COALESCE($7, entities.embedding),
            updated_at = NOW()
        RETURNING id
        """
        async with self.db.acquire() as conn:
            await conn.fetchval(
                query,
                entity.id,
                entity.name,
                entity.type,
                entity.description,
                entity.aliases,
                json.dumps(entity.attributes),
                embedding,
            )
            return entity.id

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get entity by ID."""
        query = """
        SELECT id, name, type, description, aliases, attributes
        FROM entities WHERE id = $1
        """
        row = await self.db.fetchrow(query, entity_id)
        if row is None:
            return None
        return self._row_to_entity(row)

    async def find_entities_by_name(
        self, name_query: str, limit: int = 10
    ) -> list[Entity]:
        """Find entities by name (fuzzy match)."""
        query = """
        SELECT id, name, type, description, aliases, attributes
        FROM entities
        WHERE name ILIKE $1 OR $1 = ANY(aliases)
        LIMIT $2
        """
        rows = await self.db.fetch(query, f"%{name_query}%", limit)
        return [self._row_to_entity(row) for row in rows]
    
    async def find_similar_entities(
        self, embedding: list[float], limit: int = 10, threshold: float = 0.7
    ) -> list[tuple[Entity, float]]:
        """Find entities with similar embeddings."""
        query = """
        SELECT id, name, type, description, aliases, attributes,
               1 - (embedding <=> $1) as similarity
        FROM entities
        WHERE embedding IS NOT NULL
          AND 1 - (embedding <=> $1) >= $2
        ORDER BY embedding <=> $1
        LIMIT $3
        """
        rows = await self.db.fetch(query, embedding, threshold, limit)
        return [(self._row_to_entity(row), row["similarity"]) for row in rows]

    async def create_relationship(self, relationship: Relationship) -> UUID:
        """Create a relationship between entities."""
        query = """
        INSERT INTO relationships (
            id, source_entity_id, target_entity_id, type, weight, attributes
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (source_entity_id, target_entity_id, type) DO UPDATE SET
            weight = EXCLUDED.weight,
            attributes = EXCLUDED.attributes,
            updated_at = NOW()
        RETURNING id
        """
        async with self.db.acquire() as conn:
            await conn.fetchval(
                query,
                relationship.id,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.type,
                relationship.weight,
                json.dumps(relationship.attributes),
            )
            return relationship.id

    async def get_related_entities(
        self,
        entity_id: UUID,
        direction: str = "outgoing",
        rel_type: RelationshipType | None = None,
    ) -> list[tuple[Relationship, Entity]]:
        """
        Get entities related to the given entity.
        direction: 'outgoing', 'incoming', or 'both'
        """
        conditions = []
        params = [entity_id]
        
        if rel_type:
            conditions.append(f"r.type = ${len(params) + 1}")
            params.append(rel_type)

        where_extra = " AND ".join(conditions)
        if where_extra:
            where_extra = " AND " + where_extra

        results = []
        
        # Outgoing
        if direction in ["outgoing", "both"]:
            query = f"""
            SELECT r.id as rel_id, r.source_entity_id, r.target_entity_id, r.type as rel_type, 
                   r.weight, r.attributes as rel_attributes,
                   e.id as entity_id, e.name, e.type as entity_type, e.description, 
                   e.aliases, e.attributes as entity_attributes
            FROM relationships r
            JOIN entities e ON r.target_entity_id = e.id
            WHERE r.source_entity_id = $1 {where_extra}
            """
            rows = await self.db.fetch(query, *params)
            results.extend([self._row_to_rel_entity(row) for row in rows])

        # Incoming
        if direction in ["incoming", "both"]:
            query = f"""
            SELECT r.id as rel_id, r.source_entity_id, r.target_entity_id, r.type as rel_type, 
                   r.weight, r.attributes as rel_attributes,
                   e.id as entity_id, e.name, e.type as entity_type, e.description, 
                   e.aliases, e.attributes as entity_attributes
            FROM relationships r
            JOIN entities e ON r.source_entity_id = e.id
            WHERE r.target_entity_id = $1 {where_extra}
            """
            rows = await self.db.fetch(query, *params)
            results.extend([self._row_to_rel_entity(row) for row in rows])

        return results

    def _row_to_entity(self, row: asyncpg.Record) -> Entity:
        """Convert database row to Entity model."""
        attributes = row["attributes"]
        if isinstance(attributes, str):
            attributes = json.loads(attributes)
            
        return Entity(
            id=row["id"],
            name=row["name"],
            type=EntityType(row["type"]),
            description=row["description"],
            aliases=row["aliases"] if row["aliases"] else [],
            attributes=attributes or {},
        )

    def _row_to_rel_entity(self, row: asyncpg.Record) -> tuple[Relationship, Entity]:
        """Convert database row to (Relationship, Entity) tuple."""
        rel_attrs = row["rel_attributes"]
        if isinstance(rel_attrs, str):
            rel_attrs = json.loads(rel_attrs)

        ent_attrs = row["entity_attributes"]
        if isinstance(ent_attrs, str):
            ent_attrs = json.loads(ent_attrs)

        rel = Relationship(
            id=row["rel_id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            type=RelationshipType(row["rel_type"]),
            weight=row["weight"],
            attributes=rel_attrs or {},
        )
        
        entity = Entity(
            id=row["entity_id"],
            name=row["name"],
            type=EntityType(row["entity_type"]),
            description=row["description"],
            aliases=row["aliases"] if row["aliases"] else [],
            attributes=ent_attrs or {},
        )
        
        return rel, entity


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


class MarketRepository:
    """Repository for market data."""

    def __init__(self, db: Database):
        self.db = db

    async def upsert_batch(self, data: list[MarketOHLCV]) -> int:
        """
        Insert or update a batch of market data points.
        Returns the number of records inserted/updated.
        """
        if not data:
            return 0

        query = """
        INSERT INTO market_ohlcv (
            symbol, date, open, high, low, close, volume, adjusted_close
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            adjusted_close = EXCLUDED.adjusted_close,
            updated_at = NOW()
        """

        async with self.db.transaction() as conn:
            # asyncpg execute_many is efficient for batches
            await conn.executemany(
                query,
                [
                    (
                        d.symbol,
                        d.date,
                        d.open,
                        d.high,
                        d.low,
                        d.close,
                        d.volume,
                        d.adjusted_close,
                    )
                    for d in data
                ],
            )
            return len(data)

    async def get_history(
        self,
        symbol: str,
        limit: int = 30,
    ) -> list[MarketOHLCV]:
        """Get historical data for a symbol."""
        query = """
        SELECT symbol, date, open, high, low, close, volume, adjusted_close
        FROM market_ohlcv
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT $2
        """
        rows = await self.db.fetch(query, symbol, limit)
        # Return in chronological order (oldest first) usually expected for charts/analysis,
        # but query gets newest first. Let's reverse.
        return [
            MarketOHLCV(
                symbol=row["symbol"],
                date=row["date"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                adjusted_close=row["adjusted_close"],
            )
            for row in reversed(rows)
        ]

    async def count(self) -> int:
        """Count total market data points."""
        query = "SELECT COUNT(*) FROM market_ohlcv"
        return await self.db.fetchval(query)

    async def get_symbols(self) -> list[str]:
        """Get list of unique symbols."""
        query = "SELECT DISTINCT symbol FROM market_ohlcv ORDER BY symbol"
        rows = await self.db.fetch(query)
        return [row["symbol"] for row in rows]


class PodcastRepository:
    """Repository for podcast episodes."""

    def __init__(self, db: Database):
        self.db = db

    async def upsert(
        self,
        episode: PodcastEpisode,
        embedding: list[float] | None = None,
    ) -> int:
        """
        Insert or update a podcast episode.
        Returns the episode ID.
        """
        query = """
        INSERT INTO podcast_episodes (
            external_id, title, podcast_name, published,
            summary, content, duration_seconds, url,
            topics, entities, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (external_id) DO UPDATE SET
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            content = EXCLUDED.content,
            topics = EXCLUDED.topics,
            entities = EXCLUDED.entities,
            updated_at = NOW()
        RETURNING id
        """
        async with self.db.acquire() as conn:
            episode_id = await conn.fetchval(
                query,
                episode.id,  # internal ID vs external ID naming convention?
                             # Model has 'id' as string from source, which maps to external_id
                episode.title,
                episode.podcast_name,
                episode.published_date,
                episode.summary,
                episode.content,
                episode.duration_seconds,
                episode.url,
                episode.topics,
                episode.entities,
                embedding,
            )
            return episode_id

    async def get_recent(
        self,
        limit: int = 10,
        podcast_name: str | None = None,
    ) -> list[PodcastEpisode]:
        """Get recent episodes."""
        conditions = ["1=1"]
        params: list[Any] = []

        if podcast_name:
            conditions.append(f"podcast_name = ${len(params) + 1}")
            params.append(podcast_name)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT external_id, title, podcast_name, published,
               summary, content, duration_seconds, url,
               topics, entities
        FROM podcast_episodes
        WHERE {where_clause}
        ORDER BY published DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_episode(row) for row in rows]

    async def search_by_text(
        self,
        search_text: str,
        limit: int = 20,
    ) -> list[PodcastEpisode]:
        """Search episodes by text (title, summary)."""
        query = """
        SELECT external_id, title, podcast_name, published,
               summary, content, duration_seconds, url,
               topics, entities
        FROM podcast_episodes
        WHERE title ILIKE $1 OR summary ILIKE $1
        ORDER BY published DESC
        LIMIT $2
        """
        rows = await self.db.fetch(query, f"%{search_text}%", limit)
        return [self._row_to_episode(row) for row in rows]

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[PodcastEpisode, float]]:
        """Find semantically similar episodes."""
        query = """
        SELECT external_id, title, podcast_name, published,
               summary, content, duration_seconds, url,
               topics, entities,
               1 - (embedding <=> $1) as similarity
        FROM podcast_episodes
        WHERE embedding IS NOT NULL
          AND 1 - (embedding <=> $1) >= $2
        ORDER BY embedding <=> $1
        LIMIT $3
        """
        rows = await self.db.fetch(query, embedding, threshold, limit)
        return [
            (self._row_to_episode(row), row["similarity"])
            for row in rows
        ]

    async def count(self) -> int:
        """Count total episodes."""
        query = "SELECT COUNT(*) FROM podcast_episodes"
        return await self.db.fetchval(query)

    async def get_podcasts(self) -> list[str]:
        """Get list of unique podcast names."""
        query = "SELECT DISTINCT podcast_name FROM podcast_episodes ORDER BY podcast_name"
        rows = await self.db.fetch(query)
        return [row["podcast_name"] for row in rows]

    def _row_to_episode(self, row: asyncpg.Record) -> PodcastEpisode:
        """Convert database row to PodcastEpisode model."""
        return PodcastEpisode(
            id=row["external_id"],
            title=row["title"],
            podcast_name=row["podcast_name"],
            published_date=row["published"],
            summary=row["summary"],
            content=row["content"],
            duration_seconds=row["duration_seconds"],
            url=row["url"],
            topics=row["topics"] if row["topics"] else [],
            entities=row["entities"] if row["entities"] else [],
        )
