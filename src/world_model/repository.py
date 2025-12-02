"""Repository for knowledge graph operations."""

from datetime import datetime
from typing import Any

import asyncpg
import structlog

from src.database import Database
from src.world_model.entities import (
    Domain,
    Entity,
    EntityType,
)
from src.world_model.relationships import (
    EntityMention,
    EntityVersion,
    Relationship,
    RelationshipType,
)

logger = structlog.get_logger()


class EntityRepository:
    """Repository for entity CRUD operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(
        self,
        entity: Entity,
        embedding: list[float] | None = None,
    ) -> int:
        """Create a new entity. Returns the entity ID."""
        query = """
        INSERT INTO entities (
            external_id, name, entity_type, domain, description,
            source, source_url, confidence, mention_count,
            first_seen, last_seen, embedding, properties
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id
        """
        async with self.db.acquire() as conn:
            entity_id = await conn.fetchval(
                query,
                entity.external_id,
                entity.name,
                entity.entity_type.value,
                entity.domain.value,
                entity.description,
                entity.source,
                entity.source_url,
                entity.confidence,
                entity.mention_count,
                entity.first_seen,
                entity.last_seen,
                embedding,
                entity.properties,
            )
            logger.debug(
                "Entity created",
                id=entity_id,
                name=entity.name,
                type=entity.entity_type.value,
            )
            return entity_id

    async def upsert(
        self,
        entity: Entity,
        embedding: list[float] | None = None,
    ) -> tuple[int, bool]:
        """
        Insert or update an entity based on name and type.
        Returns (entity_id, is_new).
        """
        query = """
        INSERT INTO entities (
            external_id, name, entity_type, domain, description,
            source, source_url, confidence, mention_count,
            first_seen, last_seen, embedding, properties
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (name, entity_type) DO UPDATE SET
            description = COALESCE(EXCLUDED.description, entities.description),
            mention_count = entities.mention_count + 1,
            last_seen = EXCLUDED.last_seen,
            confidence = GREATEST(entities.confidence, EXCLUDED.confidence),
            updated_at = NOW()
        RETURNING id, (xmax = 0) as is_new
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                query,
                entity.external_id,
                entity.name,
                entity.entity_type.value,
                entity.domain.value,
                entity.description,
                entity.source,
                entity.source_url,
                entity.confidence,
                entity.mention_count,
                entity.first_seen,
                entity.last_seen,
                embedding,
                entity.properties,
            )
            is_new = row["is_new"]
            logger.debug(
                "Entity upserted",
                id=row["id"],
                is_new=is_new,
                name=entity.name,
            )
            return row["id"], is_new

    async def get_by_id(self, entity_id: int) -> Entity | None:
        """Get entity by ID."""
        query = """
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities WHERE id = $1
        """
        row = await self.db.fetchrow(query, entity_id)
        if row is None:
            return None
        return self._row_to_entity(row)

    async def get_by_name_and_type(
        self,
        name: str,
        entity_type: EntityType,
    ) -> Entity | None:
        """Get entity by name and type."""
        query = """
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities WHERE name = $1 AND entity_type = $2
        """
        row = await self.db.fetchrow(query, name, entity_type.value)
        if row is None:
            return None
        return self._row_to_entity(row)

    async def find_by_type(
        self,
        entity_type: EntityType,
        limit: int = 50,
        domain: Domain | None = None,
    ) -> list[Entity]:
        """Find entities by type, optionally filtered by domain."""
        conditions = ["entity_type = $1"]
        params: list[Any] = [entity_type.value]

        if domain:
            conditions.append(f"domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities
        WHERE {where_clause}
        ORDER BY mention_count DESC, last_seen DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_entity(row) for row in rows]

    async def find_by_domain(
        self,
        domain: Domain,
        limit: int = 50,
    ) -> list[Entity]:
        """Find all entities in a domain."""
        query = """
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities
        WHERE domain = $1
        ORDER BY mention_count DESC, last_seen DESC
        LIMIT $2
        """
        rows = await self.db.fetch(query, domain.value, limit)
        return [self._row_to_entity(row) for row in rows]

    async def find_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        entity_type: EntityType | None = None,
    ) -> list[tuple[Entity, float]]:
        """Find entities with similar embeddings."""
        conditions = [
            "embedding IS NOT NULL",
            "1 - (embedding <=> $1) >= $2",
        ]
        params: list[Any] = [embedding, threshold]

        if entity_type:
            conditions.append(f"entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties,
               1 - (embedding <=> $1) as similarity
        FROM entities
        WHERE {where_clause}
        ORDER BY embedding <=> $1
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [(self._row_to_entity(row), row["similarity"]) for row in rows]

    async def search_by_name(
        self,
        search_text: str,
        limit: int = 20,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """Search entities by name (case-insensitive)."""
        conditions = ["name ILIKE $1"]
        params: list[Any] = [f"%{search_text}%"]

        if entity_type:
            conditions.append(f"entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities
        WHERE {where_clause}
        ORDER BY mention_count DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_entity(row) for row in rows]

    async def get_trending(
        self,
        days: int = 7,
        limit: int = 20,
        domain: Domain | None = None,
    ) -> list[Entity]:
        """Get entities with most recent mentions (trending)."""
        conditions = [f"last_seen >= NOW() - INTERVAL '{days} days'"]
        params: list[Any] = []

        if domain:
            conditions.append(f"domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities
        WHERE {where_clause}
        ORDER BY mention_count DESC, last_seen DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_entity(row) for row in rows]

    async def count(
        self,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
    ) -> int:
        """Count entities matching filters."""
        conditions = ["1=1"]
        params: list[Any] = []

        if entity_type:
            conditions.append(f"entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM entities WHERE {where_clause}"

        return await self.db.fetchval(query, *params)

    async def increment_mention_count(
        self,
        entity_id: int,
        last_seen: datetime | None = None,
    ) -> None:
        """Increment mention count and update last_seen."""
        if last_seen is None:
            last_seen = datetime.utcnow()

        query = """
        UPDATE entities
        SET mention_count = mention_count + 1,
            last_seen = $2,
            updated_at = NOW()
        WHERE id = $1
        """
        await self.db.execute(query, entity_id, last_seen)

    def _row_to_entity(self, row: asyncpg.Record) -> Entity:
        """Convert database row to Entity model."""
        return Entity(
            id=row["id"],
            external_id=row["external_id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            domain=Domain(row["domain"]),
            description=row["description"],
            source=row["source"],
            source_url=row["source_url"],
            confidence=row["confidence"],
            mention_count=row["mention_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            properties=row["properties"] if row["properties"] else {},
        )


class RelationshipRepository:
    """Repository for relationship CRUD operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, relationship: Relationship) -> int:
        """Create a new relationship. Returns the relationship ID."""
        query = """
        INSERT INTO relationships (
            relationship_type, source_entity_id, target_entity_id,
            weight, confidence, source, source_url,
            first_seen, last_seen, valid_from, valid_until, properties
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
        """
        async with self.db.acquire() as conn:
            rel_id = await conn.fetchval(
                query,
                relationship.relationship_type.value,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.weight,
                relationship.confidence,
                relationship.source,
                relationship.source_url,
                relationship.first_seen,
                relationship.last_seen,
                relationship.valid_from,
                relationship.valid_until,
                relationship.properties,
            )
            logger.debug(
                "Relationship created",
                id=rel_id,
                type=relationship.relationship_type.value,
            )
            return rel_id

    async def upsert(self, relationship: Relationship) -> tuple[int, bool]:
        """
        Insert or update a relationship.
        Returns (relationship_id, is_new).
        """
        query = """
        INSERT INTO relationships (
            relationship_type, source_entity_id, target_entity_id,
            weight, confidence, source, source_url,
            first_seen, last_seen, valid_from, valid_until, properties
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (relationship_type, source_entity_id, target_entity_id)
        DO UPDATE SET
            weight = GREATEST(relationships.weight, EXCLUDED.weight),
            confidence = GREATEST(
                relationships.confidence, EXCLUDED.confidence
            ),
            last_seen = EXCLUDED.last_seen,
            updated_at = NOW()
        RETURNING id, (xmax = 0) as is_new
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                query,
                relationship.relationship_type.value,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.weight,
                relationship.confidence,
                relationship.source,
                relationship.source_url,
                relationship.first_seen,
                relationship.last_seen,
                relationship.valid_from,
                relationship.valid_until,
                relationship.properties,
            )
            return row["id"], row["is_new"]

    async def get_outgoing(
        self,
        entity_id: int,
        relationship_type: RelationshipType | None = None,
        include_expired: bool = False,
    ) -> list[Relationship]:
        """Get relationships where entity is the source."""
        conditions = ["source_entity_id = $1"]
        params: list[Any] = [entity_id]

        if relationship_type:
            conditions.append(f"relationship_type = ${len(params) + 1}")
            params.append(relationship_type.value)

        if not include_expired:
            conditions.append("(valid_until IS NULL OR valid_until > NOW())")

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, relationship_type, source_entity_id, target_entity_id,
               weight, confidence, source, source_url,
               first_seen, last_seen, valid_from, valid_until,
               created_at, updated_at, properties
        FROM relationships
        WHERE {where_clause}
        ORDER BY weight DESC, confidence DESC
        """
        rows = await self.db.fetch(query, *params)
        return [self._row_to_relationship(row) for row in rows]

    async def get_incoming(
        self,
        entity_id: int,
        relationship_type: RelationshipType | None = None,
        include_expired: bool = False,
    ) -> list[Relationship]:
        """Get relationships where entity is the target."""
        conditions = ["target_entity_id = $1"]
        params: list[Any] = [entity_id]

        if relationship_type:
            conditions.append(f"relationship_type = ${len(params) + 1}")
            params.append(relationship_type.value)

        if not include_expired:
            conditions.append("(valid_until IS NULL OR valid_until > NOW())")

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, relationship_type, source_entity_id, target_entity_id,
               weight, confidence, source, source_url,
               first_seen, last_seen, valid_from, valid_until,
               created_at, updated_at, properties
        FROM relationships
        WHERE {where_clause}
        ORDER BY weight DESC, confidence DESC
        """
        rows = await self.db.fetch(query, *params)
        return [self._row_to_relationship(row) for row in rows]

    async def find_path(
        self,
        source_id: int,
        target_id: int,
        max_depth: int = 3,
    ) -> list[list[Relationship]]:
        """
        Find paths between two entities.
        Uses recursive CTE for graph traversal.
        Returns list of paths (each path is a list of relationships).
        """
        # This is a simplified version - full path finding would need
        # more sophisticated graph algorithms
        query = """
        WITH RECURSIVE paths AS (
            -- Base case: direct relationships from source
            SELECT
                ARRAY[r.id] as path,
                r.target_entity_id as current_node,
                1 as depth
            FROM relationships r
            WHERE r.source_entity_id = $1
              AND (r.valid_until IS NULL OR r.valid_until > NOW())

            UNION ALL

            -- Recursive case: extend paths
            SELECT
                p.path || r.id,
                r.target_entity_id,
                p.depth + 1
            FROM paths p
            JOIN relationships r ON r.source_entity_id = p.current_node
            WHERE p.depth < $3
              AND NOT (r.id = ANY(p.path))  -- Avoid cycles
              AND (r.valid_until IS NULL OR r.valid_until > NOW())
        )
        SELECT path FROM paths
        WHERE current_node = $2
        ORDER BY array_length(path, 1)
        LIMIT 10
        """
        rows = await self.db.fetch(query, source_id, target_id, max_depth)

        # Convert path IDs to relationships
        all_paths = []
        for row in rows:
            path_ids = row["path"]
            if path_ids:
                path_rels = []
                for rel_id in path_ids:
                    rel = await self.get_by_id(rel_id)
                    if rel:
                        path_rels.append(rel)
                if path_rels:
                    all_paths.append(path_rels)

        return all_paths

    async def get_by_id(self, relationship_id: int) -> Relationship | None:
        """Get relationship by ID."""
        query = """
        SELECT id, relationship_type, source_entity_id, target_entity_id,
               weight, confidence, source, source_url,
               first_seen, last_seen, valid_from, valid_until,
               created_at, updated_at, properties
        FROM relationships WHERE id = $1
        """
        row = await self.db.fetchrow(query, relationship_id)
        if row is None:
            return None
        return self._row_to_relationship(row)

    async def count(
        self,
        relationship_type: RelationshipType | None = None,
    ) -> int:
        """Count relationships matching filters."""
        conditions = ["1=1"]
        params: list[Any] = []

        if relationship_type:
            conditions.append(f"relationship_type = ${len(params) + 1}")
            params.append(relationship_type.value)

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM relationships WHERE {where_clause}"

        return await self.db.fetchval(query, *params)

    def _row_to_relationship(self, row: asyncpg.Record) -> Relationship:
        """Convert database row to Relationship model."""
        return Relationship(
            id=row["id"],
            relationship_type=RelationshipType(row["relationship_type"]),
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            weight=row["weight"],
            confidence=row["confidence"],
            source=row["source"],
            source_url=row["source_url"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            properties=row["properties"] if row["properties"] else {},
        )


class MentionRepository:
    """Repository for entity mention operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, mention: EntityMention) -> int:
        """Create a new mention. Returns the mention ID."""
        query = """
        INSERT INTO entity_mentions (
            entity_id, content_type, content_id,
            context, sentiment, mentioned_at
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        async with self.db.acquire() as conn:
            mention_id = await conn.fetchval(
                query,
                mention.entity_id,
                mention.content_type,
                mention.content_id,
                mention.context,
                mention.sentiment,
                mention.mentioned_at,
            )
            return mention_id

    async def upsert(self, mention: EntityMention) -> tuple[int, bool]:
        """Insert or update a mention. Returns (mention_id, is_new)."""
        query = """
        INSERT INTO entity_mentions (
            entity_id, content_type, content_id,
            context, sentiment, mentioned_at
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (entity_id, content_type, content_id) DO UPDATE SET
            context = COALESCE(EXCLUDED.context, entity_mentions.context),
            sentiment = COALESCE(EXCLUDED.sentiment, entity_mentions.sentiment)
        RETURNING id, (xmax = 0) as is_new
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                query,
                mention.entity_id,
                mention.content_type,
                mention.content_id,
                mention.context,
                mention.sentiment,
                mention.mentioned_at,
            )
            return row["id"], row["is_new"]

    async def get_by_entity(
        self,
        entity_id: int,
        content_type: str | None = None,
        limit: int = 50,
    ) -> list[EntityMention]:
        """Get mentions for an entity."""
        conditions = ["entity_id = $1"]
        params: list[Any] = [entity_id]

        if content_type:
            conditions.append(f"content_type = ${len(params) + 1}")
            params.append(content_type)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, entity_id, content_type, content_id,
               context, sentiment, mentioned_at, created_at
        FROM entity_mentions
        WHERE {where_clause}
        ORDER BY mentioned_at DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_mention(row) for row in rows]

    async def count_by_entity(
        self,
        entity_id: int,
        since: datetime | None = None,
    ) -> int:
        """Count mentions for an entity, optionally since a date."""
        conditions = ["entity_id = $1"]
        params: list[Any] = [entity_id]

        if since:
            conditions.append(f"mentioned_at >= ${len(params) + 1}")
            params.append(since)

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM entity_mentions WHERE {where_clause}"

        return await self.db.fetchval(query, *params)

    def _row_to_mention(self, row: asyncpg.Record) -> EntityMention:
        """Convert database row to EntityMention model."""
        return EntityMention(
            id=row["id"],
            entity_id=row["entity_id"],
            content_type=row["content_type"],
            content_id=row["content_id"],
            context=row["context"],
            sentiment=row["sentiment"],
            mentioned_at=row["mentioned_at"],
            created_at=row["created_at"],
        )


class VersionRepository:
    """Repository for entity version history."""

    def __init__(self, db: Database):
        self.db = db

    async def create_version(
        self,
        entity_id: int,
        name: str,
        description: str | None,
        properties: dict,
        change_type: str,
        change_source: str | None = None,
        changed_fields: list[str] | None = None,
    ) -> int:
        """Create a new version record. Returns the version ID."""
        # Get next version number
        version_num = await self.db.fetchval(
            """
            SELECT COALESCE(MAX(version), 0) + 1
            FROM entity_versions WHERE entity_id = $1
            """,
            entity_id,
        )

        # Close previous version
        await self.db.execute(
            """
            UPDATE entity_versions
            SET valid_until = NOW()
            WHERE entity_id = $1 AND valid_until IS NULL
            """,
            entity_id,
        )

        # Create new version
        query = """
        INSERT INTO entity_versions (
            entity_id, version, name, description, properties,
            change_type, change_source, changed_fields
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """
        async with self.db.acquire() as conn:
            version_id = await conn.fetchval(
                query,
                entity_id,
                version_num,
                name,
                description,
                properties,
                change_type,
                change_source,
                changed_fields or [],
            )
            return version_id

    async def get_history(
        self,
        entity_id: int,
        limit: int = 10,
    ) -> list[EntityVersion]:
        """Get version history for an entity."""
        query = """
        SELECT id, entity_id, version, name, description, properties,
               change_type, change_source, changed_fields,
               valid_from, valid_until, created_at
        FROM entity_versions
        WHERE entity_id = $1
        ORDER BY version DESC
        LIMIT $2
        """
        rows = await self.db.fetch(query, entity_id, limit)
        return [self._row_to_version(row) for row in rows]

    async def get_at_time(
        self,
        entity_id: int,
        timestamp: datetime,
    ) -> EntityVersion | None:
        """Get entity state at a specific point in time."""
        query = """
        SELECT id, entity_id, version, name, description, properties,
               change_type, change_source, changed_fields,
               valid_from, valid_until, created_at
        FROM entity_versions
        WHERE entity_id = $1
          AND valid_from <= $2
          AND (valid_until IS NULL OR valid_until > $2)
        ORDER BY version DESC
        LIMIT 1
        """
        row = await self.db.fetchrow(query, entity_id, timestamp)
        if row is None:
            return None
        return self._row_to_version(row)

    def _row_to_version(self, row: asyncpg.Record) -> EntityVersion:
        """Convert database row to EntityVersion model."""
        return EntityVersion(
            id=row["id"],
            entity_id=row["entity_id"],
            version=row["version"],
            name=row["name"],
            description=row["description"],
            properties=row["properties"] if row["properties"] else {},
            change_type=row["change_type"],
            change_source=row["change_source"],
            changed_fields=row["changed_fields"] if row["changed_fields"] else [],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            created_at=row["created_at"],
        )
