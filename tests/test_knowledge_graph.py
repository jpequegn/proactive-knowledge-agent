"""Tests for knowledge graph schema and repositories."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database import Database
from src.world_model import (
    Company,
    Concept,
    Domain,
    Entity,
    EntityMention,
    EntityRepository,
    EntityType,
    Event,
    MentionRepository,
    Metric,
    Person,
    Relationship,
    RelationshipRepository,
    RelationshipType,
    Technology,
    VersionRepository,
    create_entity,
    validate_relationship,
)


class TestEntityModels:
    """Tests for entity models."""

    def test_entity_type_enum(self):
        """Test EntityType enum values."""
        assert EntityType.TECHNOLOGY.value == "technology"
        assert EntityType.COMPANY.value == "company"
        assert EntityType.PERSON.value == "person"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.METRIC.value == "metric"
        assert EntityType.EVENT.value == "event"

    def test_domain_enum(self):
        """Test Domain enum values."""
        assert Domain.TECH.value == "tech"
        assert Domain.FITNESS.value == "fitness"
        assert Domain.FINANCE.value == "finance"
        assert Domain.GENERAL.value == "general"

    def test_base_entity_creation(self):
        """Test creating a base entity."""
        entity = Entity(
            name="Test Entity",
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            description="A test entity",
            confidence=0.9,
        )

        assert entity.name == "Test Entity"
        assert entity.entity_type == EntityType.TECHNOLOGY
        assert entity.domain == Domain.TECH
        assert entity.confidence == 0.9
        assert entity.mention_count == 1
        assert entity.first_seen is not None
        assert entity.last_seen is not None

    def test_technology_entity(self):
        """Test Technology entity with specific attributes."""
        tech = Technology(
            name="Python",
            category="language",
            version="3.11",
            github_url="https://github.com/python/cpython",
        )

        assert tech.entity_type == EntityType.TECHNOLOGY
        assert tech.domain == Domain.TECH
        assert tech.category == "language"
        assert tech.version == "3.11"

    def test_company_entity(self):
        """Test Company entity with specific attributes."""
        company = Company(
            name="Anthropic",
            industry="AI",
            company_type="startup",
            stock_symbol=None,
            domain=Domain.TECH,
        )

        assert company.entity_type == EntityType.COMPANY
        assert company.industry == "AI"
        assert company.company_type == "startup"

    def test_person_entity(self):
        """Test Person entity with specific attributes."""
        person = Person(
            name="Dario Amodei",
            role="founder",
            title="CEO",
            organization="Anthropic",
        )

        assert person.entity_type == EntityType.PERSON
        assert person.role == "founder"
        assert person.title == "CEO"

    def test_concept_entity(self):
        """Test Concept entity with specific attributes."""
        concept = Concept(
            name="Machine Learning",
            category="trend",
            parent_concept="Artificial Intelligence",
        )

        assert concept.entity_type == EntityType.CONCEPT
        assert concept.category == "trend"

    def test_metric_entity(self):
        """Test Metric entity with specific attributes."""
        metric = Metric(
            name="CTL",
            domain=Domain.FITNESS,
            unit="TSS",
            value=75.5,
            value_type="current",
        )

        assert metric.entity_type == EntityType.METRIC
        assert metric.domain == Domain.FITNESS
        assert metric.unit == "TSS"
        assert metric.value == 75.5

    def test_event_entity(self):
        """Test Event entity with specific attributes."""
        event = Event(
            name="GPT-4 Release",
            domain=Domain.TECH,
            event_type="announcement",
            event_date=datetime(2023, 3, 14, tzinfo=UTC),
            significance="major",
        )

        assert event.entity_type == EntityType.EVENT
        assert event.event_type == "announcement"
        assert event.significance == "major"

    def test_create_entity_factory(self):
        """Test factory function for creating entities."""
        tech = create_entity(
            EntityType.TECHNOLOGY,
            name="Rust",
            category="language",
        )
        assert isinstance(tech, Technology)
        assert tech.name == "Rust"

        company = create_entity(
            EntityType.COMPANY,
            name="OpenAI",
            industry="AI",
        )
        assert isinstance(company, Company)

    def test_create_entity_invalid_type(self):
        """Test factory with invalid type raises error."""
        with pytest.raises(ValueError):
            create_entity("invalid_type", name="Test")  # type: ignore

    def test_entity_properties(self):
        """Test flexible properties dictionary."""
        entity = Entity(
            name="Test",
            entity_type=EntityType.TECHNOLOGY,
            properties={"custom_field": "value", "score": 42},
        )

        assert entity.properties["custom_field"] == "value"
        assert entity.properties["score"] == 42


class TestRelationshipModels:
    """Tests for relationship models."""

    def test_relationship_type_enum(self):
        """Test RelationshipType enum values."""
        assert RelationshipType.USED_BY.value == "used_by"
        assert RelationshipType.FOUNDED.value == "founded"
        assert RelationshipType.ANNOUNCED.value == "announced"
        assert RelationshipType.RELATES_TO.value == "relates_to"
        assert RelationshipType.IMPACTS.value == "impacts"

    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            relationship_type=RelationshipType.USED_BY,
            source_entity_id=1,
            target_entity_id=2,
            weight=0.8,
            confidence=0.95,
        )

        assert rel.relationship_type == RelationshipType.USED_BY
        assert rel.source_entity_id == 1
        assert rel.target_entity_id == 2
        assert rel.weight == 0.8
        assert rel.confidence == 0.95
        assert rel.valid_until is None  # Still valid

    def test_relationship_temporal_attributes(self):
        """Test relationship temporal tracking."""
        now = datetime.now(UTC)
        rel = Relationship(
            relationship_type=RelationshipType.WORKS_AT,
            source_entity_id=1,
            target_entity_id=2,
            valid_from=now,
        )

        assert rel.valid_from == now
        assert rel.valid_until is None  # Currently valid
        assert rel.first_seen is not None

    def test_validate_relationship_valid(self):
        """Test relationship validation for valid pairs."""
        assert validate_relationship(
            RelationshipType.USED_BY, "technology", "company"
        )
        assert validate_relationship(
            RelationshipType.FOUNDED, "person", "company"
        )
        assert validate_relationship(
            RelationshipType.IMPACTS, "event", "technology"
        )
        assert validate_relationship(
            RelationshipType.IMPACTS, "event", "metric"
        )

    def test_validate_relationship_invalid(self):
        """Test relationship validation for invalid pairs."""
        assert not validate_relationship(
            RelationshipType.USED_BY, "person", "company"
        )
        assert not validate_relationship(
            RelationshipType.FOUNDED, "company", "person"
        )
        assert not validate_relationship(
            RelationshipType.IMPACTS, "technology", "event"
        )

    def test_entity_mention(self):
        """Test EntityMention model."""
        mention = EntityMention(
            entity_id=1,
            content_type="article",
            content_id=42,
            context="mentioned in paragraph about AI",
            sentiment=0.8,
        )

        assert mention.entity_id == 1
        assert mention.content_type == "article"
        assert mention.sentiment == 0.8


class TestEntityRepository:
    """Tests for EntityRepository."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        db.fetch = AsyncMock()
        db.fetchrow = AsyncMock()
        db.fetchval = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_create_entity(self, mock_db):
        """Test creating an entity."""
        repo = EntityRepository(mock_db)

        entity = Technology(name="React", category="framework")
        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchval.return_value = 1

        entity_id = await repo.create(entity)

        assert entity_id == 1
        conn.fetchval.assert_called_once()
        call_args = conn.fetchval.call_args
        assert "INSERT INTO entities" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_upsert_entity(self, mock_db):
        """Test upserting an entity."""
        repo = EntityRepository(mock_db)

        entity = Technology(name="Vue", category="framework")
        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchrow.return_value = {"id": 2, "is_new": True}

        entity_id, is_new = await repo.upsert(entity)

        assert entity_id == 2
        assert is_new is True
        call_args = conn.fetchrow.call_args
        assert "ON CONFLICT (name, entity_type)" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db):
        """Test getting entity by ID."""
        repo = EntityRepository(mock_db)

        mock_db.fetchrow.return_value = {
            "id": 1,
            "external_id": "ext-1",
            "name": "TypeScript",
            "entity_type": "technology",
            "domain": "tech",
            "description": "JS superset",
            "source": "article",
            "source_url": "http://example.com",
            "confidence": 0.95,
            "mention_count": 10,
            "first_seen": datetime.now(UTC),
            "last_seen": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "properties": {"version": "5.0"},
        }

        entity = await repo.get_by_id(1)

        assert entity is not None
        assert entity.name == "TypeScript"
        assert entity.entity_type == EntityType.TECHNOLOGY
        assert entity.properties["version"] == "5.0"

    @pytest.mark.asyncio
    async def test_find_by_type(self, mock_db):
        """Test finding entities by type."""
        repo = EntityRepository(mock_db)

        mock_db.fetch.return_value = [
            {
                "id": 1,
                "external_id": None,
                "name": "Python",
                "entity_type": "technology",
                "domain": "tech",
                "description": None,
                "source": None,
                "source_url": None,
                "confidence": 1.0,
                "mention_count": 100,
                "first_seen": datetime.now(UTC),
                "last_seen": datetime.now(UTC),
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "properties": {},
            }
        ]

        entities = await repo.find_by_type(EntityType.TECHNOLOGY)

        assert len(entities) == 1
        assert entities[0].name == "Python"

    @pytest.mark.asyncio
    async def test_search_by_name(self, mock_db):
        """Test searching entities by name."""
        repo = EntityRepository(mock_db)

        mock_db.fetch.return_value = []
        entities = await repo.search_by_name("nonexistent")

        assert len(entities) == 0
        mock_db.fetch.assert_called_once()
        call_args = mock_db.fetch.call_args
        assert "ILIKE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_count(self, mock_db):
        """Test counting entities."""
        repo = EntityRepository(mock_db)

        mock_db.fetchval.return_value = 42
        count = await repo.count(entity_type=EntityType.COMPANY)

        assert count == 42


class TestRelationshipRepository:
    """Tests for RelationshipRepository."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        db.fetch = AsyncMock()
        db.fetchrow = AsyncMock()
        db.fetchval = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_create_relationship(self, mock_db):
        """Test creating a relationship."""
        repo = RelationshipRepository(mock_db)

        rel = Relationship(
            relationship_type=RelationshipType.USED_BY,
            source_entity_id=1,
            target_entity_id=2,
        )
        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchval.return_value = 1

        rel_id = await repo.create(rel)

        assert rel_id == 1
        call_args = conn.fetchval.call_args
        assert "INSERT INTO relationships" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_outgoing(self, mock_db):
        """Test getting outgoing relationships."""
        repo = RelationshipRepository(mock_db)

        now = datetime.now(UTC)
        mock_db.fetch.return_value = [
            {
                "id": 1,
                "relationship_type": "used_by",
                "source_entity_id": 1,
                "target_entity_id": 2,
                "weight": 0.9,
                "confidence": 1.0,
                "source": None,
                "source_url": None,
                "first_seen": now,
                "last_seen": now,
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "updated_at": now,
                "properties": {},
            }
        ]

        rels = await repo.get_outgoing(1)

        assert len(rels) == 1
        assert rels[0].relationship_type == RelationshipType.USED_BY
        assert rels[0].target_entity_id == 2


class TestMentionRepository:
    """Tests for MentionRepository."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        db.fetch = AsyncMock()
        db.fetchval = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_create_mention(self, mock_db):
        """Test creating a mention."""
        repo = MentionRepository(mock_db)

        mention = EntityMention(
            entity_id=1,
            content_type="article",
            content_id=10,
            sentiment=0.5,
        )
        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchval.return_value = 1

        mention_id = await repo.create(mention)

        assert mention_id == 1

    @pytest.mark.asyncio
    async def test_count_by_entity(self, mock_db):
        """Test counting mentions for an entity."""
        repo = MentionRepository(mock_db)

        mock_db.fetchval.return_value = 15
        count = await repo.count_by_entity(1)

        assert count == 15


class TestVersionRepository:
    """Tests for VersionRepository."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        db.fetch = AsyncMock()
        db.fetchrow = AsyncMock()
        db.fetchval = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_create_version(self, mock_db):
        """Test creating a version record."""
        repo = VersionRepository(mock_db)

        mock_db.fetchval.side_effect = [2, 1]  # Version 2, then version ID 1
        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchval.return_value = 1

        version_id = await repo.create_version(
            entity_id=1,
            name="Test Entity",
            description="Updated description",
            properties={"key": "value"},
            change_type="updated",
            changed_fields=["description"],
        )

        assert version_id == 1

    @pytest.mark.asyncio
    async def test_get_history(self, mock_db):
        """Test getting version history."""
        repo = VersionRepository(mock_db)

        now = datetime.now(UTC)
        mock_db.fetch.return_value = [
            {
                "id": 2,
                "entity_id": 1,
                "version": 2,
                "name": "Entity v2",
                "description": "Updated",
                "properties": {},
                "change_type": "updated",
                "change_source": "article",
                "changed_fields": ["description"],
                "valid_from": now,
                "valid_until": None,
                "created_at": now,
            },
            {
                "id": 1,
                "entity_id": 1,
                "version": 1,
                "name": "Entity v1",
                "description": "Initial",
                "properties": {},
                "change_type": "created",
                "change_source": None,
                "changed_fields": [],
                "valid_from": now,
                "valid_until": now,
                "created_at": now,
            },
        ]

        history = await repo.get_history(1)

        assert len(history) == 2
        assert history[0].version == 2
        assert history[1].version == 1


class TestKnowledgeGraphSchema:
    """Tests for schema SQL generation."""

    def test_schema_contains_entities_table(self):
        """Test schema includes entities table."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        assert "CREATE TABLE IF NOT EXISTS entities" in schema
        assert "entity_type TEXT NOT NULL" in schema
        assert "domain TEXT DEFAULT 'general'" in schema

    def test_schema_contains_relationships_table(self):
        """Test schema includes relationships table."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        assert "CREATE TABLE IF NOT EXISTS relationships" in schema
        assert "source_entity_id INT NOT NULL" in schema
        assert "target_entity_id INT NOT NULL" in schema

    def test_schema_contains_mentions_table(self):
        """Test schema includes entity_mentions table."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        assert "CREATE TABLE IF NOT EXISTS entity_mentions" in schema
        assert "content_type TEXT NOT NULL" in schema

    def test_schema_contains_versions_table(self):
        """Test schema includes entity_versions table."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        assert "CREATE TABLE IF NOT EXISTS entity_versions" in schema
        assert "change_type TEXT NOT NULL" in schema

    def test_schema_contains_temporal_attributes(self):
        """Test schema includes temporal attributes per issue requirements."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        # Entities temporal
        assert "first_seen TIMESTAMPTZ" in schema
        assert "last_seen TIMESTAMPTZ" in schema
        # Relationships temporal
        assert "valid_from TIMESTAMPTZ" in schema
        assert "valid_until TIMESTAMPTZ" in schema

    def test_schema_contains_vector_support(self):
        """Test schema includes vector embedding support."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        assert "embedding vector(1536)" in schema
        assert "vector_cosine_ops" in schema

    def test_schema_contains_correlations_table(self):
        """Test schema includes correlations for cross-domain queries."""
        from src.world_model.schema import get_knowledge_graph_schema

        schema = get_knowledge_graph_schema()
        assert "CREATE TABLE IF NOT EXISTS correlations" in schema
        assert "correlation_value FLOAT NOT NULL" in schema
