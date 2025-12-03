"""Tests for knowledge graph repository."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Entity, Relationship, EntityType, RelationshipType
from src.repositories import KnowledgeGraphRepository


class TestKnowledgeGraphRepository:
    """Tests for KnowledgeGraphRepository."""

    @pytest.fixture
    def mock_db(self):
        with patch("src.repositories.Database") as MockDB:
            mock_instance = MockDB.return_value
            mock_instance.acquire.return_value.__aenter__.return_value = mock_instance
            mock_instance.fetchval = AsyncMock()
            mock_instance.fetchrow = AsyncMock()
            mock_instance.fetch = AsyncMock()
            yield mock_instance

    @pytest.mark.asyncio
    async def test_upsert_entity(self, mock_db):
        """Test upserting an entity."""
        repo = KnowledgeGraphRepository(mock_db)
        
        entity = Entity(
            name="Test Tech",
            type=EntityType.TECHNOLOGY,
            description="A test technology",
            attributes={"key": "value"}
        )
        
        mock_db.fetchval.return_value = entity.id
        
        result_id = await repo.upsert_entity(entity)
        
        assert result_id == entity.id
        mock_db.fetchval.assert_called_once()
        args = mock_db.fetchval.call_args[0]
        assert args[1] == entity.id
        assert args[2] == "Test Tech"
        assert args[3] == "technology"
        assert args[6] == json.dumps(entity.attributes)

    @pytest.mark.asyncio
    async def test_get_entity(self, mock_db):
        """Test getting an entity by ID."""
        repo = KnowledgeGraphRepository(mock_db)
        entity_id = uuid.uuid4()
        
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda x: {
            "id": entity_id,
            "name": "Test Tech",
            "type": "technology",
            "description": "Desc",
            "aliases": ["TT"],
            "attributes": '{"key": "value"}'
        }[x]
        
        mock_db.fetchrow.return_value = mock_row
        
        entity = await repo.get_entity(entity_id)
        
        assert entity is not None
        assert entity.id == entity_id
        assert entity.name == "Test Tech"
        assert entity.type == EntityType.TECHNOLOGY
        assert entity.attributes == {"key": "value"}

    @pytest.mark.asyncio
    async def test_create_relationship(self, mock_db):
        """Test creating a relationship."""
        repo = KnowledgeGraphRepository(mock_db)
        
        rel = Relationship(
            source_entity_id=uuid.uuid4(),
            target_entity_id=uuid.uuid4(),
            type=RelationshipType.USED_BY
        )
        
        mock_db.fetchval.return_value = rel.id
        
        result_id = await repo.create_relationship(rel)
        
        assert result_id == rel.id
        mock_db.fetchval.assert_called_once()
        args = mock_db.fetchval.call_args[0]
        assert args[1] == rel.id
        assert args[4] == "used_by"

    @pytest.mark.asyncio
    async def test_get_related_entities(self, mock_db):
        """Test getting related entities."""
        repo = KnowledgeGraphRepository(mock_db)
        entity_id = uuid.uuid4()
        
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda x: {
            "rel_id": uuid.uuid4(),
            "source_entity_id": entity_id,
            "target_entity_id": uuid.uuid4(),
            "rel_type": "used_by",
            "weight": 1.0,
            "rel_attributes": "{}",
            "entity_id": uuid.uuid4(),
            "name": "Target Entity",
            "entity_type": "company",
            "description": "Desc",
            "aliases": [],
            "entity_attributes": "{}"
        }[x]
        
        mock_db.fetch.return_value = [mock_row]
        
        results = await repo.get_related_entities(entity_id, direction="outgoing")
        
        assert len(results) == 1
        rel, entity = results[0]
        assert rel.type == RelationshipType.USED_BY
        assert entity.name == "Target Entity"
        assert entity.type == EntityType.COMPANY
