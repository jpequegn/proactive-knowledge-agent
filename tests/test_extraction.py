"""Tests for entity extraction pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database import Database
from src.world_model import (
    Domain,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    extracted_to_entity,
    extracted_to_relationship,
)
from src.world_model.extraction import EntityExtractor
from src.world_model.pipeline import (
    EntityDisambiguator,
    ExtractionPipeline,
    PipelineStats,
)


class TestExtractedModels:
    """Tests for extraction result models."""

    def test_extracted_entity_defaults(self):
        """Test ExtractedEntity with minimal fields."""
        entity = ExtractedEntity(
            name="Python",
            entity_type="technology",
        )
        assert entity.name == "Python"
        assert entity.entity_type == "technology"
        assert entity.domain == "general"
        assert entity.confidence == 0.8
        assert entity.attributes == {}

    def test_extracted_entity_full(self):
        """Test ExtractedEntity with all fields."""
        entity = ExtractedEntity(
            name="OpenAI",
            entity_type="company",
            domain="tech",
            description="AI research company",
            confidence=0.95,
            attributes={"industry": "AI", "company_type": "startup"},
        )
        assert entity.name == "OpenAI"
        assert entity.domain == "tech"
        assert entity.confidence == 0.95
        assert entity.attributes["industry"] == "AI"

    def test_extracted_relationship(self):
        """Test ExtractedRelationship model."""
        rel = ExtractedRelationship(
            source_entity="Sam Altman",
            relationship_type="founded",
            target_entity="OpenAI",
            confidence=0.9,
        )
        assert rel.source_entity == "Sam Altman"
        assert rel.relationship_type == "founded"
        assert rel.target_entity == "OpenAI"

    def test_extraction_result(self):
        """Test ExtractionResult model."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Python", entity_type="technology"),
                ExtractedEntity(name="Google", entity_type="company"),
            ],
            relationships=[
                ExtractedRelationship(
                    source_entity="Python",
                    relationship_type="used_by",
                    target_entity="Google",
                )
            ],
            content_summary="Article about Python at Google",
            source_type="article",
            source_id=123,
            token_count=500,
        )
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.source_type == "article"


class TestEntityConversion:
    """Tests for converting extracted entities to Entity models."""

    def test_extracted_to_technology(self):
        """Test converting to Technology entity."""
        extracted = ExtractedEntity(
            name="React",
            entity_type="technology",
            domain="tech",
            description="JavaScript UI library",
            attributes={"category": "framework", "github_url": "https://github.com/facebook/react"},
        )
        entity = extracted_to_entity(extracted, source="article")

        assert entity.name == "React"
        assert entity.entity_type == EntityType.TECHNOLOGY
        assert entity.domain == Domain.TECH
        assert entity.source == "article"
        # Technology-specific
        assert hasattr(entity, "category")
        assert entity.category == "framework"

    def test_extracted_to_company(self):
        """Test converting to Company entity."""
        extracted = ExtractedEntity(
            name="Anthropic",
            entity_type="company",
            domain="tech",
            attributes={"industry": "AI", "company_type": "startup"},
        )
        entity = extracted_to_entity(extracted)

        assert entity.entity_type == EntityType.COMPANY
        assert hasattr(entity, "industry")
        assert entity.industry == "AI"

    def test_extracted_to_person(self):
        """Test converting to Person entity."""
        extracted = ExtractedEntity(
            name="Dario Amodei",
            entity_type="person",
            attributes={"role": "founder", "title": "CEO", "organization": "Anthropic"},
        )
        entity = extracted_to_entity(extracted)

        assert entity.entity_type == EntityType.PERSON
        assert entity.role == "founder"
        assert entity.title == "CEO"

    def test_extracted_to_metric(self):
        """Test converting to Metric entity."""
        extracted = ExtractedEntity(
            name="CTL",
            entity_type="metric",
            domain="fitness",
            attributes={"unit": "TSS", "value": 75.5, "value_type": "current"},
        )
        entity = extracted_to_entity(extracted)

        assert entity.entity_type == EntityType.METRIC
        assert entity.domain == Domain.FITNESS
        assert entity.unit == "TSS"
        assert entity.value == 75.5

    def test_extracted_to_event(self):
        """Test converting to Event entity."""
        extracted = ExtractedEntity(
            name="GPT-4 Release",
            entity_type="event",
            attributes={"event_type": "announcement", "location": "San Francisco"},
        )
        entity = extracted_to_entity(extracted)

        assert entity.entity_type == EntityType.EVENT
        assert entity.event_type == "announcement"

    def test_invalid_entity_type_fallback(self):
        """Test fallback for invalid entity type."""
        extracted = ExtractedEntity(
            name="Unknown Thing",
            entity_type="invalid_type",
        )
        entity = extracted_to_entity(extracted)

        # Should fall back to CONCEPT
        assert entity.entity_type == EntityType.CONCEPT

    def test_extracted_to_relationship(self):
        """Test converting extracted relationship."""
        extracted = ExtractedRelationship(
            source_entity="Python",
            relationship_type="used_by",
            target_entity="Google",
            confidence=0.9,
        )
        rel = extracted_to_relationship(
            extracted, source_entity_id=1, target_entity_id=2
        )

        assert rel.source_entity_id == 1
        assert rel.target_entity_id == 2
        assert rel.confidence == 0.9


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock()
        return client

    def test_parse_json_response_clean(self):
        """Test parsing clean JSON response."""
        extractor = EntityExtractor(api_key="test")

        response = (
            '{"entities": [{"name": "Python", "entity_type": "technology"}], '
            '"relationships": []}'
        )
        result = extractor._parse_json_response(response)

        assert "entities" in result
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Python"

    def test_parse_json_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        extractor = EntityExtractor(api_key="test")

        response = """```json
{"entities": [{"name": "React", "entity_type": "technology"}], "relationships": []}
```"""
        result = extractor._parse_json_response(response)

        assert "entities" in result
        assert result["entities"][0]["name"] == "React"

    def test_parse_json_response_invalid(self):
        """Test parsing invalid JSON returns empty result."""
        extractor = EntityExtractor(api_key="test")

        response = "This is not valid JSON at all"
        result = extractor._parse_json_response(response)

        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_extract_empty_content(self):
        """Test extraction with empty content."""
        extractor = EntityExtractor(api_key="test")

        result = await extractor.extract("")

        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_extract_with_mock_response(self):
        """Test extraction with mocked API response."""
        with patch("src.world_model.extraction.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # Mock the response
            mock_response = MagicMock()
            mock_response.content = [
                MagicMock(
                    text=json.dumps(
                        {
                            "entities": [
                                {
                                    "name": "Claude",
                                    "entity_type": "technology",
                                    "domain": "tech",
                                    "confidence": 0.95,
                                }
                            ],
                            "relationships": [],
                            "content_summary": "About Claude AI",
                        }
                    )
                )
            ]
            mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            extractor = EntityExtractor(api_key="test")
            result = await extractor.extract("Claude is an AI assistant by Anthropic")

            assert len(result.entities) == 1
            assert result.entities[0].name == "Claude"
            assert result.token_count == 150


class TestEntityDisambiguator:
    """Tests for EntityDisambiguator."""

    @pytest.fixture
    def mock_entity_repo(self):
        """Create a mock EntityRepository."""
        repo = MagicMock()
        repo.get_by_name_and_type = AsyncMock(return_value=None)
        repo.find_similar = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    async def test_disambiguate_apple_company(self, mock_entity_repo):
        """Test disambiguation of Apple as company with tech context."""
        from src.world_model.entities import Entity

        disambiguator = EntityDisambiguator(entity_repo=mock_entity_repo)

        entity = Entity(
            name="Apple",
            entity_type=EntityType.COMPANY,
            domain=Domain.GENERAL,
        )

        (
            result_entity, existing_id, was_disambiguated
        ) = await disambiguator.disambiguate(
            entity, context="Tim Cook announced new iPhone features at WWDC"
        )

        assert result_entity.domain == Domain.TECH
        assert was_disambiguated is True
        assert existing_id is None

    @pytest.mark.asyncio
    async def test_disambiguate_python_language(self, mock_entity_repo):
        """Test disambiguation of Python as programming language."""
        from src.world_model.entities import Entity

        disambiguator = EntityDisambiguator(entity_repo=mock_entity_repo)

        entity = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.GENERAL,
        )

        (
            result_entity, existing_id, was_disambiguated
        ) = await disambiguator.disambiguate(
            entity, context="Python is great for machine learning with pip packages"
        )

        assert result_entity.domain == Domain.TECH
        assert was_disambiguated is True

    @pytest.mark.asyncio
    async def test_disambiguate_finds_existing(self, mock_entity_repo):
        """Test disambiguation finds existing entity."""
        from src.world_model.entities import Entity

        # Mock existing entity
        existing_entity = MagicMock()
        existing_entity.id = 42
        mock_entity_repo.get_by_name_and_type.return_value = existing_entity

        disambiguator = EntityDisambiguator(entity_repo=mock_entity_repo)

        entity = Entity(
            name="OpenAI",
            entity_type=EntityType.COMPANY,
        )

        (
            result_entity, existing_id, was_disambiguated
        ) = await disambiguator.disambiguate(entity)

        assert existing_id == 42

    @pytest.mark.asyncio
    async def test_disambiguate_no_match(self, mock_entity_repo):
        """Test disambiguation when no existing entity found."""
        from src.world_model.entities import Entity

        disambiguator = EntityDisambiguator(entity_repo=mock_entity_repo)

        entity = Entity(
            name="BrandNewCompany",
            entity_type=EntityType.COMPANY,
        )

        (
            result_entity, existing_id, was_disambiguated
        ) = await disambiguator.disambiguate(entity)

        assert existing_id is None
        assert was_disambiguated is False


class TestPipelineStats:
    """Tests for PipelineStats."""

    def test_initial_stats(self):
        """Test initial stats are zero."""
        stats = PipelineStats()
        assert stats.entities_extracted == 0
        assert stats.entities_new == 0
        assert stats.errors == []

    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = PipelineStats()
        stats.entities_extracted = 5
        stats.entities_new = 3
        stats.tokens_used = 1000

        d = stats.to_dict()

        assert d["entities_extracted"] == 5
        assert d["entities_new"] == 3
        assert d["tokens_used"] == 1000


class TestExtractionPipeline:
    """Tests for ExtractionPipeline."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock Database."""
        db = MagicMock(spec=Database)
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        db.fetch = AsyncMock(return_value=[])
        db.fetchrow = AsyncMock(return_value=None)
        db.fetchval = AsyncMock(return_value=1)
        db.execute = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_process_content_empty(self, mock_db):
        """Test processing empty content."""
        with patch("src.world_model.pipeline.EntityExtractor") as mock_extractor_cls:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=ExtractionResult())
            mock_extractor_cls.return_value = mock_extractor

            pipeline = ExtractionPipeline(db=mock_db)
            stats = await pipeline.process_content(
                content="",
                source_type="article",
                source_id=1,
            )

            assert stats.entities_extracted == 0

    @pytest.mark.asyncio
    async def test_process_article_domain_hints(self, mock_db):
        """Test that article processing applies domain hints correctly."""
        with patch("src.world_model.pipeline.EntityExtractor") as mock_extractor_cls:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=ExtractionResult())
            mock_extractor_cls.return_value = mock_extractor

            pipeline = ExtractionPipeline(db=mock_db)

            # Test tech category
            await pipeline.process_article(
                article_id=1,
                title="Test",
                content="Content",
                source="test",
                category="AI/Technology",
            )

            # Verify extract was called with tech domain hint
            call_args = mock_extractor.extract.call_args
            assert call_args.kwargs.get("domain_hint") == "tech"

    @pytest.mark.asyncio
    async def test_process_article_fitness_domain(self, mock_db):
        """Test fitness category maps to fitness domain."""
        with patch("src.world_model.pipeline.EntityExtractor") as mock_extractor_cls:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(return_value=ExtractionResult())
            mock_extractor_cls.return_value = mock_extractor

            pipeline = ExtractionPipeline(db=mock_db)

            await pipeline.process_article(
                article_id=1,
                title="Test",
                content="Content",
                source="test",
                category="Health & Fitness",
            )

            call_args = mock_extractor.extract.call_args
            assert call_args.kwargs.get("domain_hint") == "fitness"

    @pytest.mark.asyncio
    async def test_pipeline_with_entities(self, mock_db):
        """Test pipeline processing with entities."""
        with patch("src.world_model.pipeline.EntityExtractor") as mock_extractor_cls:
            mock_extractor = MagicMock()
            mock_extractor.extract = AsyncMock(
                return_value=ExtractionResult(
                    entities=[
                        ExtractedEntity(
                            name="TechCorp",
                            entity_type="company",
                            domain="tech",
                            confidence=0.9,
                        )
                    ],
                    relationships=[],
                    token_count=100,
                )
            )
            mock_extractor_cls.return_value = mock_extractor

            # Mock upsert to return new entity
            conn = mock_db.acquire.return_value.__aenter__.return_value
            conn.fetchrow.return_value = {"id": 1, "is_new": True}
            conn.fetchval.return_value = 1

            pipeline = ExtractionPipeline(db=mock_db)
            stats = await pipeline.process_content(
                content="TechCorp is a great company",
                source_type="article",
                source_id=1,
            )

            assert stats.entities_extracted == 1
            assert stats.tokens_used == 100
