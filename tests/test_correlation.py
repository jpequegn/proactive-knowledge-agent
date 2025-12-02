"""Tests for cross-source correlation service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.world_model.correlation import (
    Correlation,
    CorrelationRepository,
    CorrelationResult,
    CorrelationService,
    CorrelationStrength,
    CorrelationType,
    CrossSourceLink,
    _rank_data,
    pearson_correlation,
    spearman_correlation,
)
from src.world_model.entities import Domain, EntityType


class TestStatisticalFunctions:
    """Tests for statistical correlation functions."""

    def test_pearson_perfect_positive(self):
        """Test Pearson correlation with perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2x
        r, p = pearson_correlation(x, y)
        assert r == pytest.approx(1.0, rel=0.01)
        assert p < 0.05

    def test_pearson_perfect_negative(self):
        """Test Pearson correlation with perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]  # y = -2x + 12
        r, p = pearson_correlation(x, y)
        assert r == pytest.approx(-1.0, rel=0.01)
        assert p < 0.05

    def test_pearson_no_correlation(self):
        """Test Pearson correlation with uncorrelated data."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 2.0, 4.0, 1.0, 3.0]  # Random order
        r, p = pearson_correlation(x, y)
        assert abs(r) < 0.5  # Should be weak or no correlation
        assert p > 0.1  # Not significant

    def test_pearson_insufficient_data(self):
        """Test Pearson returns 0, 1 with insufficient data."""
        r, p = pearson_correlation([1.0, 2.0], [3.0, 4.0])  # Only 2 points
        assert r == 0.0
        assert p == 1.0

    def test_pearson_mismatched_lengths(self):
        """Test Pearson handles mismatched array lengths."""
        r, p = pearson_correlation([1.0, 2.0, 3.0], [4.0, 5.0])
        assert r == 0.0
        assert p == 1.0

    def test_pearson_zero_variance(self):
        """Test Pearson handles zero variance data."""
        x = [5.0, 5.0, 5.0, 5.0, 5.0]  # No variance
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        r, p = pearson_correlation(x, y)
        assert r == 0.0
        assert p == 1.0

    def test_spearman_monotonic(self):
        """Test Spearman correlation with monotonic relationship."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]  # y = x^2 (monotonic but not linear)
        r, p = spearman_correlation(x, y)
        assert r == pytest.approx(1.0, rel=0.01)  # Perfect monotonic
        assert p < 0.05

    def test_spearman_with_ties(self):
        """Test Spearman handles tied values correctly."""
        x = [1.0, 2.0, 2.0, 4.0, 5.0]  # Tie at 2.0
        y = [1.0, 3.0, 2.0, 4.0, 5.0]
        r, p = spearman_correlation(x, y)
        assert abs(r) <= 1.0  # Valid correlation value
        assert 0.0 <= p <= 1.0

    def test_rank_data_simple(self):
        """Test rank data function."""
        data = [3.0, 1.0, 4.0, 1.5, 2.0]
        ranks = _rank_data(data)
        assert ranks[1] == 1.0  # 1.0 is smallest
        assert ranks[3] == 2.0  # 1.5 is second
        assert ranks[4] == 3.0  # 2.0 is third
        assert ranks[0] == 4.0  # 3.0 is fourth
        assert ranks[2] == 5.0  # 4.0 is largest

    def test_rank_data_with_ties(self):
        """Test rank data with tied values."""
        data = [1.0, 2.0, 2.0, 4.0]  # Two 2.0s
        ranks = _rank_data(data)
        assert ranks[0] == 1.0  # 1.0 is rank 1
        assert ranks[1] == 2.5  # First 2.0 gets average of 2 and 3
        assert ranks[2] == 2.5  # Second 2.0 gets same average
        assert ranks[3] == 4.0  # 4.0 is rank 4


class TestCorrelationModel:
    """Tests for Correlation model."""

    def test_correlation_defaults(self):
        """Test Correlation with default values."""
        corr = Correlation()
        assert corr.correlation_type == CorrelationType.MENTION_FREQUENCY
        assert corr.correlation_value == 0.0
        assert corr.significance == 1.0
        assert corr.sample_size == 0

    def test_correlation_strength_very_strong(self):
        """Test correlation strength classification - very strong."""
        corr = Correlation(correlation_value=0.85)
        assert corr.strength == CorrelationStrength.VERY_STRONG

        corr_neg = Correlation(correlation_value=-0.85)
        assert corr_neg.strength == CorrelationStrength.VERY_STRONG

    def test_correlation_strength_strong(self):
        """Test correlation strength classification - strong."""
        corr = Correlation(correlation_value=0.65)
        assert corr.strength == CorrelationStrength.STRONG

    def test_correlation_strength_moderate(self):
        """Test correlation strength classification - moderate."""
        corr = Correlation(correlation_value=0.45)
        assert corr.strength == CorrelationStrength.MODERATE

    def test_correlation_strength_weak(self):
        """Test correlation strength classification - weak."""
        corr = Correlation(correlation_value=0.25)
        assert corr.strength == CorrelationStrength.WEAK

    def test_correlation_strength_negligible(self):
        """Test correlation strength classification - negligible."""
        corr = Correlation(correlation_value=0.1)
        assert corr.strength == CorrelationStrength.NEGLIGIBLE

    def test_is_significant_true(self):
        """Test significance check - significant."""
        corr = Correlation(significance=0.01)
        assert corr.is_significant is True

    def test_is_significant_false(self):
        """Test significance check - not significant."""
        corr = Correlation(significance=0.10)
        assert corr.is_significant is False

    def test_correlation_direction(self):
        """Test correlation direction property."""
        assert Correlation(correlation_value=0.5).direction == "positive"
        assert Correlation(correlation_value=-0.5).direction == "negative"
        assert Correlation(correlation_value=0.0).direction == "none"


class TestCrossSourceLink:
    """Tests for CrossSourceLink model."""

    def test_cross_source_link_creation(self):
        """Test creating a CrossSourceLink."""
        link = CrossSourceLink(
            entity_id=1,
            entity_name="OpenAI",
            entity_type=EntityType.COMPANY,
            sources=["article", "podcast"],
            mention_counts={"article": 10, "podcast": 5},
            first_seen_per_source={
                "article": datetime(2024, 1, 1),
                "podcast": datetime(2024, 1, 5),
            },
            total_mentions=15,
            source_agreement_score=0.8,
        )
        assert link.entity_name == "OpenAI"
        assert len(link.sources) == 2
        assert link.total_mentions == 15


class TestCorrelationType:
    """Tests for CorrelationType enum."""

    def test_correlation_types(self):
        """Test all correlation types exist."""
        assert CorrelationType.MENTION_FREQUENCY == "mention_frequency"
        assert CorrelationType.MENTION_TIMING == "mention_timing"
        assert CorrelationType.TECH_MARKET == "tech_market"
        assert CorrelationType.FITNESS_SLEEP == "fitness_sleep"
        assert CorrelationType.CROSS_SOURCE == "cross_source"


class TestCorrelationRepository:
    """Tests for CorrelationRepository."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock Database."""
        db = MagicMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchrow = AsyncMock(return_value=None)
        db.fetchval = AsyncMock(return_value=1)
        db.execute = AsyncMock(return_value="DELETE 0")
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        conn.fetchval = AsyncMock(return_value=1)
        conn.fetchrow = AsyncMock(return_value={"id": 1, "is_new": True})
        return db

    @pytest.fixture
    def repo(self, mock_db):
        """Create a CorrelationRepository with mock db."""
        return CorrelationRepository(db=mock_db)

    @pytest.mark.asyncio
    async def test_create(self, repo, mock_db):
        """Test creating a correlation."""
        corr = Correlation(
            source_entity_id=1,
            target_entity_id=2,
            correlation_type=CorrelationType.MENTION_FREQUENCY,
            correlation_value=0.75,
            significance=0.01,
        )

        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchval.return_value = 42

        corr_id = await repo.create(corr)
        assert corr_id == 42

    @pytest.mark.asyncio
    async def test_upsert(self, repo, mock_db):
        """Test upserting a correlation."""
        corr = Correlation(
            source_entity_id=1,
            target_entity_id=2,
            correlation_type=CorrelationType.MENTION_FREQUENCY,
            correlation_value=0.75,
            significance=0.01,
        )

        corr_id, is_new = await repo.upsert(corr)
        assert corr_id == 1
        assert is_new is True

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repo, mock_db):
        """Test getting non-existent correlation."""
        result = await repo.get_by_id(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_for_entity_empty(self, repo, mock_db):
        """Test getting correlations for entity with no results."""
        results = await repo.get_for_entity(entity_id=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_significant_empty(self, repo, mock_db):
        """Test getting significant correlations with no results."""
        results = await repo.get_significant()
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_old(self, repo, mock_db):
        """Test deleting old correlations."""
        mock_db.execute.return_value = "DELETE 5"
        count = await repo.delete_old(older_than_days=90)
        assert count == 5


class TestCorrelationService:
    """Tests for CorrelationService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock Database."""
        db = MagicMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchrow = AsyncMock(return_value=None)
        db.fetchval = AsyncMock(return_value=0)
        db.execute = AsyncMock()
        db.acquire = MagicMock()
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        conn.fetchval = AsyncMock(return_value=1)
        conn.fetchrow = AsyncMock(return_value={"id": 1, "is_new": True})
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create a CorrelationService with mock db."""
        return CorrelationService(db=mock_db)

    @pytest.mark.asyncio
    async def test_find_cross_source_entities_empty(self, service, mock_db):
        """Test finding cross-source entities with no data."""
        mock_db.fetch.return_value = []
        links = await service.find_cross_source_entities()
        assert links == []

    @pytest.mark.asyncio
    async def test_find_cross_source_entities_with_results(self, service, mock_db):
        """Test finding cross-source entities with data."""
        mock_db.fetch.return_value = [
            {
                "entity_id": 1,
                "name": "OpenAI",
                "entity_type": "company",
                "source_count": 3,
                "total_mentions": 15,
                "sources": ["article", "podcast", "market"],
                "mention_counts": {"article": 10, "podcast": 3, "market": 2},
                "first_seen": {
                    "article": "2024-01-01T00:00:00",
                    "podcast": "2024-01-05T00:00:00",
                    "market": "2024-01-10T00:00:00",
                },
            }
        ]

        links = await service.find_cross_source_entities(min_sources=2)

        assert len(links) == 1
        assert links[0].entity_name == "OpenAI"
        assert len(links[0].sources) == 3
        assert links[0].total_mentions == 15

    @pytest.mark.asyncio
    async def test_detect_mention_correlations_empty(self, service, mock_db):
        """Test detecting mention correlations with no data."""
        mock_db.fetch.return_value = []
        correlations = await service.detect_mention_correlations()
        assert correlations == []

    @pytest.mark.asyncio
    async def test_detect_mention_correlations_with_data(self, service, mock_db):
        """Test detecting mention correlations with correlated data."""
        # Create mock mention data for two entities with correlated patterns
        now = datetime.utcnow()
        mock_data = []

        for day_offset in range(30):
            day = (now - timedelta(days=day_offset)).date()
            # Entity 1: base pattern
            mock_data.append({
                "entity_id": 1,
                "day": day,
                "count": 10 + day_offset,
            })
            # Entity 2: correlated pattern (similar increase)
            mock_data.append({
                "entity_id": 2,
                "day": day,
                "count": 5 + day_offset,
            })

        mock_db.fetch.return_value = mock_data

        correlations = await service.detect_mention_correlations(
            window_days=30,
            min_mentions=5,
        )

        # Should find correlation between entities 1 and 2
        assert len(correlations) >= 1
        # Should be positive correlation
        assert correlations[0].correlation_value > 0

    @pytest.mark.asyncio
    async def test_detect_cross_domain_correlations_empty(self, service, mock_db):
        """Test cross-domain correlations with no data."""
        mock_db.fetch.return_value = []
        correlations = await service.detect_cross_domain_correlations(
            source_domain=Domain.TECH,
            target_domain=Domain.FINANCE,
        )
        assert correlations == []

    @pytest.mark.asyncio
    async def test_analyze_correlations(self, service, mock_db):
        """Test full correlation analysis."""
        mock_db.fetch.return_value = []

        result = await service.analyze_correlations(
            window_days=30,
            persist=False,
        )

        assert isinstance(result, CorrelationResult)
        assert result.correlations == []
        assert result.cross_source_links == []

    @pytest.mark.asyncio
    async def test_get_correlated_entities_empty(self, service, mock_db):
        """Test getting correlated entities with no results."""
        mock_db.fetch.return_value = []

        results = await service.get_correlated_entities(entity_id=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_strongest_correlations(self, service, mock_db):
        """Test getting strongest correlations."""
        mock_db.fetch.return_value = []

        results = await service.get_strongest_correlations(limit=10)
        assert results == []

    def test_get_cross_domain_type_tech_market(self, service):
        """Test getting correlation type for tech-finance pair."""
        corr_type = service._get_cross_domain_type(Domain.TECH, Domain.FINANCE)
        assert corr_type == CorrelationType.TECH_MARKET

    def test_get_cross_domain_type_fitness_general(self, service):
        """Test getting correlation type for fitness-general pair."""
        corr_type = service._get_cross_domain_type(Domain.FITNESS, Domain.GENERAL)
        assert corr_type == CorrelationType.FITNESS_SLEEP

    def test_get_cross_domain_type_default(self, service):
        """Test getting default correlation type for unknown pair."""
        corr_type = service._get_cross_domain_type(Domain.GENERAL, Domain.GENERAL)
        assert corr_type == CorrelationType.CROSS_SOURCE


class TestCorrelationResult:
    """Tests for CorrelationResult model."""

    def test_correlation_result_creation(self):
        """Test creating a CorrelationResult."""
        now = datetime.utcnow()
        result = CorrelationResult(
            correlations=[],
            cross_source_links=[],
            analysis_window=(now - timedelta(days=30), now),
            total_entities_analyzed=10,
            significant_correlations=5,
            summary="Test summary",
        )

        assert result.total_entities_analyzed == 10
        assert result.significant_correlations == 5
        assert result.summary == "Test summary"


class TestIntegrationScenarios:
    """Integration-like tests for realistic scenarios."""

    def test_pearson_moderate_correlation(self):
        """Test detecting moderate correlation in realistic data."""
        # Simulate daily mention counts with some noise
        x = [10.0, 12.0, 8.0, 15.0, 18.0, 20.0, 22.0, 19.0, 25.0, 28.0]
        # y follows x with noise
        y = [8.0, 10.0, 9.0, 13.0, 15.0, 18.0, 19.0, 16.0, 22.0, 24.0]

        r, p = pearson_correlation(x, y)

        assert r > 0.8  # Strong positive correlation
        assert p < 0.05  # Statistically significant

    def test_spearman_rank_correlation(self):
        """Test Spearman on ranked data."""
        # Rankings from different sources
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = [2.0, 1.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 10.0]  # Similar but not exact

        r, p = spearman_correlation(x, y)

        assert r > 0.9  # Very high rank correlation
        assert p < 0.05

    def test_correlation_strength_boundary_values(self):
        """Test correlation strength at exact boundaries."""
        corr_08 = Correlation(correlation_value=0.8)
        assert corr_08.strength == CorrelationStrength.VERY_STRONG
        corr_079 = Correlation(correlation_value=0.79)
        assert corr_079.strength == CorrelationStrength.STRONG
        corr_06 = Correlation(correlation_value=0.6)
        assert corr_06.strength == CorrelationStrength.STRONG
        corr_059 = Correlation(correlation_value=0.59)
        assert corr_059.strength == CorrelationStrength.MODERATE
        corr_04 = Correlation(correlation_value=0.4)
        assert corr_04.strength == CorrelationStrength.MODERATE
        corr_039 = Correlation(correlation_value=0.39)
        assert corr_039.strength == CorrelationStrength.WEAK
        corr_02 = Correlation(correlation_value=0.2)
        assert corr_02.strength == CorrelationStrength.WEAK
        corr_019 = Correlation(correlation_value=0.19)
        assert corr_019.strength == CorrelationStrength.NEGLIGIBLE
