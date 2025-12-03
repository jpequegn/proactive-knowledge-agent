"""Tests for temporal reasoning service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.world_model.temporal import (
    Anomaly,
    AnomalyType,
    ChangeEvent,
    ExponentialDecay,
    LinearDecay,
    MentionStats,
    StepDecay,
    TemporalReasoningService,
    TrendDirection,
    TrendResult,
    WhatsNewResult,
)


class TestDecayFunctions:
    """Tests for decay function implementations."""

    def test_exponential_decay_at_zero(self):
        """Test exponential decay returns 1.0 at age 0."""
        decay = ExponentialDecay(half_life_days=7.0)
        assert decay.calculate(0) == pytest.approx(1.0)

    def test_exponential_decay_at_half_life(self):
        """Test exponential decay returns 0.5 at half-life."""
        decay = ExponentialDecay(half_life_days=7.0)
        assert decay.calculate(7.0) == pytest.approx(0.5, rel=0.01)

    def test_exponential_decay_at_two_half_lives(self):
        """Test exponential decay returns 0.25 at two half-lives."""
        decay = ExponentialDecay(half_life_days=7.0)
        assert decay.calculate(14.0) == pytest.approx(0.25, rel=0.01)

    def test_exponential_decay_negative_age(self):
        """Test exponential decay returns 1.0 for negative age."""
        decay = ExponentialDecay(half_life_days=7.0)
        assert decay.calculate(-5.0) == 1.0

    def test_linear_decay_at_zero(self):
        """Test linear decay returns 1.0 at age 0."""
        decay = LinearDecay(max_age_days=30.0)
        assert decay.calculate(0) == 1.0

    def test_linear_decay_at_half(self):
        """Test linear decay returns 0.5 at half max age."""
        decay = LinearDecay(max_age_days=30.0)
        assert decay.calculate(15.0) == pytest.approx(0.5)

    def test_linear_decay_at_max(self):
        """Test linear decay returns 0.0 at max age."""
        decay = LinearDecay(max_age_days=30.0)
        assert decay.calculate(30.0) == 0.0

    def test_linear_decay_beyond_max(self):
        """Test linear decay returns 0.0 beyond max age."""
        decay = LinearDecay(max_age_days=30.0)
        assert decay.calculate(100.0) == 0.0

    def test_linear_decay_negative_age(self):
        """Test linear decay returns 1.0 for negative age."""
        decay = LinearDecay(max_age_days=30.0)
        assert decay.calculate(-5.0) == 1.0

    def test_step_decay_fresh(self):
        """Test step decay returns fresh weight within fresh period."""
        decay = StepDecay(
            fresh_days=1.0,
            recent_days=7.0,
            fresh_weight=1.0,
            recent_weight=0.5,
            old_weight=0.1,
        )
        assert decay.calculate(0.5) == 1.0

    def test_step_decay_recent(self):
        """Test step decay returns recent weight in recent period."""
        decay = StepDecay(
            fresh_days=1.0,
            recent_days=7.0,
            fresh_weight=1.0,
            recent_weight=0.5,
            old_weight=0.1,
        )
        assert decay.calculate(3.0) == 0.5

    def test_step_decay_old(self):
        """Test step decay returns old weight beyond recent period."""
        decay = StepDecay(
            fresh_days=1.0,
            recent_days=7.0,
            fresh_weight=1.0,
            recent_weight=0.5,
            old_weight=0.1,
        )
        assert decay.calculate(10.0) == 0.1

    def test_step_decay_negative_age(self):
        """Test step decay returns fresh weight for negative age."""
        decay = StepDecay()
        assert decay.calculate(-1.0) == 1.0


class TestMentionStats:
    """Tests for MentionStats model."""

    def test_trend_new_entity(self):
        """Test trend is NEW when previous count is 0 but current > 0."""
        stats = MentionStats(
            entity_id=1,
            entity_name="Test",
            period_start=datetime.utcnow() - timedelta(days=7),
            period_end=datetime.utcnow(),
            mention_count=5,
            previous_count=0,
            change_ratio=1.0,
            change_absolute=5,
        )
        assert stats.trend == TrendDirection.NEW

    def test_trend_rising(self):
        """Test trend is RISING when change ratio >= 0.5."""
        stats = MentionStats(
            entity_id=1,
            entity_name="Test",
            period_start=datetime.utcnow() - timedelta(days=7),
            period_end=datetime.utcnow(),
            mention_count=15,
            previous_count=10,
            change_ratio=0.5,
            change_absolute=5,
        )
        assert stats.trend == TrendDirection.RISING

    def test_trend_falling(self):
        """Test trend is FALLING when change ratio <= -0.5."""
        stats = MentionStats(
            entity_id=1,
            entity_name="Test",
            period_start=datetime.utcnow() - timedelta(days=7),
            period_end=datetime.utcnow(),
            mention_count=5,
            previous_count=10,
            change_ratio=-0.5,
            change_absolute=-5,
        )
        assert stats.trend == TrendDirection.FALLING

    def test_trend_stable(self):
        """Test trend is STABLE when change is small."""
        stats = MentionStats(
            entity_id=1,
            entity_name="Test",
            period_start=datetime.utcnow() - timedelta(days=7),
            period_end=datetime.utcnow(),
            mention_count=11,
            previous_count=10,
            change_ratio=0.1,
            change_absolute=1,
        )
        assert stats.trend == TrendDirection.STABLE


class TestTrendResult:
    """Tests for TrendResult model."""

    def test_trend_result_creation(self):
        """Test TrendResult can be created with all fields."""
        result = TrendResult(
            entity_id=1,
            entity_name="Python",
            entity_type="technology",
            domain="tech",
            direction=TrendDirection.RISING,
            change_ratio=0.75,
            current_count=35,
            previous_count=20,
            period_days=7,
            confidence=0.9,
        )
        assert result.entity_name == "Python"
        assert result.direction == TrendDirection.RISING
        assert result.change_ratio == 0.75


class TestAnomaly:
    """Tests for Anomaly model."""

    def test_anomaly_creation(self):
        """Test Anomaly can be created with all fields."""
        anomaly = Anomaly(
            entity_id=1,
            entity_name="Claude",
            anomaly_type=AnomalyType.SPIKE,
            severity=0.8,
            detected_at=datetime.utcnow(),
            expected_value=5.0,
            actual_value=25.0,
            description="Claude had unusual spike in mentions",
            metadata={"std_deviation": 4.0},
        )
        assert anomaly.anomaly_type == AnomalyType.SPIKE
        assert anomaly.severity == 0.8


class TestChangeEvent:
    """Tests for ChangeEvent model."""

    def test_change_event_creation(self):
        """Test ChangeEvent can be created with all fields."""
        event = ChangeEvent(
            entity_id=1,
            entity_name="OpenAI",
            change_type="updated",
            changed_at=datetime.utcnow(),
            changed_fields=["description", "properties.industry"],
            old_values={"description": "Old desc"},
            new_values={"description": "New desc"},
            source="article",
        )
        assert event.change_type == "updated"
        assert "description" in event.changed_fields


class TestWhatsNewResult:
    """Tests for WhatsNewResult model."""

    def test_whats_new_result_creation(self):
        """Test WhatsNewResult can be created with all fields."""
        result = WhatsNewResult(
            since=datetime.utcnow() - timedelta(days=1),
            new_entities=[],
            updated_entities=[],
            trending_entities=[],
            anomalies=[],
            total_mentions=100,
        )
        assert result.total_mentions == 100
        assert len(result.new_entities) == 0


class TestTemporalReasoningService:
    """Tests for TemporalReasoningService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock Database."""
        db = MagicMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchrow = AsyncMock(return_value=None)
        db.fetchval = AsyncMock(return_value=0)
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create a TemporalReasoningService with mock db."""
        return TemporalReasoningService(db=mock_db)

    def test_init_default_decay(self, mock_db):
        """Test service initializes with default exponential decay."""
        service = TemporalReasoningService(db=mock_db)
        assert isinstance(service.decay, ExponentialDecay)

    def test_init_custom_decay(self, mock_db):
        """Test service can use custom decay function."""
        custom_decay = LinearDecay(max_age_days=14.0)
        service = TemporalReasoningService(db=mock_db, decay_function=custom_decay)
        assert isinstance(service.decay, LinearDecay)

    def test_calculate_relevance_fresh(self, service):
        """Test relevance calculation for fresh entity."""
        now = datetime.utcnow()
        relevance = service.calculate_relevance(
            last_seen=now,
            base_score=1.0,
            reference_time=now,
        )
        assert relevance == pytest.approx(1.0)

    def test_calculate_relevance_old(self, service):
        """Test relevance calculation for old entity."""
        now = datetime.utcnow()
        old_time = now - timedelta(days=14)  # Two half-lives
        relevance = service.calculate_relevance(
            last_seen=old_time,
            base_score=1.0,
            reference_time=now,
        )
        # With 7-day half-life, 14 days should give ~0.25
        assert relevance == pytest.approx(0.25, rel=0.1)

    def test_calculate_relevance_with_base_score(self, service):
        """Test relevance calculation scales with base score."""
        now = datetime.utcnow()
        last_seen = now - timedelta(days=7)  # One half-life
        relevance = service.calculate_relevance(
            last_seen=last_seen,
            base_score=2.0,
            reference_time=now,
        )
        # Should be ~1.0 (2.0 * 0.5)
        assert relevance == pytest.approx(1.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_get_changes_since_empty(self, service, mock_db):
        """Test get_changes_since returns empty list when no changes."""
        mock_db.fetch.return_value = []

        changes = await service.get_changes_since(
            since=datetime.utcnow() - timedelta(days=1)
        )

        assert changes == []

    @pytest.mark.asyncio
    async def test_get_changes_since_with_results(self, service, mock_db):
        """Test get_changes_since returns change events."""
        mock_db.fetch.return_value = [
            {
                "entity_id": 1,
                "entity_name": "Test Entity",
                "change_type": "updated",
                "changed_at": datetime.utcnow(),
                "changed_fields": ["description"],
                "change_source": "article",
                "new_values": {"description": "New"},
                "old_values": {"description": "Old"},
            }
        ]

        changes = await service.get_changes_since(
            since=datetime.utcnow() - timedelta(days=1)
        )

        assert len(changes) == 1
        assert changes[0].entity_name == "Test Entity"
        assert changes[0].change_type == "updated"

    @pytest.mark.asyncio
    async def test_get_mention_changes(self, service, mock_db):
        """Test get_mention_changes compares periods correctly."""
        mock_db.fetchrow.return_value = {"name": "Test Entity"}
        # Mock current count = 10, previous count = 5
        mock_db.fetchval.side_effect = [10, 5]

        stats = await service.get_mention_changes(entity_id=1, current_period_days=7)

        assert stats.entity_id == 1
        assert stats.mention_count == 10
        assert stats.previous_count == 5
        assert stats.change_ratio == pytest.approx(1.0)  # 100% increase

    @pytest.mark.asyncio
    async def test_analyze_trends_empty(self, service, mock_db):
        """Test analyze_trends returns empty list when no data."""
        mock_db.fetch.return_value = []

        trends = await service.analyze_trends(period_days=7)

        assert trends == []

    @pytest.mark.asyncio
    async def test_analyze_trends_with_results(self, service, mock_db):
        """Test analyze_trends returns trend results."""
        mock_db.fetch.return_value = [
            {
                "entity_id": 1,
                "entity_name": "Rising Entity",
                "entity_type": "technology",
                "domain": "tech",
                "current_count": 20,
                "previous_count": 10,
                "change_ratio": 1.0,
            },
            {
                "entity_id": 2,
                "entity_name": "Falling Entity",
                "entity_type": "company",
                "domain": "tech",
                "current_count": 5,
                "previous_count": 10,
                "change_ratio": -0.5,
            },
        ]

        trends = await service.analyze_trends(period_days=7)

        assert len(trends) == 2
        assert trends[0].direction == TrendDirection.RISING
        assert trends[1].direction == TrendDirection.FALLING

    @pytest.mark.asyncio
    async def test_get_rising_entities(self, service, mock_db):
        """Test get_rising_entities filters correctly."""
        mock_db.fetch.return_value = [
            {
                "entity_id": 1,
                "entity_name": "Rising",
                "entity_type": "technology",
                "domain": "tech",
                "current_count": 20,
                "previous_count": 10,
                "change_ratio": 1.0,
            },
            {
                "entity_id": 2,
                "entity_name": "Stable",
                "entity_type": "technology",
                "domain": "tech",
                "current_count": 10,
                "previous_count": 10,
                "change_ratio": 0.0,
            },
        ]

        rising = await service.get_rising_entities(period_days=7)

        assert len(rising) == 1
        assert rising[0].entity_name == "Rising"

    @pytest.mark.asyncio
    async def test_get_falling_entities(self, service, mock_db):
        """Test get_falling_entities filters correctly."""
        mock_db.fetch.return_value = [
            {
                "entity_id": 1,
                "entity_name": "Falling",
                "entity_type": "technology",
                "domain": "tech",
                "current_count": 5,
                "previous_count": 20,
                "change_ratio": -0.75,
            },
            {
                "entity_id": 2,
                "entity_name": "Stable",
                "entity_type": "technology",
                "domain": "tech",
                "current_count": 10,
                "previous_count": 10,
                "change_ratio": 0.0,
            },
        ]

        falling = await service.get_falling_entities(period_days=7)

        assert len(falling) == 1
        assert falling[0].entity_name == "Falling"

    @pytest.mark.asyncio
    async def test_detect_anomalies_empty(self, service, mock_db):
        """Test detect_anomalies returns empty list when no anomalies."""
        mock_db.fetch.return_value = []

        anomalies = await service.detect_anomalies(lookback_days=30)

        assert anomalies == []

    @pytest.mark.asyncio
    async def test_detect_anomalies_spike(self, service, mock_db):
        """Test detect_anomalies detects spikes."""
        mock_db.fetch.side_effect = [
            # First call: daily counts with spike
            [
                {
                    "entity_id": 1,
                    "entity_name": "Spiking Entity",
                    "day": datetime.utcnow(),
                    "count": 50,
                    "avg_count": 10.0,
                    "std_count": 5.0,
                    "last_seen_day": datetime.utcnow(),
                }
            ],
            # Second call: silence check (empty)
            [],
        ]

        anomalies = await service.detect_anomalies(lookback_days=30)

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.SPIKE

    @pytest.mark.asyncio
    async def test_whats_new_comprehensive(self, service, mock_db):
        """Test whats_new returns comprehensive results."""
        # Mock various database calls
        mock_db.fetch.return_value = []
        mock_db.fetchval.return_value = 50

        result = await service.whats_new(
            since=datetime.utcnow() - timedelta(days=1),
            include_trends=False,
            include_anomalies=False,
        )

        assert isinstance(result, WhatsNewResult)
        assert result.total_mentions == 50
        assert len(result.new_entities) == 0

    @pytest.mark.asyncio
    async def test_get_entities_with_decay(self, service, mock_db):
        """Test get_entities_with_decay applies decay correctly."""
        now = datetime.utcnow()
        mock_db.fetch.return_value = [
            {
                "id": 1,
                "external_id": None,
                "name": "Fresh Entity",
                "entity_type": "technology",
                "domain": "tech",
                "description": "A fresh entity",
                "source": "test",
                "source_url": None,
                "confidence": 0.9,
                "mention_count": 10,
                "first_seen": now,
                "last_seen": now,
                "created_at": now,
                "updated_at": now,
                "properties": {},
            },
            {
                "id": 2,
                "external_id": None,
                "name": "Old Entity",
                "entity_type": "technology",
                "domain": "tech",
                "description": "An old entity",
                "source": "test",
                "source_url": None,
                "confidence": 0.9,
                "mention_count": 10,
                "first_seen": now - timedelta(days=30),
                "last_seen": now - timedelta(days=14),
                "created_at": now - timedelta(days=30),
                "updated_at": now - timedelta(days=14),
                "properties": {},
            },
        ]

        results = await service.get_entities_with_decay(limit=10)

        assert len(results) == 2
        # Fresh entity should have higher relevance
        fresh_entity, fresh_relevance = results[0]
        old_entity, old_relevance = results[1]
        assert fresh_entity.name == "Fresh Entity"
        assert fresh_relevance > old_relevance


class TestVersionRepositoryExtensions:
    """Tests for extended VersionRepository methods."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock Database."""
        db = MagicMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchrow = AsyncMock(return_value=None)
        db.fetchval = AsyncMock(return_value=0)
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def repo(self, mock_db):
        """Create a VersionRepository with mock db."""
        from src.world_model.repository import VersionRepository

        return VersionRepository(db=mock_db)

    @pytest.mark.asyncio
    async def test_get_changes_since_empty(self, repo, mock_db):
        """Test get_changes_since returns empty list when no changes."""
        mock_db.fetch.return_value = []

        changes = await repo.get_changes_since(
            since=datetime.utcnow() - timedelta(days=1)
        )

        assert changes == []

    @pytest.mark.asyncio
    async def test_get_changes_since_with_filter(self, repo, mock_db):
        """Test get_changes_since filters by change_type."""
        mock_db.fetch.return_value = [
            {
                "id": 1,
                "entity_id": 1,
                "version": 2,
                "name": "Test",
                "description": None,
                "properties": {},
                "change_type": "updated",
                "change_source": "article",
                "changed_fields": ["description"],
                "valid_from": datetime.utcnow(),
                "valid_until": None,
                "created_at": datetime.utcnow(),
            }
        ]

        changes = await repo.get_changes_since(
            since=datetime.utcnow() - timedelta(days=1),
            change_type="updated",
        )

        assert len(changes) == 1
        assert changes[0].change_type == "updated"

    @pytest.mark.asyncio
    async def test_get_version_count(self, repo, mock_db):
        """Test get_version_count returns correct count."""
        mock_db.fetchval.return_value = 5

        count = await repo.get_version_count(entity_id=1)

        assert count == 5

    @pytest.mark.asyncio
    async def test_compare_versions_empty(self, repo, mock_db):
        """Test compare_versions returns empty dict when versions not found."""
        mock_db.fetch.return_value = []

        changes = await repo.compare_versions(entity_id=1, version_a=1, version_b=2)

        assert changes == {}

    @pytest.mark.asyncio
    async def test_compare_versions_with_changes(self, repo, mock_db):
        """Test compare_versions returns changed fields."""
        mock_db.fetch.return_value = [
            {
                "version": 1,
                "name": "Old Name",
                "description": "Old desc",
                "properties": {"key": "old"},
            },
            {
                "version": 2,
                "name": "New Name",
                "description": "New desc",
                "properties": {"key": "new"},
            },
        ]

        changes = await repo.compare_versions(entity_id=1, version_a=1, version_b=2)

        assert "name" in changes
        assert changes["name"] == ("Old Name", "New Name")
        assert "description" in changes
        assert "properties.key" in changes
