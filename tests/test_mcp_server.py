"""Tests for MCP Server integration."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents import AlertLevel, Insight, InsightType
from src.outputs.alerts import Alert, AlertStatus
from src.outputs.mcp_server import (
    PKAMCPServer,
    format_alert,
    format_entity,
    format_insight,
    format_trend,
)
from src.world_model import Domain, Entity, EntityType, TrendDirection, TrendResult

# ============================================================================
# Formatter Tests
# ============================================================================


class TestFormatters:
    """Tests for response formatters."""

    def test_format_entity(self) -> None:
        """Test entity formatting."""
        entity = Entity(
            id=1,
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            description="Programming language",
            confidence=0.95,
            mention_count=100,
            first_seen=datetime(2024, 1, 1, tzinfo=UTC),
            last_seen=datetime(2024, 12, 1, tzinfo=UTC),
            properties={"version": "3.12"},
        )

        result = format_entity(entity)

        assert result["id"] == 1
        assert result["name"] == "Python"
        assert result["type"] == "technology"
        assert result["domain"] == "tech"
        assert result["description"] == "Programming language"
        assert result["confidence"] == 0.95
        assert result["mention_count"] == 100
        assert result["properties"] == {"version": "3.12"}
        assert "2024-01-01" in result["first_seen"]
        assert "2024-12-01" in result["last_seen"]

    def test_format_entity_default_dates(self) -> None:
        """Test entity formatting with default dates."""
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            domain=Domain.GENERAL,
        )

        result = format_entity(entity)

        # Default dates are set automatically
        assert result["first_seen"] is not None
        assert result["last_seen"] is not None

    def test_format_trend(self) -> None:
        """Test trend formatting."""
        trend = TrendResult(
            entity_id=1,
            entity_name="AI",
            entity_type="concept",
            domain="tech",
            direction=TrendDirection.RISING,
            change_ratio=0.75,
            current_count=35,
            previous_count=20,
            period_days=7,
            confidence=0.9,
        )

        result = format_trend(trend)

        assert result["entity_id"] == 1
        assert result["entity_name"] == "AI"
        assert result["direction"] == "rising"
        assert result["change_ratio"] == 75.0  # Converted to percentage
        assert result["current_count"] == 35
        assert result["previous_count"] == 20
        assert result["period_days"] == 7
        assert result["confidence"] == 0.9

    def test_format_alert(self) -> None:
        """Test alert formatting."""
        alert = Alert(
            alert_id="alert_001",
            title="Test Alert",
            description="This is a test",
            level=AlertLevel.ACTION,
            source="test",
            status=AlertStatus.PENDING,
            created_at=datetime(2024, 12, 1, tzinfo=UTC),
        )

        result = format_alert(alert)

        assert result["id"] == "alert_001"
        assert result["title"] == "Test Alert"
        assert result["description"] == "This is a test"
        assert result["level"] == "action"
        assert result["source"] == "test"
        assert result["status"] == "pending"
        assert result["is_actionable"] is True

    def test_format_insight(self) -> None:
        """Test insight formatting."""
        insight = Insight(
            title="Test Insight",
            description="Important finding",
            insight_type=InsightType.TREND_CHANGE,
            level=AlertLevel.WATCH,
            confidence=0.85,
            relevance_score=0.9,
            created_at=datetime(2024, 12, 1, tzinfo=UTC),
            metadata={"key": "value"},
        )

        result = format_insight(insight)

        assert result["title"] == "Test Insight"
        assert result["description"] == "Important finding"
        assert result["insight_type"] == "trend_change"
        assert result["level"] == "watch"
        assert result["confidence"] == 0.85
        # priority_score is computed: 0.5 * 0.85 * 0.9 = 0.3825
        assert 0.38 <= result["priority_score"] <= 0.39
        assert result["metadata"] == {"key": "value"}


# ============================================================================
# MCP Server Tests
# ============================================================================


class TestPKAMCPServer:
    """Tests for PKAMCPServer."""

    def test_server_initialization(self) -> None:
        """Test server initializes correctly."""
        server = PKAMCPServer(database_url="postgresql://test:test@localhost/test")

        assert server.database_url == "postgresql://test:test@localhost/test"
        assert server._db is None
        assert server._entity_repo is None
        assert server.server is not None

    def test_server_default_database_url(self) -> None:
        """Test server uses environment variable for database URL."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://env@localhost"}):
            server = PKAMCPServer()
            assert server.database_url == "postgresql://env@localhost"

    @pytest.mark.asyncio
    async def test_ensure_connected(self) -> None:
        """Test database connection is established on first call."""
        server = PKAMCPServer(database_url="postgresql://test:test@localhost/test")

        with patch.object(
            server, "_db", new_callable=lambda: None
        ), patch(
            "src.outputs.mcp_server.Database"
        ) as mock_db_class:
            mock_db = AsyncMock()
            mock_db_class.return_value = mock_db

            await server._ensure_connected()

            mock_db_class.assert_called_once_with(
                "postgresql://test:test@localhost/test"
            )
            mock_db.connect.assert_called_once()


class TestSearchHandler:
    """Tests for pka_search tool handler."""

    @pytest.mark.asyncio
    async def test_handle_search_basic(self) -> None:
        """Test basic search functionality."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        # Mock entity repository
        mock_repo = AsyncMock()
        mock_entity = Entity(
            id=1,
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            description="Programming language",
        )
        mock_repo.search_by_name.return_value = [mock_entity]
        server._entity_repo = mock_repo

        result = await server._handle_search({"query": "Python"})

        assert result["query"] == "Python"
        assert result["count"] == 1
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Python"

    @pytest.mark.asyncio
    async def test_handle_search_with_filters(self) -> None:
        """Test search with type and domain filters."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_repo = AsyncMock()
        mock_entity = Entity(
            id=1,
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
        )
        mock_repo.search_by_name.return_value = [mock_entity]
        server._entity_repo = mock_repo

        result = await server._handle_search({
            "query": "Python",
            "entity_type": "technology",
            "domain": "tech",
            "limit": 5,
        })

        mock_repo.search_by_name.assert_called_once_with(
            search_text="Python",
            limit=5,
            entity_type=EntityType.TECHNOLOGY,
        )
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_handle_search_no_repo(self) -> None:
        """Test search returns error when repo not connected."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")
        server._entity_repo = None

        result = await server._handle_search({"query": "test"})

        assert "error" in result


class TestTrendsHandler:
    """Tests for pka_trends tool handler."""

    @pytest.mark.asyncio
    async def test_handle_trends_basic(self) -> None:
        """Test basic trends functionality."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_service = AsyncMock()
        mock_trend = TrendResult(
            entity_id=1,
            entity_name="AI",
            entity_type="concept",
            domain="tech",
            direction=TrendDirection.RISING,
            change_ratio=0.5,
            current_count=30,
            previous_count=20,
            period_days=7,
        )
        mock_service.analyze_trends.return_value = [mock_trend]
        server._temporal_service = mock_service

        result = await server._handle_trends({"domain": "tech"})

        assert result["domain"] == "tech"
        assert result["period"] == "week"
        assert result["period_days"] == 7
        assert result["summary"]["rising"] == 1
        assert len(result["trends"]) == 1

    @pytest.mark.asyncio
    async def test_handle_trends_with_period(self) -> None:
        """Test trends with different period."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_service = AsyncMock()
        mock_service.analyze_trends.return_value = []
        server._temporal_service = mock_service

        result = await server._handle_trends({
            "domain": "tech",
            "period": "month",
            "limit": 20,
        })

        assert result["period_days"] == 30
        mock_service.analyze_trends.assert_called_once_with(
            domain=Domain.TECH,
            period_days=30,
            limit=20,
        )


class TestAlertsHandler:
    """Tests for pka_alerts tool handler."""

    @pytest.mark.asyncio
    async def test_handle_alerts_basic(self) -> None:
        """Test basic alerts functionality."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_manager = MagicMock()
        mock_alert = Alert(
            alert_id="alert_001",
            title="Test Alert",
            description="Test",
            level=AlertLevel.ACTION,
            source="test",
        )
        mock_manager.get_pending_alerts.return_value = [mock_alert]
        mock_manager.get_stats.return_value = {
            "total_alerts": 10,
            "pending_count": 5,
            "actionable_count": 3,
        }
        server._alert_manager = mock_manager

        result = await server._handle_alerts({})

        assert result["count"] == 1
        assert result["stats"]["total"] == 10
        assert result["stats"]["pending"] == 5
        assert len(result["alerts"]) == 1

    @pytest.mark.asyncio
    async def test_handle_alerts_filter_by_level(self) -> None:
        """Test alerts filtered by level."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_manager = MagicMock()
        alerts = [
            Alert(
                alert_id="1",
                title="Info",
                description="Info",
                level=AlertLevel.INFO,
                source="test",
            ),
            Alert(
                alert_id="2",
                title="Action",
                description="Action",
                level=AlertLevel.ACTION,
                source="test",
            ),
            Alert(
                alert_id="3",
                title="Urgent",
                description="Urgent",
                level=AlertLevel.URGENT,
                source="test",
            ),
        ]
        mock_manager.get_pending_alerts.return_value = alerts
        mock_manager.get_stats.return_value = {
            "total_alerts": 3,
            "pending_count": 3,
            "actionable_count": 2,
        }
        server._alert_manager = mock_manager

        result = await server._handle_alerts({"level": "action"})

        # Should only include ACTION and URGENT (>= action level)
        assert result["count"] == 2


class TestReportHandler:
    """Tests for pka_report tool handler."""

    @pytest.mark.asyncio
    async def test_handle_report_basic(self) -> None:
        """Test basic report functionality."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_service = AsyncMock()
        mock_whats_new = MagicMock()
        mock_whats_new.new_entities = []
        mock_whats_new.updated_entities = []
        mock_whats_new.new_relationships = []
        mock_service.get_whats_new.return_value = mock_whats_new
        mock_service.analyze_trends.return_value = []
        server._temporal_service = mock_service

        result = await server._handle_report({})

        assert result["type"] == "weekly"
        assert result["period_days"] == 7
        assert "whats_new" in result
        assert "domains" in result


class TestEntityHandler:
    """Tests for pka_entity tool handler."""

    @pytest.mark.asyncio
    async def test_handle_entity_found(self) -> None:
        """Test entity lookup when found."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_repo = AsyncMock()
        mock_entity = Entity(
            id=1,
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            description="Programming language",
        )
        mock_repo.search_by_name.return_value = [mock_entity]
        server._entity_repo = mock_repo

        mock_service = AsyncMock()
        mock_service.analyze_trends.return_value = []
        server._temporal_service = mock_service

        result = await server._handle_entity({"name": "Python"})

        assert result["name"] == "Python"
        assert result["type"] == "technology"

    @pytest.mark.asyncio
    async def test_handle_entity_not_found(self) -> None:
        """Test entity lookup when not found."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_repo = AsyncMock()
        mock_repo.search_by_name.return_value = []
        server._entity_repo = mock_repo
        server._temporal_service = AsyncMock()

        result = await server._handle_entity({"name": "Nonexistent"})

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_entity_exact_match(self) -> None:
        """Test entity lookup prefers exact match."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_repo = AsyncMock()
        # Return multiple entities, with exact match not first
        entities = [
            Entity(
                id=1,
                name="Python 3",
                entity_type=EntityType.TECHNOLOGY,
                domain=Domain.TECH,
            ),
            Entity(
                id=2,
                name="python",
                entity_type=EntityType.TECHNOLOGY,
                domain=Domain.TECH,
            ),
        ]
        mock_repo.search_by_name.return_value = entities
        server._entity_repo = mock_repo

        mock_service = AsyncMock()
        mock_service.analyze_trends.return_value = []
        server._temporal_service = mock_service

        result = await server._handle_entity({"name": "python"})

        # Should return the exact match (case-insensitive)
        assert result["name"] == "python"
        assert result["id"] == 2


class TestServerLifecycle:
    """Tests for server lifecycle."""

    @pytest.mark.asyncio
    async def test_close_with_db(self) -> None:
        """Test server closes database connection."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        mock_db = AsyncMock()
        server._db = mock_db

        await server.close()

        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_db(self) -> None:
        """Test server close is safe without database."""
        server = PKAMCPServer(database_url="postgresql://test@localhost/test")

        # Should not raise
        await server.close()
