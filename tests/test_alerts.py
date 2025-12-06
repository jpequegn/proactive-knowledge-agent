"""Tests for Proactive Alert System."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents import AlertLevel, Insight, InsightType
from src.outputs import (
    Alert,
    AlertBatch,
    AlertManager,
    AlertManagerConfig,
    AlertStatus,
    ConsoleChannel,
    ConsoleConfig,
    EmailChannel,
    EmailConfig,
    SlackChannel,
    SlackConfig,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_alert():
    """Create a sample alert."""
    return Alert(
        alert_id="test_alert_1",
        title="Test Alert",
        description="This is a test alert.",
        level=AlertLevel.ACTION,
        source="TestAgent",
    )


@pytest.fixture
def urgent_alert():
    """Create an urgent alert."""
    return Alert(
        alert_id="urgent_alert_1",
        title="Urgent Issue",
        description="This requires immediate attention.",
        level=AlertLevel.URGENT,
        source="TestAgent",
    )


@pytest.fixture
def sample_insight():
    """Create a sample insight."""
    return Insight(
        insight_type=InsightType.TREND_CHANGE,
        title="Market Trend Detected",
        description="Significant market movement observed.",
        level=AlertLevel.WATCH,
        confidence=0.85,
        relevance_score=0.90,
        source_ids=["source_1"],
        metadata={},
    )


@pytest.fixture
def console_config():
    """Create console channel config."""
    return ConsoleConfig(
        enabled=True,
        min_level=AlertLevel.INFO,
        batch_window_seconds=60,
        cooldown_seconds=1,
    )


@pytest.fixture
def slack_config():
    """Create Slack channel config."""
    return SlackConfig(
        enabled=True,
        webhook_url="https://hooks.slack.com/test",
        channel="#alerts",
        min_level=AlertLevel.WATCH,
        batch_window_seconds=60,
        cooldown_seconds=1,
    )


@pytest.fixture
def email_config():
    """Create email channel config."""
    return EmailConfig(
        enabled=True,
        smtp_host="localhost",
        smtp_port=587,
        from_address="test@example.com",
        to_addresses=["user@example.com"],
        min_level=AlertLevel.ACTION,
        batch_window_seconds=60,
        cooldown_seconds=1,
    )


@pytest.fixture
def alert_manager():
    """Create an alert manager with console only."""
    config = AlertManagerConfig(
        console=ConsoleConfig(enabled=True),
        flush_interval_seconds=1,
    )
    return AlertManager(config)


# ============================================================================
# Alert Model Tests
# ============================================================================


class TestAlertModel:
    """Tests for Alert model."""

    def test_alert_creation(self, sample_alert):
        """Test basic alert creation."""
        assert sample_alert.alert_id == "test_alert_1"
        assert sample_alert.title == "Test Alert"
        assert sample_alert.level == AlertLevel.ACTION
        assert sample_alert.status == AlertStatus.PENDING

    def test_is_actionable(self):
        """Test is_actionable property."""
        action_alert = Alert(
            alert_id="1",
            title="Action",
            description="Test",
            level=AlertLevel.ACTION,
            source="Test",
        )
        urgent_alert = Alert(
            alert_id="2",
            title="Urgent",
            description="Test",
            level=AlertLevel.URGENT,
            source="Test",
        )
        info_alert = Alert(
            alert_id="3",
            title="Info",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )

        assert action_alert.is_actionable is True
        assert urgent_alert.is_actionable is True
        assert info_alert.is_actionable is False

    def test_is_snoozed(self):
        """Test is_snoozed property."""
        alert = Alert(
            alert_id="1",
            title="Test",
            description="Test",
            level=AlertLevel.ACTION,
            source="Test",
        )

        assert alert.is_snoozed is False

        # Snooze for future
        alert.snoozed_until = datetime.now(UTC) + timedelta(hours=1)
        assert alert.is_snoozed is True

        # Snooze expired
        alert.snoozed_until = datetime.now(UTC) - timedelta(hours=1)
        assert alert.is_snoozed is False


class TestAlertBatch:
    """Tests for AlertBatch model."""

    def test_batch_level_calculation(self):
        """Test that batch level is highest among alerts."""
        alerts = [
            Alert(
                alert_id="1",
                title="Info",
                description="Test",
                level=AlertLevel.INFO,
                source="Test",
            ),
            Alert(
                alert_id="2",
                title="Action",
                description="Test",
                level=AlertLevel.ACTION,
                source="Test",
            ),
            Alert(
                alert_id="3",
                title="Watch",
                description="Test",
                level=AlertLevel.WATCH,
                source="Test",
            ),
        ]

        batch = AlertBatch(batch_id="test_batch", alerts=alerts)
        assert batch.level == AlertLevel.ACTION

    def test_empty_batch(self):
        """Test batch with no alerts."""
        batch = AlertBatch(batch_id="empty_batch", alerts=[])
        assert batch.level is None


# ============================================================================
# Channel Tests
# ============================================================================


class TestConsoleChannel:
    """Tests for ConsoleChannel."""

    def test_channel_name(self, console_config):
        """Test channel name."""
        channel = ConsoleChannel(console_config)
        assert channel.name == "console"

    def test_should_accept_by_level(self):
        """Test alert acceptance based on level."""
        config = ConsoleConfig(min_level=AlertLevel.WATCH)
        channel = ConsoleChannel(config)

        info_alert = Alert(
            alert_id="1",
            title="Info",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )
        watch_alert = Alert(
            alert_id="2",
            title="Watch",
            description="Test",
            level=AlertLevel.WATCH,
            source="Test",
        )

        assert channel.should_accept(info_alert) is False
        assert channel.should_accept(watch_alert) is True

    def test_add_alert(self, console_config, sample_alert):
        """Test adding alert to channel."""
        channel = ConsoleChannel(console_config)

        assert channel.add_alert(sample_alert) is True
        assert channel.get_pending_count() == 1

    def test_reject_snoozed_alert(self, console_config):
        """Test that snoozed alerts are rejected."""
        channel = ConsoleChannel(console_config)

        alert = Alert(
            alert_id="1",
            title="Snoozed",
            description="Test",
            level=AlertLevel.ACTION,
            source="Test",
            snoozed_until=datetime.now(UTC) + timedelta(hours=1),
        )

        assert channel.add_alert(alert) is False

    def test_should_flush_batch_full(self, console_config):
        """Test flush when batch is full."""
        config = ConsoleConfig(max_alerts_per_batch=2)
        channel = ConsoleChannel(config)

        for i in range(2):
            alert = Alert(
                alert_id=f"alert_{i}",
                title=f"Alert {i}",
                description="Test",
                level=AlertLevel.INFO,
                source="Test",
            )
            channel.add_alert(alert)

        assert channel.should_flush() is True

    def test_should_flush_urgent(self, console_config, urgent_alert):
        """Test immediate flush for urgent alerts."""
        channel = ConsoleChannel(console_config)
        channel.add_alert(urgent_alert)

        assert channel.should_flush() is True

    @pytest.mark.asyncio
    async def test_deliver_console(self, console_config, sample_alert, capsys):
        """Test console delivery."""
        channel = ConsoleChannel(console_config)
        channel.add_alert(sample_alert)

        batch = await channel.flush()

        assert batch is not None
        assert len(batch.alerts) == 1
        assert batch.alerts[0].status == AlertStatus.DELIVERED

        captured = capsys.readouterr()
        assert "PKA ALERT BATCH" in captured.out
        assert "Test Alert" in captured.out


class TestSlackChannel:
    """Tests for SlackChannel."""

    def test_channel_name(self, slack_config):
        """Test channel name."""
        channel = SlackChannel(slack_config)
        assert channel.name == "slack"

    @pytest.mark.asyncio
    async def test_deliver_slack(self, slack_config, sample_alert):
        """Test Slack webhook delivery."""
        channel = SlackChannel(slack_config)
        channel.add_alert(sample_alert)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            batch = await channel.flush()

            assert batch is not None
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == slack_config.webhook_url

    @pytest.mark.asyncio
    async def test_slack_no_webhook_fails(self):
        """Test that missing webhook URL raises error."""
        config = SlackConfig(webhook_url="")
        channel = SlackChannel(config)

        alert = Alert(
            alert_id="1",
            title="Test",
            description="Test",
            level=AlertLevel.WATCH,
            source="Test",
        )
        channel.add_alert(alert)

        batch = await channel.flush()
        assert batch is None  # Failed delivery


class TestEmailChannel:
    """Tests for EmailChannel."""

    def test_channel_name(self, email_config):
        """Test channel name."""
        channel = EmailChannel(email_config)
        assert channel.name == "email"

    def test_format_text(self, email_config, sample_alert):
        """Test text email formatting."""
        channel = EmailChannel(email_config)

        batch = AlertBatch(batch_id="test", alerts=[sample_alert])
        text = channel._format_text(batch)

        assert "PKA Alert Summary" in text
        assert "Test Alert" in text
        assert "TestAgent" in text

    def test_format_html(self, email_config, sample_alert):
        """Test HTML email formatting."""
        channel = EmailChannel(email_config)

        batch = AlertBatch(batch_id="test", alerts=[sample_alert])
        html = channel._format_html(batch)

        assert "<html>" in html
        assert "Test Alert" in html
        assert "ACTION" in html


# ============================================================================
# Alert Manager Tests
# ============================================================================


class TestAlertManager:
    """Tests for AlertManager."""

    def test_create_alert(self, alert_manager):
        """Test creating an alert."""
        alert = alert_manager.create_alert(
            title="New Alert",
            description="This is a new alert.",
            level=AlertLevel.WATCH,
            source="TestAgent",
        )

        assert alert.alert_id.startswith("alert_")
        assert alert.title == "New Alert"
        assert alert.status == AlertStatus.PENDING

    def test_create_from_insight(self, alert_manager, sample_insight):
        """Test creating alert from insight."""
        alert = alert_manager.create_from_insight(
            insight=sample_insight,
            source="TechAgent",
        )

        assert alert.title == sample_insight.title
        assert alert.description == sample_insight.description
        assert alert.level == sample_insight.level
        assert alert.insight == sample_insight

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        alert = alert_manager.create_alert(
            title="Test",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )

        assert alert_manager.acknowledge(alert.alert_id) is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None

    def test_acknowledge_nonexistent(self, alert_manager):
        """Test acknowledging non-existent alert."""
        assert alert_manager.acknowledge("nonexistent") is False

    def test_snooze_alert(self, alert_manager):
        """Test snoozing an alert."""
        alert = alert_manager.create_alert(
            title="Test",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )

        assert alert_manager.snooze(alert.alert_id, duration_minutes=30) is True
        assert alert.status == AlertStatus.SNOOZED
        assert alert.snoozed_until is not None
        assert alert.is_snoozed is True

    def test_get_pending_alerts(self, alert_manager):
        """Test getting pending alerts."""
        for i in range(3):
            alert_manager.create_alert(
                title=f"Alert {i}",
                description="Test",
                level=AlertLevel.INFO,
                source="Test",
            )

        pending = alert_manager.get_pending_alerts()
        assert len(pending) == 3

    def test_get_actionable_alerts(self, alert_manager):
        """Test getting actionable alerts."""
        alert_manager.create_alert(
            title="Info",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )
        alert_manager.create_alert(
            title="Action",
            description="Test",
            level=AlertLevel.ACTION,
            source="Test",
        )
        alert_manager.create_alert(
            title="Urgent",
            description="Test",
            level=AlertLevel.URGENT,
            source="Test",
        )

        actionable = alert_manager.get_actionable_alerts()
        assert len(actionable) == 2

    @pytest.mark.asyncio
    async def test_flush_all(self, alert_manager):
        """Test flushing all channels."""
        for i in range(3):
            alert_manager.create_alert(
                title=f"Alert {i}",
                description="Test",
                level=AlertLevel.URGENT,  # Urgent triggers immediate flush
                source="Test",
            )

        batches = await alert_manager.flush_all()
        assert len(batches) >= 1

    @pytest.mark.asyncio
    async def test_force_flush(self, alert_manager):
        """Test force flushing."""
        alert_manager.create_alert(
            title="Test",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )

        batches = await alert_manager.force_flush()
        assert len(batches) == 1

    def test_cleanup_expired(self, alert_manager):
        """Test cleanup of expired alerts."""
        # Create alert with past creation time
        alert = alert_manager.create_alert(
            title="Old Alert",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )
        alert.created_at = datetime.now(UTC) - timedelta(hours=25)

        expired_count = alert_manager.cleanup_expired()
        assert expired_count == 1
        assert alert.alert_id not in alert_manager._alerts

    def test_get_stats(self, alert_manager):
        """Test getting statistics."""
        alert_manager.create_alert(
            title="Info",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )
        alert_manager.create_alert(
            title="Action",
            description="Test",
            level=AlertLevel.ACTION,
            source="Test",
        )

        stats = alert_manager.get_stats()
        assert stats["total_alerts"] == 2
        assert stats["by_level"]["info"] == 1
        assert stats["by_level"]["action"] == 1
        assert len(stats["channels"]) >= 1


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_cooldown_respected(self, console_config, sample_alert):
        """Test that cooldown between batches is respected."""
        config = ConsoleConfig(cooldown_seconds=3600)  # 1 hour cooldown
        channel = ConsoleChannel(config)

        channel.add_alert(sample_alert)

        # First delivery should work
        channel._last_delivery = datetime.now(UTC)

        # Should not flush due to cooldown
        assert channel._check_rate_limit() is False

    def test_hourly_limit(self, console_config, sample_alert):
        """Test hourly rate limit."""
        config = ConsoleConfig(rate_limit_per_hour=2, cooldown_seconds=0)
        channel = ConsoleChannel(config)

        # Simulate max deliveries
        channel._delivery_count_this_hour = 2

        assert channel._check_rate_limit() is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestAlertIntegration:
    """Integration tests for alert system."""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test complete alert flow."""
        config = AlertManagerConfig(
            console=ConsoleConfig(
                enabled=True,
                batch_window_seconds=1,
                cooldown_seconds=0,
            ),
        )
        manager = AlertManager(config)

        # Create alerts
        alert1 = manager.create_alert(
            title="First Alert",
            description="Test 1",
            level=AlertLevel.INFO,
            source="Test",
        )
        alert2 = manager.create_alert(
            title="Second Alert",
            description="Test 2",
            level=AlertLevel.URGENT,
            source="Test",
        )

        # Flush
        batches = await manager.force_flush()

        assert len(batches) == 1
        assert alert1.status == AlertStatus.DELIVERED
        assert alert2.status == AlertStatus.DELIVERED

        # Acknowledge one
        manager.acknowledge(alert1.alert_id)
        assert alert1.status == AlertStatus.ACKNOWLEDGED

        # Snooze another
        manager.snooze(alert2.alert_id, duration_minutes=60)
        assert alert2.status == AlertStatus.SNOOZED

        # Get stats
        stats = manager.get_stats()
        assert stats["total_alerts"] == 2

    @pytest.mark.asyncio
    async def test_multi_channel_routing(self):
        """Test alerts routed to multiple channels."""
        config = AlertManagerConfig(
            console=ConsoleConfig(enabled=True, min_level=AlertLevel.INFO),
            slack=SlackConfig(
                enabled=True,
                webhook_url="https://hooks.slack.com/test",
                min_level=AlertLevel.ACTION,
            ),
        )
        manager = AlertManager(config)

        # INFO should only go to console
        _info_alert = manager.create_alert(
            title="Info Alert",
            description="Test",
            level=AlertLevel.INFO,
            source="Test",
        )

        # ACTION should go to both
        _action_alert = manager.create_alert(
            title="Action Alert",
            description="Test",
            level=AlertLevel.ACTION,
            source="Test",
        )

        console_channel = manager._channels[0]
        slack_channel = manager._channels[1]

        assert console_channel.get_pending_count() == 2
        assert slack_channel.get_pending_count() == 1
