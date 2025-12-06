"""Proactive Alert System with Multiple Channels.

Provides a flexible alert system with:
- Multiple channels (console, Slack, email)
- Alert batching to prevent spam
- Acknowledge/snooze functionality
- Configurable thresholds per channel
"""

import asyncio
import smtplib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import httpx
import structlog

from src.agents import AlertLevel, Insight

logger = structlog.get_logger()


# ============================================================================
# Alert Models
# ============================================================================


class AlertStatus(str, Enum):
    """Status of an alert."""

    PENDING = "pending"  # Not yet delivered
    DELIVERED = "delivered"  # Sent to channel
    ACKNOWLEDGED = "acknowledged"  # User acknowledged
    SNOOZED = "snoozed"  # Temporarily dismissed
    EXPIRED = "expired"  # No longer relevant


@dataclass
class Alert:
    """An alert to be delivered through channels."""

    alert_id: str
    title: str
    description: str
    level: AlertLevel
    source: str  # Which agent generated this
    insight: Insight | None = None
    status: AlertStatus = AlertStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None
    acknowledged_at: datetime | None = None
    snoozed_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Check if alert requires action."""
        return self.level in (AlertLevel.ACTION, AlertLevel.URGENT)

    @property
    def is_snoozed(self) -> bool:
        """Check if alert is currently snoozed."""
        if self.snoozed_until is None:
            return False
        return datetime.now(UTC) < self.snoozed_until


@dataclass
class AlertBatch:
    """A batch of alerts for delivery."""

    batch_id: str
    alerts: list[Alert]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    level: AlertLevel | None = None  # Highest level in batch

    def __post_init__(self) -> None:
        """Calculate batch level after initialization."""
        if self.alerts and self.level is None:
            level_order = [
                AlertLevel.INFO,
                AlertLevel.WATCH,
                AlertLevel.ACTION,
                AlertLevel.URGENT,
            ]
            self.level = max(
                (a.level for a in self.alerts),
                key=lambda x: level_order.index(x),
            )


# ============================================================================
# Channel Configuration
# ============================================================================


@dataclass
class ChannelConfig:
    """Configuration for an alert channel."""

    enabled: bool = True
    min_level: AlertLevel = AlertLevel.INFO
    batch_window_seconds: int = 300  # 5 minutes
    max_alerts_per_batch: int = 10
    cooldown_seconds: int = 60  # Minimum time between batches
    rate_limit_per_hour: int = 20


@dataclass
class SlackConfig(ChannelConfig):
    """Slack-specific configuration."""

    webhook_url: str = ""
    channel: str = "#alerts"
    username: str = "PKA Alert Bot"
    icon_emoji: str = ":robot_face:"


@dataclass
class EmailConfig(ChannelConfig):
    """Email-specific configuration."""

    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    from_address: str = "alerts@pka.local"
    to_addresses: list[str] = field(default_factory=list)
    subject_prefix: str = "[PKA Alert]"


@dataclass
class ConsoleConfig(ChannelConfig):
    """Console output configuration."""

    colorize: bool = True
    show_metadata: bool = False


# ============================================================================
# Alert Channels
# ============================================================================


class AlertChannel(ABC):
    """Abstract base class for alert channels."""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self._pending_alerts: list[Alert] = []
        self._batch_start: datetime | None = None
        self._last_delivery: datetime | None = None
        self._delivery_count_this_hour: int = 0
        self._hour_start: datetime = datetime.now(UTC)

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name for logging."""
        pass

    def should_accept(self, alert: Alert) -> bool:
        """Check if channel should accept this alert."""
        if not self.config.enabled:
            return False

        level_order = [
            AlertLevel.INFO,
            AlertLevel.WATCH,
            AlertLevel.ACTION,
            AlertLevel.URGENT,
        ]
        return level_order.index(alert.level) >= level_order.index(
            self.config.min_level
        )

    def add_alert(self, alert: Alert) -> bool:
        """Add alert to pending queue."""
        if not self.should_accept(alert):
            return False

        if alert.is_snoozed:
            return False

        self._pending_alerts.append(alert)

        if self._batch_start is None:
            self._batch_start = datetime.now(UTC)

        return True

    def should_flush(self) -> bool:
        """Check if pending alerts should be flushed."""
        if not self._pending_alerts:
            return False

        # Check if batch window expired
        if self._batch_start:
            elapsed = (datetime.now(UTC) - self._batch_start).total_seconds()
            if elapsed >= self.config.batch_window_seconds:
                return True

        # Check if batch is full
        if len(self._pending_alerts) >= self.config.max_alerts_per_batch:
            return True

        # Check for urgent alerts (deliver immediately)
        if any(a.level == AlertLevel.URGENT for a in self._pending_alerts):
            return True

        return False

    def _check_rate_limit(self) -> bool:
        """Check if within rate limits."""
        now = datetime.now(UTC)

        # Reset hourly counter
        if (now - self._hour_start).total_seconds() >= 3600:
            self._hour_start = now
            self._delivery_count_this_hour = 0

        # Check hourly limit
        if self._delivery_count_this_hour >= self.config.rate_limit_per_hour:
            logger.warning(
                "Rate limit reached",
                channel=self.name,
                limit=self.config.rate_limit_per_hour,
            )
            return False

        # Check cooldown
        if self._last_delivery:
            cooldown = (now - self._last_delivery).total_seconds()
            if cooldown < self.config.cooldown_seconds:
                return False

        return True

    async def flush(self) -> AlertBatch | None:
        """Flush pending alerts as a batch."""
        if not self._pending_alerts:
            return None

        if not self._check_rate_limit():
            return None

        # Create batch
        batch_id = f"batch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{self.name}"
        batch = AlertBatch(
            batch_id=batch_id,
            alerts=self._pending_alerts[: self.config.max_alerts_per_batch],
        )

        # Deliver
        try:
            await self._deliver(batch)

            # Update state
            now = datetime.now(UTC)
            for alert in batch.alerts:
                alert.status = AlertStatus.DELIVERED
                alert.delivered_at = now

            self._last_delivery = now
            self._delivery_count_this_hour += 1

            # Clear delivered alerts from pending
            delivered_ids = {a.alert_id for a in batch.alerts}
            self._pending_alerts = [
                a for a in self._pending_alerts if a.alert_id not in delivered_ids
            ]
            self._batch_start = None

            logger.info(
                "Alert batch delivered",
                channel=self.name,
                batch_id=batch_id,
                count=len(batch.alerts),
                level=batch.level.value if batch.level else None,
            )

            return batch

        except Exception as e:
            logger.error(
                "Failed to deliver alert batch",
                channel=self.name,
                batch_id=batch_id,
                error=str(e),
            )
            return None

    @abstractmethod
    async def _deliver(self, batch: AlertBatch) -> None:
        """Deliver a batch of alerts. Implement in subclasses."""
        pass

    def get_pending_count(self) -> int:
        """Get number of pending alerts."""
        return len(self._pending_alerts)


class ConsoleChannel(AlertChannel):
    """Console output channel for testing and development."""

    def __init__(self, config: ConsoleConfig | None = None):
        super().__init__(config or ConsoleConfig())
        self._console_config = config or ConsoleConfig()

    @property
    def name(self) -> str:
        return "console"

    async def _deliver(self, batch: AlertBatch) -> None:
        """Print alerts to console."""
        level_colors = {
            AlertLevel.INFO: "\033[94m",  # Blue
            AlertLevel.WATCH: "\033[93m",  # Yellow
            AlertLevel.ACTION: "\033[95m",  # Magenta
            AlertLevel.URGENT: "\033[91m",  # Red
        }
        reset = "\033[0m"

        print("\n" + "=" * 60)
        print(f"PKA ALERT BATCH - {batch.batch_id}")
        print(f"Time: {batch.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Alerts: {len(batch.alerts)}")
        print("=" * 60)

        for alert in batch.alerts:
            if self._console_config.colorize:
                color = level_colors.get(alert.level, "")
            else:
                color = ""
            level_str = f"[{alert.level.value.upper()}]"

            print(f"\n{color}{level_str}{reset} {alert.title}")
            print(f"  Source: {alert.source}")
            print(f"  {alert.description}")

            if self._console_config.show_metadata and alert.metadata:
                print(f"  Metadata: {alert.metadata}")

        print("\n" + "=" * 60 + "\n")


class SlackChannel(AlertChannel):
    """Slack webhook channel."""

    def __init__(self, config: SlackConfig):
        super().__init__(config)
        self._slack_config = config

    @property
    def name(self) -> str:
        return "slack"

    def _format_alert_block(self, alert: Alert) -> dict[str, Any]:
        """Format an alert as a Slack block."""
        emoji = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WATCH: ":eyes:",
            AlertLevel.ACTION: ":zap:",
            AlertLevel.URGENT: ":rotating_light:",
        }.get(alert.level, ":bell:")

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{emoji} *[{alert.level.value.upper()}]* {alert.title}\n"
                    f"_{alert.source}_ - {alert.description}"
                ),
            },
        }

    async def _deliver(self, batch: AlertBatch) -> None:
        """Send alerts to Slack webhook."""
        if not self._slack_config.webhook_url:
            raise ValueError("Slack webhook URL not configured")

        # Build message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"PKA Alerts ({len(batch.alerts)} items)",
                },
            },
            {"type": "divider"},
        ]

        for alert in batch.alerts:
            blocks.append(self._format_alert_block(alert))

        payload = {
            "channel": self._slack_config.channel,
            "username": self._slack_config.username,
            "icon_emoji": self._slack_config.icon_emoji,
            "blocks": blocks,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._slack_config.webhook_url,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()


class EmailChannel(AlertChannel):
    """Email alert channel."""

    def __init__(self, config: EmailConfig):
        super().__init__(config)
        self._email_config = config

    @property
    def name(self) -> str:
        return "email"

    def _format_html(self, batch: AlertBatch) -> str:
        """Format batch as HTML email."""
        level_colors = {
            AlertLevel.INFO: "#3498db",
            AlertLevel.WATCH: "#f39c12",
            AlertLevel.ACTION: "#9b59b6",
            AlertLevel.URGENT: "#e74c3c",
        }

        alerts_html = ""
        for alert in batch.alerts:
            color = level_colors.get(alert.level, "#333")
            div_style = (
                f"margin-bottom: 15px; padding: 10px; "
                f"border-left: 4px solid {color};"
            )
            alerts_html += f"""
            <div style="{div_style}">
                <strong style="color: {color};">[{alert.level.value.upper()}]</strong>
                <strong>{alert.title}</strong><br>
                <small style="color: #666;">{alert.source}</small><br>
                <p style="margin: 5px 0;">{alert.description}</p>
            </div>
            """

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #333;">PKA Alert Summary</h2>
            <p>You have {len(batch.alerts)} new alert(s):</p>
            {alerts_html}
            <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
            <small style="color: #999;">
                Generated at {batch.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
            </small>
        </body>
        </html>
        """

    def _format_text(self, batch: AlertBatch) -> str:
        """Format batch as plain text email."""
        lines = [
            f"PKA Alert Summary - {len(batch.alerts)} alert(s)",
            "=" * 50,
            "",
        ]

        for alert in batch.alerts:
            lines.extend([
                f"[{alert.level.value.upper()}] {alert.title}",
                f"Source: {alert.source}",
                alert.description,
                "",
            ])

        lines.extend([
            "-" * 50,
            f"Generated at {batch.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ])

        return "\n".join(lines)

    async def _deliver(self, batch: AlertBatch) -> None:
        """Send alerts via email."""
        if not self._email_config.to_addresses:
            raise ValueError("No email recipients configured")

        # Build email
        msg = MIMEMultipart("alternative")
        level_str = batch.level.value.upper() if batch.level else "INFO"
        msg["Subject"] = (
            f"{self._email_config.subject_prefix} "
            f"{len(batch.alerts)} Alert(s) - {level_str}"
        )
        msg["From"] = self._email_config.from_address
        msg["To"] = ", ".join(self._email_config.to_addresses)

        # Add text and HTML parts
        text_part = MIMEText(self._format_text(batch), "plain")
        html_part = MIMEText(self._format_html(batch), "html")
        msg.attach(text_part)
        msg.attach(html_part)

        # Send email (run in thread to not block)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_email, msg)

    def _send_email(self, msg: MIMEMultipart) -> None:
        """Send email synchronously."""
        with smtplib.SMTP(
            self._email_config.smtp_host,
            self._email_config.smtp_port,
        ) as server:
            if self._email_config.use_tls:
                server.starttls()

            if self._email_config.smtp_user:
                server.login(
                    self._email_config.smtp_user,
                    self._email_config.smtp_password,
                )

            server.send_message(msg)


# ============================================================================
# Alert Manager
# ============================================================================


@dataclass
class AlertManagerConfig:
    """Configuration for the alert manager."""

    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    slack: SlackConfig | None = None
    email: EmailConfig | None = None
    flush_interval_seconds: int = 30
    alert_expiry_hours: int = 24


class AlertManager:
    """
    Central manager for the alert system.

    Handles:
    - Alert creation and routing
    - Channel management
    - Batching and delivery
    - Acknowledge/snooze functionality
    - Alert persistence (in-memory for now)
    """

    def __init__(self, config: AlertManagerConfig | None = None):
        self.config = config or AlertManagerConfig()
        self._channels: list[AlertChannel] = []
        self._alerts: dict[str, Alert] = {}
        self._alert_counter: int = 0
        self._running: bool = False
        self._flush_task: asyncio.Task | None = None

        self._setup_channels()

    def _setup_channels(self) -> None:
        """Initialize configured channels."""
        # Console is always available
        self._channels.append(ConsoleChannel(self.config.console))

        # Optional channels
        if self.config.slack and self.config.slack.webhook_url:
            self._channels.append(SlackChannel(self.config.slack))
            logger.info("Slack channel configured")

        if self.config.email and self.config.email.to_addresses:
            self._channels.append(EmailChannel(self.config.email))
            logger.info("Email channel configured")

    def create_alert(
        self,
        title: str,
        description: str,
        level: AlertLevel,
        source: str,
        insight: Insight | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """Create a new alert and queue for delivery."""
        self._alert_counter += 1
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        alert_id = f"alert_{timestamp}_{self._alert_counter}"

        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            level=level,
            source=source,
            insight=insight,
            metadata=metadata or {},
        )

        self._alerts[alert_id] = alert

        # Queue for all applicable channels
        for channel in self._channels:
            channel.add_alert(alert)

        logger.info(
            "Alert created",
            alert_id=alert_id,
            level=level.value,
            source=source,
        )

        return alert

    def create_from_insight(self, insight: Insight, source: str) -> Alert:
        """Create an alert from an insight."""
        return self.create_alert(
            title=insight.title,
            description=insight.description,
            level=insight.level,
            source=source,
            insight=insight,
            metadata={
                "insight_type": insight.insight_type.value,
                "confidence": insight.confidence,
                "relevance_score": insight.relevance_score,
            },
        )

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(UTC)

        logger.info("Alert acknowledged", alert_id=alert_id)
        return True

    def snooze(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """Snooze an alert for a duration."""
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        alert.status = AlertStatus.SNOOZED
        alert.snoozed_until = datetime.now(UTC) + timedelta(minutes=duration_minutes)

        logger.info(
            "Alert snoozed",
            alert_id=alert_id,
            until=alert.snoozed_until.isoformat(),
        )
        return True

    def get_alert(self, alert_id: str) -> Alert | None:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    def get_pending_alerts(self) -> list[Alert]:
        """Get all pending alerts."""
        return [
            a for a in self._alerts.values()
            if a.status == AlertStatus.PENDING
        ]

    def get_unacknowledged_alerts(self) -> list[Alert]:
        """Get alerts that haven't been acknowledged."""
        return [
            a for a in self._alerts.values()
            if a.status in (AlertStatus.PENDING, AlertStatus.DELIVERED)
        ]

    def get_actionable_alerts(self) -> list[Alert]:
        """Get alerts that require action."""
        return [
            a for a in self._alerts.values()
            if a.is_actionable and a.status != AlertStatus.ACKNOWLEDGED
        ]

    async def flush_all(self) -> list[AlertBatch]:
        """Flush all channels that are ready."""
        batches = []
        for channel in self._channels:
            if channel.should_flush():
                batch = await channel.flush()
                if batch:
                    batches.append(batch)
        return batches

    async def force_flush(self) -> list[AlertBatch]:
        """Force flush all channels regardless of timing."""
        batches = []
        for channel in self._channels:
            if channel.get_pending_count() > 0:
                batch = await channel.flush()
                if batch:
                    batches.append(batch)
        return batches

    def cleanup_expired(self) -> int:
        """Remove expired alerts."""
        now = datetime.now(UTC)
        expiry = timedelta(hours=self.config.alert_expiry_hours)
        expired = []

        for alert_id, alert in self._alerts.items():
            if now - alert.created_at > expiry:
                alert.status = AlertStatus.EXPIRED
                expired.append(alert_id)

        for alert_id in expired:
            del self._alerts[alert_id]

        if expired:
            logger.info("Cleaned up expired alerts", count=len(expired))

        return len(expired)

    async def start(self) -> None:
        """Start the alert manager background task."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop the alert manager."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.force_flush()
        logger.info("Alert manager stopped")

    async def _flush_loop(self) -> None:
        """Background loop for flushing alerts."""
        while self._running:
            try:
                await self.flush_all()
                self.cleanup_expired()
                await asyncio.sleep(self.config.flush_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in flush loop", error=str(e))
                await asyncio.sleep(5)

    def get_stats(self) -> dict[str, Any]:
        """Get alert system statistics."""
        status_counts: dict[str, int] = defaultdict(int)
        level_counts: dict[str, int] = defaultdict(int)

        for alert in self._alerts.values():
            status_counts[alert.status.value] += 1
            level_counts[alert.level.value] += 1

        return {
            "total_alerts": len(self._alerts),
            "by_status": dict(status_counts),
            "by_level": dict(level_counts),
            "channels": [
                {
                    "name": ch.name,
                    "enabled": ch.config.enabled,
                    "pending": ch.get_pending_count(),
                }
                for ch in self._channels
            ],
        }
