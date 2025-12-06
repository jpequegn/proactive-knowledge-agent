"""Report generation, alerts, and MCP integration."""

from src.outputs.alerts import (
    Alert,
    AlertBatch,
    AlertChannel,
    AlertManager,
    AlertManagerConfig,
    AlertStatus,
    ChannelConfig,
    ConsoleChannel,
    ConsoleConfig,
    EmailChannel,
    EmailConfig,
    SlackChannel,
    SlackConfig,
)
from src.outputs.mcp_server import PKAMCPServer
from src.outputs.reports import (
    DomainSection,
    ReportConfig,
    ReportFormat,
    ReportType,
    WeeklyReport,
    WeeklyReportGenerator,
    generate_weekly_report,
)

__all__ = [
    # Alert System
    "Alert",
    "AlertBatch",
    "AlertChannel",
    "AlertManager",
    "AlertManagerConfig",
    "AlertStatus",
    "ChannelConfig",
    "ConsoleChannel",
    "ConsoleConfig",
    "EmailChannel",
    "EmailConfig",
    "SlackChannel",
    "SlackConfig",
    # Report Generation
    "DomainSection",
    "ReportConfig",
    "ReportFormat",
    "ReportType",
    "WeeklyReport",
    "WeeklyReportGenerator",
    "generate_weekly_report",
    # MCP Server
    "PKAMCPServer",
]
