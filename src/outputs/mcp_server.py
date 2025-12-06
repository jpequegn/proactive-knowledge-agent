"""MCP Server for Claude Code Integration.

Provides tools for querying the Proactive Knowledge Agent:
- pka_search: Search knowledge graph entities
- pka_trends: Get trend analysis for a domain
- pka_alerts: Get pending alerts
- pka_report: Generate reports
- pka_entity: Get entity details
"""

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.agents import AlertLevel, Insight
from src.database import Database
from src.outputs.alerts import Alert, AlertManager, AlertManagerConfig, AlertStatus
from src.outputs.reports import ReportType
from src.world_model import (
    Domain,
    Entity,
    EntityRepository,
    EntityType,
    TemporalReasoningService,
    TrendDirection,
    TrendResult,
)

logger = structlog.get_logger()


# ============================================================================
# Response Formatters
# ============================================================================


def format_entity(entity: Entity) -> dict[str, Any]:
    """Format entity for JSON response."""
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.entity_type.value,
        "domain": entity.domain.value,
        "description": entity.description,
        "confidence": entity.confidence,
        "mention_count": entity.mention_count,
        "first_seen": (
            entity.first_seen.isoformat() if entity.first_seen else None
        ),
        "last_seen": (
            entity.last_seen.isoformat() if entity.last_seen else None
        ),
        "properties": entity.properties,
    }


def format_trend(trend: TrendResult) -> dict[str, Any]:
    """Format trend result for JSON response."""
    return {
        "entity_id": trend.entity_id,
        "entity_name": trend.entity_name,
        "entity_type": trend.entity_type,
        "domain": trend.domain,
        "direction": trend.direction.value,
        "change_ratio": round(trend.change_ratio * 100, 1),  # As percentage
        "current_count": trend.current_count,
        "previous_count": trend.previous_count,
        "period_days": trend.period_days,
        "confidence": trend.confidence,
    }


def format_alert(alert: Alert) -> dict[str, Any]:
    """Format alert for JSON response."""
    return {
        "id": alert.alert_id,
        "title": alert.title,
        "description": alert.description,
        "level": alert.level.value,
        "source": alert.source,
        "status": alert.status.value,
        "created_at": alert.created_at.isoformat(),
        "is_actionable": alert.is_actionable,
    }


def format_insight(insight: Insight) -> dict[str, Any]:
    """Format insight for JSON response."""
    return {
        "title": insight.title,
        "description": insight.description,
        "insight_type": insight.insight_type.value,
        "level": insight.level.value,
        "confidence": insight.confidence,
        "priority_score": insight.priority_score,
        "created_at": insight.created_at.isoformat(),
        "metadata": insight.metadata,
    }


# ============================================================================
# MCP Server Implementation
# ============================================================================


class PKAMCPServer:
    """MCP Server for Proactive Knowledge Agent."""

    def __init__(
        self,
        database_url: str | None = None,
    ):
        """Initialize the MCP server.

        Args:
            database_url: PostgreSQL connection URL. Defaults to DATABASE_URL env var.
        """
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self._db: Database | None = None
        self._entity_repo: EntityRepository | None = None
        self._temporal_service: TemporalReasoningService | None = None
        self._alert_manager: AlertManager | None = None

        # Create MCP server
        self.server = Server("pka")
        self._setup_handlers()

    async def _ensure_connected(self) -> None:
        """Ensure database connection is established."""
        if self._db is None:
            self._db = Database(self.database_url)
            await self._db.connect()
            self._entity_repo = EntityRepository(self._db)
            self._temporal_service = TemporalReasoningService(self._db)
            self._alert_manager = AlertManager(AlertManagerConfig())
            logger.info("MCP server connected to database")

    def _setup_handlers(self) -> None:
        """Setup MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available PKA tools."""
            return [
                Tool(
                    name="pka_search",
                    description=(
                        "Search the PKA knowledge graph for entities. "
                        "Returns matching entities with their types, domains, "
                        "and descriptions."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (entity name or keyword)",
                            },
                            "entity_type": {
                                "type": "string",
                                "enum": [t.value for t in EntityType],
                                "description": "Optional: filter by entity type",
                            },
                            "domain": {
                                "type": "string",
                                "enum": [d.value for d in Domain],
                                "description": "Optional: filter by domain",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="pka_trends",
                    description=(
                        "Get trend analysis for entities in a domain. "
                        "Shows rising, falling, and new trends over a time period."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "enum": [d.value for d in Domain],
                                "description": "Domain to analyze",
                            },
                            "period": {
                                "type": "string",
                                "enum": ["day", "week", "month"],
                                "description": "Time period for trend analysis",
                                "default": "week",
                            },
                            "direction": {
                                "type": "string",
                                "enum": [d.value for d in TrendDirection],
                                "description": "Optional: filter by trend direction",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default 10)",
                                "default": 10,
                            },
                        },
                        "required": ["domain"],
                    },
                ),
                Tool(
                    name="pka_alerts",
                    description=(
                        "Get pending alerts from the PKA system. "
                        "Returns alerts requiring attention, optionally filtered "
                        "by level."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "enum": [lvl.value for lvl in AlertLevel],
                                "description": (
                                    "Optional: filter by minimum alert level"
                                ),
                            },
                            "status": {
                                "type": "string",
                                "enum": [s.value for s in AlertStatus],
                                "description": "Optional: filter by status",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default 20)",
                                "default": 20,
                            },
                        },
                    },
                ),
                Tool(
                    name="pka_report",
                    description=(
                        "Generate a PKA intelligence report. "
                        "Summarizes insights across domains for a time period."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [t.value for t in ReportType],
                                "description": "Report type",
                                "default": "weekly",
                            },
                            "period_days": {
                                "type": "integer",
                                "description": "Days to cover (default 7)",
                                "default": 7,
                            },
                            "domain": {
                                "type": "string",
                                "enum": [d.value for d in Domain],
                                "description": (
                                    "Optional: focus on specific domain"
                                ),
                            },
                        },
                    },
                ),
                Tool(
                    name="pka_entity",
                    description=(
                        "Get detailed information about a specific entity. "
                        "Returns full entity data including relationships "
                        "and recent activity."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Entity name to look up",
                            },
                            "entity_type": {
                                "type": "string",
                                "enum": [t.value for t in EntityType],
                                "description": (
                                    "Optional: entity type for disambiguation"
                                ),
                            },
                        },
                        "required": ["name"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: dict[str, Any],
        ) -> list[TextContent]:
            """Handle tool calls."""
            await self._ensure_connected()

            try:
                if name == "pka_search":
                    result = await self._handle_search(arguments)
                elif name == "pka_trends":
                    result = await self._handle_trends(arguments)
                elif name == "pka_alerts":
                    result = await self._handle_alerts(arguments)
                elif name == "pka_report":
                    result = await self._handle_report(arguments)
                elif name == "pka_entity":
                    result = await self._handle_entity(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str),
                )]

            except Exception as e:
                logger.error("Tool call failed", tool=name, error=str(e))
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2),
                )]

    async def _handle_search(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle pka_search tool call."""
        query = arguments.get("query", "")
        entity_type_str = arguments.get("entity_type")
        domain_str = arguments.get("domain")
        limit = arguments.get("limit", 10)

        if not self._entity_repo:
            return {"error": "Database not connected"}

        entity_type = EntityType(entity_type_str) if entity_type_str else None

        # Search by name
        entities = await self._entity_repo.search_by_name(
            search_text=query,
            limit=limit,
            entity_type=entity_type,
        )

        # Filter by domain if specified
        if domain_str:
            domain = Domain(domain_str)
            entities = [e for e in entities if e.domain == domain]

        return {
            "query": query,
            "count": len(entities),
            "entities": [format_entity(e) for e in entities],
        }

    async def _handle_trends(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle pka_trends tool call."""
        domain_str = arguments.get("domain", "tech")
        period = arguments.get("period", "week")
        direction_str = arguments.get("direction")
        limit = arguments.get("limit", 10)

        if not self._temporal_service:
            return {"error": "Database not connected"}

        domain = Domain(domain_str)

        # Calculate period days
        period_days = {"day": 1, "week": 7, "month": 30}.get(period, 7)

        # Get trends
        trends = await self._temporal_service.analyze_trends(
            domain=domain,
            period_days=period_days,
            limit=limit,
        )

        # Filter by direction if specified
        if direction_str:
            direction = TrendDirection(direction_str)
            trends = [t for t in trends if t.direction == direction]

        # Summarize by direction
        summary = {
            "rising": len([t for t in trends if t.direction == TrendDirection.RISING]),
            "falling": len([
                t for t in trends if t.direction == TrendDirection.FALLING
            ]),
            "stable": len([t for t in trends if t.direction == TrendDirection.STABLE]),
            "new": len([t for t in trends if t.direction == TrendDirection.NEW]),
        }

        return {
            "domain": domain_str,
            "period": period,
            "period_days": period_days,
            "summary": summary,
            "trends": [format_trend(t) for t in trends],
        }

    async def _handle_alerts(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle pka_alerts tool call."""
        level_str = arguments.get("level")
        status_str = arguments.get("status")
        limit = arguments.get("limit", 20)

        if not self._alert_manager:
            return {"error": "Alert manager not initialized"}

        # Get pending alerts
        alerts = self._alert_manager.get_pending_alerts()

        # Filter by level if specified
        if level_str:
            min_level = AlertLevel(level_str)
            level_priority = {
                AlertLevel.INFO: 0,
                AlertLevel.WATCH: 1,
                AlertLevel.ACTION: 2,
                AlertLevel.URGENT: 3,
            }
            min_priority = level_priority.get(min_level, 0)
            alerts = [
                a for a in alerts
                if level_priority.get(a.level, 0) >= min_priority
            ]

        # Filter by status if specified
        if status_str:
            status = AlertStatus(status_str)
            alerts = [a for a in alerts if a.status == status]

        # Apply limit
        alerts = alerts[:limit]

        # Get stats
        stats = self._alert_manager.get_stats()

        return {
            "count": len(alerts),
            "stats": {
                "total": stats["total_alerts"],
                "pending": stats["pending_count"],
                "actionable": stats["actionable_count"],
            },
            "alerts": [format_alert(a) for a in alerts],
        }

    async def _handle_report(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle pka_report tool call."""
        report_type_str = arguments.get("type", "weekly")
        period_days = arguments.get("period_days", 7)
        domain_str = arguments.get("domain")

        if not self._temporal_service:
            return {"error": "Database not connected"}

        # Generate a summary report
        now = datetime.now(UTC)
        period_start = now - timedelta(days=period_days)

        # Get "what's new" data
        whats_new = await self._temporal_service.get_whats_new(
            since=period_start,
            limit=20,
        )

        # Collect entities by domain
        domains_to_analyze = (
            [Domain(domain_str)] if domain_str else list(Domain)
        )

        domain_summaries = {}
        for domain in domains_to_analyze:
            trends = await self._temporal_service.analyze_trends(
                domain=domain,
                period_days=period_days,
                limit=5,
            )

            domain_summaries[domain.value] = {
                "total_trends": len(trends),
                "rising": [
                    t.entity_name for t in trends
                    if t.direction == TrendDirection.RISING
                ][:3],
                "new": [
                    t.entity_name for t in trends
                    if t.direction == TrendDirection.NEW
                ][:3],
            }

        return {
            "type": report_type_str,
            "period_start": period_start.isoformat(),
            "period_end": now.isoformat(),
            "period_days": period_days,
            "whats_new": {
                "new_entities": len(whats_new.new_entities),
                "updated_entities": len(whats_new.updated_entities),
                "new_relationships": len(whats_new.new_relationships),
            },
            "domains": domain_summaries,
        }

    async def _handle_entity(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle pka_entity tool call."""
        name = arguments.get("name", "")
        entity_type_str = arguments.get("entity_type")

        if not self._entity_repo or not self._temporal_service:
            return {"error": "Database not connected"}

        # Search for entity
        entity_type = EntityType(entity_type_str) if entity_type_str else None
        entities = await self._entity_repo.search_by_name(
            search_text=name,
            limit=5,
            entity_type=entity_type,
        )

        if not entities:
            return {"error": f"Entity not found: {name}"}

        # Get best match (exact match preferred)
        entity = next(
            (e for e in entities if e.name.lower() == name.lower()),
            entities[0],
        )

        # Get trend data for this entity
        trend_data = None
        if entity.id:
            trends = await self._temporal_service.analyze_trends(
                domain=entity.domain,
                period_days=7,
                limit=50,
            )
            entity_trend = next(
                (t for t in trends if t.entity_id == entity.id),
                None,
            )
            if entity_trend:
                trend_data = format_trend(entity_trend)

        result = format_entity(entity)
        result["trend"] = trend_data

        return result

    async def run(self) -> None:
        """Run the MCP server."""
        logger.info("Starting PKA MCP server")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    async def close(self) -> None:
        """Close database connections."""
        if self._db:
            await self._db.close()
            logger.info("MCP server database connection closed")


# ============================================================================
# Entry Point
# ============================================================================


async def main() -> None:
    """Main entry point for the MCP server."""
    server = PKAMCPServer()
    try:
        await server.run()
    finally:
        await server.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
