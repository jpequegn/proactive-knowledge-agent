"""Weekly Intelligence Report Generation System.

Generates comprehensive reports combining insights from all domain agents:
- Technology Intelligence
- Fitness Intelligence
- Finance Intelligence
- Cross-Dimensional Synthesis (when available)

Reports are generated in Markdown format and can be:
- Saved to a designated directory
- Archived for historical reference
- Formatted for email delivery
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.agents import (
    AgentReport,
    AlertLevel,
    Insight,
    ProjectIdea,
)
from src.world_model import Domain

logger = structlog.get_logger()


# ============================================================================
# Report Configuration
# ============================================================================


class ReportType(str, Enum):
    """Types of reports that can be generated."""

    WEEKLY = "weekly"
    DAILY = "daily"
    ON_DEMAND = "on_demand"


class ReportFormat(str, Enum):
    """Output formats for reports."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: Path = field(default_factory=lambda: Path("reports"))
    archive_dir: Path = field(default_factory=lambda: Path("reports/archive"))
    report_type: ReportType = ReportType.WEEKLY
    format: ReportFormat = ReportFormat.MARKDOWN
    include_project_ideas: bool = True
    max_insights_per_section: int = 10
    include_metadata: bool = False


# ============================================================================
# Report Models
# ============================================================================


@dataclass
class DomainSection:
    """A section of the report for a specific domain."""

    domain: Domain
    title: str
    insights: list[Insight]
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    project_ideas: list[ProjectIdea] = field(default_factory=list)


@dataclass
class WeeklyReport:
    """Complete weekly intelligence report."""

    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    sections: dict[Domain, DomainSection]
    cross_dimensional_insights: list[Insight] = field(default_factory=list)
    executive_summary: str = ""
    total_insights: int = 0
    urgent_count: int = 0
    action_count: int = 0

    @property
    def filename(self) -> str:
        """Generate filename for the report."""
        date_str = self.generated_at.strftime("%Y-%m-%d")
        return f"weekly_report_{date_str}.md"


# ============================================================================
# Report Generator
# ============================================================================


class WeeklyReportGenerator:
    """
    Generates weekly intelligence reports from agent reports.

    Combines insights from all domain agents into a unified report
    with executive summary, domain sections, and cross-dimensional insights.
    """

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure output directories exist."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.archive_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        agent_reports: dict[Domain, AgentReport],
        period_days: int = 7,
    ) -> WeeklyReport:
        """
        Generate a weekly report from agent reports.

        Args:
            agent_reports: Reports from each domain agent
            period_days: Number of days covered by the report

        Returns:
            Complete WeeklyReport object
        """
        now = datetime.now(UTC)
        period_start = now - timedelta(days=period_days)

        report_id = f"report_{now.strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            "Generating weekly report",
            report_id=report_id,
            domains=list(agent_reports.keys()),
        )

        # Build sections for each domain
        sections: dict[Domain, DomainSection] = {}

        for domain, agent_report in agent_reports.items():
            section = self._build_domain_section(domain, agent_report)
            sections[domain] = section

        # Extract cross-dimensional insights (if synthesis agent ran)
        cross_dimensional = self._extract_cross_dimensional_insights(agent_reports)

        # Calculate totals
        total_insights = sum(len(s.insights) for s in sections.values())
        total_insights += len(cross_dimensional)

        urgent_count = sum(
            1 for s in sections.values()
            for i in s.insights if i.level == AlertLevel.URGENT
        )
        action_count = sum(
            1 for s in sections.values()
            for i in s.insights if i.level == AlertLevel.ACTION
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            sections, cross_dimensional, urgent_count, action_count
        )

        report = WeeklyReport(
            report_id=report_id,
            generated_at=now,
            period_start=period_start,
            period_end=now,
            sections=sections,
            cross_dimensional_insights=cross_dimensional,
            executive_summary=executive_summary,
            total_insights=total_insights,
            urgent_count=urgent_count,
            action_count=action_count,
        )

        logger.info(
            "Weekly report generated",
            report_id=report_id,
            total_insights=total_insights,
            urgent=urgent_count,
            action=action_count,
        )

        return report

    def _build_domain_section(
        self, domain: Domain, agent_report: AgentReport
    ) -> DomainSection:
        """Build a report section for a domain."""
        # Get top insights (sorted by priority)
        sorted_insights = sorted(
            agent_report.insights,
            key=lambda i: i.priority_score,
            reverse=True,
        )
        top_insights = sorted_insights[:self.config.max_insights_per_section]

        # Generate domain-specific summary
        summary = self._generate_domain_summary(domain, agent_report)

        # Extract recommendations from insights
        recommendations = self._extract_recommendations(top_insights)

        # Domain title mapping
        titles = {
            Domain.TECH: "Technology Intelligence",
            Domain.FITNESS: "Fitness Intelligence",
            Domain.FINANCE: "Finance Intelligence",
            Domain.GENERAL: "General Intelligence",
        }

        return DomainSection(
            domain=domain,
            title=titles.get(domain, domain.value.title()),
            insights=top_insights,
            summary=summary,
            metrics={
                "total_insights": len(agent_report.insights),
                "trends_analyzed": agent_report.trends_analyzed,
                "entities_scanned": agent_report.entities_scanned,
                "alerts_generated": agent_report.alerts_generated,
            },
            recommendations=recommendations,
            project_ideas=agent_report.project_ideas[:3],  # Top 3 ideas
        )

    def _generate_domain_summary(
        self, domain: Domain, agent_report: AgentReport
    ) -> str:
        """Generate a summary for a domain section."""
        insights = agent_report.insights
        if not insights:
            return f"No significant {domain.value} insights this period."

        urgent = len([i for i in insights if i.level == AlertLevel.URGENT])
        action = len([i for i in insights if i.level == AlertLevel.ACTION])
        watch = len([i for i in insights if i.level == AlertLevel.WATCH])

        parts = []
        if urgent > 0:
            parts.append(f"{urgent} urgent issue{'s' if urgent > 1 else ''}")
        if action > 0:
            parts.append(f"{action} action item{'s' if action > 1 else ''}")
        if watch > 0:
            parts.append(f"{watch} item{'s' if watch > 1 else ''} to watch")

        summary = f"Found {len(insights)} insights"
        if parts:
            summary += f": {', '.join(parts)}"

        return summary + "."

    def _extract_recommendations(self, insights: list[Insight]) -> list[str]:
        """Extract actionable recommendations from insights."""
        recommendations = []

        for insight in insights:
            if insight.level in (AlertLevel.ACTION, AlertLevel.URGENT):
                # Extract action from metadata if available
                actions = insight.metadata.get("recommended_actions", [])
                if actions:
                    recommendations.extend(actions[:2])  # Top 2 per insight
                else:
                    # Use insight title as recommendation
                    recommendations.append(insight.title)

        return recommendations[:5]  # Max 5 recommendations

    def _extract_cross_dimensional_insights(
        self, agent_reports: dict[Domain, AgentReport]
    ) -> list[Insight]:
        """Extract cross-dimensional insights from synthesis agent."""
        # Look for GENERAL domain (synthesis agent)
        synthesis_report = agent_reports.get(Domain.GENERAL)
        if synthesis_report:
            return synthesis_report.insights

        # If no synthesis report, look for cross-domain insights in any report
        cross_dimensional = []
        for report in agent_reports.values():
            for insight in report.insights:
                if insight.insight_type.value == "cross_domain":
                    cross_dimensional.append(insight)

        return cross_dimensional

    def _generate_executive_summary(
        self,
        sections: dict[Domain, DomainSection],
        cross_dimensional: list[Insight],
        urgent_count: int,
        action_count: int,
    ) -> str:
        """Generate executive summary for the report."""
        lines = []

        # Overall status
        if urgent_count > 0:
            lines.append(
                f"**Attention Required**: {urgent_count} urgent "
                f"issue{'s' if urgent_count > 1 else ''} detected."
            )
        elif action_count > 0:
            s = "s" if action_count > 1 else ""
            lines.append(f"**Action Items**: {action_count} item{s} require attention.")
        else:
            lines.append("**Status**: All systems nominal. No urgent issues.")

        lines.append("")

        # Domain highlights
        for domain, section in sections.items():
            if section.insights:
                top_insight = section.insights[0]
                emoji = self._get_domain_emoji(domain)
                lines.append(
                    f"{emoji} **{section.title}**: {top_insight.title}"
                )

        # Cross-dimensional highlight
        if cross_dimensional:
            top_cross = cross_dimensional[0]
            lines.append(f"🔗 **Cross-Domain**: {top_cross.title}")

        return "\n".join(lines)

    def _get_domain_emoji(self, domain: Domain) -> str:
        """Get emoji for domain."""
        return {
            Domain.TECH: "💻",
            Domain.FITNESS: "🏃",
            Domain.FINANCE: "📈",
            Domain.GENERAL: "🔗",
        }.get(domain, "📊")

    # ========================================================================
    # Formatting
    # ========================================================================

    def format_markdown(self, report: WeeklyReport) -> str:
        """Format report as Markdown."""
        lines = []

        # Header
        lines.append(
            f"# Weekly Intelligence Report - "
            f"{report.generated_at.strftime('%Y-%m-%d')}"
        )
        lines.append("")
        period_str = (
            f"{report.period_start.strftime('%Y-%m-%d')} to "
            f"{report.period_end.strftime('%Y-%m-%d')}"
        )
        lines.append(f"*Report Period: {period_str}*")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(report.executive_summary)
        lines.append("")

        # Domain sections
        domain_order = [Domain.TECH, Domain.FITNESS, Domain.FINANCE]
        for domain in domain_order:
            if domain in report.sections:
                section = report.sections[domain]
                lines.extend(self._format_domain_section(section))

        # Cross-Dimensional section
        if report.cross_dimensional_insights:
            lines.append("## Cross-Dimensional Insights")
            lines.append("")
            for insight in report.cross_dimensional_insights[:5]:
                lines.extend(self._format_insight(insight))

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(
            f"*Generated at {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Report ID: {report.report_id}*"
        )

        return "\n".join(lines)

    def _format_domain_section(self, section: DomainSection) -> list[str]:
        """Format a domain section."""
        lines = []
        emoji = self._get_domain_emoji(section.domain)

        lines.append(f"## {emoji} {section.title}")
        lines.append("")
        lines.append(section.summary)
        lines.append("")

        # Insights
        if section.insights:
            for insight in section.insights[:5]:  # Top 5
                lines.extend(self._format_insight(insight))

        # Recommendations
        if section.recommendations:
            lines.append("### Recommendations")
            lines.append("")
            for rec in section.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Project Ideas
        if self.config.include_project_ideas and section.project_ideas:
            lines.append("### Project Ideas")
            lines.append("")
            for idea in section.project_ideas:
                lines.append(f"**{idea.title}** ({idea.difficulty})")
                lines.append(f": {idea.description[:100]}...")
                lines.append("")

        return lines

    def _format_insight(self, insight: Insight) -> list[str]:
        """Format a single insight."""
        lines = []
        emoji = {
            AlertLevel.URGENT: "🚨",
            AlertLevel.ACTION: "⚡",
            AlertLevel.WATCH: "👀",
            AlertLevel.INFO: "ℹ️",
        }.get(insight.level, "📊")

        lines.append(f"### {emoji} {insight.title}")
        lines.append("")
        lines.append(
            f"**Level**: {insight.level.value.upper()} | "
            f"**Confidence**: {insight.confidence:.0%}"
        )
        lines.append("")
        lines.append(insight.description)
        lines.append("")

        return lines

    # ========================================================================
    # Storage & Archiving
    # ========================================================================

    def save_report(self, report: WeeklyReport) -> Path:
        """
        Save report to the output directory.

        Args:
            report: The report to save

        Returns:
            Path to the saved report
        """
        content = self.format_markdown(report)
        output_path = self.config.output_dir / report.filename

        output_path.write_text(content, encoding="utf-8")

        logger.info(
            "Report saved",
            path=str(output_path),
            report_id=report.report_id,
        )

        return output_path

    def archive_report(self, report: WeeklyReport) -> Path:
        """
        Archive report to the archive directory.

        Creates a dated subdirectory structure: archive/YYYY/MM/

        Args:
            report: The report to archive

        Returns:
            Path to the archived report
        """
        # Create dated subdirectory
        year_month = report.generated_at.strftime("%Y/%m")
        archive_subdir = self.config.archive_dir / year_month
        archive_subdir.mkdir(parents=True, exist_ok=True)

        content = self.format_markdown(report)
        archive_path = archive_subdir / report.filename

        archive_path.write_text(content, encoding="utf-8")

        logger.info(
            "Report archived",
            path=str(archive_path),
            report_id=report.report_id,
        )

        return archive_path

    def save_and_archive(self, report: WeeklyReport) -> tuple[Path, Path]:
        """
        Save report to output and archive directories.

        Args:
            report: The report to save

        Returns:
            Tuple of (output_path, archive_path)
        """
        output_path = self.save_report(report)
        archive_path = self.archive_report(report)
        return output_path, archive_path

    def list_archived_reports(
        self,
        year: int | None = None,
        month: int | None = None,
    ) -> list[Path]:
        """
        List archived reports.

        Args:
            year: Optional year filter
            month: Optional month filter (requires year)

        Returns:
            List of paths to archived reports
        """
        if year and month:
            search_dir = self.config.archive_dir / f"{year}/{month:02d}"
        elif year:
            search_dir = self.config.archive_dir / str(year)
        else:
            search_dir = self.config.archive_dir

        if not search_dir.exists():
            return []

        return sorted(search_dir.glob("**/*.md"), reverse=True)

    def get_latest_report(self) -> Path | None:
        """Get the path to the most recent report in output directory."""
        reports = sorted(self.config.output_dir.glob("*.md"), reverse=True)
        return reports[0] if reports else None


# ============================================================================
# Convenience Functions
# ============================================================================


def generate_weekly_report(
    agent_reports: dict[Domain, AgentReport],
    output_dir: Path | None = None,
    archive: bool = True,
) -> tuple[WeeklyReport, Path]:
    """
    Generate and save a weekly report.

    Convenience function for common use case.

    Args:
        agent_reports: Reports from domain agents
        output_dir: Optional output directory
        archive: Whether to also archive the report

    Returns:
        Tuple of (WeeklyReport, output_path)
    """
    config = ReportConfig()
    if output_dir:
        config.output_dir = output_dir
        config.archive_dir = output_dir / "archive"

    generator = WeeklyReportGenerator(config)
    report = generator.generate_report(agent_reports)

    if archive:
        output_path, _ = generator.save_and_archive(report)
    else:
        output_path = generator.save_report(report)

    return report, output_path
