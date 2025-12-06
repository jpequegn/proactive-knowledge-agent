"""Tests for Weekly Report Generation System."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.agents import (
    AgentReport,
    AlertLevel,
    Insight,
    InsightType,
    ProjectIdea,
)
from src.outputs import (
    DomainSection,
    ReportConfig,
    ReportFormat,
    ReportType,
    WeeklyReport,
    WeeklyReportGenerator,
    generate_weekly_report,
)
from src.world_model import Domain

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_insights():
    """Create sample insights for testing."""
    return [
        Insight(
            insight_type=InsightType.EMERGING_TECH,
            title="Critical Security Update",
            description="A critical security vulnerability was found.",
            level=AlertLevel.URGENT,
            confidence=0.95,
            relevance_score=0.95,
            source_ids=["article_1"],
            metadata={"recommended_actions": ["Update immediately", "Review access"]},
        ),
        Insight(
            insight_type=InsightType.NEW_FRAMEWORK,
            title="New Framework Release",
            description="React 19 has been released with new features.",
            level=AlertLevel.ACTION,
            confidence=0.85,
            relevance_score=0.80,
            source_ids=["article_2"],
            metadata={},
        ),
        Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Performance Optimization Tips",
            description="New techniques for optimizing Python code.",
            level=AlertLevel.WATCH,
            confidence=0.80,
            relevance_score=0.70,
            source_ids=["article_3"],
            metadata={},
        ),
    ]


@pytest.fixture
def sample_fitness_insights():
    """Create sample fitness insights."""
    return [
        Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Training Load Warning",
            description="Your training load is elevated.",
            level=AlertLevel.ACTION,
            confidence=0.90,
            relevance_score=0.85,
            source_ids=["strava_data"],
            metadata={"recommended_actions": ["Take a rest day"]},
        ),
    ]


@pytest.fixture
def sample_finance_insights():
    """Create sample finance insights."""
    return [
        Insight(
            insight_type=InsightType.ANOMALY,
            title="Market Volatility Alert",
            description="VIX has spiked significantly.",
            level=AlertLevel.WATCH,
            confidence=0.80,
            relevance_score=0.70,
            source_ids=["market_data"],
            metadata={},
        ),
    ]


@pytest.fixture
def sample_project_ideas():
    """Create sample project ideas."""
    return [
        ProjectIdea(
            title="AI-Powered Code Review",
            description="Build a tool that uses AI to review code.",
            technologies=["Python", "LLM APIs"],
            learning_path=["AI integration", "Code analysis"],
            difficulty="intermediate",
            estimated_hours=40,
            rationale="AI code review tools are increasingly valuable.",
            source_trends=["LLM adoption", "Developer productivity"],
            relevance_score=0.85,
        ),
    ]


@pytest.fixture
def sample_agent_report(sample_insights, sample_project_ideas):
    """Create a sample agent report."""
    return AgentReport(
        agent_name="TechIntelligenceAgent",
        domain=Domain.TECH,
        run_at=datetime.now(UTC),
        insights=sample_insights,
        trends_analyzed=10,
        entities_scanned=25,
        alerts_generated=3,
        project_ideas=sample_project_ideas,
    )


@pytest.fixture
def sample_fitness_report(sample_fitness_insights):
    """Create a sample fitness agent report."""
    return AgentReport(
        agent_name="FitnessIntelligenceAgent",
        domain=Domain.FITNESS,
        run_at=datetime.now(UTC),
        insights=sample_fitness_insights,
        trends_analyzed=5,
        entities_scanned=15,
        alerts_generated=1,
        project_ideas=[],
    )


@pytest.fixture
def sample_finance_report(sample_finance_insights):
    """Create a sample finance agent report."""
    return AgentReport(
        agent_name="FinanceIntelligenceAgent",
        domain=Domain.FINANCE,
        run_at=datetime.now(UTC),
        insights=sample_finance_insights,
        trends_analyzed=8,
        entities_scanned=20,
        alerts_generated=1,
        project_ideas=[],
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "reports"
    archive_dir = tmp_path / "reports" / "archive"
    output_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    return output_dir, archive_dir


# ============================================================================
# ReportConfig Tests
# ============================================================================


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()

        assert config.output_dir == Path("reports")
        assert config.archive_dir == Path("reports/archive")
        assert config.report_type == ReportType.WEEKLY
        assert config.format == ReportFormat.MARKDOWN
        assert config.include_project_ideas is True
        assert config.max_insights_per_section == 10
        assert config.include_metadata is False

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = ReportConfig(
            output_dir=tmp_path / "custom",
            archive_dir=tmp_path / "custom/archive",
            report_type=ReportType.DAILY,
            format=ReportFormat.HTML,
            include_project_ideas=False,
            max_insights_per_section=5,
        )

        assert config.output_dir == tmp_path / "custom"
        assert config.report_type == ReportType.DAILY
        assert config.include_project_ideas is False
        assert config.max_insights_per_section == 5


# ============================================================================
# WeeklyReport Tests
# ============================================================================


class TestWeeklyReport:
    """Tests for WeeklyReport model."""

    def test_filename_generation(self):
        """Test filename generation from date."""
        now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        report = WeeklyReport(
            report_id="test_report",
            generated_at=now,
            period_start=now - timedelta(days=7),
            period_end=now,
            sections={},
        )

        assert report.filename == "weekly_report_2024-01-15.md"

    def test_report_with_sections(self, sample_insights):
        """Test report with domain sections."""
        now = datetime.now(UTC)
        section = DomainSection(
            domain=Domain.TECH,
            title="Technology Intelligence",
            insights=sample_insights,
            summary="Found 3 insights.",
        )

        report = WeeklyReport(
            report_id="test_report",
            generated_at=now,
            period_start=now - timedelta(days=7),
            period_end=now,
            sections={Domain.TECH: section},
            total_insights=3,
            urgent_count=1,
            action_count=1,
        )

        assert len(report.sections) == 1
        assert report.total_insights == 3
        assert report.urgent_count == 1


# ============================================================================
# DomainSection Tests
# ============================================================================


class TestDomainSection:
    """Tests for DomainSection model."""

    def test_domain_section_creation(self, sample_insights, sample_project_ideas):
        """Test creating a domain section."""
        section = DomainSection(
            domain=Domain.TECH,
            title="Technology Intelligence",
            insights=sample_insights,
            summary="Found 3 insights.",
            metrics={"total_insights": 3},
            recommendations=["Update systems"],
            project_ideas=sample_project_ideas,
        )

        assert section.domain == Domain.TECH
        assert section.title == "Technology Intelligence"
        assert len(section.insights) == 3
        assert len(section.project_ideas) == 1


# ============================================================================
# WeeklyReportGenerator Tests
# ============================================================================


class TestWeeklyReportGenerator:
    """Tests for WeeklyReportGenerator."""

    def test_generator_initialization(self, tmp_output_dir):
        """Test generator initialization creates directories."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)

        generator = WeeklyReportGenerator(config)

        assert generator.config.output_dir.exists()
        assert generator.config.archive_dir.exists()

    def test_generate_report_single_domain(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test generating report with single domain."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)

        assert report.report_id.startswith("report_")
        assert Domain.TECH in report.sections
        assert report.total_insights == 3
        assert report.urgent_count == 1
        assert report.action_count == 1

    def test_generate_report_multiple_domains(
        self,
        tmp_output_dir,
        sample_agent_report,
        sample_fitness_report,
        sample_finance_report,
    ):
        """Test generating report with multiple domains."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {
            Domain.TECH: sample_agent_report,
            Domain.FITNESS: sample_fitness_report,
            Domain.FINANCE: sample_finance_report,
        }
        report = generator.generate_report(agent_reports)

        assert len(report.sections) == 3
        assert Domain.TECH in report.sections
        assert Domain.FITNESS in report.sections
        assert Domain.FINANCE in report.sections
        assert report.total_insights == 5  # 3 + 1 + 1

    def test_generate_executive_summary_with_urgent(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test executive summary generation with urgent items."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)

        assert "Attention Required" in report.executive_summary
        assert "urgent" in report.executive_summary.lower()

    def test_generate_executive_summary_nominal(self, tmp_output_dir):
        """Test executive summary when all systems nominal."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        # Create report with only INFO level insights
        info_insight = Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Information Only",
            description="Just FYI.",
            level=AlertLevel.INFO,
            confidence=0.80,
            relevance_score=0.50,
            source_ids=["source"],
            metadata={},
        )

        now = datetime.now(UTC)
        agent_report = AgentReport(
            agent_name="TechAgent",
            domain=Domain.TECH,
            run_at=now,
            insights=[info_insight],
            trends_analyzed=1,
            entities_scanned=1,
            alerts_generated=0,
            project_ideas=[],
        )

        report = generator.generate_report({Domain.TECH: agent_report})

        assert "All systems nominal" in report.executive_summary

    def test_format_markdown(self, tmp_output_dir, sample_agent_report):
        """Test Markdown formatting."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        markdown = generator.format_markdown(report)

        assert "# Weekly Intelligence Report" in markdown
        assert "## Executive Summary" in markdown
        assert "## " in markdown  # Domain section header
        assert "Critical Security Update" in markdown
        assert "**Level**: URGENT" in markdown

    def test_format_markdown_with_project_ideas(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test Markdown includes project ideas when enabled."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(
            output_dir=output_dir,
            archive_dir=archive_dir,
            include_project_ideas=True,
        )
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        markdown = generator.format_markdown(report)

        assert "### Project Ideas" in markdown
        assert "AI-Powered Code Review" in markdown

    def test_save_report(self, tmp_output_dir, sample_agent_report):
        """Test saving report to output directory."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        output_path = generator.save_report(report)

        assert output_path.exists()
        assert output_path.parent == output_dir
        assert output_path.suffix == ".md"

        content = output_path.read_text()
        assert "Weekly Intelligence Report" in content

    def test_archive_report(self, tmp_output_dir, sample_agent_report):
        """Test archiving report."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        archive_path = generator.archive_report(report)

        assert archive_path.exists()
        # Should be in archive/YYYY/MM/ structure
        assert archive_dir in archive_path.parents or archive_path.is_relative_to(
            archive_dir
        )

    def test_save_and_archive(self, tmp_output_dir, sample_agent_report):
        """Test saving and archiving together."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        output_path, archive_path = generator.save_and_archive(report)

        assert output_path.exists()
        assert archive_path.exists()

        # Both should have same content
        assert output_path.read_text() == archive_path.read_text()

    def test_list_archived_reports_empty(self, tmp_output_dir):
        """Test listing archived reports when empty."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        reports = generator.list_archived_reports()
        assert reports == []

    def test_list_archived_reports_with_filter(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test listing archived reports with year/month filter."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        # Create and archive a report
        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        generator.archive_report(report)

        # List all
        all_reports = generator.list_archived_reports()
        assert len(all_reports) == 1

        # List by year
        year = report.generated_at.year
        year_reports = generator.list_archived_reports(year=year)
        assert len(year_reports) == 1

    def test_get_latest_report(self, tmp_output_dir, sample_agent_report):
        """Test getting the latest report."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        # No reports yet
        assert generator.get_latest_report() is None

        # Save a report
        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)
        generator.save_report(report)

        latest = generator.get_latest_report()
        assert latest is not None
        assert latest.name == report.filename

    def test_domain_summary_generation(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test domain summary generation."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)

        section = report.sections[Domain.TECH]
        assert "Found 3 insights" in section.summary
        assert "urgent" in section.summary.lower()

    def test_recommendations_extraction(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test recommendations are extracted from insights."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)

        section = report.sections[Domain.TECH]
        # Should have extracted recommended actions
        assert len(section.recommendations) > 0

    def test_insights_sorted_by_priority(
        self, tmp_output_dir, sample_agent_report
    ):
        """Test insights are sorted by priority score."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        agent_reports = {Domain.TECH: sample_agent_report}
        report = generator.generate_report(agent_reports)

        section = report.sections[Domain.TECH]
        priorities = [i.priority_score for i in section.insights]

        # Should be sorted descending
        assert priorities == sorted(priorities, reverse=True)


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestGenerateWeeklyReportFunction:
    """Tests for generate_weekly_report convenience function."""

    def test_generate_with_archive(self, tmp_output_dir, sample_agent_report):
        """Test convenience function with archiving."""
        output_dir, _ = tmp_output_dir
        agent_reports = {Domain.TECH: sample_agent_report}

        report, output_path = generate_weekly_report(
            agent_reports,
            output_dir=output_dir,
            archive=True,
        )

        assert report is not None
        assert output_path.exists()

        # Archive should also exist
        archive_dir = output_dir / "archive"
        archive_files = list(archive_dir.glob("**/*.md"))
        assert len(archive_files) == 1

    def test_generate_without_archive(self, tmp_output_dir, sample_agent_report):
        """Test convenience function without archiving."""
        output_dir, _ = tmp_output_dir
        agent_reports = {Domain.TECH: sample_agent_report}

        report, output_path = generate_weekly_report(
            agent_reports,
            output_dir=output_dir,
            archive=False,
        )

        assert output_path.exists()

        # Archive should be empty
        archive_dir = output_dir / "archive"
        archive_files = list(archive_dir.glob("**/*.md"))
        assert len(archive_files) == 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_agent_reports(self, tmp_output_dir):
        """Test handling of empty agent reports."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        report = generator.generate_report({})

        assert report.total_insights == 0
        assert len(report.sections) == 0

    def test_agent_report_with_no_insights(self, tmp_output_dir):
        """Test agent report with no insights."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        now = datetime.now(UTC)
        empty_report = AgentReport(
            agent_name="EmptyAgent",
            domain=Domain.TECH,
            run_at=now,
            insights=[],
            trends_analyzed=0,
            entities_scanned=0,
            alerts_generated=0,
            project_ideas=[],
        )

        report = generator.generate_report({Domain.TECH: empty_report})

        section = report.sections[Domain.TECH]
        assert "No significant" in section.summary

    def test_max_insights_limit(self, tmp_output_dir):
        """Test that max_insights_per_section is respected."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(
            output_dir=output_dir,
            archive_dir=archive_dir,
            max_insights_per_section=2,
        )
        generator = WeeklyReportGenerator(config)

        # Create many insights
        insights = []
        for i in range(10):
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title=f"Insight {i}",
                    description=f"Description {i}",
                    level=AlertLevel.INFO,
                    confidence=0.80,
                    relevance_score=0.50 + i * 0.01,
                    source_ids=["source"],
                    metadata={},
                )
            )

        now = datetime.now(UTC)
        agent_report = AgentReport(
            agent_name="TestAgent",
            domain=Domain.TECH,
            run_at=now,
            insights=insights,
            trends_analyzed=10,
            entities_scanned=10,
            alerts_generated=0,
            project_ideas=[],
        )

        report = generator.generate_report({Domain.TECH: agent_report})

        # Section should only have 2 insights
        assert len(report.sections[Domain.TECH].insights) == 2

    def test_cross_dimensional_insights(self, tmp_output_dir):
        """Test extraction of cross-dimensional insights."""
        output_dir, archive_dir = tmp_output_dir
        config = ReportConfig(output_dir=output_dir, archive_dir=archive_dir)
        generator = WeeklyReportGenerator(config)

        cross_insight = Insight(
            insight_type=InsightType.CROSS_DOMAIN,
            title="Cross-Domain Pattern",
            description="Pattern across domains.",
            level=AlertLevel.ACTION,
            confidence=0.85,
            relevance_score=0.80,
            source_ids=["multiple"],
            metadata={},
        )

        now = datetime.now(UTC)
        synthesis_report = AgentReport(
            agent_name="SynthesisAgent",
            domain=Domain.GENERAL,
            run_at=now,
            insights=[cross_insight],
            trends_analyzed=0,
            entities_scanned=0,
            alerts_generated=1,
            project_ideas=[],
        )

        report = generator.generate_report({Domain.GENERAL: synthesis_report})

        assert len(report.cross_dimensional_insights) == 1
        assert report.cross_dimensional_insights[0].title == "Cross-Domain Pattern"
