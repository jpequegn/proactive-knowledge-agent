"""Tests for Synthesis Intelligence Agent."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents import (
    AgentConfig,
    AlertLevel,
    Insight,
    InsightType,
    UserProfile,
)
from src.agents.synthesis_agent import (
    CrossDomainPattern,
    DomainSignal,
    DomainState,
    SynthesisAgent,
    SynthesisInsightType,
)
from src.world_model import Domain

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=0)
    return db


@pytest.fixture
def default_config():
    """Create default agent configuration."""
    return AgentConfig(
        enabled=True,
        schedule="0 8 * * *",
        thresholds={
            "min_domains": 2,
            "impact_threshold": 0.5,
            "lookback_hours": 24,
            "min_confidence": 0.5,
            "min_relevance": 0.3,
        },
        focus_areas=[],
        output={"weekly_report": True, "cross_domain_alerts": True},
    )


@pytest.fixture
def user_profile():
    """Create a sample user profile."""
    return UserProfile(
        role="developer",
        experience_level="senior",
        interests=["AI/ML", "Python", "Running"],
        known_technologies=["Python", "TypeScript", "React"],
        learning_goals=["Rust", "Machine Learning"],
    )


@pytest.fixture
def tech_insights():
    """Create sample tech domain insights."""
    return [
        Insight(
            insight_type=InsightType.EMERGING_TECH,
            title="LangChain Rising",
            description="LangChain adoption increasing 150%",
            level=AlertLevel.ACTION,
            confidence=0.85,
            relevance_score=0.9,
            entity_names=["LangChain"],
            metadata={"category": "ai_ml"},
        ),
        Insight(
            insight_type=InsightType.SKILL_GAP,
            title="Rust Skill Gap Detected",
            description="Rust is trending but not in your skillset",
            level=AlertLevel.WATCH,
            confidence=0.7,
            relevance_score=0.8,
            entity_names=["Rust"],
            metadata={},
        ),
    ]


@pytest.fixture
def fitness_insights():
    """Create sample fitness domain insights."""
    return [
        Insight(
            insight_type=InsightType.ANOMALY,
            title="Training Overload Detected",
            description="ATL:CTL ratio above safe threshold",
            level=AlertLevel.ACTION,
            confidence=0.9,
            relevance_score=0.85,
            metadata={
                "fitness_insight_type": "overtraining_risk",
                "training_zone": "OVERREACHED",
            },
        ),
        Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Recovery Week Needed",
            description="Training stress accumulated, rest recommended",
            level=AlertLevel.WATCH,
            confidence=0.75,
            relevance_score=0.8,
            metadata={"training_zone": "TIRED"},
        ),
    ]


@pytest.fixture
def finance_insights():
    """Create sample finance domain insights."""
    return [
        Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Market Volatility Spike",
            description="VIX increased significantly",
            level=AlertLevel.WATCH,
            confidence=0.8,
            relevance_score=0.7,
            metadata={
                "market_regime": "VOLATILE",
                "risk_level": "HIGH",
            },
        ),
    ]


@pytest.fixture
def positive_finance_insights():
    """Create positive finance domain insights."""
    return [
        Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Bull Market Confirmed",
            description="Strong uptrend across indices",
            level=AlertLevel.INFO,
            confidence=0.85,
            relevance_score=0.75,
            metadata={"market_regime": "BULL"},
        ),
        Insight(
            insight_type=InsightType.EMERGING_TECH,
            title="Fintech Opportunity",
            description="AI fintech sector showing growth",
            level=AlertLevel.ACTION,
            confidence=0.8,
            relevance_score=0.8,
            metadata={},
        ),
    ]


# ============================================================================
# Test Synthesis-Specific Models
# ============================================================================


class TestDomainSignal:
    """Tests for DomainSignal enum."""

    def test_tech_signals_exist(self):
        """Test tech domain signals are defined."""
        assert DomainSignal.TECH_TREND_RISING
        assert DomainSignal.TECH_SKILL_GAP
        assert DomainSignal.TECH_LEARNING_OPPORTUNITY

    def test_fitness_signals_exist(self):
        """Test fitness domain signals are defined."""
        assert DomainSignal.FITNESS_OVERTRAINING
        assert DomainSignal.FITNESS_PEAK_FORM
        assert DomainSignal.FITNESS_RECOVERY_NEEDED

    def test_finance_signals_exist(self):
        """Test finance domain signals are defined."""
        assert DomainSignal.FINANCE_BULL_MARKET
        assert DomainSignal.FINANCE_BEAR_MARKET
        assert DomainSignal.FINANCE_HIGH_VOLATILITY


class TestSynthesisInsightType:
    """Tests for SynthesisInsightType enum."""

    def test_insight_types_exist(self):
        """Test all synthesis insight types are defined."""
        assert SynthesisInsightType.COMPOUND_RISK
        assert SynthesisInsightType.COMPOUND_OPPORTUNITY
        assert SynthesisInsightType.DOMAIN_CONFLICT
        assert SynthesisInsightType.TIMING_ALIGNMENT
        assert SynthesisInsightType.LIFESTYLE_BALANCE


class TestDomainState:
    """Tests for DomainState dataclass."""

    def test_domain_state_creation(self, tech_insights):
        """Test creating a domain state."""
        state = DomainState(
            domain=Domain.TECH,
            active_signals=[DomainSignal.TECH_TREND_RISING],
            risk_level=0.3,
            opportunity_level=0.7,
            recent_insights=tech_insights,
        )

        assert state.domain == Domain.TECH
        assert DomainSignal.TECH_TREND_RISING in state.active_signals
        assert state.risk_level == 0.3
        assert state.opportunity_level == 0.7
        assert len(state.recent_insights) == 2


class TestCrossDomainPattern:
    """Tests for CrossDomainPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a cross-domain pattern."""
        pattern = CrossDomainPattern(
            pattern_type=SynthesisInsightType.COMPOUND_RISK,
            domains_involved=[Domain.TECH, Domain.FITNESS],
            signals=[DomainSignal.TECH_SKILL_GAP, DomainSignal.FITNESS_OVERTRAINING],
            description="High stress across domains",
            reasoning="Work pressure combined with training overload",
            impact_score=0.8,
            confidence=0.75,
            recommended_actions=["Reduce training", "Focus on core work"],
        )

        assert pattern.pattern_type == SynthesisInsightType.COMPOUND_RISK
        assert len(pattern.domains_involved) == 2
        assert pattern.impact_score == 0.8


# ============================================================================
# Test Synthesis Intelligence Agent
# ============================================================================


class TestSynthesisAgent:
    """Tests for SynthesisAgent."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent with mocked dependencies."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    @pytest.fixture
    def agent_with_insights(
        self, mock_db, default_config, user_profile,
        tech_insights, fitness_insights, finance_insights
    ):
        """Create agent with pre-loaded domain insights."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
            domain_insights={
                Domain.TECH: tech_insights,
                Domain.FITNESS: fitness_insights,
                Domain.FINANCE: finance_insights,
            },
        )

    async def test_agent_initialization(self, agent, default_config):
        """Test agent initializes with correct config."""
        assert agent.name == "synthesis_intelligence"
        assert agent.domain == Domain.GENERAL
        assert agent.min_domains_for_pattern == 2
        assert agent.impact_threshold == 0.5
        assert agent.lookback_hours == 24

    async def test_agent_with_no_insights(self, agent):
        """Test agent handles empty domain insights."""
        corr_svc = agent.correlation_service
        corr_svc.find_cross_source_entities = AsyncMock(return_value=[])

        data = await agent.monitor()

        assert "domain_states" in data
        assert len(data["domain_states"]) == 3  # tech, fitness, finance

        # All states should have no signals
        for domain, state in data["domain_states"].items():
            assert len(state.active_signals) == 0


class TestSignalExtraction:
    """Tests for signal extraction from insights."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent for signal extraction tests."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    def test_extract_tech_signals_emerging(self, agent):
        """Test extracting tech signals from emerging tech insight."""
        insight = Insight(
            insight_type=InsightType.EMERGING_TECH,
            title="New Framework Rising",
            description="Test",
            level=AlertLevel.ACTION,
            confidence=0.8,
            relevance_score=0.9,
        )

        signals = agent._extract_tech_signals(insight)

        assert DomainSignal.TECH_TREND_RISING in signals

    def test_extract_tech_signals_skill_gap(self, agent):
        """Test extracting tech signals from skill gap insight."""
        insight = Insight(
            insight_type=InsightType.SKILL_GAP,
            title="Skill Gap Detected",
            description="Test",
            level=AlertLevel.WATCH,
            confidence=0.7,
            relevance_score=0.8,
        )

        signals = agent._extract_tech_signals(insight)

        assert DomainSignal.TECH_SKILL_GAP in signals

    def test_extract_tech_signals_learning_opportunity(self, agent):
        """Test high relevance action insight triggers learning opportunity."""
        insight = Insight(
            insight_type=InsightType.EMERGING_TECH,
            title="Hot Technology",
            description="Test",
            level=AlertLevel.ACTION,
            confidence=0.9,
            relevance_score=0.85,
        )

        signals = agent._extract_tech_signals(insight)

        assert DomainSignal.TECH_LEARNING_OPPORTUNITY in signals

    def test_extract_fitness_signals_overtraining(self, agent):
        """Test extracting overtraining signal."""
        insight = Insight(
            insight_type=InsightType.ANOMALY,
            title="Overtraining Risk",
            description="Test",
            level=AlertLevel.ACTION,
            confidence=0.9,
            relevance_score=0.85,
            metadata={"training_zone": "OVERREACHED"},
        )

        signals = agent._extract_fitness_signals(insight)

        assert DomainSignal.FITNESS_OVERTRAINING in signals

    def test_extract_fitness_signals_peak_form(self, agent):
        """Test extracting peak form signal."""
        insight = Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Peak Performance Window",
            description="Test",
            level=AlertLevel.INFO,
            confidence=0.8,
            relevance_score=0.9,
            metadata={"training_zone": "OPTIMAL"},
        )

        signals = agent._extract_fitness_signals(insight)

        assert DomainSignal.FITNESS_PEAK_FORM in signals

    def test_extract_fitness_signals_recovery(self, agent):
        """Test extracting recovery signal."""
        insight = Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Recovery Recommended",
            description="Test",
            level=AlertLevel.WATCH,
            confidence=0.75,
            relevance_score=0.8,
            metadata={},
        )

        signals = agent._extract_fitness_signals(insight)

        assert DomainSignal.FITNESS_RECOVERY_NEEDED in signals

    def test_extract_finance_signals_bull_market(self, agent):
        """Test extracting bull market signal."""
        insight = Insight(
            insight_type=InsightType.TREND_CHANGE,
            title="Market Uptrend",
            description="Test",
            level=AlertLevel.INFO,
            confidence=0.85,
            relevance_score=0.7,
            metadata={"market_regime": "BULL"},
        )

        signals = agent._extract_finance_signals(insight)

        assert DomainSignal.FINANCE_BULL_MARKET in signals

    def test_extract_finance_signals_volatility(self, agent):
        """Test extracting high volatility signal."""
        insight = Insight(
            insight_type=InsightType.ANOMALY,
            title="Volatility Spike Detected",
            description="Test",
            level=AlertLevel.WATCH,
            confidence=0.8,
            relevance_score=0.75,
            metadata={"market_regime": "VOLATILE"},
        )

        signals = agent._extract_finance_signals(insight)

        assert DomainSignal.FINANCE_HIGH_VOLATILITY in signals

    def test_extract_finance_signals_high_risk(self, agent):
        """Test extracting high risk signal."""
        insight = Insight(
            insight_type=InsightType.ANOMALY,
            title="Risk Alert",
            description="Test",
            level=AlertLevel.URGENT,
            confidence=0.9,
            relevance_score=0.85,
            metadata={"risk_level": "HIGH"},
        )

        signals = agent._extract_finance_signals(insight)

        assert DomainSignal.FINANCE_RISK_HIGH in signals


class TestPatternMatching:
    """Tests for cross-domain pattern matching."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent for pattern matching tests."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    def test_match_stress_overload_pattern(self, agent):
        """Test matching stress overload pattern."""
        from src.agents.synthesis_agent import CROSS_DOMAIN_PATTERNS

        stress_pattern = next(
            p for p in CROSS_DOMAIN_PATTERNS if p["name"] == "stress_overload"
        )

        active_signals = {
            DomainSignal.FITNESS_OVERTRAINING,
            DomainSignal.TECH_SKILL_GAP,
        }

        matched = agent._match_pattern(stress_pattern, active_signals)

        assert matched is not None
        assert DomainSignal.FITNESS_OVERTRAINING in matched
        assert DomainSignal.TECH_SKILL_GAP in matched

    def test_match_tech_finance_opportunity(self, agent):
        """Test matching tech-finance opportunity pattern."""
        from src.agents.synthesis_agent import CROSS_DOMAIN_PATTERNS

        opp_pattern = next(
            p for p in CROSS_DOMAIN_PATTERNS if p["name"] == "tech_finance_opportunity"
        )

        active_signals = {
            DomainSignal.TECH_TREND_RISING,
            DomainSignal.FINANCE_BULL_MARKET,
        }

        matched = agent._match_pattern(opp_pattern, active_signals)

        assert matched is not None

    def test_no_match_when_signals_missing(self, agent):
        """Test no match when required signals are missing."""
        from src.agents.synthesis_agent import CROSS_DOMAIN_PATTERNS

        stress_pattern = next(
            p for p in CROSS_DOMAIN_PATTERNS if p["name"] == "stress_overload"
        )

        # Only one of the required signals
        active_signals = {DomainSignal.FITNESS_OVERTRAINING}

        matched = agent._match_pattern(stress_pattern, active_signals)

        assert matched is None

    def test_triple_stress_pattern_urgent(self, agent):
        """Test triple stress pattern generates urgent alert."""
        from src.agents.synthesis_agent import CROSS_DOMAIN_PATTERNS

        triple_stress = next(
            p for p in CROSS_DOMAIN_PATTERNS if p["name"] == "triple_stress"
        )

        active_signals = {
            DomainSignal.FITNESS_OVERTRAINING,
            DomainSignal.FINANCE_HIGH_VOLATILITY,
            DomainSignal.TECH_SKILL_GAP,
        }

        matched = agent._match_pattern(triple_stress, active_signals)

        assert matched is not None
        assert triple_stress["level"] == AlertLevel.URGENT


class TestRiskAndOpportunityCalculation:
    """Tests for risk and opportunity level calculations."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent for calculation tests."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    def test_calculate_domain_risk_high(self, agent):
        """Test risk calculation with high-risk insights."""
        insights = [
            Insight(
                insight_type=InsightType.ANOMALY,
                title="High Risk Alert",
                description="Test",
                level=AlertLevel.URGENT,
                confidence=0.9,
                relevance_score=0.8,
            ),
        ]

        risk = agent._calculate_domain_risk(Domain.TECH, insights)

        assert risk > 0.5  # Should be high

    def test_calculate_domain_risk_low(self, agent):
        """Test risk calculation with low-risk insights."""
        insights = [
            Insight(
                insight_type=InsightType.EMERGING_TECH,
                title="New Opportunity",
                description="Test",
                level=AlertLevel.INFO,
                confidence=0.7,
                relevance_score=0.6,
            ),
        ]

        risk = agent._calculate_domain_risk(Domain.TECH, insights)

        assert risk == 0.0  # No risk indicators

    def test_calculate_domain_opportunity(self, agent):
        """Test opportunity calculation."""
        insights = [
            Insight(
                insight_type=InsightType.EMERGING_TECH,
                title="Emerging Opportunity",
                description="Test",
                level=AlertLevel.ACTION,
                confidence=0.85,
                relevance_score=0.9,
            ),
        ]

        opportunity = agent._calculate_domain_opportunity(Domain.TECH, insights)

        assert opportunity > 0.5  # Should be high

    def test_calculate_opportunity_empty(self, agent):
        """Test opportunity calculation with no insights."""
        opportunity = agent._calculate_domain_opportunity(Domain.TECH, [])

        assert opportunity == 0.0


class TestAnalyzePhase:
    """Tests for the analyze phase."""

    @pytest.fixture
    def agent_with_risk_insights(
        self, mock_db, default_config, user_profile,
        tech_insights, fitness_insights, finance_insights
    ):
        """Create agent with insights that trigger patterns."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
            domain_insights={
                Domain.TECH: tech_insights,
                Domain.FITNESS: fitness_insights,
                Domain.FINANCE: finance_insights,
            },
        )

    async def test_analyze_detects_compound_risk(self, agent_with_risk_insights):
        """Test analyze phase detects compound risk patterns."""
        agent = agent_with_risk_insights
        corr_svc = agent.correlation_service
        corr_svc.find_cross_source_entities = AsyncMock(return_value=[])

        # Run monitor to get domain states
        monitoring_data = await agent.monitor()

        # Run analyze
        insights = await agent.analyze(monitoring_data)

        # Should detect stress_overload pattern (tech skill gap + fitness overtraining)
        pattern_insights = [
            i for i in insights
            if i.insight_type == InsightType.CROSS_DOMAIN
        ]

        assert len(pattern_insights) > 0

    async def test_analyze_generates_cross_domain_insights(
        self, agent_with_risk_insights
    ):
        """Test analyze generates properly formatted insights."""
        agent = agent_with_risk_insights
        corr_svc = agent.correlation_service
        corr_svc.find_cross_source_entities = AsyncMock(return_value=[])

        monitoring_data = await agent.monitor()
        insights = await agent.analyze(monitoring_data)

        for insight in insights:
            assert insight.insight_type == InsightType.CROSS_DOMAIN
            assert insight.confidence > 0
            assert insight.relevance_score > 0
            assert "pattern_type" in insight.metadata or "avg_risk" in insight.metadata


class TestLifestyleBalance:
    """Tests for lifestyle balance analysis."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent for balance tests."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    def test_excellent_balance_detected(self, agent):
        """Test detection of excellent life balance."""
        domain_states = {
            Domain.TECH: DomainState(
                domain=Domain.TECH,
                active_signals=[DomainSignal.TECH_LEARNING_OPPORTUNITY],
                risk_level=0.1,
                opportunity_level=0.8,
                recent_insights=[],
            ),
            Domain.FITNESS: DomainState(
                domain=Domain.FITNESS,
                active_signals=[DomainSignal.FITNESS_PEAK_FORM],
                risk_level=0.2,
                opportunity_level=0.7,
                recent_insights=[],
            ),
            Domain.FINANCE: DomainState(
                domain=Domain.FINANCE,
                active_signals=[DomainSignal.FINANCE_BULL_MARKET],
                risk_level=0.1,
                opportunity_level=0.75,
                recent_insights=[],
            ),
        }

        insights = agent._analyze_lifestyle_balance(domain_states)

        assert len(insights) == 1
        assert "Excellent Life Balance" in insights[0].title
        assert insights[0].level == AlertLevel.INFO

    def test_elevated_risk_detected(self, agent):
        """Test detection of elevated multi-domain risk."""
        domain_states = {
            Domain.TECH: DomainState(
                domain=Domain.TECH,
                active_signals=[DomainSignal.TECH_SKILL_GAP],
                risk_level=0.7,
                opportunity_level=0.2,
                recent_insights=[],
            ),
            Domain.FITNESS: DomainState(
                domain=Domain.FITNESS,
                active_signals=[DomainSignal.FITNESS_OVERTRAINING],
                risk_level=0.8,
                opportunity_level=0.1,
                recent_insights=[],
            ),
            Domain.FINANCE: DomainState(
                domain=Domain.FINANCE,
                active_signals=[DomainSignal.FINANCE_HIGH_VOLATILITY],
                risk_level=0.6,
                opportunity_level=0.2,
                recent_insights=[],
            ),
        }

        insights = agent._analyze_lifestyle_balance(domain_states)

        assert len(insights) == 1
        assert "Elevated Multi-Domain Risk" in insights[0].title
        assert insights[0].level == AlertLevel.WATCH


class TestProjectIdeaGeneration:
    """Tests for cross-domain project idea generation."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent for project idea tests."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    async def test_generate_fintech_project_idea(self, agent):
        """Test generation of fintech project idea."""
        insights = [
            Insight(
                insight_type=InsightType.CROSS_DOMAIN,
                title="Tech-Finance Opportunity",
                description="Rising tech meets bull market",
                level=AlertLevel.ACTION,
                confidence=0.8,
                relevance_score=0.85,
                metadata={
                    "signals": ["tech_trend_rising", "finance_bull_market"],
                },
            ),
        ]

        ideas = await agent.generate_project_ideas(insights)

        assert len(ideas) >= 1
        fintech_idea = next(
            (i for i in ideas if "Finance" in i.title or "AI" in i.title), None
        )
        assert fintech_idea is not None
        assert fintech_idea.difficulty in ["beginner", "intermediate", "advanced"]
        assert len(fintech_idea.learning_path) > 0

    async def test_generate_healthtech_project_idea(self, agent):
        """Test generation of health-tech project idea."""
        insights = [
            Insight(
                insight_type=InsightType.CROSS_DOMAIN,
                title="Tech-Fitness Synergy",
                description="Tech and fitness signals aligned",
                level=AlertLevel.INFO,
                confidence=0.75,
                relevance_score=0.8,
                metadata={
                    "signals": ["tech_trend_rising", "fitness_peak_form"],
                },
            ),
        ]

        ideas = await agent.generate_project_ideas(insights)

        assert len(ideas) >= 1
        health_idea = next(
            (i for i in ideas if "Training" in i.title or "Fitness" in i.title), None
        )
        assert health_idea is not None


class TestReportFormatting:
    """Tests for report generation."""

    @pytest.fixture
    def agent(self, mock_db, default_config, user_profile):
        """Create agent for report tests."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
        )

    def test_format_report_empty(self, agent):
        """Test formatting report with no insights."""
        from src.agents import AgentReport

        report = AgentReport(
            agent_name="synthesis_intelligence",
            domain=Domain.GENERAL,
            run_at=datetime.utcnow(),
            insights=[],
            project_ideas=[],
        )

        formatted = agent.format_report(report)

        assert "Synthesis Intelligence Report" in formatted
        assert "Summary" in formatted

    def test_format_report_with_insights(self, agent):
        """Test formatting report with cross-domain insights."""
        from src.agents import AgentReport

        insights = [
            Insight(
                insight_type=InsightType.CROSS_DOMAIN,
                title="Cross-Domain Pattern: Stress Overload",
                description="High stress detected across domains",
                level=AlertLevel.ACTION,
                confidence=0.85,
                relevance_score=0.9,
                metadata={
                    "domains_involved": ["tech", "fitness"],
                    "pattern_type": "compound_risk",
                },
            ),
        ]

        report = AgentReport(
            agent_name="synthesis_intelligence",
            domain=Domain.GENERAL,
            run_at=datetime.utcnow(),
            insights=insights,
            project_ideas=[],
        )

        formatted = agent.format_report(report)

        assert "Cross-Dimensional Insights" in formatted
        assert "Stress Overload" in formatted
        assert "ACTION" in formatted
        assert "85%" in formatted  # Confidence


class TestFullAgentRun:
    """Integration tests for full agent run."""

    @pytest.fixture
    def agent_with_all_insights(
        self, mock_db, default_config, user_profile,
        tech_insights, fitness_insights, finance_insights
    ):
        """Create agent with comprehensive insights."""
        return SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
            domain_insights={
                Domain.TECH: tech_insights,
                Domain.FITNESS: fitness_insights,
                Domain.FINANCE: finance_insights,
            },
        )

    async def test_full_agent_run(self, agent_with_all_insights):
        """Test complete agent run produces valid report."""
        agent = agent_with_all_insights
        corr_svc = agent.correlation_service
        corr_svc.find_cross_source_entities = AsyncMock(return_value=[])

        report = await agent.run()

        assert report.agent_name == "synthesis_intelligence"
        assert report.domain == Domain.GENERAL
        assert report.execution_time_ms >= 0
        assert isinstance(report.insights, list)
        assert isinstance(report.project_ideas, list)

    async def test_agent_run_with_positive_signals(
        self, mock_db, default_config, user_profile, positive_finance_insights
    ):
        """Test agent run with opportunity signals."""
        tech_opportunities = [
            Insight(
                insight_type=InsightType.EMERGING_TECH,
                title="AI Framework Rising",
                description="New AI framework gaining traction",
                level=AlertLevel.ACTION,
                confidence=0.85,
                relevance_score=0.9,
                entity_names=["NewAIFramework"],
            ),
        ]

        fitness_peak = [
            Insight(
                insight_type=InsightType.TREND_CHANGE,
                title="Peak Performance",
                description="Optimal training zone reached",
                level=AlertLevel.INFO,
                confidence=0.8,
                relevance_score=0.85,
                metadata={"training_zone": "OPTIMAL"},
            ),
        ]

        agent = SynthesisAgent(
            db=mock_db,
            config=default_config,
            user_profile=user_profile,
            domain_insights={
                Domain.TECH: tech_opportunities,
                Domain.FITNESS: fitness_peak,
                Domain.FINANCE: positive_finance_insights,
            },
        )
        corr_svc = agent.correlation_service
        corr_svc.find_cross_source_entities = AsyncMock(return_value=[])

        report = await agent.run()

        # Should detect opportunity patterns
        assert report.agent_name == "synthesis_intelligence"
        # May have lifestyle balance or opportunity insights
        assert isinstance(report.insights, list)
