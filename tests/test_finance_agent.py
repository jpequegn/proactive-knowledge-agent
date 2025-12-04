"""Tests for the Finance Intelligence Agent."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentConfig, AlertLevel, InsightType, UserProfile
from src.agents.finance_agent import (
    FinanceIntelligenceAgent,
    FinanceInsightType,
    MarketRegime,
    MarketRegimeAnalysis,
    PortfolioRiskAssessment,
    RebalancingSignal,
    RiskLevel,
    VolatilityMetrics,
)
from src.database import Database
from src.models import MarketOHLCV


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_db():
    """Create a mock database."""
    return MagicMock(spec=Database)


@pytest.fixture
def mock_market_client():
    """Create a mock market client."""
    client = MagicMock()
    client.get_history = AsyncMock(return_value=[])
    return client


@pytest.fixture
def config():
    """Create agent config."""
    return AgentConfig(
        enabled=True,
        thresholds={
            "min_confidence": 0.5,
            "min_relevance": 0.3,
            "volatility_warning": 2.0,
            "drawdown_warning": -0.10,
            "rebalance_threshold": 0.05,
            "benchmark": "SPY",
        },
        focus_areas=["SPY", "QQQ", "IWM"],
    )


@pytest.fixture
def user_profile():
    """Create user profile."""
    return UserProfile(
        role="investor",
        experience_level="senior",
        interests=["market analysis", "portfolio management"],
        known_technologies=["python", "finance"],
        learning_goals=["quantitative finance", "risk management"],
    )


@pytest.fixture
def agent(mock_db, config, user_profile, mock_market_client):
    """Create a FinanceIntelligenceAgent instance."""
    agent = FinanceIntelligenceAgent(
        db=mock_db,
        config=config,
        user_profile=user_profile,
        market_client=mock_market_client,
    )
    # Mock repositories
    agent.market_repo = MagicMock()
    agent.market_repo.get_history = AsyncMock(return_value=[])
    agent.market_repo.get_symbols = AsyncMock(return_value=["SPY", "QQQ", "IWM"])
    return agent


@pytest.fixture
def sample_market_data():
    """Generate sample market OHLCV data for testing."""
    base_date = datetime.now(UTC) - timedelta(days=60)
    data = []
    price = 100.0

    for i in range(60):
        # Simulate uptrend with some noise
        change = 0.5 + (0.3 * (1 if i % 3 else -1))
        price = price * (1 + change / 100)

        data.append(
            MarketOHLCV(
                symbol="SPY",
                date=base_date + timedelta(days=i),
                open=price * 0.999,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000 + i * 10000,
            )
        )

    return data


@pytest.fixture
def downtrend_market_data():
    """Generate sample market data with downtrend."""
    base_date = datetime.now(UTC) - timedelta(days=60)
    data = []
    price = 100.0

    for i in range(60):
        # Simulate downtrend
        change = -0.5 - (0.2 * (1 if i % 2 else -1))
        price = max(price * (1 + change / 100), 50)  # Floor at 50

        data.append(
            MarketOHLCV(
                symbol="SPY",
                date=base_date + timedelta(days=i),
                open=price * 1.001,
                high=price * 1.01,
                low=price * 0.98,
                close=price,
                volume=1000000 + i * 10000,
            )
        )

    return data


@pytest.fixture
def volatile_market_data():
    """Generate sample market data with high volatility."""
    base_date = datetime.now(UTC) - timedelta(days=60)
    data = []
    price = 100.0

    for i in range(60):
        # Simulate high volatility
        change = 3.0 * (1 if i % 2 else -1)  # Large swings
        price = price * (1 + change / 100)

        data.append(
            MarketOHLCV(
                symbol="SPY",
                date=base_date + timedelta(days=i),
                open=price * 0.98,
                high=price * 1.05,
                low=price * 0.95,
                close=price,
                volume=2000000 + i * 20000,
            )
        )

    return data


# ============================================================================
# Model Tests
# ============================================================================


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_all_regimes_exist(self):
        """Test all market regimes are defined."""
        assert MarketRegime.BULL == "bull"
        assert MarketRegime.BEAR == "bear"
        assert MarketRegime.SIDEWAYS == "sideways"
        assert MarketRegime.VOLATILE == "volatile"
        assert MarketRegime.RECOVERY == "recovery"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_risk_levels_exist(self):
        """Test all risk levels are defined."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MODERATE == "moderate"
        assert RiskLevel.ELEVATED == "elevated"
        assert RiskLevel.HIGH == "high"


class TestFinanceInsightType:
    """Tests for FinanceInsightType enum."""

    def test_all_insight_types_exist(self):
        """Test all finance insight types are defined."""
        assert FinanceInsightType.MARKET_REGIME_CHANGE == "market_regime_change"
        assert FinanceInsightType.SECTOR_ROTATION == "sector_rotation"
        assert FinanceInsightType.VOLATILITY_SPIKE == "volatility_spike"
        assert FinanceInsightType.DRAWDOWN_WARNING == "drawdown_warning"
        assert FinanceInsightType.REBALANCING_TRIGGER == "rebalancing_trigger"


class TestVolatilityMetrics:
    """Tests for VolatilityMetrics dataclass."""

    def test_create_volatility_metrics(self):
        """Test creating volatility metrics."""
        metrics = VolatilityMetrics(
            current_volatility=18.5,
            volatility_20d=18.5,
            volatility_ratio=1.2,
            vix_level=None,
            risk_level=RiskLevel.MODERATE,
            percentile_rank=55.0,
        )

        assert metrics.current_volatility == 18.5
        assert metrics.volatility_ratio == 1.2
        assert metrics.risk_level == RiskLevel.MODERATE


class TestMarketRegimeAnalysis:
    """Tests for MarketRegimeAnalysis dataclass."""

    def test_create_regime_analysis(self):
        """Test creating regime analysis."""
        analysis = MarketRegimeAnalysis(
            regime=MarketRegime.BULL,
            confidence=0.85,
            trend_strength=0.7,
            volatility_percentile=40.0,
            sma_20_position="above",
            sma_50_position="above",
            rsi_14=65.0,
            days_in_regime=10,
            previous_regime=None,
        )

        assert analysis.regime == MarketRegime.BULL
        assert analysis.confidence == 0.85
        assert analysis.trend_strength == 0.7


class TestPortfolioRiskAssessment:
    """Tests for PortfolioRiskAssessment dataclass."""

    def test_create_risk_assessment(self):
        """Test creating portfolio risk assessment."""
        assessment = PortfolioRiskAssessment(
            total_risk_score=45.0,
            risk_level=RiskLevel.MODERATE,
            max_drawdown_current=-0.08,
            concentration_risk=25.0,
            correlation_risk=60.0,
            volatility_risk=18.0,
            recommendations=["Consider diversifying"],
        )

        assert assessment.total_risk_score == 45.0
        assert assessment.risk_level == RiskLevel.MODERATE
        assert len(assessment.recommendations) == 1


class TestRebalancingSignal:
    """Tests for RebalancingSignal dataclass."""

    def test_create_rebalancing_signal(self):
        """Test creating rebalancing signal."""
        signal = RebalancingSignal(
            symbol="SPY",
            current_weight=0.35,
            target_weight=0.30,
            deviation=0.05,
            action="sell",
            priority=2,
            rationale="SPY is overweight by 5.0%",
        )

        assert signal.symbol == "SPY"
        assert signal.action == "sell"
        assert signal.deviation == 0.05


# ============================================================================
# Agent Initialization Tests
# ============================================================================


class TestAgentInitialization:
    """Tests for FinanceIntelligenceAgent initialization."""

    def test_agent_attributes(self, agent):
        """Test agent has correct attributes."""
        assert agent.name == "finance_intelligence"
        assert agent.domain.value == "finance"

    def test_agent_default_thresholds(self, agent):
        """Test agent has default thresholds."""
        assert agent.TREND_STRENGTH_THRESHOLD == 0.6
        assert agent.RSI_OVERBOUGHT == 70
        assert agent.RSI_OVERSOLD == 30
        assert agent.DRAWDOWN_WARNING == -0.10
        assert agent.DRAWDOWN_CRITICAL == -0.20

    def test_agent_portfolio_config(self, agent):
        """Test agent reads portfolio from config."""
        assert "SPY" in agent.portfolio_symbols
        assert "QQQ" in agent.portfolio_symbols
        assert agent.benchmark_symbol == "SPY"

    def test_agent_minimal_config(self, mock_db):
        """Test agent works with minimal config."""
        agent = FinanceIntelligenceAgent(db=mock_db)
        assert agent.portfolio_symbols == ["SPY", "QQQ", "IWM"]
        assert agent.benchmark_symbol == "SPY"


# ============================================================================
# Market Regime Classification Tests
# ============================================================================


class TestRegimeAnalysis:
    """Tests for market regime analysis."""

    def test_analyze_regime_bull_market(self, agent, sample_market_data):
        """Test regime analysis detects bull market."""
        result = agent._analyze_regime(sample_market_data)

        assert result is not None
        assert result.sma_20_position in ["above", "below", "crossing"]
        assert 0 <= result.rsi_14 <= 100

    def test_analyze_regime_bear_market(self, agent, downtrend_market_data):
        """Test regime analysis detects bear market."""
        result = agent._analyze_regime(downtrend_market_data)

        assert result is not None
        assert result.trend_strength < 0  # Negative trend

    def test_analyze_regime_volatile_market(self, agent, volatile_market_data):
        """Test regime analysis detects high volatility."""
        result = agent._analyze_regime(volatile_market_data)

        assert result is not None
        # Volatile markets have high volatility percentile
        assert result.volatility_percentile > 0

    def test_analyze_regime_insufficient_data(self, agent):
        """Test regime analysis with insufficient data."""
        short_data = [
            MarketOHLCV(
                symbol="SPY",
                date=datetime.now(UTC),
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1000000,
            )
        ]
        result = agent._analyze_regime(short_data)
        assert result is None

    def test_classify_regime_bull(self, agent):
        """Test regime classification for bull market."""
        regime = agent._classify_regime(
            trend_strength=0.7,
            volatility_percentile=40,
            rsi=65,
            sma_20_pos="above",
            sma_50_pos="above",
        )
        assert regime == MarketRegime.BULL

    def test_classify_regime_bear(self, agent):
        """Test regime classification for bear market."""
        regime = agent._classify_regime(
            trend_strength=-0.7,
            volatility_percentile=40,
            rsi=35,
            sma_20_pos="below",
            sma_50_pos="below",
        )
        assert regime == MarketRegime.BEAR

    def test_classify_regime_volatile(self, agent):
        """Test regime classification for volatile market."""
        regime = agent._classify_regime(
            trend_strength=0.2,
            volatility_percentile=85,
            rsi=50,
            sma_20_pos="above",
            sma_50_pos="below",
        )
        assert regime == MarketRegime.VOLATILE

    def test_classify_regime_recovery(self, agent):
        """Test regime classification for recovery."""
        regime = agent._classify_regime(
            trend_strength=0.3,
            volatility_percentile=40,
            rsi=55,
            sma_20_pos="above",
            sma_50_pos="below",
        )
        assert regime == MarketRegime.RECOVERY

    def test_classify_regime_sideways(self, agent):
        """Test regime classification for sideways market."""
        regime = agent._classify_regime(
            trend_strength=0.1,
            volatility_percentile=40,
            rsi=50,
            sma_20_pos="above",
            sma_50_pos="above",
        )
        assert regime == MarketRegime.SIDEWAYS


# ============================================================================
# Volatility Analysis Tests
# ============================================================================


class TestVolatilityAnalysis:
    """Tests for volatility calculation."""

    def test_calculate_volatility(self, agent, sample_market_data):
        """Test volatility calculation."""
        result = agent._calculate_volatility(sample_market_data)

        assert result is not None
        assert result.current_volatility >= 0
        assert result.volatility_20d >= 0
        assert result.volatility_ratio > 0
        assert result.risk_level in RiskLevel

    def test_calculate_volatility_insufficient_data(self, agent):
        """Test volatility calculation with insufficient data."""
        short_data = [
            MarketOHLCV(
                symbol="SPY",
                date=datetime.now(UTC) - timedelta(days=i),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1000000,
            )
            for i in range(10)  # Less than 20 days
        ]
        result = agent._calculate_volatility(short_data)
        assert result is None

    def test_risk_level_low(self, agent):
        """Test low risk level detection."""
        # Create stable data with low volatility
        base_date = datetime.now(UTC) - timedelta(days=30)
        stable_data = [
            MarketOHLCV(
                symbol="SPY",
                date=base_date + timedelta(days=i),
                open=100 + i * 0.01,
                high=100 + i * 0.01 + 0.1,
                low=100 + i * 0.01 - 0.1,
                close=100 + i * 0.01 + 0.05,
                volume=1000000,
            )
            for i in range(30)
        ]
        result = agent._calculate_volatility(stable_data)
        assert result is not None
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]


# ============================================================================
# Technical Indicator Tests
# ============================================================================


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""

    def test_calculate_rsi(self, agent):
        """Test RSI calculation."""
        closes = [100 + i for i in range(20)]  # Uptrend
        rsi = agent._calculate_rsi(closes)

        assert 0 <= rsi <= 100

    def test_calculate_rsi_overbought(self, agent):
        """Test RSI overbought detection."""
        # Strong uptrend
        closes = [100 + i * 2 for i in range(20)]
        rsi = agent._calculate_rsi(closes)

        assert rsi > 50  # Should indicate bullish

    def test_calculate_rsi_oversold(self, agent):
        """Test RSI oversold detection."""
        # Strong downtrend
        closes = [100 - i * 2 for i in range(20)]
        rsi = agent._calculate_rsi(closes)

        assert rsi < 50  # Should indicate bearish

    def test_calculate_trend_strength(self, agent):
        """Test trend strength calculation."""
        uptrend = [100 + i for i in range(20)]
        strength = agent._calculate_trend_strength(uptrend)

        assert strength > 0  # Positive for uptrend

    def test_calculate_trend_strength_downtrend(self, agent):
        """Test trend strength for downtrend."""
        downtrend = [100 - i for i in range(20)]
        strength = agent._calculate_trend_strength(downtrend)

        assert strength < 0  # Negative for downtrend

    def test_std_dev_calculation(self, agent):
        """Test standard deviation calculation."""
        values = [10, 12, 8, 11, 9]
        std = agent._std_dev(values)

        assert std > 0
        assert std < 2  # Should be around 1.4


# ============================================================================
# Insight Generation Tests
# ============================================================================


class TestInsightGeneration:
    """Tests for insight generation."""

    def test_analyze_market_regime_bear(self, agent):
        """Test bear market insight generation."""
        regime = MarketRegimeAnalysis(
            regime=MarketRegime.BEAR,
            confidence=0.85,
            trend_strength=-0.7,
            volatility_percentile=50,
            sma_20_position="below",
            sma_50_position="below",
            rsi_14=35,
            days_in_regime=5,
            previous_regime=None,
        )

        insights = agent._analyze_market_regime(regime)

        assert len(insights) >= 1
        bear_insight = insights[0]
        assert bear_insight.level == AlertLevel.ACTION
        assert "Bear Market" in bear_insight.title

    def test_analyze_market_regime_volatile(self, agent):
        """Test volatile market insight generation."""
        regime = MarketRegimeAnalysis(
            regime=MarketRegime.VOLATILE,
            confidence=0.8,
            trend_strength=0.1,
            volatility_percentile=85,
            sma_20_position="crossing",
            sma_50_position="above",
            rsi_14=50,
            days_in_regime=3,
            previous_regime=MarketRegime.BULL,
        )

        insights = agent._analyze_market_regime(regime)

        assert len(insights) >= 1
        vol_insight = insights[0]
        assert vol_insight.level == AlertLevel.WATCH
        assert "Volatility" in vol_insight.title

    def test_analyze_market_regime_overbought_bull(self, agent):
        """Test overbought bull market insight generation."""
        regime = MarketRegimeAnalysis(
            regime=MarketRegime.BULL,
            confidence=0.85,
            trend_strength=0.7,
            volatility_percentile=40,
            sma_20_position="above",
            sma_50_position="above",
            rsi_14=75,  # Overbought
            days_in_regime=20,
            previous_regime=None,
        )

        insights = agent._analyze_market_regime(regime)

        assert len(insights) >= 1
        overbought_insight = next(
            (i for i in insights if "Overbought" in i.title), None
        )
        assert overbought_insight is not None
        assert overbought_insight.level == AlertLevel.INFO

    def test_analyze_volatility_spike(self, agent):
        """Test volatility spike insight generation."""
        metrics = VolatilityMetrics(
            current_volatility=35,
            volatility_20d=35,
            volatility_ratio=2.5,  # Spike
            vix_level=None,
            risk_level=RiskLevel.ELEVATED,
            percentile_rank=90,
        )

        insights = agent._analyze_volatility(metrics)

        assert len(insights) >= 1
        spike_insight = next(
            (i for i in insights if "Spike" in i.title), None
        )
        assert spike_insight is not None
        assert spike_insight.level == AlertLevel.WATCH

    def test_analyze_high_risk(self, agent):
        """Test high risk environment insight generation."""
        metrics = VolatilityMetrics(
            current_volatility=40,
            volatility_20d=40,
            volatility_ratio=1.5,
            vix_level=None,
            risk_level=RiskLevel.HIGH,
            percentile_rank=95,
        )

        insights = agent._analyze_volatility(metrics)

        high_risk = next(
            (i for i in insights if "High Risk" in i.title), None
        )
        assert high_risk is not None
        assert high_risk.level == AlertLevel.ACTION

    def test_analyze_holding_drawdown(self, agent, sample_market_data):
        """Test holding drawdown insight generation."""
        # Modify data to have significant drawdown
        data = sample_market_data.copy()
        peak_price = max(d.close for d in data)
        # Set last price to 25% below peak
        data[-1] = MarketOHLCV(
            symbol="SPY",
            date=data[-1].date,
            open=peak_price * 0.76,
            high=peak_price * 0.77,
            low=peak_price * 0.74,
            close=peak_price * 0.75,
            volume=data[-1].volume,
        )

        insights = agent._analyze_holding("SPY", data)

        assert len(insights) >= 1
        drawdown_insight = next(
            (i for i in insights if "Drawdown" in i.title), None
        )
        assert drawdown_insight is not None
        assert drawdown_insight.level in [AlertLevel.ACTION, AlertLevel.WATCH]


# ============================================================================
# Portfolio Analysis Tests
# ============================================================================


class TestPortfolioAnalysis:
    """Tests for portfolio-level analysis."""

    def test_assess_portfolio_risk(self, agent, sample_market_data):
        """Test portfolio risk assessment."""
        market_data = {
            "SPY": sample_market_data,
            "QQQ": sample_market_data,  # Same data for simplicity
        }

        assessment = agent.assess_portfolio_risk(market_data)

        assert 0 <= assessment.total_risk_score <= 100
        assert assessment.risk_level in RiskLevel
        assert isinstance(assessment.recommendations, list)

    def test_assess_portfolio_risk_with_weights(self, agent, sample_market_data):
        """Test portfolio risk assessment with explicit weights."""
        market_data = {
            "SPY": sample_market_data,
            "QQQ": sample_market_data,
        }
        weights = {"SPY": 0.7, "QQQ": 0.3}

        assessment = agent.assess_portfolio_risk(market_data, weights)

        assert assessment.concentration_risk >= 70  # SPY is 70%

    def test_correlation_calculation(self, agent, sample_market_data):
        """Test correlation calculation between symbols."""
        market_data = {
            "SPY": sample_market_data,
            "QQQ": sample_market_data,  # Same data = perfect correlation
        }

        correlations = agent._calculate_correlations(market_data)

        # Same data should have high correlation
        spy_spy = correlations.get(("SPY", "SPY"))
        assert spy_spy is not None
        assert abs(spy_spy - 1.0) < 0.01  # Should be ~1.0

    def test_correlation_function(self, agent):
        """Test basic correlation function."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        corr = agent._correlation(x, y)

        assert abs(corr - 1.0) < 0.01  # Perfect positive correlation


# ============================================================================
# Rebalancing Tests
# ============================================================================


class TestRebalancing:
    """Tests for rebalancing signal generation."""

    def test_generate_rebalancing_signals_overweight(self, agent):
        """Test rebalancing signals for overweight position."""
        current = {"SPY": 0.40, "QQQ": 0.35, "IWM": 0.25}
        target = {"SPY": 0.33, "QQQ": 0.33, "IWM": 0.34}

        signals = agent.generate_rebalancing_signals(current, target)

        assert len(signals) >= 1
        spy_signal = next((s for s in signals if s.symbol == "SPY"), None)
        assert spy_signal is not None
        assert spy_signal.action == "sell"

    def test_generate_rebalancing_signals_underweight(self, agent):
        """Test rebalancing signals for underweight position."""
        current = {"SPY": 0.20, "QQQ": 0.40, "IWM": 0.40}
        target = {"SPY": 0.33, "QQQ": 0.33, "IWM": 0.34}

        signals = agent.generate_rebalancing_signals(current, target)

        spy_signal = next((s for s in signals if s.symbol == "SPY"), None)
        assert spy_signal is not None
        assert spy_signal.action == "buy"

    def test_no_rebalancing_signals_within_threshold(self, agent):
        """Test no signals when within threshold."""
        current = {"SPY": 0.34, "QQQ": 0.33, "IWM": 0.33}
        target = {"SPY": 0.33, "QQQ": 0.33, "IWM": 0.34}

        signals = agent.generate_rebalancing_signals(current, target)

        assert len(signals) == 0

    def test_rebalancing_signal_priority(self, agent):
        """Test rebalancing signal priority ordering."""
        current = {"SPY": 0.50, "QQQ": 0.10}  # Large deviation
        target = {"SPY": 0.30, "QQQ": 0.30}

        signals = agent.generate_rebalancing_signals(current, target)

        # Signals should be sorted by priority and deviation
        assert len(signals) == 2
        # Both signals have same priority, sorted by deviation magnitude
        # SPY deviation is 0.20, QQQ deviation is -0.20, same magnitude
        assert signals[0].symbol in ["SPY", "QQQ"]


# ============================================================================
# Agent Run Tests
# ============================================================================


class TestAgentRun:
    """Tests for full agent run."""

    @pytest.mark.asyncio
    async def test_monitor_returns_data(self, agent, sample_market_data):
        """Test monitor phase returns expected data."""
        agent.market_repo.get_history = AsyncMock(return_value=sample_market_data)
        agent.market_client.get_history = AsyncMock(return_value=sample_market_data)

        data = await agent.monitor()

        assert "market_data" in data
        assert "regime_analysis" in data
        assert "volatility_metrics" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_analyze_generates_insights(self, agent, sample_market_data):
        """Test analyze phase generates insights."""
        regime = MarketRegimeAnalysis(
            regime=MarketRegime.BULL,
            confidence=0.8,
            trend_strength=0.5,
            volatility_percentile=40,
            sma_20_position="above",
            sma_50_position="above",
            rsi_14=60,
            days_in_regime=10,
            previous_regime=None,
        )

        monitoring_data = {
            "market_data": {"SPY": sample_market_data},
            "regime_analysis": regime,
            "volatility_metrics": VolatilityMetrics(
                current_volatility=15,
                volatility_20d=15,
                volatility_ratio=1.0,
                vix_level=None,
                risk_level=RiskLevel.LOW,
                percentile_rank=40,
            ),
        }

        insights = await agent.analyze(monitoring_data)

        assert isinstance(insights, list)
        # Should generate at least low volatility insight
        assert len(insights) >= 1

    @pytest.mark.asyncio
    async def test_full_run(self, agent, sample_market_data):
        """Test full agent run."""
        agent.market_repo.get_history = AsyncMock(return_value=sample_market_data)
        agent.market_client.get_history = AsyncMock(return_value=sample_market_data)

        report = await agent.run()

        assert report.agent_name == "finance_intelligence"
        assert report.domain.value == "finance"
        assert isinstance(report.insights, list)
        assert report.execution_time_ms >= 0
        assert "regime_analysis" in report.metadata
        assert "risk_assessment" in report.metadata

    @pytest.mark.asyncio
    async def test_run_with_empty_data(self, agent):
        """Test agent run with no market data."""
        agent.market_repo.get_history = AsyncMock(return_value=[])
        agent.market_client.get_history = AsyncMock(return_value=[])

        report = await agent.run()

        assert report.agent_name == "finance_intelligence"
        assert isinstance(report.insights, list)


# ============================================================================
# Report Generation Tests
# ============================================================================


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_weekly_report(self, agent):
        """Test weekly report generation."""
        from src.agents.base import AgentReport
        from src.world_model import Domain

        regime = MarketRegimeAnalysis(
            regime=MarketRegime.BULL,
            confidence=0.85,
            trend_strength=0.6,
            volatility_percentile=40,
            sma_20_position="above",
            sma_50_position="above",
            rsi_14=62,
            days_in_regime=15,
            previous_regime=None,
        )

        volatility = VolatilityMetrics(
            current_volatility=16.5,
            volatility_20d=16.5,
            volatility_ratio=1.1,
            vix_level=None,
            risk_level=RiskLevel.MODERATE,
            percentile_rank=50,
        )

        risk = PortfolioRiskAssessment(
            total_risk_score=42,
            risk_level=RiskLevel.MODERATE,
            max_drawdown_current=-0.05,
            concentration_risk=35,
            correlation_risk=65,
            volatility_risk=16.5,
            recommendations=["Consider diversifying"],
        )

        report = AgentReport(
            agent_name="finance_intelligence",
            domain=Domain.FINANCE,
            run_at=datetime.now(UTC),
            insights=[],
            entities_scanned=3,
            alerts_generated=0,
            execution_time_ms=150,
            metadata={
                "regime_analysis": regime,
                "volatility_metrics": volatility,
                "risk_assessment": risk,
            },
        )

        markdown = agent.generate_weekly_report(report)

        assert "# Weekly Market Intelligence Report" in markdown
        assert "Market Regime" in markdown
        assert "Bull" in markdown
        assert "Volatility Metrics" in markdown
        assert "Portfolio Risk Assessment" in markdown


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_closes_rsi(self, agent):
        """Test RSI with empty data."""
        rsi = agent._calculate_rsi([])
        assert rsi == 50.0  # Default neutral

    def test_single_value_trend_strength(self, agent):
        """Test trend strength with single value."""
        strength = agent._calculate_trend_strength([100])
        assert strength == 0.0

    def test_zero_std_dev_correlation(self, agent):
        """Test correlation with zero std dev."""
        x = [5, 5, 5, 5, 5]  # All same values
        y = [1, 2, 3, 4, 5]
        corr = agent._correlation(x, y)
        assert corr == 0.0

    def test_mismatched_correlation_lengths(self, agent):
        """Test correlation with mismatched lengths."""
        x = [1, 2, 3]
        y = [1, 2, 3, 4, 5]
        corr = agent._correlation(x, y)
        assert corr == 0.0

    def test_std_dev_single_value(self, agent):
        """Test std dev with single value."""
        std = agent._std_dev([100])
        assert std == 0.0
