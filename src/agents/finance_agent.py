"""Finance Intelligence Agent for market monitoring and portfolio risk assessment.

Monitors:
- Market regime changes (bull, bear, sideways)
- Sector rotations
- Risk indicators (VIX, volatility)
- News sentiment on holdings

Outputs:
- Weekly "Market Report"
- Alerts for significant market events
- Portfolio risk assessment
- Rebalancing triggers
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.agents.base import (
    AgentConfig,
    AgentReport,
    AlertLevel,
    BaseAgent,
    Insight,
    InsightType,
    UserProfile,
)
from src.database import Database
from src.ingestion.market_client import MarketClient
from src.models import MarketOHLCV
from src.repositories import MarketRepository
from src.world_model import Domain

logger = structlog.get_logger()


# ============================================================================
# Finance-Specific Models
# ============================================================================


class FinanceInsightType(str, Enum):
    """Types of finance-specific insights."""

    MARKET_REGIME_CHANGE = "market_regime_change"
    SECTOR_ROTATION = "sector_rotation"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_WARNING = "drawdown_warning"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REBALANCING_TRIGGER = "rebalancing_trigger"
    MOMENTUM_SHIFT = "momentum_shift"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    TREND_REVERSAL = "trend_reversal"


class MarketRegime(str, Enum):
    """Market regime classification."""

    BULL = "bull"  # Strong uptrend, positive momentum
    BEAR = "bear"  # Downtrend, negative momentum
    SIDEWAYS = "sideways"  # Range-bound, low directional movement
    VOLATILE = "volatile"  # High uncertainty, rapid swings
    RECOVERY = "recovery"  # Transitioning from bear to bull


class RiskLevel(str, Enum):
    """Portfolio risk level assessment."""

    LOW = "low"  # VIX < 15, low volatility
    MODERATE = "moderate"  # VIX 15-25, normal conditions
    ELEVATED = "elevated"  # VIX 25-35, increased uncertainty
    HIGH = "high"  # VIX > 35, crisis conditions


@dataclass
class MarketRegimeAnalysis:
    """Analysis of current market regime."""

    regime: MarketRegime
    confidence: float
    trend_strength: float  # -1 to 1 (negative = downtrend)
    volatility_percentile: float  # 0-100
    sma_20_position: str  # "above", "below", "crossing"
    sma_50_position: str
    rsi_14: float
    days_in_regime: int
    previous_regime: MarketRegime | None


@dataclass
class VolatilityMetrics:
    """Volatility-related metrics."""

    current_volatility: float  # Annualized std dev
    volatility_20d: float  # 20-day realized volatility
    volatility_ratio: float  # Current / historical average
    vix_level: float | None  # If VIX data available
    risk_level: RiskLevel
    percentile_rank: float  # Where current vol ranks historically


@dataclass
class PortfolioRiskAssessment:
    """Overall portfolio risk assessment."""

    total_risk_score: float  # 0-100
    risk_level: RiskLevel
    max_drawdown_current: float  # Current drawdown from peak
    concentration_risk: float  # Single position risk
    correlation_risk: float  # Over-correlated positions
    volatility_risk: float  # Exposure to volatile assets
    recommendations: list[str]


@dataclass
class RebalancingSignal:
    """Signal indicating rebalancing may be needed."""

    symbol: str
    current_weight: float
    target_weight: float
    deviation: float  # Percentage deviation from target
    action: str  # "buy", "sell", "hold"
    priority: int  # 1 = highest
    rationale: str


@dataclass
class SectorPerformance:
    """Performance metrics for a market sector."""

    sector: str
    performance_1d: float
    performance_1w: float
    performance_1m: float
    momentum_score: float  # -1 to 1
    relative_strength: float  # vs market
    flow_direction: str  # "inflow", "outflow", "neutral"


# ============================================================================
# Finance Intelligence Agent
# ============================================================================


class FinanceIntelligenceAgent(BaseAgent):
    """
    Proactive agent for financial market intelligence.

    Monitors market conditions, detects regime changes,
    assesses portfolio risk, and generates rebalancing signals.
    """

    name = "finance_intelligence"
    domain = Domain.FINANCE

    # Market regime thresholds
    TREND_STRENGTH_THRESHOLD = 0.6  # For regime classification
    VOLATILITY_SPIKE_THRESHOLD = 2.0  # 2x normal volatility
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    DRAWDOWN_WARNING = -0.10  # 10% drawdown
    DRAWDOWN_CRITICAL = -0.20  # 20% drawdown
    REBALANCE_THRESHOLD = 0.05  # 5% deviation triggers rebalance

    # Risk level thresholds (based on typical VIX-like volatility)
    VOLATILITY_LOW = 15
    VOLATILITY_MODERATE = 25
    VOLATILITY_HIGH = 35

    def __init__(
        self,
        db: Database,
        config: AgentConfig | None = None,
        user_profile: UserProfile | None = None,
        market_client: MarketClient | None = None,
    ):
        super().__init__(db, config, user_profile)

        # Initialize market-specific components
        self.market_repo = MarketRepository(db)
        self.market_client = market_client or MarketClient()

        # Portfolio configuration from config or defaults
        self.portfolio_symbols = self.config.focus_areas or [
            "SPY",  # S&P 500
            "QQQ",  # Nasdaq
            "IWM",  # Russell 2000
        ]
        self.benchmark_symbol = self.config.thresholds.get("benchmark", "SPY")

        # Finance-specific thresholds from config
        self.volatility_warning = self.config.thresholds.get(
            "volatility_warning", self.VOLATILITY_SPIKE_THRESHOLD
        )
        self.drawdown_warning = self.config.thresholds.get(
            "drawdown_warning", self.DRAWDOWN_WARNING
        )
        self.rebalance_threshold = self.config.thresholds.get(
            "rebalance_threshold", self.REBALANCE_THRESHOLD
        )

        logger.info(
            "Initialized FinanceIntelligenceAgent",
            portfolio_symbols=self.portfolio_symbols,
            benchmark=self.benchmark_symbol,
        )

    # ========================================================================
    # Monitor Phase
    # ========================================================================

    async def monitor(self) -> dict[str, Any]:
        """
        Monitor market data for regime changes and risk indicators.

        Gathers:
        - Price history for portfolio symbols
        - Volatility metrics
        - Technical indicators (SMA, RSI)
        - Market breadth indicators
        """
        logger.info("Monitoring market data")

        now = datetime.now(UTC)

        # Get symbols to monitor (portfolio + benchmark + VIX proxy)
        symbols_to_monitor = list(set(self.portfolio_symbols + [self.benchmark_symbol]))

        # Fetch market data for each symbol
        market_data: dict[str, list[MarketOHLCV]] = {}
        for symbol in symbols_to_monitor:
            try:
                # Try to get from database first
                data = await self.market_repo.get_history(symbol, limit=60)
                if not data:
                    # Fetch fresh data
                    data = await self.market_client.get_history(
                        symbol, period="3mo", interval="1d"
                    )
                market_data[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}", error=str(e))
                market_data[symbol] = []

        # Calculate regime for benchmark
        benchmark_data = market_data.get(self.benchmark_symbol, [])
        regime_analysis = self._analyze_regime(benchmark_data) if benchmark_data else None

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility(benchmark_data) if benchmark_data else None

        # Get all tracked symbols count
        all_symbols = await self.market_repo.get_symbols()

        monitoring_data = {
            "market_data": market_data,
            "benchmark_data": benchmark_data,
            "regime_analysis": regime_analysis,
            "volatility_metrics": volatility_metrics,
            "symbols_monitored": len(symbols_to_monitor),
            "total_symbols": len(all_symbols),
            "timestamp": now,
        }

        logger.info(
            "Market monitoring complete",
            symbols_monitored=len(symbols_to_monitor),
            has_regime=regime_analysis is not None,
            has_volatility=volatility_metrics is not None,
        )

        return monitoring_data

    def _analyze_regime(self, data: list[MarketOHLCV]) -> MarketRegimeAnalysis | None:
        """Analyze market regime based on price action and indicators."""
        if len(data) < 50:
            return None

        closes = [d.close for d in data]

        # Calculate SMAs
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50

        # Current price position
        current_price = closes[-1]
        sma_20_position = (
            "above" if current_price > sma_20
            else "below" if current_price < sma_20
            else "crossing"
        )
        sma_50_position = (
            "above" if current_price > sma_50
            else "below" if current_price < sma_50
            else "crossing"
        )

        # Calculate RSI
        rsi = self._calculate_rsi(closes)

        # Calculate trend strength using linear regression slope
        trend_strength = self._calculate_trend_strength(closes[-20:])

        # Calculate volatility percentile
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]
        current_vol = self._std_dev(returns[-20:]) if len(returns) >= 20 else 0
        historical_vol = self._std_dev(returns) if returns else 0
        volatility_percentile = (
            (current_vol / historical_vol * 50) if historical_vol > 0 else 50
        )
        volatility_percentile = min(100, max(0, volatility_percentile))

        # Determine regime
        regime = self._classify_regime(
            trend_strength, volatility_percentile, rsi, sma_20_position, sma_50_position
        )

        return MarketRegimeAnalysis(
            regime=regime,
            confidence=0.7 + abs(trend_strength) * 0.3,
            trend_strength=trend_strength,
            volatility_percentile=volatility_percentile,
            sma_20_position=sma_20_position,
            sma_50_position=sma_50_position,
            rsi_14=rsi,
            days_in_regime=1,  # Would need historical tracking
            previous_regime=None,  # Would need historical tracking
        )

    def _classify_regime(
        self,
        trend_strength: float,
        volatility_percentile: float,
        rsi: float,
        sma_20_pos: str,
        sma_50_pos: str,
    ) -> MarketRegime:
        """Classify market regime based on multiple indicators."""
        # High volatility overrides other signals
        if volatility_percentile > 80:
            return MarketRegime.VOLATILE

        # Strong uptrend
        if (
            trend_strength > self.TREND_STRENGTH_THRESHOLD
            and sma_20_pos == "above"
            and sma_50_pos == "above"
        ):
            return MarketRegime.BULL

        # Strong downtrend
        if (
            trend_strength < -self.TREND_STRENGTH_THRESHOLD
            and sma_20_pos == "below"
            and sma_50_pos == "below"
        ):
            return MarketRegime.BEAR

        # Recovery (price above 20 SMA but below 50 SMA, positive momentum)
        if trend_strength > 0.2 and sma_20_pos == "above" and sma_50_pos == "below":
            return MarketRegime.RECOVERY

        # Default to sideways
        return MarketRegime.SIDEWAYS

    def _calculate_rsi(self, closes: list[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(-period, 0):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_trend_strength(self, prices: list[float]) -> float:
        """Calculate trend strength using normalized slope."""
        if len(prices) < 2:
            return 0.0

        n = len(prices)
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n

        numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        # Normalize by average price to get relative slope
        normalized_slope = (slope / y_mean) * n if y_mean != 0 else 0

        # Clamp to -1 to 1 range
        return max(-1.0, min(1.0, normalized_slope))

    def _std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _calculate_volatility(self, data: list[MarketOHLCV]) -> VolatilityMetrics | None:
        """Calculate volatility metrics."""
        if len(data) < 20:
            return None

        closes = [d.close for d in data]

        # Calculate returns
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]

        # 20-day volatility (annualized)
        vol_20d = self._std_dev(returns[-20:]) * (252 ** 0.5) * 100  # Percentage

        # Historical average volatility
        vol_historical = self._std_dev(returns) * (252 ** 0.5) * 100

        # Volatility ratio
        vol_ratio = vol_20d / vol_historical if vol_historical > 0 else 1.0

        # Percentile rank (simplified)
        percentile = min(100, (vol_20d / 30) * 100)  # Assuming 30% is 100th percentile

        # Determine risk level
        if vol_20d < self.VOLATILITY_LOW:
            risk_level = RiskLevel.LOW
        elif vol_20d < self.VOLATILITY_MODERATE:
            risk_level = RiskLevel.MODERATE
        elif vol_20d < self.VOLATILITY_HIGH:
            risk_level = RiskLevel.ELEVATED
        else:
            risk_level = RiskLevel.HIGH

        return VolatilityMetrics(
            current_volatility=vol_20d,
            volatility_20d=vol_20d,
            volatility_ratio=vol_ratio,
            vix_level=None,  # Would need VIX data
            risk_level=risk_level,
            percentile_rank=percentile,
        )

    # ========================================================================
    # Analyze Phase
    # ========================================================================

    async def analyze(self, monitoring_data: dict[str, Any]) -> list[Insight]:
        """
        Analyze market data and generate insights.

        Generates insights for:
        - Market regime changes
        - Volatility spikes
        - Overbought/oversold conditions
        - Drawdown warnings
        - Sector rotations
        """
        insights = []

        regime = monitoring_data.get("regime_analysis")
        volatility = monitoring_data.get("volatility_metrics")
        market_data = monitoring_data.get("market_data", {})

        # 1. Analyze market regime
        if regime:
            regime_insights = self._analyze_market_regime(regime)
            insights.extend(regime_insights)

        # 2. Analyze volatility
        if volatility:
            vol_insights = self._analyze_volatility(volatility)
            insights.extend(vol_insights)

        # 3. Analyze individual holdings
        for symbol, data in market_data.items():
            if data:
                holding_insights = self._analyze_holding(symbol, data)
                insights.extend(holding_insights)

        # 4. Analyze portfolio-level risks
        portfolio_insights = await self._analyze_portfolio(market_data)
        insights.extend(portfolio_insights)

        logger.info(
            "Market analysis complete",
            total_insights=len(insights),
        )

        return insights

    def _analyze_market_regime(self, regime: MarketRegimeAnalysis) -> list[Insight]:
        """Generate insights from market regime analysis."""
        insights = []

        # Alert on extreme regimes
        if regime.regime == MarketRegime.BEAR:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title="Bear Market Detected",
                    description=(
                        f"Market is in a bearish regime with trend strength of "
                        f"{regime.trend_strength:.2f}. Price is below both 20 and 50 SMAs. "
                        f"Consider defensive positioning."
                    ),
                    level=AlertLevel.ACTION,
                    confidence=regime.confidence,
                    relevance_score=0.95,
                    metadata={
                        "regime": regime.regime.value,
                        "trend_strength": regime.trend_strength,
                        "rsi": regime.rsi_14,
                        "finance_insight_type": FinanceInsightType.MARKET_REGIME_CHANGE.value,
                    },
                )
            )
        elif regime.regime == MarketRegime.VOLATILE:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="High Market Volatility",
                    description=(
                        f"Market volatility is elevated (percentile: {regime.volatility_percentile:.0f}%). "
                        f"Consider reducing position sizes and widening stop losses."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=regime.confidence,
                    relevance_score=0.9,
                    metadata={
                        "regime": regime.regime.value,
                        "volatility_percentile": regime.volatility_percentile,
                        "finance_insight_type": FinanceInsightType.VOLATILITY_SPIKE.value,
                    },
                )
            )
        elif regime.regime == MarketRegime.BULL and regime.rsi_14 > self.RSI_OVERBOUGHT:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title="Overbought Market Conditions",
                    description=(
                        f"Market is overbought with RSI at {regime.rsi_14:.1f}. "
                        f"While trend is bullish, a pullback may be imminent."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.7,
                    relevance_score=0.7,
                    metadata={
                        "regime": regime.regime.value,
                        "rsi": regime.rsi_14,
                        "finance_insight_type": FinanceInsightType.OVERBOUGHT.value,
                    },
                )
            )
        elif regime.regime == MarketRegime.RECOVERY:
            insights.append(
                Insight(
                    insight_type=InsightType.EMERGING_TECH,
                    title="Market Recovery in Progress",
                    description=(
                        f"Market shows signs of recovery. Price is above 20 SMA with "
                        f"positive momentum ({regime.trend_strength:.2f}). "
                        f"Consider gradual re-entry into risk assets."
                    ),
                    level=AlertLevel.INFO,
                    confidence=regime.confidence,
                    relevance_score=0.8,
                    metadata={
                        "regime": regime.regime.value,
                        "trend_strength": regime.trend_strength,
                        "finance_insight_type": FinanceInsightType.MARKET_REGIME_CHANGE.value,
                    },
                )
            )

        return insights

    def _analyze_volatility(self, metrics: VolatilityMetrics) -> list[Insight]:
        """Generate insights from volatility metrics."""
        insights = []

        # Volatility spike warning
        if metrics.volatility_ratio > self.VOLATILITY_SPIKE_THRESHOLD:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="Volatility Spike Detected",
                    description=(
                        f"Current volatility ({metrics.current_volatility:.1f}%) is "
                        f"{metrics.volatility_ratio:.1f}x the historical average. "
                        f"This indicates increased market uncertainty."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=0.85,
                    relevance_score=0.9,
                    metadata={
                        "current_volatility": metrics.current_volatility,
                        "volatility_ratio": metrics.volatility_ratio,
                        "risk_level": metrics.risk_level.value,
                        "finance_insight_type": FinanceInsightType.VOLATILITY_SPIKE.value,
                    },
                )
            )

        # Risk level changes
        if metrics.risk_level == RiskLevel.HIGH:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="High Risk Environment",
                    description=(
                        f"Market volatility ({metrics.current_volatility:.1f}%) indicates "
                        f"a high-risk environment. Review position sizing and hedging strategies."
                    ),
                    level=AlertLevel.ACTION,
                    confidence=0.9,
                    relevance_score=0.95,
                    metadata={
                        "risk_level": metrics.risk_level.value,
                        "volatility": metrics.current_volatility,
                        "finance_insight_type": FinanceInsightType.VOLATILITY_SPIKE.value,
                    },
                )
            )
        elif metrics.risk_level == RiskLevel.LOW:
            insights.append(
                Insight(
                    insight_type=InsightType.EMERGING_TECH,
                    title="Low Volatility Environment",
                    description=(
                        f"Market volatility is low ({metrics.current_volatility:.1f}%). "
                        f"Good conditions for trend-following strategies."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.8,
                    relevance_score=0.6,
                    metadata={
                        "risk_level": metrics.risk_level.value,
                        "volatility": metrics.current_volatility,
                        "finance_insight_type": FinanceInsightType.VOLATILITY_SPIKE.value,
                    },
                )
            )

        return insights

    def _analyze_holding(self, symbol: str, data: list[MarketOHLCV]) -> list[Insight]:
        """Analyze individual holding for signals."""
        insights = []

        if len(data) < 14:
            return insights

        closes = [d.close for d in data]
        current_price = closes[-1]
        high_52w = max(closes) if len(closes) >= 252 else max(closes)

        # Calculate drawdown from peak
        drawdown = (current_price - high_52w) / high_52w

        # RSI analysis
        rsi = self._calculate_rsi(closes)

        # Drawdown warning
        if drawdown < self.DRAWDOWN_CRITICAL:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title=f"Critical Drawdown: {symbol}",
                    description=(
                        f"{symbol} is down {abs(drawdown) * 100:.1f}% from its high. "
                        f"Review position and consider stop-loss or hedging."
                    ),
                    level=AlertLevel.ACTION,
                    confidence=0.9,
                    relevance_score=0.95,
                    entity_names=[symbol],
                    metadata={
                        "symbol": symbol,
                        "drawdown": drawdown,
                        "current_price": current_price,
                        "finance_insight_type": FinanceInsightType.DRAWDOWN_WARNING.value,
                    },
                )
            )
        elif drawdown < self.DRAWDOWN_WARNING:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title=f"Drawdown Alert: {symbol}",
                    description=(
                        f"{symbol} is down {abs(drawdown) * 100:.1f}% from its high. "
                        f"Monitor closely for further weakness."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=0.85,
                    relevance_score=0.8,
                    entity_names=[symbol],
                    metadata={
                        "symbol": symbol,
                        "drawdown": drawdown,
                        "finance_insight_type": FinanceInsightType.DRAWDOWN_WARNING.value,
                    },
                )
            )

        # Overbought/oversold
        if rsi > self.RSI_OVERBOUGHT:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title=f"Overbought: {symbol}",
                    description=(
                        f"{symbol} RSI is {rsi:.1f}, indicating overbought conditions. "
                        f"Consider taking profits or tightening stops."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.7,
                    relevance_score=0.7,
                    entity_names=[symbol],
                    metadata={
                        "symbol": symbol,
                        "rsi": rsi,
                        "finance_insight_type": FinanceInsightType.OVERBOUGHT.value,
                    },
                )
            )
        elif rsi < self.RSI_OVERSOLD:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title=f"Oversold: {symbol}",
                    description=(
                        f"{symbol} RSI is {rsi:.1f}, indicating oversold conditions. "
                        f"May present a buying opportunity if fundamentals are intact."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.7,
                    relevance_score=0.7,
                    entity_names=[symbol],
                    metadata={
                        "symbol": symbol,
                        "rsi": rsi,
                        "finance_insight_type": FinanceInsightType.OVERSOLD.value,
                    },
                )
            )

        return insights

    async def _analyze_portfolio(
        self, market_data: dict[str, list[MarketOHLCV]]
    ) -> list[Insight]:
        """Analyze portfolio-level risks and correlations."""
        insights = []

        if len(market_data) < 2:
            return insights

        # Calculate correlations between holdings
        correlations = self._calculate_correlations(market_data)

        # Check for high correlations (correlation risk)
        high_correlations = [
            (pair, corr) for pair, corr in correlations.items()
            if corr > 0.9 and pair[0] != pair[1]
        ]

        if high_correlations:
            pairs_str = ", ".join(f"{p[0]}/{p[1]}" for p, _ in high_correlations[:3])
            insights.append(
                Insight(
                    insight_type=InsightType.CORRELATION,
                    title="High Portfolio Correlation",
                    description=(
                        f"High correlation detected between: {pairs_str}. "
                        f"Consider diversifying to reduce concentration risk."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=0.8,
                    relevance_score=0.75,
                    metadata={
                        "correlations": {
                            f"{p[0]}/{p[1]}": c for p, c in high_correlations[:5]
                        },
                        "finance_insight_type": FinanceInsightType.CORRELATION_BREAKDOWN.value,
                    },
                )
            )

        return insights

    def _calculate_correlations(
        self, market_data: dict[str, list[MarketOHLCV]]
    ) -> dict[tuple[str, str], float]:
        """Calculate pairwise correlations between symbols."""
        correlations = {}

        symbols = list(market_data.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i:]:
                data1 = market_data[sym1]
                data2 = market_data[sym2]

                if len(data1) < 20 or len(data2) < 20:
                    continue

                # Get overlapping dates
                dates1 = {d.date.date(): d.close for d in data1}
                dates2 = {d.date.date(): d.close for d in data2}

                common_dates = set(dates1.keys()) & set(dates2.keys())
                if len(common_dates) < 20:
                    continue

                sorted_dates = sorted(common_dates)
                closes1 = [dates1[d] for d in sorted_dates]
                closes2 = [dates2[d] for d in sorted_dates]

                # Calculate returns
                returns1 = [
                    (closes1[i] - closes1[i - 1]) / closes1[i - 1]
                    for i in range(1, len(closes1))
                ]
                returns2 = [
                    (closes2[i] - closes2[i - 1]) / closes2[i - 1]
                    for i in range(1, len(closes2))
                ]

                # Calculate correlation
                corr = self._correlation(returns1, returns2)
                correlations[(sym1, sym2)] = corr

        return correlations

    def _correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return numerator / (n * std_x * std_y)

    # ========================================================================
    # Portfolio Risk Assessment
    # ========================================================================

    def assess_portfolio_risk(
        self,
        market_data: dict[str, list[MarketOHLCV]],
        weights: dict[str, float] | None = None,
    ) -> PortfolioRiskAssessment:
        """Generate comprehensive portfolio risk assessment."""
        if weights is None:
            # Equal weight by default
            n = len(market_data)
            weights = {sym: 1.0 / n for sym in market_data.keys()}

        recommendations = []

        # Calculate individual metrics
        drawdowns = {}
        volatilities = {}

        for symbol, data in market_data.items():
            if len(data) < 20:
                continue

            closes = [d.close for d in data]
            high = max(closes)
            current = closes[-1]
            drawdowns[symbol] = (current - high) / high

            returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
            ]
            volatilities[symbol] = self._std_dev(returns[-20:]) * (252 ** 0.5)

        # Portfolio-level metrics
        max_drawdown = min(drawdowns.values()) if drawdowns else 0
        avg_volatility = (
            sum(volatilities.get(s, 0) * weights.get(s, 0) for s in market_data)
            if volatilities else 0
        )

        # Concentration risk (largest position)
        max_weight = max(weights.values()) if weights else 0
        concentration_risk = max_weight * 100

        # Correlation risk (simplified)
        correlations = self._calculate_correlations(market_data)
        avg_correlation = (
            sum(correlations.values()) / len(correlations)
            if correlations else 0
        )
        correlation_risk = avg_correlation * 100

        # Total risk score (0-100)
        total_risk = (
            abs(max_drawdown) * 100 * 0.3 +  # Drawdown component
            avg_volatility * 0.3 +  # Volatility component
            concentration_risk * 0.2 +  # Concentration component
            abs(correlation_risk) * 0.2  # Correlation component
        )
        total_risk = min(100, max(0, total_risk))

        # Determine risk level
        if total_risk < 25:
            risk_level = RiskLevel.LOW
        elif total_risk < 50:
            risk_level = RiskLevel.MODERATE
        elif total_risk < 75:
            risk_level = RiskLevel.ELEVATED
        else:
            risk_level = RiskLevel.HIGH

        # Generate recommendations
        if max_drawdown < -0.15:
            recommendations.append(
                "Consider reducing exposure to positions with significant drawdowns"
            )
        if concentration_risk > 30:
            recommendations.append(
                "Portfolio is concentrated. Consider diversifying across more positions"
            )
        if avg_correlation > 0.8:
            recommendations.append(
                "Holdings are highly correlated. Add uncorrelated assets for diversification"
            )
        if avg_volatility > 0.25:
            recommendations.append(
                "Portfolio volatility is elevated. Consider defensive positioning"
            )

        return PortfolioRiskAssessment(
            total_risk_score=total_risk,
            risk_level=risk_level,
            max_drawdown_current=max_drawdown,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            volatility_risk=avg_volatility * 100,
            recommendations=recommendations,
        )

    # ========================================================================
    # Rebalancing Signals
    # ========================================================================

    def generate_rebalancing_signals(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> list[RebalancingSignal]:
        """Generate rebalancing signals based on weight deviations."""
        signals = []

        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            deviation = current - target

            if abs(deviation) < self.rebalance_threshold:
                continue

            if deviation > 0:
                action = "sell"
                priority = 2
            else:
                action = "buy"
                priority = 1  # Buying to target is usually higher priority

            # Increase priority for large deviations
            if abs(deviation) > self.rebalance_threshold * 2:
                priority = 1

            signals.append(
                RebalancingSignal(
                    symbol=symbol,
                    current_weight=current,
                    target_weight=target,
                    deviation=deviation,
                    action=action,
                    priority=priority,
                    rationale=(
                        f"{symbol} is {'overweight' if deviation > 0 else 'underweight'} "
                        f"by {abs(deviation) * 100:.1f}%"
                    ),
                )
            )

        return sorted(signals, key=lambda s: (s.priority, -abs(s.deviation)))

    # ========================================================================
    # Report Generation
    # ========================================================================

    def generate_weekly_report(self, report: AgentReport) -> str:
        """Generate a formatted weekly market report."""
        lines = [
            "# Weekly Market Intelligence Report",
            f"Generated: {report.run_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        # Extract regime analysis from metadata
        regime = report.metadata.get("regime_analysis")
        if regime:
            lines.extend([
                "## Market Regime",
                f"- **Current Regime**: {regime.regime.value.title()}",
                f"- **Trend Strength**: {regime.trend_strength:.2f}",
                f"- **RSI (14)**: {regime.rsi_14:.1f}",
                f"- **Confidence**: {regime.confidence:.0%}",
                "",
            ])

        # Volatility metrics
        volatility = report.metadata.get("volatility_metrics")
        if volatility:
            lines.extend([
                "## Volatility Metrics",
                f"- **Current Volatility**: {volatility.current_volatility:.1f}%",
                f"- **Risk Level**: {volatility.risk_level.value.title()}",
                f"- **Volatility Ratio**: {volatility.volatility_ratio:.2f}x",
                "",
            ])

        # Urgent/Action items
        urgent_action = [
            i for i in report.insights
            if i.level in (AlertLevel.URGENT, AlertLevel.ACTION)
        ]
        if urgent_action:
            lines.append("## Action Required")
            for insight in urgent_action:
                lines.append(f"- **{insight.title}**: {insight.description}")
            lines.append("")

        # Watch items
        watch = [i for i in report.insights if i.level == AlertLevel.WATCH]
        if watch:
            lines.append("## Items to Watch")
            for insight in watch:
                lines.append(f"- **{insight.title}**: {insight.description}")
            lines.append("")

        # Portfolio risk
        risk_assessment = report.metadata.get("risk_assessment")
        if risk_assessment:
            lines.extend([
                "## Portfolio Risk Assessment",
                f"- **Risk Score**: {risk_assessment.total_risk_score:.0f}/100",
                f"- **Risk Level**: {risk_assessment.risk_level.value.title()}",
                f"- **Max Drawdown**: {risk_assessment.max_drawdown_current:.1%}",
                "",
            ])
            if risk_assessment.recommendations:
                lines.append("### Recommendations")
                for rec in risk_assessment.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")

        # Summary stats
        lines.extend([
            "## Summary",
            f"- Symbols monitored: {report.entities_scanned}",
            f"- Insights generated: {len(report.insights)}",
            f"- Alerts: {report.alerts_generated}",
            "",
        ])

        return "\n".join(lines)

    # ========================================================================
    # Override run() to include finance-specific data
    # ========================================================================

    async def run(self) -> AgentReport:
        """Execute the finance agent with additional metadata."""
        start_time = datetime.now(UTC)
        logger.info(f"Starting {self.name} agent run")

        try:
            # Monitor
            monitoring_data = await self.monitor()

            # Analyze
            insights = await self.analyze(monitoring_data)

            # Decide (filter)
            filtered_insights = await self.decide(insights)

            # Generate portfolio risk assessment
            market_data = monitoring_data.get("market_data", {})
            risk_assessment = None
            if market_data:
                risk_assessment = self.assess_portfolio_risk(market_data)

            # Calculate execution time
            execution_time = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            report = AgentReport(
                agent_name=self.name,
                domain=self.domain,
                run_at=start_time,
                insights=filtered_insights,
                trends_analyzed=0,
                entities_scanned=monitoring_data.get("symbols_monitored", 0),
                alerts_generated=len([i for i in filtered_insights if i.level != AlertLevel.INFO]),
                execution_time_ms=execution_time,
                metadata={
                    "regime_analysis": monitoring_data.get("regime_analysis"),
                    "volatility_metrics": monitoring_data.get("volatility_metrics"),
                    "risk_assessment": risk_assessment,
                    "symbols_monitored": monitoring_data.get("symbols_monitored", 0),
                    "total_symbols": monitoring_data.get("total_symbols", 0),
                },
            )

            logger.info(
                f"Completed {self.name} agent run",
                insights=len(filtered_insights),
                risk_level=risk_assessment.risk_level.value if risk_assessment else "unknown",
                execution_time_ms=execution_time,
            )

            return report

        except Exception as e:
            logger.error(f"Error in {self.name} agent run", error=str(e))
            raise
