"""Synthesis Agent for cross-dimensional intelligence and compound insights.

Monitors:
- Recent insights from tech, fitness, and finance agents
- Cross-domain correlations and patterns
- Compound signals across multiple dimensions

Outputs:
- Cross-dimensional pattern alerts
- Compound insights with multi-domain reasoning
- Priority-ranked opportunities and risks
- Explanations linking domain-specific signals
"""

from dataclasses import dataclass, field
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
    ProjectIdea,
    UserProfile,
)
from src.database import Database
from src.world_model import Domain

logger = structlog.get_logger()


# ============================================================================
# Synthesis-Specific Models
# ============================================================================


class SynthesisInsightType(str, Enum):
    """Types of cross-dimensional insights."""

    COMPOUND_RISK = "compound_risk"  # Multiple domains signal risk
    COMPOUND_OPPORTUNITY = "compound_opportunity"  # Multiple domains signal opportunity
    DOMAIN_CONFLICT = "domain_conflict"  # Domains have conflicting signals
    TIMING_ALIGNMENT = "timing_alignment"  # Multiple domains align for action
    PATTERN_EMERGENCE = "pattern_emergence"  # New cross-domain pattern detected
    LIFESTYLE_BALANCE = "lifestyle_balance"  # Work/fitness/finance balance insight


class DomainSignal(str, Enum):
    """Signal types from each domain."""

    # Tech signals
    TECH_TREND_RISING = "tech_trend_rising"
    TECH_SKILL_GAP = "tech_skill_gap"
    TECH_LEARNING_OPPORTUNITY = "tech_learning_opportunity"

    # Fitness signals
    FITNESS_OVERTRAINING = "fitness_overtraining"
    FITNESS_PEAK_FORM = "fitness_peak_form"
    FITNESS_RECOVERY_NEEDED = "fitness_recovery_needed"
    FITNESS_GOAL_READY = "fitness_goal_ready"

    # Finance signals
    FINANCE_BULL_MARKET = "finance_bull_market"
    FINANCE_BEAR_MARKET = "finance_bear_market"
    FINANCE_HIGH_VOLATILITY = "finance_high_volatility"
    FINANCE_OPPORTUNITY = "finance_opportunity"
    FINANCE_RISK_HIGH = "finance_risk_high"


@dataclass
class CrossDomainPattern:
    """A detected cross-domain pattern."""

    pattern_type: SynthesisInsightType
    domains_involved: list[Domain]
    signals: list[DomainSignal]
    description: str
    reasoning: str
    impact_score: float  # 0.0-1.0, how significant is this pattern
    confidence: float  # 0.0-1.0
    recommended_actions: list[str]
    source_insights: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DomainState:
    """Current state summary from a domain agent."""

    domain: Domain
    active_signals: list[DomainSignal]
    risk_level: float  # 0.0-1.0
    opportunity_level: float  # 0.0-1.0
    recent_insights: list[Insight]
    key_metrics: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Cross-Domain Pattern Definitions
# ============================================================================


CROSS_DOMAIN_PATTERNS = [
    # Risk patterns
    {
        "name": "stress_overload",
        "type": SynthesisInsightType.COMPOUND_RISK,
        "signals_required": [
            (DomainSignal.FITNESS_OVERTRAINING, DomainSignal.TECH_SKILL_GAP),
            (DomainSignal.FITNESS_RECOVERY_NEEDED, DomainSignal.TECH_SKILL_GAP),
        ],
        "description": (
            "High training stress combined with work skill pressure "
            "increases injury and burnout risk"
        ),
        "level": AlertLevel.ACTION,
        "actions": [
            "Reduce training volume temporarily",
            "Focus on maintenance workouts",
            "Defer non-urgent skill development",
        ],
    },
    {
        "name": "financial_fitness_conflict",
        "type": SynthesisInsightType.DOMAIN_CONFLICT,
        "signals_required": [
            (
                DomainSignal.FINANCE_HIGH_VOLATILITY,
                DomainSignal.FITNESS_GOAL_READY,
            ),
        ],
        "description": (
            "Market volatility may impact race/event investments "
            "despite peak fitness"
        ),
        "level": AlertLevel.WATCH,
        "actions": [
            "Review event-related expenses",
            "Consider local alternatives if travel involved",
            "Maintain training while monitoring market",
        ],
    },
    {
        "name": "triple_stress",
        "type": SynthesisInsightType.COMPOUND_RISK,
        "signals_required": [
            (
                DomainSignal.FITNESS_OVERTRAINING,
                DomainSignal.FINANCE_HIGH_VOLATILITY,
                DomainSignal.TECH_SKILL_GAP,
            ),
        ],
        "description": (
            "High stress across all dimensions - "
            "physical, financial, and professional"
        ),
        "level": AlertLevel.URGENT,
        "actions": [
            "Immediate stress reduction protocol",
            "Pause non-essential training",
            "Avoid major financial decisions",
            "Focus on core work responsibilities only",
        ],
    },
    # Opportunity patterns
    {
        "name": "tech_finance_opportunity",
        "type": SynthesisInsightType.COMPOUND_OPPORTUNITY,
        "signals_required": [
            (DomainSignal.TECH_TREND_RISING, DomainSignal.FINANCE_BULL_MARKET),
            (
                DomainSignal.TECH_LEARNING_OPPORTUNITY,
                DomainSignal.FINANCE_OPPORTUNITY,
            ),
        ],
        "description": (
            "Rising tech trend aligns with positive market conditions - "
            "good time to invest in skills"
        ),
        "level": AlertLevel.ACTION,
        "actions": [
            "Prioritize learning the emerging technology",
            "Consider related investment opportunities",
            "Network in the growing space",
        ],
    },
    {
        "name": "peak_performance_window",
        "type": SynthesisInsightType.TIMING_ALIGNMENT,
        "signals_required": [
            (DomainSignal.FITNESS_PEAK_FORM, DomainSignal.FINANCE_OPPORTUNITY),
        ],
        "description": (
            "Physical peak form with financial stability - "
            "optimal window for ambitious goals"
        ),
        "level": AlertLevel.ACTION,
        "actions": [
            "Schedule key race or event",
            "Consider career moves or negotiations",
            "Maximize current momentum",
        ],
    },
    {
        "name": "recovery_investment",
        "type": SynthesisInsightType.LIFESTYLE_BALANCE,
        "signals_required": [
            (
                DomainSignal.FITNESS_RECOVERY_NEEDED,
                DomainSignal.TECH_LEARNING_OPPORTUNITY,
            ),
        ],
        "description": (
            "Recovery period is ideal for skill development - "
            "redirect energy to learning"
        ),
        "level": AlertLevel.INFO,
        "actions": [
            "Use recovery days for online courses",
            "Light study during active recovery",
            "Balance rest with mental stimulation",
        ],
    },
    {
        "name": "financial_learning_window",
        "type": SynthesisInsightType.TIMING_ALIGNMENT,
        "signals_required": [
            (
                DomainSignal.FINANCE_BEAR_MARKET,
                DomainSignal.TECH_LEARNING_OPPORTUNITY,
            ),
        ],
        "description": (
            "Bear market is ideal time to invest in skills - "
            "lower opportunity cost"
        ),
        "level": AlertLevel.INFO,
        "actions": [
            "Focus on skill building over active trading",
            "Use downtime to prepare for next cycle",
            "Build expertise for market recovery",
        ],
    },
]


# ============================================================================
# Synthesis Intelligence Agent
# ============================================================================


class SynthesisAgent(BaseAgent):
    """
    Proactive agent for cross-dimensional intelligence.

    Synthesizes insights from tech, fitness, and finance domains
    to detect compound patterns and generate holistic recommendations.
    """

    name = "synthesis_intelligence"
    domain = Domain.GENERAL

    def __init__(
        self,
        db: Database,
        config: AgentConfig | None = None,
        user_profile: UserProfile | None = None,
        domain_insights: dict[Domain, list[Insight]] | None = None,
    ):
        super().__init__(db, config, user_profile)

        # Pre-loaded insights from domain agents (optional)
        self.domain_insights = domain_insights or {}

        # Synthesis-specific thresholds
        self.min_domains_for_pattern = int(
            self.config.thresholds.get("min_domains", 2)
        )
        self.impact_threshold = self.config.thresholds.get("impact_threshold", 0.5)
        self.lookback_hours = int(self.config.thresholds.get("lookback_hours", 24))

        logger.info(
            "Initialized SynthesisAgent",
            min_domains=self.min_domains_for_pattern,
            impact_threshold=self.impact_threshold,
            lookback_hours=self.lookback_hours,
        )

    # ========================================================================
    # Monitor Phase
    # ========================================================================

    async def monitor(self) -> dict[str, Any]:
        """
        Monitor all domains for cross-dimensional patterns.

        Gathers:
        - Recent insights from each domain agent
        - Cross-domain correlations
        - Domain state summaries
        """
        logger.info("Monitoring for cross-dimensional patterns")

        # Build domain states from available insights
        domain_states: dict[Domain, DomainState] = {}

        for domain in [Domain.TECH, Domain.FITNESS, Domain.FINANCE]:
            insights = self.domain_insights.get(domain, [])
            signals = self._extract_signals_from_insights(domain, insights)
            risk_level = self._calculate_domain_risk(domain, insights)
            opportunity_level = self._calculate_domain_opportunity(domain, insights)

            domain_states[domain] = DomainState(
                domain=domain,
                active_signals=signals,
                risk_level=risk_level,
                opportunity_level=opportunity_level,
                recent_insights=insights,
            )

            logger.debug(
                "Domain state extracted",
                domain=domain.value,
                signals=[s.value for s in signals],
                risk=risk_level,
                opportunity=opportunity_level,
            )

        # Look for cross-domain correlations in the world model
        cross_domain_correlations = await self._find_cross_domain_correlations()

        # Count entities and trends across domains
        entities_count = sum(
            len(state.recent_insights) for state in domain_states.values()
        )

        return {
            "domain_states": domain_states,
            "cross_correlations": cross_domain_correlations,
            "entities_count": entities_count,
            "trends_count": len(cross_domain_correlations),
        }

    def _extract_signals_from_insights(
        self, domain: Domain, insights: list[Insight]
    ) -> list[DomainSignal]:
        """Extract domain signals from insights."""
        signals = []

        for insight in insights:
            # Map insight types to signals based on domain and alert level
            if domain == Domain.TECH:
                signals.extend(self._extract_tech_signals(insight))
            elif domain == Domain.FITNESS:
                signals.extend(self._extract_fitness_signals(insight))
            elif domain == Domain.FINANCE:
                signals.extend(self._extract_finance_signals(insight))

        return list(set(signals))  # Deduplicate

    def _extract_tech_signals(self, insight: Insight) -> list[DomainSignal]:
        """Extract tech domain signals from an insight."""
        signals = []

        if insight.insight_type == InsightType.EMERGING_TECH:
            signals.append(DomainSignal.TECH_TREND_RISING)
        elif insight.insight_type == InsightType.SKILL_GAP:
            signals.append(DomainSignal.TECH_SKILL_GAP)
        elif insight.insight_type == InsightType.PROJECT_IDEA:
            signals.append(DomainSignal.TECH_LEARNING_OPPORTUNITY)
        elif insight.insight_type == InsightType.ADOPTION_SPIKE:
            signals.append(DomainSignal.TECH_TREND_RISING)

        # Learning opportunity from any high-priority tech insight
        if insight.level in (AlertLevel.ACTION, AlertLevel.URGENT):
            if insight.relevance_score > 0.7:
                signals.append(DomainSignal.TECH_LEARNING_OPPORTUNITY)

        return signals

    def _extract_fitness_signals(self, insight: Insight) -> list[DomainSignal]:
        """Extract fitness domain signals from an insight."""
        signals = []
        metadata = insight.metadata

        # Check for specific fitness insight types in metadata
        fitness_type = metadata.get("fitness_insight_type", "")
        title_lower = insight.title.lower()
        is_overtraining = (
            "overtraining" in title_lower or "overreached" in fitness_type.lower()
        )
        if is_overtraining:
            signals.append(DomainSignal.FITNESS_OVERTRAINING)
        if "recovery" in title_lower or "rest" in title_lower:
            signals.append(DomainSignal.FITNESS_RECOVERY_NEEDED)
        if "peak" in title_lower or "optimal" in title_lower:
            signals.append(DomainSignal.FITNESS_PEAK_FORM)
        if "ready" in title_lower or metadata.get("goal_ready"):
            signals.append(DomainSignal.FITNESS_GOAL_READY)

        # Check training zone from metadata
        training_zone = metadata.get("training_zone", "")
        if training_zone == "OVERREACHED":
            signals.append(DomainSignal.FITNESS_OVERTRAINING)
        elif training_zone == "OPTIMAL":
            signals.append(DomainSignal.FITNESS_PEAK_FORM)
        elif training_zone in ("FRESH", "TRANSITION"):
            signals.append(DomainSignal.FITNESS_RECOVERY_NEEDED)

        return signals

    def _extract_finance_signals(self, insight: Insight) -> list[DomainSignal]:
        """Extract finance domain signals from an insight."""
        signals = []
        metadata = insight.metadata

        # Check market regime from metadata
        market_regime = metadata.get("market_regime", "")

        if market_regime == "BULL":
            signals.append(DomainSignal.FINANCE_BULL_MARKET)
        elif market_regime == "BEAR":
            signals.append(DomainSignal.FINANCE_BEAR_MARKET)
        elif market_regime == "VOLATILE":
            signals.append(DomainSignal.FINANCE_HIGH_VOLATILITY)

        # Check risk level
        risk_level = metadata.get("risk_level", "")
        if risk_level in ("HIGH", "EXTREME"):
            signals.append(DomainSignal.FINANCE_RISK_HIGH)

        # Check for opportunities
        if "opportunity" in insight.title.lower():
            signals.append(DomainSignal.FINANCE_OPPORTUNITY)
        if "volatility" in insight.title.lower() or "spike" in insight.title.lower():
            signals.append(DomainSignal.FINANCE_HIGH_VOLATILITY)
        if insight.level == AlertLevel.URGENT and "risk" in insight.title.lower():
            signals.append(DomainSignal.FINANCE_RISK_HIGH)

        return signals

    def _calculate_domain_risk(
        self, domain: Domain, insights: list[Insight]
    ) -> float:
        """Calculate aggregate risk level for a domain."""
        if not insights:
            return 0.0

        # Weight by alert level
        risk_weights = {
            AlertLevel.INFO: 0.1,
            AlertLevel.WATCH: 0.3,
            AlertLevel.ACTION: 0.6,
            AlertLevel.URGENT: 1.0,
        }

        risk_insights = [
            i for i in insights
            if "risk" in i.title.lower()
            or i.level in (AlertLevel.ACTION, AlertLevel.URGENT)
        ]

        if not risk_insights:
            return 0.0

        total_risk = sum(
            risk_weights.get(i.level, 0.1) * i.confidence
            for i in risk_insights
        )
        return min(1.0, total_risk / len(risk_insights))

    def _calculate_domain_opportunity(
        self, domain: Domain, insights: list[Insight]
    ) -> float:
        """Calculate aggregate opportunity level for a domain."""
        if not insights:
            return 0.0

        opportunity_keywords = ["opportunity", "emerging", "rising", "peak", "optimal"]

        opp_insights = [
            i for i in insights
            if any(kw in i.title.lower() for kw in opportunity_keywords)
            and i.level != AlertLevel.URGENT  # Urgent is usually risk
        ]

        if not opp_insights:
            return 0.0

        total_opp = sum(i.relevance_score * i.confidence for i in opp_insights)
        return min(1.0, total_opp / len(opp_insights))

    async def _find_cross_domain_correlations(self) -> list[dict[str, Any]]:
        """Find correlations across domains in the world model."""
        correlations = []

        try:
            # Look for entities mentioned across multiple domains
            corr_svc = self.correlation_service
            cross_domain_entities = await corr_svc.find_cross_source_entities(
                min_sources=2,
                limit=20,
            )

            for entity in cross_domain_entities:
                if entity.source_agreement_score > 0.5:
                    correlations.append({
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type.value,
                        "sources": entity.sources,
                        "agreement_score": entity.source_agreement_score,
                        "total_mentions": entity.total_mentions,
                    })

        except Exception as e:
            logger.warning("Could not fetch cross-domain correlations", error=str(e))

        return correlations

    # ========================================================================
    # Analyze Phase
    # ========================================================================

    async def analyze(self, monitoring_data: dict[str, Any]) -> list[Insight]:
        """
        Analyze monitoring data for cross-dimensional patterns.

        Detects:
        - Compound risk patterns
        - Compound opportunity patterns
        - Domain conflicts
        - Timing alignments
        """
        logger.info("Analyzing for cross-dimensional patterns")

        insights = []
        domain_states: dict[Domain, DomainState] = monitoring_data["domain_states"]
        cross_correlations = monitoring_data["cross_correlations"]

        # Collect all active signals across domains
        all_signals: set[DomainSignal] = set()
        for state in domain_states.values():
            all_signals.update(state.active_signals)

        logger.debug(
            "Active signals across domains",
            signals=[s.value for s in all_signals],
        )

        # Check each defined pattern
        for pattern_def in CROSS_DOMAIN_PATTERNS:
            matched_pattern = self._match_pattern(pattern_def, all_signals)
            if matched_pattern:
                insight = self._create_pattern_insight(
                    pattern_def, matched_pattern, domain_states
                )
                insights.append(insight)

        # Generate insights from cross-domain correlations
        correlation_insights = self._analyze_correlations(
            cross_correlations, domain_states
        )
        insights.extend(correlation_insights)

        # Check for lifestyle balance insights
        balance_insights = self._analyze_lifestyle_balance(domain_states)
        insights.extend(balance_insights)

        logger.info(
            "Cross-dimensional analysis complete",
            patterns_detected=len(insights),
        )

        return insights

    def _match_pattern(
        self, pattern_def: dict[str, Any], active_signals: set[DomainSignal]
    ) -> list[DomainSignal] | None:
        """Check if a pattern matches the current signals."""
        signals_required = pattern_def["signals_required"]

        for signal_combo in signals_required:
            # Ensure signal_combo is a tuple
            if not isinstance(signal_combo, tuple):
                signal_combo = (signal_combo,)

            if all(signal in active_signals for signal in signal_combo):
                return list(signal_combo)

        return None

    def _create_pattern_insight(
        self,
        pattern_def: dict[str, Any],
        matched_signals: list[DomainSignal],
        domain_states: dict[Domain, DomainState],
    ) -> Insight:
        """Create an insight from a matched pattern."""
        # Determine which domains are involved
        involved_domains = set()
        for signal in matched_signals:
            if signal.value.startswith("tech_"):
                involved_domains.add(Domain.TECH)
            elif signal.value.startswith("fitness_"):
                involved_domains.add(Domain.FITNESS)
            elif signal.value.startswith("finance_"):
                involved_domains.add(Domain.FINANCE)

        # Calculate confidence based on domain states
        confidences = []
        for domain in involved_domains:
            state = domain_states.get(domain)
            if state and state.recent_insights:
                avg_confidence = sum(
                    i.confidence for i in state.recent_insights
                ) / len(state.recent_insights)
                confidences.append(avg_confidence)

        confidence = sum(confidences) / len(confidences) if confidences else 0.7

        # Build the description with reasoning
        actions_str = "\n".join(f"- {a}" for a in pattern_def["actions"])
        signals_str = ", ".join(s.value for s in matched_signals)
        description = (
            f"{pattern_def['description']}\n\n"
            f"**Signals detected:** {signals_str}\n\n"
            f"**Recommended actions:**\n{actions_str}"
        )

        # Collect source insight metadata
        source_insights = []
        for domain in involved_domains:
            state = domain_states.get(domain)
            if state:
                for insight in state.recent_insights[:3]:  # Top 3 per domain
                    source_insights.append({
                        "domain": domain.value,
                        "title": insight.title,
                        "level": insight.level.value,
                    })

        pattern_title = pattern_def["name"].replace("_", " ").title()
        return Insight(
            insight_type=InsightType.CROSS_DOMAIN,
            title=f"Cross-Domain Pattern: {pattern_title}",
            description=description,
            level=pattern_def["level"],
            confidence=confidence,
            relevance_score=0.8,  # Cross-domain patterns are inherently relevant
            entity_names=[d.value for d in involved_domains],
            metadata={
                "pattern_name": pattern_def["name"],
                "pattern_type": pattern_def["type"].value,
                "signals": [s.value for s in matched_signals],
                "domains_involved": [d.value for d in involved_domains],
                "recommended_actions": pattern_def["actions"],
                "source_insights": source_insights,
            },
        )

    def _analyze_correlations(
        self,
        correlations: list[dict[str, Any]],
        domain_states: dict[Domain, DomainState],
    ) -> list[Insight]:
        """Generate insights from cross-domain correlations."""
        insights = []

        for corr in correlations:
            # High agreement across sources suggests important entity
            if corr["agreement_score"] > 0.7 and corr["total_mentions"] >= 5:
                entity_name = corr["entity_name"]
                sources_str = ", ".join(corr["sources"])
                agreement_pct = f"{corr['agreement_score']:.0%}"
                insights.append(
                    Insight(
                        insight_type=InsightType.CROSS_DOMAIN,
                        title=f"Cross-Source Signal: {entity_name}",
                        description=(
                            f"Entity '{entity_name}' is appearing across "
                            f"multiple sources ({sources_str}) with high "
                            f"agreement ({agreement_pct}). This may indicate "
                            f"an emerging trend worth monitoring."
                        ),
                        level=AlertLevel.INFO,
                        confidence=corr["agreement_score"],
                        relevance_score=0.6,
                        entity_names=[entity_name],
                        metadata={
                            "entity_type": corr["entity_type"],
                            "sources": corr["sources"],
                            "total_mentions": corr["total_mentions"],
                        },
                    )
                )

        return insights

    def _analyze_lifestyle_balance(
        self, domain_states: dict[Domain, DomainState]
    ) -> list[Insight]:
        """Generate lifestyle balance insights."""
        insights = []

        # Calculate overall balance score
        risk_levels = [
            state.risk_level for state in domain_states.values()
        ]
        opportunity_levels = [
            state.opportunity_level for state in domain_states.values()
        ]

        avg_risk = sum(risk_levels) / len(risk_levels) if risk_levels else 0
        if opportunity_levels:
            avg_opportunity = sum(opportunity_levels) / len(opportunity_levels)
        else:
            avg_opportunity = 0

        # High opportunity, low risk = good balance
        if avg_opportunity > 0.6 and avg_risk < 0.3:
            insights.append(
                Insight(
                    insight_type=InsightType.CROSS_DOMAIN,
                    title="Excellent Life Balance",
                    description=(
                        "All domains are showing positive signals with low risk. "
                        "This is an optimal time for ambitious goals."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.8,
                    relevance_score=0.9,
                    metadata={
                        "pattern_type": SynthesisInsightType.LIFESTYLE_BALANCE.value,
                        "avg_risk": avg_risk,
                        "avg_opportunity": avg_opportunity,
                    },
                )
            )
        # High risk across multiple domains
        elif avg_risk > 0.6:
            insights.append(
                Insight(
                    insight_type=InsightType.CROSS_DOMAIN,
                    title="Elevated Multi-Domain Risk",
                    description=(
                        "Risk levels are elevated across multiple life dimensions. "
                        "Consider a conservative approach and focus on stability."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=0.7,
                    relevance_score=0.85,
                    metadata={
                        "pattern_type": SynthesisInsightType.LIFESTYLE_BALANCE.value,
                        "avg_risk": avg_risk,
                        "avg_opportunity": avg_opportunity,
                    },
                )
            )

        return insights

    # ========================================================================
    # Project Ideas
    # ========================================================================

    async def generate_project_ideas(
        self, insights: list[Insight]
    ) -> list[ProjectIdea]:
        """
        Generate project ideas that span multiple domains.

        Cross-domain projects leverage insights from tech, fitness, and finance.
        """
        ideas = []

        # Look for patterns that suggest cross-domain project opportunities
        for insight in insights:
            if insight.insight_type != InsightType.CROSS_DOMAIN:
                continue

            metadata = insight.metadata
            signals = metadata.get("signals", [])

            has_tech = any("tech" in s for s in signals)
            has_finance = any("finance" in s for s in signals)
            has_fitness = any("fitness" in s for s in signals)

            # Tech + Finance opportunity → Fintech project
            if has_tech and has_finance:
                if "opportunity" in insight.title.lower():
                    tech_trends = [s for s in signals if "tech" in s]
                    ideas.append(
                        ProjectIdea(
                            title="Personal Finance Dashboard with AI",
                            description=(
                                "Build a dashboard that combines market data "
                                "with ML predictions. Leverages current tech "
                                "trends in AI/ML with finance domain knowledge."
                            ),
                            technologies=[
                                "Python",
                                "TensorFlow",
                                "React",
                                "Financial APIs",
                            ],
                            learning_path=[
                                "Learn financial data APIs",
                                "Study ML for time series prediction",
                                "Build data pipeline for real-time updates",
                                "Create visualization dashboard",
                            ],
                            difficulty="intermediate",
                            estimated_hours=40,
                            rationale=insight.description,
                            source_trends=tech_trends,
                            relevance_score=insight.relevance_score,
                        )
                    )

            # Tech + Fitness → Health tech project
            if has_tech and has_fitness:
                source_trends = [
                    s for s in signals if "tech" in s or "fitness" in s
                ]
                ideas.append(
                    ProjectIdea(
                        title="Training Load Optimizer",
                        description=(
                            "Build an app that analyzes training data and "
                            "suggests optimal workout schedules based on "
                            "recovery and performance metrics."
                        ),
                        technologies=[
                            "Python",
                            "Data Science",
                            "Mobile App",
                            "Wearable APIs",
                        ],
                        learning_path=[
                            "Study sports science metrics (TSS, ATL, CTL)",
                            "Learn wearable device APIs",
                            "Implement optimization algorithms",
                            "Build mobile-friendly interface",
                        ],
                        difficulty="intermediate",
                        estimated_hours=35,
                        rationale=insight.description,
                        source_trends=source_trends,
                        relevance_score=insight.relevance_score,
                    )
                )

        return ideas

    # ========================================================================
    # Report Generation
    # ========================================================================

    def format_report(self, report: AgentReport) -> str:
        """Format the synthesis report for output."""
        lines = [
            f"# {report.agent_name.replace('_', ' ').title()} Report",
            f"**Generated**: {report.run_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            report.summary,
            "",
        ]

        # Group insights by pattern type
        if report.insights:
            lines.append("## Cross-Dimensional Insights")
            lines.append("")

            for insight in report.insights:
                emoji = {
                    AlertLevel.URGENT: "🚨",
                    AlertLevel.ACTION: "⚡",
                    AlertLevel.WATCH: "👀",
                    AlertLevel.INFO: "ℹ️",
                }.get(insight.level, "📊")

                lines.append(f"### {emoji} {insight.title}")
                lines.append(f"**Level**: {insight.level.value.upper()} | "
                           f"**Confidence**: {insight.confidence:.0%}")
                lines.append("")
                lines.append(insight.description)
                lines.append("")

                # Add domains involved
                domains = insight.metadata.get("domains_involved", [])
                if domains:
                    lines.append(f"**Domains**: {', '.join(domains)}")
                    lines.append("")

        # Project ideas
        if report.project_ideas:
            lines.append("## Cross-Domain Project Ideas")
            lines.append("")

            for idea in report.project_ideas:
                lines.append(f"### {idea.title}")
                lines.append(f"**Difficulty**: {idea.difficulty} | "
                           f"**Estimated**: {idea.estimated_hours}h")
                lines.append("")
                lines.append(idea.description)
                lines.append("")
                lines.append("**Learning Path:**")
                for step in idea.learning_path:
                    lines.append(f"1. {step}")
                lines.append("")

        return "\n".join(lines)
