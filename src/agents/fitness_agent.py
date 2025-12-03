"""Fitness Intelligence Agent for training optimization and injury prevention.

Monitors:
- Training load trends (acute vs chronic)
- Recovery indicators (HRV, sleep)
- Performance progression (pace, power)
- Injury risk patterns

Outputs:
- Weekly "Fitness Report"
- Alerts for overtraining risk
- Recovery recommendations
- Goal readiness assessments
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
from src.fitness_repository import ActivityRepository, FitnessMetricsRepository
from src.ingestion.fitness_client import TrainingMetricsCalculator
from src.models import Activity, ActivityMetrics
from src.world_model import Domain

logger = structlog.get_logger()


# ============================================================================
# Fitness-Specific Models
# ============================================================================


class FitnessInsightType(str, Enum):
    """Types of fitness-specific insights."""

    OVERTRAINING_RISK = "overtraining_risk"
    RECOVERY_NEEDED = "recovery_needed"
    PERFORMANCE_GAIN = "performance_gain"
    PERFORMANCE_DECLINE = "performance_decline"
    INJURY_RISK = "injury_risk"
    GOAL_READY = "goal_ready"
    TRAINING_CONSISTENCY = "training_consistency"
    REST_DAY_RECOMMENDED = "rest_day_recommended"
    PEAK_FORM = "peak_form"


class TrainingZone(str, Enum):
    """Training status zones based on TSB."""

    TRANSITION = "transition"  # TSB > 25, risk of detraining
    FRESH = "fresh"  # TSB 5-25, ready to perform
    OPTIMAL = "optimal"  # TSB -10 to 5, ideal for training
    TIRED = "tired"  # TSB -30 to -10, need recovery
    OVERREACHED = "overreached"  # TSB < -30, high fatigue


@dataclass
class TrainingLoadAnalysis:
    """Analysis of current training load status."""

    atl: float  # Acute Training Load
    ctl: float  # Chronic Training Load
    tsb: float  # Training Stress Balance
    atl_ctl_ratio: float  # Acute:Chronic ratio
    training_zone: TrainingZone
    injury_risk_level: str
    form_status: str
    weekly_tss: float
    avg_daily_tss: float
    trend: str  # "building", "maintaining", "deloading"


@dataclass
class PerformanceMetric:
    """Track a specific performance metric over time."""

    name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend: str  # "improving", "declining", "stable"
    unit: str


@dataclass
class RecoveryRecommendation:
    """Actionable recovery recommendation."""

    priority: int  # 1 = highest
    title: str
    description: str
    action: str
    duration: str | None = None
    rationale: str | None = None


@dataclass
class GoalReadiness:
    """Assessment of readiness for a specific goal."""

    goal_name: str
    target_date: datetime | None
    readiness_score: float  # 0-100
    current_fitness: float  # CTL
    recommended_fitness: float
    gap_percent: float
    recommendation: str
    weeks_to_ready: int | None = None


# ============================================================================
# Fitness Intelligence Agent
# ============================================================================


class FitnessIntelligenceAgent(BaseAgent):
    """
    Proactive agent for fitness intelligence.

    Monitors training load, detects overtraining risk,
    provides recovery recommendations, and assesses goal readiness.
    """

    name = "fitness_intelligence"
    domain = Domain.FITNESS

    # Training load thresholds
    ATL_CTL_WARNING_RATIO = 1.3  # 30% above chronic load = warning
    ATL_CTL_DANGER_RATIO = 1.5  # 50% above chronic load = danger
    TSB_OVERREACHED_THRESHOLD = -30
    TSB_TIRED_THRESHOLD = -10
    TSB_FRESH_THRESHOLD = 5
    TSB_TRANSITION_THRESHOLD = 25

    def __init__(
        self,
        db: Database,
        config: AgentConfig | None = None,
        user_profile: UserProfile | None = None,
    ):
        super().__init__(db, config, user_profile)

        # Initialize fitness-specific repositories
        self.activity_repo = ActivityRepository(db)
        self.metrics_repo = FitnessMetricsRepository(db)
        self.calculator = TrainingMetricsCalculator()

        # Fitness-specific thresholds from config
        self.training_load_warning = self.config.thresholds.get("training_load_warning", 1.3)
        self.recovery_warning = self.config.thresholds.get("recovery_warning", 60)
        self.fatigue_warning = self.config.thresholds.get("fatigue_warning", 80)

        logger.info(
            "Initialized FitnessIntelligenceAgent",
            training_load_warning=self.training_load_warning,
            recovery_warning=self.recovery_warning,
        )

    # ========================================================================
    # Monitor Phase
    # ========================================================================

    async def monitor(self) -> dict[str, Any]:
        """
        Monitor fitness data for training load and recovery indicators.

        Gathers:
        - Recent activities (last 7 days)
        - Training load metrics (ATL, CTL, TSB)
        - Performance trends
        - Activity patterns
        """
        logger.info("Monitoring fitness data")

        now = datetime.now(UTC)

        # Get recent activities (last 7 days)
        week_ago = now - timedelta(days=7)
        recent_activities = await self.activity_repo.get_by_date_range(
            start_date=week_ago,
            end_date=now,
        )

        # Get activities for last 6 weeks (for CTL calculation)
        six_weeks_ago = now - timedelta(days=42)
        all_activities = await self.activity_repo.get_by_date_range(
            start_date=six_weeks_ago,
            end_date=now,
        )

        # Get latest metrics
        latest_metrics = await self.metrics_repo.get_latest()

        # Calculate training load if we have activities
        training_load = await self._calculate_training_load(all_activities)

        # Get activity count
        total_activities = await self.activity_repo.count()

        monitoring_data = {
            "recent_activities": recent_activities,
            "all_activities": all_activities,
            "latest_metrics": latest_metrics,
            "training_load": training_load,
            "activities_count": len(recent_activities),
            "total_activities": total_activities,
            "timestamp": now,
        }

        logger.info(
            "Fitness monitoring complete",
            recent_activities=len(recent_activities),
            total_activities=total_activities,
            has_metrics=latest_metrics is not None,
        )

        return monitoring_data

    async def _calculate_training_load(
        self, activities: list[Activity]
    ) -> TrainingLoadAnalysis | None:
        """Calculate comprehensive training load analysis."""
        if not activities:
            return None

        now = datetime.now(UTC)

        # Calculate daily TSS for last 42 days
        daily_tss: dict[str, float] = {}
        for activity in activities:
            day_key = activity.start_date.strftime("%Y-%m-%d")
            tss = activity.tss or self.calculator.estimate_tss(activity)
            daily_tss[day_key] = daily_tss.get(day_key, 0) + tss

        # Fill missing days with 0
        start_date = now - timedelta(days=42)
        filled_tss = []
        current = start_date
        while current <= now:
            day_key = current.strftime("%Y-%m-%d")
            filled_tss.append(daily_tss.get(day_key, 0))
            current += timedelta(days=1)

        # Calculate metrics
        metrics = self.calculator.calculate_training_load(filled_tss)

        # Calculate ATL:CTL ratio
        atl_ctl_ratio = metrics.atl / metrics.ctl if metrics.ctl > 0 else 0

        # Determine training zone based on TSB
        if metrics.tsb > self.TSB_TRANSITION_THRESHOLD:
            zone = TrainingZone.TRANSITION
        elif metrics.tsb > self.TSB_FRESH_THRESHOLD:
            zone = TrainingZone.FRESH
        elif metrics.tsb > self.TSB_TIRED_THRESHOLD:
            zone = TrainingZone.OPTIMAL
        elif metrics.tsb > self.TSB_OVERREACHED_THRESHOLD:
            zone = TrainingZone.TIRED
        else:
            zone = TrainingZone.OVERREACHED

        # Calculate weekly TSS
        week_start = now - timedelta(days=7)
        weekly_tss = sum(
            a.tss or self.calculator.estimate_tss(a)
            for a in activities
            if a.start_date >= week_start
        )

        # Determine trend
        if atl_ctl_ratio > 1.1:
            trend = "building"
        elif atl_ctl_ratio < 0.9:
            trend = "deloading"
        else:
            trend = "maintaining"

        return TrainingLoadAnalysis(
            atl=metrics.atl,
            ctl=metrics.ctl,
            tsb=metrics.tsb,
            atl_ctl_ratio=atl_ctl_ratio,
            training_zone=zone,
            injury_risk_level=metrics.injury_risk,
            form_status=metrics.form_status,
            weekly_tss=weekly_tss,
            avg_daily_tss=weekly_tss / 7,
            trend=trend,
        )

    # ========================================================================
    # Analyze Phase
    # ========================================================================

    async def analyze(self, monitoring_data: dict[str, Any]) -> list[Insight]:
        """
        Analyze fitness data and generate insights.

        Generates insights for:
        - Overtraining risk
        - Recovery needs
        - Performance changes
        - Training consistency
        """
        insights = []

        training_load = monitoring_data.get("training_load")
        recent_activities = monitoring_data.get("recent_activities", [])

        if training_load:
            # 1. Analyze training load and injury risk
            load_insights = self._analyze_training_load(training_load)
            insights.extend(load_insights)

            # 2. Analyze recovery needs
            recovery_insights = self._analyze_recovery_needs(training_load)
            insights.extend(recovery_insights)

        # 3. Analyze training consistency
        consistency_insights = self._analyze_consistency(recent_activities)
        insights.extend(consistency_insights)

        # 4. Analyze performance trends
        if recent_activities:
            performance_insights = await self._analyze_performance(recent_activities)
            insights.extend(performance_insights)

        logger.info(
            "Fitness analysis complete",
            total_insights=len(insights),
        )

        return insights

    def _analyze_training_load(self, load: TrainingLoadAnalysis) -> list[Insight]:
        """Generate insights from training load analysis."""
        insights = []

        # Check for overtraining risk
        if load.atl_ctl_ratio >= self.ATL_CTL_DANGER_RATIO:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="🚨 High Overtraining Risk",
                    description=(
                        f"Your acute training load is {int(load.atl_ctl_ratio * 100)}% of "
                        f"your chronic load (ATL: {load.atl:.0f}, CTL: {load.ctl:.0f}). "
                        f"This significantly increases injury risk."
                    ),
                    level=AlertLevel.URGENT,
                    confidence=0.9,
                    relevance_score=1.0,
                    metadata={
                        "atl": load.atl,
                        "ctl": load.ctl,
                        "ratio": load.atl_ctl_ratio,
                        "fitness_insight_type": FitnessInsightType.OVERTRAINING_RISK.value,
                    },
                )
            )
        elif load.atl_ctl_ratio >= self.ATL_CTL_WARNING_RATIO:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="⚠️ Elevated Training Load",
                    description=(
                        f"Your acute training load is {int(load.atl_ctl_ratio * 100)}% of "
                        f"your chronic load. Consider moderating intensity."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=0.85,
                    relevance_score=0.9,
                    metadata={
                        "atl": load.atl,
                        "ctl": load.ctl,
                        "ratio": load.atl_ctl_ratio,
                        "fitness_insight_type": FitnessInsightType.OVERTRAINING_RISK.value,
                    },
                )
            )

        # Check injury risk
        if load.injury_risk_level == "High":
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="🏥 High Injury Risk Detected",
                    description=(
                        f"Your training pattern indicates high injury risk. "
                        f"ATL:CTL ratio of {load.atl_ctl_ratio:.2f} is in the danger zone. "
                        f"Reduce training volume immediately."
                    ),
                    level=AlertLevel.ACTION,
                    confidence=0.9,
                    relevance_score=1.0,
                    metadata={
                        "injury_risk": load.injury_risk_level,
                        "fitness_insight_type": FitnessInsightType.INJURY_RISK.value,
                    },
                )
            )

        # Check for overreached state
        if load.training_zone == TrainingZone.OVERREACHED:
            insights.append(
                Insight(
                    insight_type=InsightType.ANOMALY,
                    title="😴 Functional Overreaching Detected",
                    description=(
                        f"Your TSB of {load.tsb:.0f} indicates significant fatigue. "
                        f"You're in the overreached zone. Plan a recovery week."
                    ),
                    level=AlertLevel.ACTION,
                    confidence=0.85,
                    relevance_score=0.95,
                    metadata={
                        "tsb": load.tsb,
                        "zone": load.training_zone.value,
                        "fitness_insight_type": FitnessInsightType.RECOVERY_NEEDED.value,
                    },
                )
            )

        # Positive: Check for peak form
        if load.training_zone == TrainingZone.FRESH and load.ctl > 50:
            insights.append(
                Insight(
                    insight_type=InsightType.EMERGING_TECH,  # Reusing for positive insight
                    title="🎯 Peak Form Detected",
                    description=(
                        f"You're fresh with good fitness (CTL: {load.ctl:.0f}, "
                        f"TSB: {load.tsb:.0f}). Great time for a race or hard effort!"
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.8,
                    relevance_score=0.8,
                    metadata={
                        "ctl": load.ctl,
                        "tsb": load.tsb,
                        "fitness_insight_type": FitnessInsightType.PEAK_FORM.value,
                    },
                )
            )

        return insights

    def _analyze_recovery_needs(self, load: TrainingLoadAnalysis) -> list[Insight]:
        """Generate recovery recommendations based on current state."""
        insights = []

        # Check if rest day is recommended
        if load.training_zone in (TrainingZone.TIRED, TrainingZone.OVERREACHED):
            insights.append(
                Insight(
                    insight_type=InsightType.SKILL_GAP,  # Reusing for recommendation
                    title="🛌 Rest Day Recommended",
                    description=(
                        f"Based on your current fatigue level (TSB: {load.tsb:.0f}), "
                        f"today would benefit from rest or very light activity."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=0.8,
                    relevance_score=0.85,
                    metadata={
                        "tsb": load.tsb,
                        "zone": load.training_zone.value,
                        "fitness_insight_type": FitnessInsightType.REST_DAY_RECOMMENDED.value,
                    },
                )
            )

        # Check for detraining risk
        if load.training_zone == TrainingZone.TRANSITION:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title="📉 Fitness May Be Declining",
                    description=(
                        f"Your TSB of {load.tsb:.0f} suggests you're very fresh but may be "
                        f"losing fitness. Consider adding some training stress."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.7,
                    relevance_score=0.6,
                    metadata={
                        "tsb": load.tsb,
                        "fitness_insight_type": FitnessInsightType.PERFORMANCE_DECLINE.value,
                    },
                )
            )

        return insights

    def _analyze_consistency(self, activities: list[Activity]) -> list[Insight]:
        """Analyze training consistency over the week."""
        insights = []

        if not activities:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title="📊 No Recent Activities",
                    description=(
                        "No activities recorded in the last 7 days. "
                        "Consistent training is key to progress."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.9,
                    relevance_score=0.7,
                    metadata={
                        "activity_count": 0,
                        "fitness_insight_type": FitnessInsightType.TRAINING_CONSISTENCY.value,
                    },
                )
            )
            return insights

        # Count training days
        training_days = len(set(a.start_date.date() for a in activities))

        if training_days >= 5:
            insights.append(
                Insight(
                    insight_type=InsightType.EMERGING_TECH,
                    title="✅ Great Training Consistency",
                    description=(
                        f"You trained {training_days} days this week with "
                        f"{len(activities)} activities. Keep it up!"
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.9,
                    relevance_score=0.6,
                    metadata={
                        "training_days": training_days,
                        "activity_count": len(activities),
                        "fitness_insight_type": FitnessInsightType.TRAINING_CONSISTENCY.value,
                    },
                )
            )
        elif training_days <= 2:
            insights.append(
                Insight(
                    insight_type=InsightType.TREND_CHANGE,
                    title="📉 Low Training Frequency",
                    description=(
                        f"Only {training_days} training days this week. "
                        f"Consider increasing frequency for better progress."
                    ),
                    level=AlertLevel.INFO,
                    confidence=0.8,
                    relevance_score=0.5,
                    metadata={
                        "training_days": training_days,
                        "fitness_insight_type": FitnessInsightType.TRAINING_CONSISTENCY.value,
                    },
                )
            )

        return insights

    async def _analyze_performance(self, activities: list[Activity]) -> list[Insight]:
        """Analyze performance trends across activities."""
        insights = []

        # Analyze by activity type
        runs = [a for a in activities if a.activity_type == "Run"]
        rides = [a for a in activities if a.activity_type == "Ride"]

        # Analyze running pace if enough data
        if len(runs) >= 2:
            paces = [
                a.moving_time_seconds / (a.distance_meters / 1000)
                for a in runs
                if a.distance_meters > 0
            ]
            if len(paces) >= 2:
                recent_pace = sum(paces[:2]) / 2
                older_pace = sum(paces[2:4]) / 2 if len(paces) >= 4 else paces[-1]

                if older_pace > 0:
                    pace_change = ((older_pace - recent_pace) / older_pace) * 100

                    if pace_change > 3:  # 3% faster
                        insights.append(
                            Insight(
                                insight_type=InsightType.EMERGING_TECH,
                                title="🏃 Running Pace Improving",
                                description=(
                                    f"Your recent runs are {pace_change:.1f}% faster. "
                                    f"Great progress!"
                                ),
                                level=AlertLevel.INFO,
                                confidence=0.75,
                                relevance_score=0.7,
                                metadata={
                                    "pace_change_percent": pace_change,
                                    "fitness_insight_type": FitnessInsightType.PERFORMANCE_GAIN.value,
                                },
                            )
                        )
                    elif pace_change < -5:  # 5% slower
                        insights.append(
                            Insight(
                                insight_type=InsightType.TREND_CHANGE,
                                title="🏃 Running Pace Declining",
                                description=(
                                    f"Your recent runs are {abs(pace_change):.1f}% slower. "
                                    f"Consider checking recovery or training load."
                                ),
                                level=AlertLevel.WATCH,
                                confidence=0.7,
                                relevance_score=0.7,
                                metadata={
                                    "pace_change_percent": pace_change,
                                    "fitness_insight_type": FitnessInsightType.PERFORMANCE_DECLINE.value,
                                },
                            )
                        )

        return insights

    # ========================================================================
    # Recovery Recommendations
    # ========================================================================

    def generate_recovery_recommendations(
        self, load: TrainingLoadAnalysis
    ) -> list[RecoveryRecommendation]:
        """Generate actionable recovery recommendations."""
        recommendations = []

        if load.training_zone == TrainingZone.OVERREACHED:
            recommendations.append(
                RecoveryRecommendation(
                    priority=1,
                    title="Immediate Recovery Week",
                    description="Your body needs rest to adapt to training stress.",
                    action="Reduce training volume by 50% this week",
                    duration="7 days",
                    rationale=f"TSB of {load.tsb:.0f} indicates significant accumulated fatigue",
                )
            )
            recommendations.append(
                RecoveryRecommendation(
                    priority=2,
                    title="Prioritize Sleep",
                    description="Sleep is crucial for physical recovery.",
                    action="Aim for 8+ hours of sleep per night",
                    duration="7 days",
                    rationale="Sleep deprivation compounds training fatigue",
                )
            )

        elif load.training_zone == TrainingZone.TIRED:
            recommendations.append(
                RecoveryRecommendation(
                    priority=1,
                    title="Active Recovery Day",
                    description="Light activity promotes recovery without adding stress.",
                    action="Easy walk, yoga, or mobility work today",
                    duration="30-45 minutes",
                    rationale=f"TSB of {load.tsb:.0f} suggests moderate fatigue",
                )
            )

        if load.atl_ctl_ratio > self.ATL_CTL_WARNING_RATIO:
            recommendations.append(
                RecoveryRecommendation(
                    priority=1 if load.atl_ctl_ratio > self.ATL_CTL_DANGER_RATIO else 2,
                    title="Reduce Training Intensity",
                    description="High acute load relative to chronic load increases injury risk.",
                    action="Lower intensity to Zone 2 for next 3-4 days",
                    duration="3-4 days",
                    rationale=f"ATL:CTL ratio of {load.atl_ctl_ratio:.2f} is elevated",
                )
            )

        return sorted(recommendations, key=lambda r: r.priority)

    # ========================================================================
    # Goal Readiness Assessment
    # ========================================================================

    def assess_goal_readiness(
        self,
        load: TrainingLoadAnalysis,
        goal_name: str,
        target_ctl: float,
        target_date: datetime | None = None,
    ) -> GoalReadiness:
        """Assess readiness for a specific fitness goal."""
        gap = target_ctl - load.ctl
        gap_percent = (gap / target_ctl) * 100 if target_ctl > 0 else 0

        # Calculate readiness score (0-100)
        if load.ctl >= target_ctl:
            readiness = min(100, 70 + (load.tsb + 10) * 2)  # Boost for freshness
        else:
            readiness = max(0, (load.ctl / target_ctl) * 80)

        # Estimate weeks to ready
        weeks_to_ready = None
        if gap > 0:
            # CTL typically increases ~3-5 per week with consistent training
            weekly_ctl_gain = 4
            weeks_to_ready = int(gap / weekly_ctl_gain)

        # Generate recommendation
        if readiness >= 80:
            if load.tsb > 0:
                recommendation = "You're race ready! Maintain current form."
            else:
                recommendation = "Fitness is good but consider a mini-taper for peak performance."
        elif readiness >= 60:
            recommendation = f"Getting close! {int(gap):.0f} more CTL points needed."
        else:
            recommendation = f"Need more training time. Current CTL: {load.ctl:.0f}, Target: {target_ctl:.0f}."

        return GoalReadiness(
            goal_name=goal_name,
            target_date=target_date,
            readiness_score=readiness,
            current_fitness=load.ctl,
            recommended_fitness=target_ctl,
            gap_percent=gap_percent,
            recommendation=recommendation,
            weeks_to_ready=weeks_to_ready,
        )

    # ========================================================================
    # Report Generation
    # ========================================================================

    def generate_weekly_report(self, report: AgentReport) -> str:
        """Generate a formatted weekly fitness report."""
        lines = [
            "# Weekly Fitness Intelligence Report",
            f"Generated: {report.run_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        # Extract training load from metadata
        training_load = report.metadata.get("training_load")
        if training_load:
            lines.extend([
                "## Training Load Summary",
                f"- **Acute Training Load (ATL)**: {training_load.atl:.0f}",
                f"- **Chronic Training Load (CTL)**: {training_load.ctl:.0f}",
                f"- **Training Stress Balance (TSB)**: {training_load.tsb:.0f}",
                f"- **Form Status**: {training_load.form_status}",
                f"- **Injury Risk**: {training_load.injury_risk_level}",
                f"- **Training Zone**: {training_load.training_zone.value.title()}",
                f"- **Weekly TSS**: {training_load.weekly_tss:.0f}",
                f"- **Training Trend**: {training_load.trend.title()}",
                "",
            ])

        # Urgent/Action items
        urgent_action = [
            i for i in report.insights
            if i.level in (AlertLevel.URGENT, AlertLevel.ACTION)
        ]
        if urgent_action:
            lines.append("## 🚨 Attention Required")
            for insight in urgent_action:
                lines.append(f"- **{insight.title}**: {insight.description}")
            lines.append("")

        # Recovery recommendations
        recommendations = report.metadata.get("recommendations", [])
        if recommendations:
            lines.append("## 💡 Recovery Recommendations")
            for rec in recommendations:
                lines.append(f"### {rec.title}")
                lines.append(f"{rec.description}")
                lines.append(f"- **Action**: {rec.action}")
                if rec.duration:
                    lines.append(f"- **Duration**: {rec.duration}")
                lines.append("")

        # Positive insights
        positive = [
            i for i in report.insights
            if i.level == AlertLevel.INFO
            and "improving" in i.title.lower() or "great" in i.title.lower() or "peak" in i.title.lower()
        ]
        if positive:
            lines.append("## ✅ Positive Notes")
            for insight in positive:
                lines.append(f"- {insight.title}")
            lines.append("")

        # Activity summary
        activities_count = report.metadata.get("activities_count", 0)
        lines.extend([
            "## 📊 Activity Summary",
            f"- Activities this week: {activities_count}",
            f"- Total activities tracked: {report.entities_scanned}",
            "",
        ])

        return "\n".join(lines)

    # ========================================================================
    # Override run() to include fitness-specific data
    # ========================================================================

    async def run(self) -> AgentReport:
        """Execute the fitness agent with additional metadata."""
        start_time = datetime.now(UTC)
        logger.info(f"Starting {self.name} agent run")

        try:
            # Monitor
            monitoring_data = await self.monitor()

            # Analyze
            insights = await self.analyze(monitoring_data)

            # Decide (filter)
            filtered_insights = await self.decide(insights)

            # Generate recovery recommendations
            training_load = monitoring_data.get("training_load")
            recommendations = []
            if training_load:
                recommendations = self.generate_recovery_recommendations(training_load)

            # Calculate execution time
            execution_time = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            report = AgentReport(
                agent_name=self.name,
                domain=self.domain,
                run_at=start_time,
                insights=filtered_insights,
                trends_analyzed=0,
                entities_scanned=monitoring_data.get("total_activities", 0),
                alerts_generated=len([i for i in filtered_insights if i.level != AlertLevel.INFO]),
                execution_time_ms=execution_time,
                metadata={
                    "training_load": training_load,
                    "recommendations": recommendations,
                    "activities_count": monitoring_data.get("activities_count", 0),
                },
            )

            logger.info(
                f"Completed {self.name} agent run",
                insights=len(filtered_insights),
                recommendations=len(recommendations),
                execution_time_ms=execution_time,
            )

            return report

        except Exception as e:
            logger.error(f"Error in {self.name} agent run", error=str(e))
            raise
