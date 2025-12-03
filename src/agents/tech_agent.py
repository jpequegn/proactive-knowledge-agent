"""Tech Intelligence Agent for surfacing emerging technologies and learning opportunities.

Monitors:
- New technologies gaining mentions
- Tools with accelerating adoption
- Skills gaps relative to trends
- Project opportunities

Outputs:
- Weekly "Emerging Tech Report"
- Real-time alerts for significant announcements
- Project ideas with learning paths
- Relevance scores based on user profile
"""

import json
from datetime import datetime, timedelta
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
from src.world_model import (
    Anomaly,
    AnomalyType,
    Domain,
    EntityType,
    TrendDirection,
    TrendResult,
)

logger = structlog.get_logger()


# ============================================================================
# Tech-Specific Models
# ============================================================================


TECH_CATEGORIES = [
    "framework",
    "library",
    "language",
    "tool",
    "platform",
    "database",
    "cloud",
    "devops",
    "ai_ml",
    "frontend",
    "backend",
    "mobile",
    "security",
]

# Default focus areas if none configured
DEFAULT_FOCUS_AREAS = [
    "AI/ML frameworks",
    "Developer tools",
    "Cloud infrastructure",
    "Programming languages",
    "Web frameworks",
]


# ============================================================================
# Tech Intelligence Agent
# ============================================================================


class TechIntelligenceAgent(BaseAgent):
    """
    Proactive agent for technology intelligence.

    Surfaces emerging technologies, generates project ideas,
    and identifies skill gaps relative to current trends.
    """

    name = "tech_intelligence"
    domain = Domain.TECH

    def __init__(
        self,
        db: Database,
        config: AgentConfig | None = None,
        user_profile: UserProfile | None = None,
        llm_client: Any | None = None,  # For project idea generation
    ):
        super().__init__(db, config, user_profile)
        self.llm_client = llm_client

        # Tech-specific thresholds
        self.mention_increase_threshold = self.config.thresholds.get("mention_increase", 1.5)
        self.new_tech_min_mentions = int(self.config.thresholds.get("new_tech_mentions", 3))
        self.anomaly_severity_threshold = self.config.thresholds.get("anomaly_severity", 0.6)

        # Focus areas
        self.focus_areas = self.config.focus_areas or DEFAULT_FOCUS_AREAS

        logger.info(
            "Initialized TechIntelligenceAgent",
            mention_threshold=self.mention_increase_threshold,
            new_tech_min=self.new_tech_min_mentions,
            focus_areas=self.focus_areas,
        )

    # ========================================================================
    # Monitor Phase
    # ========================================================================

    async def monitor(self) -> dict[str, Any]:
        """
        Monitor the world model for tech-related changes.

        Gathers:
        - Rising technology trends
        - New technologies (first seen recently)
        - Anomalies in tech mentions
        - Cross-source correlations
        """
        logger.info("Monitoring tech trends")

        # Get trending technologies (last 7 days)
        trends = await self.temporal_service.analyze_trends(
            period_days=7,
            min_mentions=self.new_tech_min_mentions,
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            limit=50,
        )

        # Get rising trends specifically
        rising_trends = [
            t for t in trends
            if t.direction in (TrendDirection.RISING, TrendDirection.NEW)
        ]

        # Get what's new in the last 24 hours
        since_yesterday = datetime.utcnow() - timedelta(days=1)
        whats_new = await self.temporal_service.whats_new(
            since=since_yesterday,
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            include_trends=False,  # We already have trends
            include_anomalies=True,
        )

        # Get anomalies
        anomalies = await self.temporal_service.detect_anomalies(
            lookback_days=30,
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
            limit=20,
        )

        # Filter significant anomalies
        significant_anomalies = [
            a for a in anomalies
            if a.severity >= self.anomaly_severity_threshold
        ]

        # Get cross-source correlations for tech entities
        correlations = await self.correlation_service.find_correlations(
            entity_type=EntityType.TECHNOLOGY,
            min_significance=0.7,
            limit=20,
        )

        # Count entities scanned
        entity_count = await self.entity_repo.count(
            entity_type=EntityType.TECHNOLOGY,
            domain=Domain.TECH,
        )

        monitoring_data = {
            "trends": trends,
            "rising_trends": rising_trends,
            "new_entities": whats_new.new_entities,
            "anomalies": significant_anomalies,
            "correlations": correlations,
            "trends_count": len(trends),
            "entities_count": entity_count,
            "timestamp": datetime.utcnow(),
        }

        logger.info(
            "Tech monitoring complete",
            total_trends=len(trends),
            rising_trends=len(rising_trends),
            new_entities=len(whats_new.new_entities),
            anomalies=len(significant_anomalies),
            correlations=len(correlations),
        )

        return monitoring_data

    # ========================================================================
    # Analyze Phase
    # ========================================================================

    async def analyze(self, monitoring_data: dict[str, Any]) -> list[Insight]:
        """
        Analyze monitoring data and generate tech insights.

        Generates insights for:
        - Emerging technologies (rising trends)
        - New frameworks/tools (first seen)
        - Adoption spikes (anomalies)
        - Skill gaps (known tech falling, new tech rising)
        """
        insights = []

        # 1. Analyze rising trends
        rising_insights = await self._analyze_rising_trends(
            monitoring_data.get("rising_trends", [])
        )
        insights.extend(rising_insights)

        # 2. Analyze new entities
        new_entity_insights = await self._analyze_new_entities(
            monitoring_data.get("new_entities", [])
        )
        insights.extend(new_entity_insights)

        # 3. Analyze anomalies
        anomaly_insights = await self._analyze_anomalies(
            monitoring_data.get("anomalies", [])
        )
        insights.extend(anomaly_insights)

        # 4. Analyze skill gaps
        skill_gap_insights = await self._analyze_skill_gaps(
            monitoring_data.get("trends", [])
        )
        insights.extend(skill_gap_insights)

        # 5. Analyze correlations for cross-domain insights
        correlation_insights = await self._analyze_correlations(
            monitoring_data.get("correlations", [])
        )
        insights.extend(correlation_insights)

        logger.info(
            "Tech analysis complete",
            total_insights=len(insights),
            rising=len(rising_insights),
            new_entities=len(new_entity_insights),
            anomalies=len(anomaly_insights),
            skill_gaps=len(skill_gap_insights),
            correlations=len(correlation_insights),
        )

        return insights

    async def _analyze_rising_trends(self, trends: list[TrendResult]) -> list[Insight]:
        """Generate insights for rising technology trends."""
        insights = []

        for trend in trends:
            # Skip if below threshold
            if trend.change_ratio < self.mention_increase_threshold - 1:
                continue

            # Determine alert level based on change magnitude
            level = self._trend_to_alert_level(trend)

            # Determine insight type
            if trend.direction == TrendDirection.NEW:
                insight_type = InsightType.NEW_FRAMEWORK
                title = f"New technology detected: {trend.entity_name}"
                description = (
                    f"{trend.entity_name} is a newly detected technology with "
                    f"{trend.current_count} mentions in the last {trend.period_days} days."
                )
            else:
                insight_type = InsightType.EMERGING_TECH
                change_pct = int(trend.change_ratio * 100)
                title = f"Rising trend: {trend.entity_name} (+{change_pct}%)"
                description = (
                    f"{trend.entity_name} mentions increased by {change_pct}% "
                    f"({trend.previous_count} → {trend.current_count}) "
                    f"over the last {trend.period_days} days."
                )

            # Calculate confidence based on mention count
            confidence = min(1.0, trend.current_count / 20)

            insights.append(
                Insight(
                    insight_type=insight_type,
                    title=title,
                    description=description,
                    level=level,
                    confidence=confidence,
                    relevance_score=0.5,  # Will be updated in decide()
                    entity_ids=[trend.entity_id],
                    entity_names=[trend.entity_name],
                    metadata={
                        "change_ratio": trend.change_ratio,
                        "current_count": trend.current_count,
                        "previous_count": trend.previous_count,
                        "period_days": trend.period_days,
                        "direction": trend.direction.value,
                    },
                )
            )

        return insights

    async def _analyze_new_entities(self, entities: list) -> list[Insight]:
        """Generate insights for newly discovered technologies."""
        insights = []

        for entity in entities:
            # Only process technology entities
            if entity.entity_type != EntityType.TECHNOLOGY:
                continue

            # Check if it has enough mentions to be significant
            if entity.mention_count < self.new_tech_min_mentions:
                continue

            insights.append(
                Insight(
                    insight_type=InsightType.NEW_FRAMEWORK,
                    title=f"New technology: {entity.name}",
                    description=(
                        f"First detected: {entity.first_seen.strftime('%Y-%m-%d')}. "
                        f"{entity.description or 'No description available.'}"
                    ),
                    level=AlertLevel.INFO,
                    confidence=entity.confidence,
                    relevance_score=0.5,
                    entity_ids=[entity.id] if entity.id else [],
                    entity_names=[entity.name],
                    metadata={
                        "first_seen": entity.first_seen.isoformat(),
                        "mention_count": entity.mention_count,
                        "source": entity.source,
                        "properties": entity.properties,
                    },
                )
            )

        return insights

    async def _analyze_anomalies(self, anomalies: list[Anomaly]) -> list[Insight]:
        """Generate insights from anomaly detection."""
        insights = []

        for anomaly in anomalies:
            # Only care about spikes for tech (drops might indicate obsolescence)
            if anomaly.anomaly_type == AnomalyType.SPIKE:
                insight_type = InsightType.ADOPTION_SPIKE
                title = f"Adoption spike: {anomaly.entity_name}"
                level = AlertLevel.ACTION if anomaly.severity >= 0.8 else AlertLevel.WATCH
            elif anomaly.anomaly_type == AnomalyType.BURST:
                insight_type = InsightType.ANOMALY
                title = f"Unusual activity: {anomaly.entity_name}"
                level = AlertLevel.WATCH
            else:
                # Skip drops and silence for now
                continue

            insights.append(
                Insight(
                    insight_type=insight_type,
                    title=title,
                    description=anomaly.description,
                    level=level,
                    confidence=anomaly.severity,
                    relevance_score=0.5,
                    entity_ids=[anomaly.entity_id],
                    entity_names=[anomaly.entity_name],
                    metadata={
                        "anomaly_type": anomaly.anomaly_type.value,
                        "severity": anomaly.severity,
                        "expected_value": anomaly.expected_value,
                        "actual_value": anomaly.actual_value,
                        "detected_at": anomaly.detected_at.isoformat(),
                    },
                )
            )

        return insights

    async def _analyze_skill_gaps(self, trends: list[TrendResult]) -> list[Insight]:
        """Identify skill gaps based on user profile and trends."""
        if not self.user_profile or not self.user_profile.known_technologies:
            return []

        insights = []
        known_lower = [t.lower() for t in self.user_profile.known_technologies]

        for trend in trends:
            # Only look at rising trends
            if trend.direction not in (TrendDirection.RISING, TrendDirection.NEW):
                continue

            # Check if this is NOT in user's known technologies
            tech_name_lower = trend.entity_name.lower()
            is_known = any(
                known in tech_name_lower or tech_name_lower in known
                for known in known_lower
            )

            if not is_known and trend.change_ratio >= 0.5:
                # Potential skill gap
                change_pct = int(trend.change_ratio * 100)
                insights.append(
                    Insight(
                        insight_type=InsightType.SKILL_GAP,
                        title=f"Skill gap opportunity: {trend.entity_name}",
                        description=(
                            f"{trend.entity_name} is trending (+{change_pct}%) "
                            f"but not in your known technologies. "
                            f"Consider exploring this technology."
                        ),
                        level=AlertLevel.INFO,
                        confidence=min(1.0, trend.current_count / 15),
                        relevance_score=0.7,  # Higher relevance for skill gaps
                        entity_ids=[trend.entity_id],
                        entity_names=[trend.entity_name],
                        metadata={
                            "change_ratio": trend.change_ratio,
                            "current_count": trend.current_count,
                            "is_learning_goal": any(
                                goal.lower() in tech_name_lower
                                for goal in self.user_profile.learning_goals
                            ),
                        },
                    )
                )

        return insights

    async def _analyze_correlations(self, correlations: list) -> list[Insight]:
        """Generate insights from cross-source correlations."""
        insights = []

        for corr in correlations:
            # Look for significant correlations
            if corr.significance < 0.7:
                continue

            insights.append(
                Insight(
                    insight_type=InsightType.CORRELATION,
                    title=f"Cross-source signal: {corr.entity_name}",
                    description=(
                        f"{corr.entity_name} is being discussed across multiple sources "
                        f"({corr.source_count} sources) with high correlation. "
                        f"This may indicate growing industry interest."
                    ),
                    level=AlertLevel.WATCH,
                    confidence=corr.significance,
                    relevance_score=0.6,
                    entity_ids=[corr.entity_id] if hasattr(corr, "entity_id") else [],
                    entity_names=[corr.entity_name] if hasattr(corr, "entity_name") else [],
                    metadata={
                        "source_count": getattr(corr, "source_count", 0),
                        "correlation_type": getattr(corr, "correlation_type", "unknown"),
                        "significance": corr.significance,
                    },
                )
            )

        return insights

    # ========================================================================
    # Project Idea Generation
    # ========================================================================

    async def generate_project_ideas(self, insights: list[Insight]) -> list[ProjectIdea]:
        """
        Generate project ideas based on tech insights.

        Uses LLM if available, otherwise generates template-based ideas.
        """
        if not insights:
            return []

        # Get top emerging technologies
        emerging_techs = [
            i for i in insights
            if i.insight_type in (InsightType.EMERGING_TECH, InsightType.NEW_FRAMEWORK)
            and i.confidence >= 0.5
        ][:5]

        if not emerging_techs:
            return []

        # Use LLM for rich project ideas if available
        if self.llm_client:
            return await self._generate_llm_project_ideas(emerging_techs)

        # Otherwise, generate template-based ideas
        return self._generate_template_project_ideas(emerging_techs)

    async def _generate_llm_project_ideas(
        self, insights: list[Insight]
    ) -> list[ProjectIdea]:
        """Generate project ideas using LLM."""
        tech_names = [i.entity_names[0] for i in insights if i.entity_names]

        prompt = f"""Based on these emerging technologies: {', '.join(tech_names)}

Generate 3 practical project ideas that would help a developer learn these technologies.
For each project, provide:
1. Title (concise, descriptive)
2. Description (2-3 sentences)
3. Technologies used (from the list above)
4. Learning path (3-5 steps)
5. Difficulty (beginner/intermediate/advanced)
6. Estimated hours to complete
7. Why this project is valuable right now

Format as JSON array."""

        try:
            response = await self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response
            content = response.content[0].text
            # Extract JSON from response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                ideas_data = json.loads(content[json_start:json_end])

                return [
                    ProjectIdea(
                        title=idea.get("title", "Untitled Project"),
                        description=idea.get("description", ""),
                        technologies=idea.get("technologies", tech_names[:2]),
                        learning_path=idea.get("learning_path", []),
                        difficulty=idea.get("difficulty", "intermediate"),
                        estimated_hours=idea.get("estimated_hours", 20),
                        rationale=idea.get("why", "Trending technology"),
                        source_trends=tech_names,
                    )
                    for idea in ideas_data
                ]
        except Exception as e:
            logger.warning("Failed to generate LLM project ideas", error=str(e))

        # Fall back to templates
        return self._generate_template_project_ideas(insights)

    def _generate_template_project_ideas(
        self, insights: list[Insight]
    ) -> list[ProjectIdea]:
        """Generate template-based project ideas."""
        ideas = []

        for insight in insights[:3]:
            if not insight.entity_names:
                continue

            tech_name = insight.entity_names[0]
            ideas.append(
                ProjectIdea(
                    title=f"Build a CLI tool with {tech_name}",
                    description=(
                        f"Create a command-line application using {tech_name} "
                        f"to solve a common developer workflow problem."
                    ),
                    technologies=[tech_name],
                    learning_path=[
                        f"Read {tech_name} documentation",
                        "Set up development environment",
                        "Build core functionality",
                        "Add tests",
                        "Polish and document",
                    ],
                    difficulty="intermediate",
                    estimated_hours=15,
                    rationale=(
                        f"{tech_name} is trending with a "
                        f"{int(insight.metadata.get('change_ratio', 0) * 100)}% increase in mentions."
                    ),
                    source_trends=[tech_name],
                    relevance_score=insight.relevance_score,
                )
            )

        return ideas

    # ========================================================================
    # Report Generation
    # ========================================================================

    def generate_weekly_report(self, report: AgentReport) -> str:
        """Generate a formatted weekly tech report."""
        lines = [
            "# Weekly Tech Intelligence Report",
            f"Generated: {report.run_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            f"- Analyzed {report.trends_analyzed} trends across {report.entities_scanned} entities",
            f"- Generated {len(report.insights)} insights ({report.alerts_generated} alerts)",
            f"- Identified {len(report.project_ideas)} project ideas",
            "",
        ]

        # Urgent/Action items first
        urgent_action = [
            i for i in report.insights
            if i.level in (AlertLevel.URGENT, AlertLevel.ACTION)
        ]
        if urgent_action:
            lines.append("## 🚨 Action Required")
            for insight in urgent_action:
                lines.append(f"- **{insight.title}**: {insight.description}")
            lines.append("")

        # Emerging technologies
        emerging = [
            i for i in report.insights
            if i.insight_type == InsightType.EMERGING_TECH
        ]
        if emerging:
            lines.append("## 📈 Emerging Technologies")
            for insight in emerging[:5]:
                change = insight.metadata.get("change_ratio", 0)
                lines.append(
                    f"- **{insight.entity_names[0]}**: +{int(change * 100)}% "
                    f"({insight.metadata.get('current_count', 0)} mentions)"
                )
            lines.append("")

        # New frameworks
        new_tech = [
            i for i in report.insights
            if i.insight_type == InsightType.NEW_FRAMEWORK
        ]
        if new_tech:
            lines.append("## 🆕 New Technologies")
            for insight in new_tech[:5]:
                lines.append(f"- **{insight.title}**")
            lines.append("")

        # Skill gaps
        skill_gaps = [
            i for i in report.insights
            if i.insight_type == InsightType.SKILL_GAP
        ]
        if skill_gaps:
            lines.append("## 🎯 Learning Opportunities")
            for insight in skill_gaps[:3]:
                lines.append(f"- {insight.title}")
            lines.append("")

        # Project ideas
        if report.project_ideas:
            lines.append("## 💡 Project Ideas")
            for idea in report.project_ideas[:3]:
                lines.append(f"### {idea.title}")
                lines.append(f"{idea.description}")
                lines.append(f"- **Technologies**: {', '.join(idea.technologies)}")
                lines.append(f"- **Difficulty**: {idea.difficulty}")
                lines.append(f"- **Time**: ~{idea.estimated_hours} hours")
                lines.append("")

        return "\n".join(lines)
