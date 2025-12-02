"""Temporal reasoning service for knowledge graph analysis.

Implements:
- Change detection ("X mentioned 3x more this week")
- Trend analysis ("growing interest in Y")
- Anomaly detection ("unusual activity in Z")
- Decay functions (relevance decreases over time)
- "What's new since yesterday?" queries
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.database import Database
from src.world_model.entities import Domain, Entity, EntityType

logger = structlog.get_logger()


# ============================================================================
# Temporal Models
# ============================================================================


class TrendDirection(str, Enum):
    """Direction of a trend."""

    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    NEW = "new"  # First seen in period


class AnomalyType(str, Enum):
    """Types of anomalies detected."""

    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    SILENCE = "silence"  # Extended period of no mentions
    BURST = "burst"  # Multiple spikes in short period


@dataclass
class MentionStats:
    """Statistics about entity mentions over a time period."""

    entity_id: int
    entity_name: str
    period_start: datetime
    period_end: datetime
    mention_count: int
    previous_count: int = 0
    change_ratio: float = 0.0
    change_absolute: int = 0

    @property
    def trend(self) -> TrendDirection:
        """Determine trend direction based on change ratio."""
        if self.previous_count == 0 and self.mention_count > 0:
            return TrendDirection.NEW
        if self.change_ratio >= 0.5:  # 50%+ increase
            return TrendDirection.RISING
        if self.change_ratio <= -0.5:  # 50%+ decrease
            return TrendDirection.FALLING
        return TrendDirection.STABLE


@dataclass
class TrendResult:
    """Result of trend analysis."""

    entity_id: int
    entity_name: str
    entity_type: str
    domain: str
    direction: TrendDirection
    change_ratio: float  # Percentage change
    current_count: int
    previous_count: int
    period_days: int
    confidence: float = 1.0


@dataclass
class Anomaly:
    """An anomaly detected in entity activity."""

    entity_id: int
    entity_name: str
    anomaly_type: AnomalyType
    severity: float  # 0.0-1.0
    detected_at: datetime
    expected_value: float
    actual_value: float
    description: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChangeEvent:
    """A change event for an entity."""

    entity_id: int
    entity_name: str
    change_type: str
    changed_at: datetime
    changed_fields: list[str]
    old_values: dict = field(default_factory=dict)
    new_values: dict = field(default_factory=dict)
    source: str | None = None


@dataclass
class WhatsNewResult:
    """Result of 'what's new since X' query."""

    since: datetime
    new_entities: list[Entity]
    updated_entities: list[tuple[Entity, list[str]]]  # Entity + changed fields
    trending_entities: list[TrendResult]
    anomalies: list[Anomaly]
    total_mentions: int


# ============================================================================
# Decay Functions
# ============================================================================


class DecayFunction:
    """Base class for decay functions."""

    def calculate(self, age_days: float) -> float:
        """Calculate decay factor (0-1) based on age in days."""
        raise NotImplementedError


class ExponentialDecay(DecayFunction):
    """Exponential decay: relevance = e^(-lambda * t)"""

    def __init__(self, half_life_days: float = 7.0):
        # lambda = ln(2) / half_life
        self.decay_rate = math.log(2) / half_life_days

    def calculate(self, age_days: float) -> float:
        if age_days < 0:
            return 1.0
        return math.exp(-self.decay_rate * age_days)


class LinearDecay(DecayFunction):
    """Linear decay: relevance = 1 - (t / max_age)"""

    def __init__(self, max_age_days: float = 30.0):
        self.max_age_days = max_age_days

    def calculate(self, age_days: float) -> float:
        if age_days < 0:
            return 1.0
        if age_days >= self.max_age_days:
            return 0.0
        return 1.0 - (age_days / self.max_age_days)


class StepDecay(DecayFunction):
    """Step decay: full relevance within window, then drops."""

    def __init__(
        self,
        fresh_days: float = 1.0,
        recent_days: float = 7.0,
        fresh_weight: float = 1.0,
        recent_weight: float = 0.5,
        old_weight: float = 0.1,
    ):
        self.fresh_days = fresh_days
        self.recent_days = recent_days
        self.fresh_weight = fresh_weight
        self.recent_weight = recent_weight
        self.old_weight = old_weight

    def calculate(self, age_days: float) -> float:
        if age_days < 0:
            return self.fresh_weight
        if age_days <= self.fresh_days:
            return self.fresh_weight
        if age_days <= self.recent_days:
            return self.recent_weight
        return self.old_weight


# ============================================================================
# Temporal Reasoning Service
# ============================================================================


class TemporalReasoningService:
    """
    Service for temporal analysis of the knowledge graph.

    Provides:
    - Change detection: Track what has changed since a timestamp
    - Trend analysis: Identify rising/falling entities
    - Anomaly detection: Find unusual patterns
    - Decay-weighted queries: Apply time decay to results
    """

    # Thresholds for anomaly detection
    SPIKE_THRESHOLD = 3.0  # 3x standard deviation
    DROP_THRESHOLD = -2.0  # 2x below average
    SILENCE_DAYS = 14  # Days without mention to trigger silence anomaly

    def __init__(
        self,
        db: Database,
        decay_function: DecayFunction | None = None,
    ):
        self.db = db
        self.decay = decay_function or ExponentialDecay(half_life_days=7.0)

    # ========================================================================
    # Change Detection
    # ========================================================================

    async def get_changes_since(
        self,
        since: datetime,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
        limit: int = 100,
    ) -> list[ChangeEvent]:
        """
        Get all entity changes since a timestamp.

        Args:
            since: Timestamp to check changes from
            entity_type: Optional filter by entity type
            domain: Optional filter by domain
            limit: Maximum number of changes to return

        Returns:
            List of ChangeEvent objects
        """
        conditions = ["ev.created_at >= $1"]
        params: list[Any] = [since]

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            ev.entity_id,
            e.name as entity_name,
            ev.change_type,
            ev.created_at as changed_at,
            ev.changed_fields,
            ev.change_source,
            ev.properties as new_values,
            prev.properties as old_values
        FROM entity_versions ev
        JOIN entities e ON e.id = ev.entity_id
        LEFT JOIN entity_versions prev ON prev.entity_id = ev.entity_id
            AND prev.version = ev.version - 1
        WHERE {where_clause}
        ORDER BY ev.created_at DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)

        changes = []
        for row in rows:
            changes.append(
                ChangeEvent(
                    entity_id=row["entity_id"],
                    entity_name=row["entity_name"],
                    change_type=row["change_type"],
                    changed_at=row["changed_at"],
                    changed_fields=row["changed_fields"] or [],
                    old_values=row["old_values"] or {},
                    new_values=row["new_values"] or {},
                    source=row["change_source"],
                )
            )

        logger.debug(
            "Retrieved changes since",
            since=since.isoformat(),
            count=len(changes),
        )
        return changes

    async def get_mention_changes(
        self,
        entity_id: int,
        current_period_days: int = 7,
    ) -> MentionStats:
        """
        Compare mention counts between current and previous period.

        Args:
            entity_id: Entity to analyze
            current_period_days: Length of comparison period

        Returns:
            MentionStats with comparison data
        """
        now = datetime.utcnow()
        period_start = now - timedelta(days=current_period_days)
        prev_start = period_start - timedelta(days=current_period_days)

        # Get entity name
        entity_row = await self.db.fetchrow(
            "SELECT name FROM entities WHERE id = $1", entity_id
        )
        entity_name = entity_row["name"] if entity_row else f"Entity {entity_id}"

        # Current period count
        current_count = await self.db.fetchval(
            """
            SELECT COUNT(*) FROM entity_mentions
            WHERE entity_id = $1 AND mentioned_at >= $2
            """,
            entity_id,
            period_start,
        )

        # Previous period count
        previous_count = await self.db.fetchval(
            """
            SELECT COUNT(*) FROM entity_mentions
            WHERE entity_id = $1 AND mentioned_at >= $2 AND mentioned_at < $3
            """,
            entity_id,
            prev_start,
            period_start,
        )

        # Calculate change
        if previous_count > 0:
            change_ratio = (current_count - previous_count) / previous_count
        elif current_count > 0:
            change_ratio = float("inf")  # New entity
        else:
            change_ratio = 0.0

        return MentionStats(
            entity_id=entity_id,
            entity_name=entity_name,
            period_start=period_start,
            period_end=now,
            mention_count=current_count,
            previous_count=previous_count,
            change_ratio=change_ratio if change_ratio != float("inf") else 1.0,
            change_absolute=current_count - previous_count,
        )

    # ========================================================================
    # Trend Analysis
    # ========================================================================

    async def analyze_trends(
        self,
        period_days: int = 7,
        min_mentions: int = 2,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
        limit: int = 50,
    ) -> list[TrendResult]:
        """
        Analyze mention trends across all entities.

        Args:
            period_days: Length of analysis period
            min_mentions: Minimum mentions to include in results
            entity_type: Optional filter by entity type
            domain: Optional filter by domain
            limit: Maximum results to return

        Returns:
            List of TrendResult objects sorted by change magnitude
        """
        now = datetime.utcnow()
        current_start = now - timedelta(days=period_days)
        prev_start = current_start - timedelta(days=period_days)

        conditions = []
        params: list[Any] = [current_start, prev_start, current_start]

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
        WITH current_counts AS (
            SELECT entity_id, COUNT(*) as current_count
            FROM entity_mentions
            WHERE mentioned_at >= $1
            GROUP BY entity_id
        ),
        previous_counts AS (
            SELECT entity_id, COUNT(*) as previous_count
            FROM entity_mentions
            WHERE mentioned_at >= $2 AND mentioned_at < $3
            GROUP BY entity_id
        )
        SELECT
            e.id as entity_id,
            e.name as entity_name,
            e.entity_type,
            e.domain,
            COALESCE(cc.current_count, 0) as current_count,
            COALESCE(pc.previous_count, 0) as previous_count,
            CASE
                WHEN pc.previous_count > 0 THEN
                    (COALESCE(cc.current_count, 0) - pc.previous_count)::float
                    / pc.previous_count
                WHEN cc.current_count > 0 THEN 1.0
                ELSE 0.0
            END as change_ratio
        FROM entities e
        LEFT JOIN current_counts cc ON cc.entity_id = e.id
        LEFT JOIN previous_counts pc ON pc.entity_id = e.id
        WHERE {where_clause}
          AND (COALESCE(cc.current_count, 0) >= {min_mentions}
               OR COALESCE(pc.previous_count, 0) >= {min_mentions})
        ORDER BY ABS(change_ratio) DESC, current_count DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)

        trends = []
        for row in rows:
            change = row["change_ratio"]
            if row["previous_count"] == 0 and row["current_count"] > 0:
                direction = TrendDirection.NEW
            elif change >= 0.5:
                direction = TrendDirection.RISING
            elif change <= -0.5:
                direction = TrendDirection.FALLING
            else:
                direction = TrendDirection.STABLE

            trends.append(
                TrendResult(
                    entity_id=row["entity_id"],
                    entity_name=row["entity_name"],
                    entity_type=row["entity_type"],
                    domain=row["domain"],
                    direction=direction,
                    change_ratio=change,
                    current_count=row["current_count"],
                    previous_count=row["previous_count"],
                    period_days=period_days,
                )
            )

        logger.info(
            "Analyzed trends",
            period_days=period_days,
            rising=len([t for t in trends if t.direction == TrendDirection.RISING]),
            falling=len([t for t in trends if t.direction == TrendDirection.FALLING]),
        )
        return trends

    async def get_rising_entities(
        self,
        period_days: int = 7,
        min_change_ratio: float = 0.5,
        limit: int = 20,
    ) -> list[TrendResult]:
        """Get entities with rising mention trends."""
        trends = await self.analyze_trends(period_days=period_days, limit=limit * 2)
        rising = [
            t for t in trends
            if t.direction == TrendDirection.RISING
            and t.change_ratio >= min_change_ratio
        ]
        return rising[:limit]

    async def get_falling_entities(
        self,
        period_days: int = 7,
        max_change_ratio: float = -0.5,
        limit: int = 20,
    ) -> list[TrendResult]:
        """Get entities with falling mention trends."""
        trends = await self.analyze_trends(period_days=period_days, limit=limit * 2)
        falling = [
            t for t in trends
            if t.direction == TrendDirection.FALLING
            and t.change_ratio <= max_change_ratio
        ]
        return falling[:limit]

    # ========================================================================
    # Anomaly Detection
    # ========================================================================

    async def detect_anomalies(
        self,
        lookback_days: int = 30,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
        limit: int = 50,
    ) -> list[Anomaly]:
        """
        Detect anomalous patterns in entity mentions.

        Detects:
        - Spikes: Sudden increases in mentions
        - Drops: Sudden decreases in mentions
        - Silence: Extended periods without mentions

        Args:
            lookback_days: Historical period to analyze
            entity_type: Optional filter by entity type
            domain: Optional filter by domain
            limit: Maximum anomalies to return

        Returns:
            List of Anomaly objects
        """
        now = datetime.utcnow()
        lookback_start = now - timedelta(days=lookback_days)

        conditions = []
        params: list[Any] = [lookback_start]

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get daily mention counts per entity
        query = f"""
        WITH daily_counts AS (
            SELECT
                em.entity_id,
                e.name as entity_name,
                DATE_TRUNC('day', em.mentioned_at) as day,
                COUNT(*) as count
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE em.mentioned_at >= $1 AND {where_clause}
            GROUP BY em.entity_id, e.name, DATE_TRUNC('day', em.mentioned_at)
        ),
        entity_stats AS (
            SELECT
                entity_id,
                entity_name,
                AVG(count) as avg_count,
                STDDEV(count) as std_count,
                MAX(day) as last_seen_day
            FROM daily_counts
            GROUP BY entity_id, entity_name
            HAVING COUNT(*) >= 3
        )
        SELECT
            dc.entity_id,
            es.entity_name,
            dc.day,
            dc.count,
            es.avg_count,
            es.std_count,
            es.last_seen_day
        FROM daily_counts dc
        JOIN entity_stats es ON es.entity_id = dc.entity_id
        WHERE es.std_count > 0
          AND ABS(dc.count - es.avg_count) > es.std_count * 2
        ORDER BY ABS(dc.count - es.avg_count) / NULLIF(es.std_count, 0) DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)

        anomalies = []
        for row in rows:
            deviation = (row["count"] - row["avg_count"]) / row["std_count"]

            if deviation >= self.SPIKE_THRESHOLD:
                anomaly_type = AnomalyType.SPIKE
                severity = min(1.0, deviation / 5.0)
                desc = (
                    f"{row['entity_name']} had {row['count']} mentions "
                    f"({deviation:.1f}x std dev above average)"
                )
            elif deviation <= self.DROP_THRESHOLD:
                anomaly_type = AnomalyType.DROP
                severity = min(1.0, abs(deviation) / 5.0)
                desc = (
                    f"{row['entity_name']} dropped to {row['count']} mentions "
                    f"({abs(deviation):.1f}x std dev below average)"
                )
            else:
                continue

            anomalies.append(
                Anomaly(
                    entity_id=row["entity_id"],
                    entity_name=row["entity_name"],
                    anomaly_type=anomaly_type,
                    severity=severity,
                    detected_at=row["day"],
                    expected_value=row["avg_count"],
                    actual_value=row["count"],
                    description=desc,
                    metadata={
                        "std_deviation": deviation,
                        "lookback_days": lookback_days,
                    },
                )
            )

        # Check for silence anomalies
        silence_anomalies = await self._detect_silence_anomalies(
            lookback_start=lookback_start,
            entity_type=entity_type,
            domain=domain,
        )
        anomalies.extend(silence_anomalies)

        spike_count = len([a for a in anomalies if a.anomaly_type == AnomalyType.SPIKE])
        drop_count = len([a for a in anomalies if a.anomaly_type == AnomalyType.DROP])
        silence_count = len(
            [a for a in anomalies if a.anomaly_type == AnomalyType.SILENCE]
        )
        logger.info(
            "Detected anomalies",
            total=len(anomalies),
            spikes=spike_count,
            drops=drop_count,
            silence=silence_count,
        )

        return sorted(anomalies, key=lambda a: a.severity, reverse=True)[:limit]

    async def _detect_silence_anomalies(
        self,
        lookback_start: datetime,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
    ) -> list[Anomaly]:
        """Detect entities that have gone silent."""
        now = datetime.utcnow()
        silence_threshold = now - timedelta(days=self.SILENCE_DAYS)

        conditions = [
            "e.last_seen < $1",  # Not seen recently
            "e.first_seen < $2",  # Was active before lookback
            "e.mention_count >= 5",  # Had significant activity
        ]
        params: list[Any] = [silence_threshold, lookback_start]

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            e.id as entity_id,
            e.name,
            e.last_seen,
            e.mention_count
        FROM entities e
        WHERE {where_clause}
        ORDER BY e.mention_count DESC
        LIMIT 20
        """

        rows = await self.db.fetch(query, *params)

        anomalies = []
        for row in rows:
            days_silent = (now - row["last_seen"]).days
            severity = min(1.0, days_silent / 30.0)

            anomalies.append(
                Anomaly(
                    entity_id=row["entity_id"],
                    entity_name=row["name"],
                    anomaly_type=AnomalyType.SILENCE,
                    severity=severity,
                    detected_at=now,
                    expected_value=row["mention_count"] / 30.0,  # avg per day
                    actual_value=0,
                    description=(
                        f"{row['name']} has been silent for {days_silent} days "
                        f"(previously had {row['mention_count']} mentions)"
                    ),
                    metadata={"days_silent": days_silent},
                )
            )

        return anomalies

    # ========================================================================
    # Decay-Weighted Queries
    # ========================================================================

    def calculate_relevance(
        self,
        last_seen: datetime,
        base_score: float = 1.0,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Calculate time-decayed relevance score.

        Args:
            last_seen: When entity was last mentioned
            base_score: Initial score (e.g., confidence, mention_count)
            reference_time: Reference point (defaults to now)

        Returns:
            Decay-adjusted score
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        age_days = (reference_time - last_seen).total_seconds() / 86400
        decay_factor = self.decay.calculate(age_days)

        return base_score * decay_factor

    async def get_entities_with_decay(
        self,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
        limit: int = 50,
    ) -> list[tuple[Entity, float]]:
        """
        Get entities ranked by decay-weighted relevance.

        Returns:
            List of (Entity, relevance_score) tuples
        """
        conditions = []
        params: list[Any] = []

        if entity_type:
            conditions.append(f"entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities
        WHERE {where_clause}
        ORDER BY last_seen DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit * 2)  # Fetch more to account for reranking

        rows = await self.db.fetch(query, *params)

        now = datetime.utcnow()
        results = []
        for row in rows:
            entity = Entity(
                id=row["id"],
                external_id=row["external_id"],
                name=row["name"],
                entity_type=EntityType(row["entity_type"]),
                domain=Domain(row["domain"]),
                description=row["description"],
                source=row["source"],
                source_url=row["source_url"],
                confidence=row["confidence"],
                mention_count=row["mention_count"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                properties=row["properties"] or {},
            )

            # Calculate relevance with decay
            relevance = self.calculate_relevance(
                last_seen=entity.last_seen,
                base_score=entity.confidence * math.log1p(entity.mention_count),
                reference_time=now,
            )
            results.append((entity, relevance))

        # Sort by relevance and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ========================================================================
    # What's New Query
    # ========================================================================

    async def whats_new(
        self,
        since: datetime,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
        include_trends: bool = True,
        include_anomalies: bool = True,
    ) -> WhatsNewResult:
        """
        Answer "What's new since X?" queries.

        Args:
            since: Timestamp to check from
            entity_type: Optional filter by entity type
            domain: Optional filter by domain
            include_trends: Include trend analysis
            include_anomalies: Include anomaly detection

        Returns:
            WhatsNewResult with comprehensive changes
        """
        # Get new entities
        new_entities = await self._get_new_entities_since(
            since, entity_type, domain
        )

        # Get updated entities
        updated_entities = await self._get_updated_entities_since(
            since, entity_type, domain
        )

        # Get mention count
        total_mentions = await self._count_mentions_since(since)

        # Optional: trends
        trending_entities = []
        if include_trends:
            days_since = max(1, (datetime.utcnow() - since).days)
            trends = await self.analyze_trends(
                period_days=days_since,
                entity_type=entity_type,
                domain=domain,
                limit=10,
            )
            rising_directions = (TrendDirection.RISING, TrendDirection.NEW)
            trending_entities = [
                t for t in trends if t.direction in rising_directions
            ]

        # Optional: anomalies
        anomalies = []
        if include_anomalies:
            days_since = max(7, (datetime.utcnow() - since).days)
            anomalies = await self.detect_anomalies(
                lookback_days=days_since,
                entity_type=entity_type,
                domain=domain,
                limit=10,
            )

        logger.info(
            "What's new query completed",
            since=since.isoformat(),
            new_entities=len(new_entities),
            updated_entities=len(updated_entities),
            total_mentions=total_mentions,
        )

        return WhatsNewResult(
            since=since,
            new_entities=new_entities,
            updated_entities=updated_entities,
            trending_entities=trending_entities,
            anomalies=anomalies,
            total_mentions=total_mentions,
        )

    async def _get_new_entities_since(
        self,
        since: datetime,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
    ) -> list[Entity]:
        """Get entities first seen since timestamp."""
        conditions = ["first_seen >= $1"]
        params: list[Any] = [since]

        if entity_type:
            conditions.append(f"entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT id, external_id, name, entity_type, domain, description,
               source, source_url, confidence, mention_count,
               first_seen, last_seen, created_at, updated_at, properties
        FROM entities
        WHERE {where_clause}
        ORDER BY first_seen DESC
        LIMIT 100
        """

        rows = await self.db.fetch(query, *params)

        return [
            Entity(
                id=row["id"],
                external_id=row["external_id"],
                name=row["name"],
                entity_type=EntityType(row["entity_type"]),
                domain=Domain(row["domain"]),
                description=row["description"],
                source=row["source"],
                source_url=row["source_url"],
                confidence=row["confidence"],
                mention_count=row["mention_count"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                properties=row["properties"] or {},
            )
            for row in rows
        ]

    async def _get_updated_entities_since(
        self,
        since: datetime,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
    ) -> list[tuple[Entity, list[str]]]:
        """Get entities updated since timestamp with changed fields."""
        conditions = [
            "ev.created_at >= $1",
            "ev.change_type = 'updated'",
        ]
        params: list[Any] = [since]

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT DISTINCT ON (e.id)
            e.id, e.external_id, e.name, e.entity_type, e.domain, e.description,
            e.source, e.source_url, e.confidence, e.mention_count,
            e.first_seen, e.last_seen, e.created_at, e.updated_at, e.properties,
            ev.changed_fields
        FROM entities e
        JOIN entity_versions ev ON ev.entity_id = e.id
        WHERE {where_clause}
        ORDER BY e.id, ev.created_at DESC
        LIMIT 100
        """

        rows = await self.db.fetch(query, *params)

        return [
            (
                Entity(
                    id=row["id"],
                    external_id=row["external_id"],
                    name=row["name"],
                    entity_type=EntityType(row["entity_type"]),
                    domain=Domain(row["domain"]),
                    description=row["description"],
                    source=row["source"],
                    source_url=row["source_url"],
                    confidence=row["confidence"],
                    mention_count=row["mention_count"],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    properties=row["properties"] or {},
                ),
                row["changed_fields"] or [],
            )
            for row in rows
        ]

    async def _count_mentions_since(self, since: datetime) -> int:
        """Count total mentions since timestamp."""
        return await self.db.fetchval(
            "SELECT COUNT(*) FROM entity_mentions WHERE mentioned_at >= $1",
            since,
        )
