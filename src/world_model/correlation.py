"""Cross-source correlation detection and analysis.

Implements:
- Cross-source entity linking (same entity in podcast AND news)
- Correlation detection algorithms (Pearson, Spearman)
- Significance scoring with p-values
- Query API for correlations
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
# Correlation Types and Models
# ============================================================================


class CorrelationType(str, Enum):
    """Types of correlations that can be detected."""

    # Mention-based correlations
    MENTION_FREQUENCY = "mention_frequency"  # Co-mention frequency
    MENTION_TIMING = "mention_timing"  # Temporal co-occurrence
    MENTION_SENTIMENT = "mention_sentiment"  # Sentiment correlation

    # Cross-domain correlations
    TECH_MARKET = "tech_market"  # Tech mentions vs market movement
    FITNESS_SLEEP = "fitness_sleep"  # Training load vs sleep
    NEWS_SOCIAL = "news_social"  # News vs social mentions

    # Source-based correlations
    CROSS_SOURCE = "cross_source"  # Same entity across sources
    SOURCE_AGREEMENT = "source_agreement"  # Sources agreeing on topic


class CorrelationStrength(str, Enum):
    """Qualitative strength of a correlation."""

    VERY_STRONG = "very_strong"  # |r| >= 0.8
    STRONG = "strong"  # 0.6 <= |r| < 0.8
    MODERATE = "moderate"  # 0.4 <= |r| < 0.6
    WEAK = "weak"  # 0.2 <= |r| < 0.4
    NEGLIGIBLE = "negligible"  # |r| < 0.2


@dataclass
class Correlation:
    """A correlation between two entities or data series."""

    id: int | None = None
    source_entity_id: int | None = None
    target_entity_id: int | None = None

    # Correlation metrics
    correlation_type: CorrelationType = CorrelationType.MENTION_FREQUENCY
    correlation_value: float = 0.0  # -1 to 1 (Pearson/Spearman coefficient)
    significance: float = 1.0  # p-value (lower = more significant)

    # Time window
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_end: datetime = field(default_factory=datetime.utcnow)
    window_size_days: int = 7

    # Metadata
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Additional data
    sample_size: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def strength(self) -> CorrelationStrength:
        """Classify correlation strength based on coefficient."""
        abs_val = abs(self.correlation_value)
        if abs_val >= 0.8:
            return CorrelationStrength.VERY_STRONG
        if abs_val >= 0.6:
            return CorrelationStrength.STRONG
        if abs_val >= 0.4:
            return CorrelationStrength.MODERATE
        if abs_val >= 0.2:
            return CorrelationStrength.WEAK
        return CorrelationStrength.NEGLIGIBLE

    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant (p < 0.05)."""
        return self.significance < 0.05

    @property
    def direction(self) -> str:
        """Get correlation direction."""
        if self.correlation_value > 0:
            return "positive"
        if self.correlation_value < 0:
            return "negative"
        return "none"


@dataclass
class CrossSourceLink:
    """A link between the same entity mentioned across different sources."""

    entity_id: int
    entity_name: str
    entity_type: EntityType
    sources: list[str]  # List of source types (article, podcast, etc.)
    mention_counts: dict[str, int]  # Count per source
    first_seen_per_source: dict[str, datetime]
    total_mentions: int
    source_agreement_score: float  # How consistently mentioned across sources


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""

    correlations: list[Correlation]
    cross_source_links: list[CrossSourceLink]
    analysis_window: tuple[datetime, datetime]
    total_entities_analyzed: int
    significant_correlations: int
    summary: str


# ============================================================================
# Statistical Functions
# ============================================================================


def pearson_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Calculate Pearson correlation coefficient and p-value.

    Args:
        x: First data series
        y: Second data series

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(x) != len(y) or len(x) < 3:
        return 0.0, 1.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Calculate covariance and standard deviations
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

    if std_x == 0 or std_y == 0:
        return 0.0, 1.0

    r = cov / (std_x * std_y)

    # Calculate t-statistic and approximate p-value
    if abs(r) >= 1.0:
        return r, 0.0

    t_stat = r * math.sqrt((n - 2) / (1 - r**2))
    # Approximate p-value using normal distribution for large n
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    return r, p_value


def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient and p-value.

    Args:
        x: First data series
        y: Second data series

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(x) != len(y) or len(x) < 3:
        return 0.0, 1.0

    # Convert to ranks
    x_ranks = _rank_data(x)
    y_ranks = _rank_data(y)

    # Use Pearson on ranks
    return pearson_correlation(x_ranks, y_ranks)


def _rank_data(data: list[float]) -> list[float]:
    """Convert data to ranks (1-based, handling ties with average rank)."""
    n = len(data)
    sorted_indices = sorted(range(n), key=lambda i: data[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find all elements with the same value (ties)
        while j < n - 1 and data[sorted_indices[j]] == data[sorted_indices[j + 1]]:
            j += 1
        # Assign average rank to all ties
        avg_rank = (i + j + 2) / 2  # +2 because ranks are 1-based
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1

    return ranks


def _normal_cdf(x: float) -> float:
    """Approximate cumulative distribution function for standard normal."""
    # Using error function approximation
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ============================================================================
# Correlation Repository
# ============================================================================


class CorrelationRepository:
    """Repository for correlation CRUD operations."""

    def __init__(self, db: Database):
        self.db = db

    async def create(self, correlation: Correlation) -> int:
        """Create a new correlation record. Returns the correlation ID."""
        query = """
        INSERT INTO correlations (
            source_entity_id, target_entity_id, correlation_type,
            correlation_value, significance,
            window_start, window_end, window_size_days, description
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """
        async with self.db.acquire() as conn:
            corr_id = await conn.fetchval(
                query,
                correlation.source_entity_id,
                correlation.target_entity_id,
                correlation.correlation_type.value,
                correlation.correlation_value,
                correlation.significance,
                correlation.window_start,
                correlation.window_end,
                correlation.window_size_days,
                correlation.description,
            )
            logger.debug(
                "Correlation created",
                id=corr_id,
                type=correlation.correlation_type.value,
                value=correlation.correlation_value,
            )
            return corr_id

    async def upsert(self, correlation: Correlation) -> tuple[int, bool]:
        """
        Insert or update a correlation.
        Returns (correlation_id, is_new).
        """
        query = """
        INSERT INTO correlations (
            source_entity_id, target_entity_id, correlation_type,
            correlation_value, significance,
            window_start, window_end, window_size_days, description
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (source_entity_id, target_entity_id, correlation_type, window_start)
        DO UPDATE SET
            correlation_value = EXCLUDED.correlation_value,
            significance = EXCLUDED.significance,
            window_end = EXCLUDED.window_end,
            description = EXCLUDED.description
        RETURNING id, (xmax = 0) as is_new
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                query,
                correlation.source_entity_id,
                correlation.target_entity_id,
                correlation.correlation_type.value,
                correlation.correlation_value,
                correlation.significance,
                correlation.window_start,
                correlation.window_end,
                correlation.window_size_days,
                correlation.description,
            )
            return row["id"], row["is_new"]

    async def get_by_id(self, correlation_id: int) -> Correlation | None:
        """Get correlation by ID."""
        query = """
        SELECT id, source_entity_id, target_entity_id, correlation_type,
               correlation_value, significance,
               window_start, window_end, window_size_days,
               description, created_at
        FROM correlations WHERE id = $1
        """
        row = await self.db.fetchrow(query, correlation_id)
        if row is None:
            return None
        return self._row_to_correlation(row)

    async def get_for_entity(
        self,
        entity_id: int,
        correlation_type: CorrelationType | None = None,
        min_significance: float = 0.05,
        limit: int = 50,
    ) -> list[Correlation]:
        """Get correlations involving an entity."""
        conditions = ["(source_entity_id = $1 OR target_entity_id = $1)"]
        params: list[Any] = [entity_id]

        if correlation_type:
            conditions.append(f"correlation_type = ${len(params) + 1}")
            params.append(correlation_type.value)

        conditions.append(f"significance <= ${len(params) + 1}")
        params.append(min_significance)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, source_entity_id, target_entity_id, correlation_type,
               correlation_value, significance,
               window_start, window_end, window_size_days,
               description, created_at
        FROM correlations
        WHERE {where_clause}
        ORDER BY ABS(correlation_value) DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_correlation(row) for row in rows]

    async def get_significant(
        self,
        correlation_type: CorrelationType | None = None,
        min_strength: float = 0.4,
        max_p_value: float = 0.05,
        window_days: int | None = None,
        limit: int = 100,
    ) -> list[Correlation]:
        """Get statistically significant correlations."""
        conditions = [
            "significance <= $1",
            "ABS(correlation_value) >= $2",
        ]
        params: list[Any] = [max_p_value, min_strength]

        if correlation_type:
            conditions.append(f"correlation_type = ${len(params) + 1}")
            params.append(correlation_type.value)

        if window_days:
            conditions.append(f"window_size_days = ${len(params) + 1}")
            params.append(window_days)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, source_entity_id, target_entity_id, correlation_type,
               correlation_value, significance,
               window_start, window_end, window_size_days,
               description, created_at
        FROM correlations
        WHERE {where_clause}
        ORDER BY ABS(correlation_value) DESC, significance ASC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_correlation(row) for row in rows]

    async def get_between_entities(
        self,
        entity_id_1: int,
        entity_id_2: int,
    ) -> list[Correlation]:
        """Get all correlations between two specific entities."""
        query = """
        SELECT id, source_entity_id, target_entity_id, correlation_type,
               correlation_value, significance,
               window_start, window_end, window_size_days,
               description, created_at
        FROM correlations
        WHERE (source_entity_id = $1 AND target_entity_id = $2)
           OR (source_entity_id = $2 AND target_entity_id = $1)
        ORDER BY window_start DESC
        """
        rows = await self.db.fetch(query, entity_id_1, entity_id_2)
        return [self._row_to_correlation(row) for row in rows]

    async def delete_old(self, older_than_days: int = 90) -> int:
        """Delete correlations older than specified days. Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        result = await self.db.execute(
            "DELETE FROM correlations WHERE window_end < $1",
            cutoff,
        )
        # Extract count from result string like "DELETE 5"
        count = int(result.split()[-1]) if result else 0
        logger.info(
            "Deleted old correlations", count=count, older_than_days=older_than_days
        )
        return count

    def _row_to_correlation(self, row) -> Correlation:
        """Convert database row to Correlation model."""
        return Correlation(
            id=row["id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            correlation_type=CorrelationType(row["correlation_type"]),
            correlation_value=row["correlation_value"],
            significance=row["significance"],
            window_start=row["window_start"],
            window_end=row["window_end"],
            window_size_days=row["window_size_days"],
            description=row["description"],
            created_at=row["created_at"],
        )


# ============================================================================
# Correlation Service
# ============================================================================


class CorrelationService:
    """
    Service for detecting and analyzing cross-source correlations.

    Provides:
    - Cross-source entity linking
    - Mention frequency correlation
    - Statistical significance testing
    - Correlation querying and reporting
    """

    # Minimum sample size for reliable correlation
    MIN_SAMPLE_SIZE = 5

    # P-value threshold for significance
    SIGNIFICANCE_THRESHOLD = 0.05

    def __init__(self, db: Database):
        self.db = db
        self.repo = CorrelationRepository(db)

    # ========================================================================
    # Cross-Source Entity Linking
    # ========================================================================

    async def find_cross_source_entities(
        self,
        min_sources: int = 2,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
        limit: int = 100,
    ) -> list[CrossSourceLink]:
        """
        Find entities mentioned across multiple sources.

        Args:
            min_sources: Minimum number of different sources
            entity_type: Optional filter by entity type
            domain: Optional filter by domain
            limit: Maximum results

        Returns:
            List of CrossSourceLink objects
        """
        conditions = []
        params: list[Any] = []

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
        WITH source_counts AS (
            SELECT
                em.entity_id,
                e.name,
                e.entity_type,
                COUNT(DISTINCT em.content_type) as source_count,
                COUNT(*) as total_mentions,
                ARRAY_AGG(DISTINCT em.content_type) as sources,
                JSON_OBJECT_AGG(
                    em.content_type,
                    (SELECT COUNT(*) FROM entity_mentions
                     WHERE entity_id = em.entity_id
                     AND content_type = em.content_type)
                ) as mention_counts,
                JSON_OBJECT_AGG(
                    em.content_type,
                    (SELECT MIN(mentioned_at) FROM entity_mentions
                     WHERE entity_id = em.entity_id
                     AND content_type = em.content_type)
                ) as first_seen
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE {where_clause}
            GROUP BY em.entity_id, e.name, e.entity_type
        )
        SELECT * FROM source_counts
        WHERE source_count >= ${len(params) + 1}
        ORDER BY source_count DESC, total_mentions DESC
        LIMIT ${len(params) + 2}
        """
        params.extend([min_sources, limit])

        rows = await self.db.fetch(query, *params)

        links = []
        for row in rows:
            # Calculate source agreement score
            mention_counts = row["mention_counts"] or {}
            if mention_counts:
                values = list(mention_counts.values())
                mean_count = sum(values) / len(values)
                if mean_count > 0:
                    # Coefficient of variation inverse (higher = more agreement)
                    std_dev = math.sqrt(
                        sum((v - mean_count) ** 2 for v in values) / len(values)
                    )
                    agreement = 1 - min(1, std_dev / mean_count)
                else:
                    agreement = 0.0
            else:
                agreement = 0.0

            # Convert first_seen JSON to datetime dict
            first_seen_data = row["first_seen"] or {}
            first_seen_per_source = {}
            for source, ts in first_seen_data.items():
                if isinstance(ts, str):
                    first_seen_per_source[source] = datetime.fromisoformat(
                        ts.replace("Z", "+00:00")
                    )
                elif isinstance(ts, datetime):
                    first_seen_per_source[source] = ts

            links.append(
                CrossSourceLink(
                    entity_id=row["entity_id"],
                    entity_name=row["name"],
                    entity_type=EntityType(row["entity_type"]),
                    sources=row["sources"] or [],
                    mention_counts=mention_counts,
                    first_seen_per_source=first_seen_per_source,
                    total_mentions=row["total_mentions"],
                    source_agreement_score=agreement,
                )
            )

        logger.info(
            "Found cross-source entities",
            count=len(links),
            min_sources=min_sources,
        )
        return links

    # ========================================================================
    # Correlation Detection
    # ========================================================================

    async def detect_mention_correlations(
        self,
        window_days: int = 30,
        min_mentions: int = 5,
        entity_type: EntityType | None = None,
        domain: Domain | None = None,
    ) -> list[Correlation]:
        """
        Detect correlations between entity mention frequencies.

        Analyzes daily mention counts and finds entities whose
        mention patterns correlate (rise and fall together).

        Args:
            window_days: Analysis window size
            min_mentions: Minimum mentions to include entity
            entity_type: Optional filter
            domain: Optional filter

        Returns:
            List of detected correlations
        """
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=window_days)

        # Get daily mention counts for all entities
        conditions = ["em.mentioned_at >= $1", "em.mentioned_at <= $2"]
        params: list[Any] = [window_start, window_end]

        if entity_type:
            conditions.append(f"e.entity_type = ${len(params) + 1}")
            params.append(entity_type.value)

        if domain:
            conditions.append(f"e.domain = ${len(params) + 1}")
            params.append(domain.value)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            em.entity_id,
            DATE_TRUNC('day', em.mentioned_at)::date as day,
            COUNT(*) as count
        FROM entity_mentions em
        JOIN entities e ON e.id = em.entity_id
        WHERE {where_clause}
        GROUP BY em.entity_id, DATE_TRUNC('day', em.mentioned_at)
        ORDER BY em.entity_id, day
        """

        rows = await self.db.fetch(query, *params)

        # Build time series per entity
        entity_series: dict[int, dict[str, int]] = {}
        for row in rows:
            entity_id = row["entity_id"]
            day_str = row["day"].isoformat()
            if entity_id not in entity_series:
                entity_series[entity_id] = {}
            entity_series[entity_id][day_str] = row["count"]

        # Filter entities with minimum mentions
        valid_entities = {
            eid: series
            for eid, series in entity_series.items()
            if sum(series.values()) >= min_mentions
        }

        if len(valid_entities) < 2:
            return []

        # Generate all days in window for alignment
        all_days = []
        current = window_start
        while current <= window_end:
            all_days.append(current.date().isoformat())
            current += timedelta(days=1)

        # Calculate correlations between all pairs
        correlations = []
        entity_ids = list(valid_entities.keys())

        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                eid1, eid2 = entity_ids[i], entity_ids[j]
                series1 = valid_entities[eid1]
                series2 = valid_entities[eid2]

                # Align to same days
                x = [float(series1.get(day, 0)) for day in all_days]
                y = [float(series2.get(day, 0)) for day in all_days]

                # Skip if not enough variation
                if sum(x) < self.MIN_SAMPLE_SIZE or sum(y) < self.MIN_SAMPLE_SIZE:
                    continue

                # Calculate correlation
                r, p_value = pearson_correlation(x, y)

                # Only keep significant correlations
                if p_value <= self.SIGNIFICANCE_THRESHOLD and abs(r) >= 0.3:
                    correlation = Correlation(
                        source_entity_id=eid1,
                        target_entity_id=eid2,
                        correlation_type=CorrelationType.MENTION_FREQUENCY,
                        correlation_value=r,
                        significance=p_value,
                        window_start=window_start,
                        window_end=window_end,
                        window_size_days=window_days,
                        sample_size=len(all_days),
                        description=(
                            f"Mention frequency correlation "
                            f"(r={r:.3f}, p={p_value:.4f})"
                        ),
                    )
                    correlations.append(correlation)

        logger.info(
            "Detected mention correlations",
            total_entities=len(valid_entities),
            correlations_found=len(correlations),
            window_days=window_days,
        )

        # Sort by strength
        correlations.sort(key=lambda c: abs(c.correlation_value), reverse=True)
        return correlations

    async def detect_cross_domain_correlations(
        self,
        source_domain: Domain,
        target_domain: Domain,
        window_days: int = 30,
    ) -> list[Correlation]:
        """
        Detect correlations between entities in different domains.

        For example: tech mentions vs market movement,
        training load vs sleep quality.

        Args:
            source_domain: First domain
            target_domain: Second domain
            window_days: Analysis window

        Returns:
            List of detected correlations
        """
        # Get entities from each domain with daily mentions
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=window_days)

        query = """
        SELECT
            em.entity_id,
            e.domain,
            e.name,
            DATE_TRUNC('day', em.mentioned_at)::date as day,
            COUNT(*) as count
        FROM entity_mentions em
        JOIN entities e ON e.id = em.entity_id
        WHERE em.mentioned_at >= $1 AND em.mentioned_at <= $2
          AND e.domain IN ($3, $4)
        GROUP BY em.entity_id, e.domain, e.name, DATE_TRUNC('day', em.mentioned_at)
        ORDER BY em.entity_id, day
        """

        rows = await self.db.fetch(
            query, window_start, window_end,
            source_domain.value, target_domain.value
        )

        # Separate by domain
        source_series: dict[int, dict[str, int]] = {}
        target_series: dict[int, dict[str, int]] = {}

        for row in rows:
            entity_id = row["entity_id"]
            day_str = row["day"].isoformat()
            domain = row["domain"]

            if domain == source_domain.value:
                if entity_id not in source_series:
                    source_series[entity_id] = {}
                source_series[entity_id][day_str] = row["count"]
            else:
                if entity_id not in target_series:
                    target_series[entity_id] = {}
                target_series[entity_id][day_str] = row["count"]

        if not source_series or not target_series:
            return []

        # Generate all days
        all_days = []
        current = window_start
        while current <= window_end:
            all_days.append(current.date().isoformat())
            current += timedelta(days=1)

        # Calculate cross-domain correlations
        correlations = []
        corr_type = self._get_cross_domain_type(source_domain, target_domain)

        for source_id, s_series in source_series.items():
            for target_id, t_series in target_series.items():
                x = [float(s_series.get(day, 0)) for day in all_days]
                y = [float(t_series.get(day, 0)) for day in all_days]

                if sum(x) < self.MIN_SAMPLE_SIZE or sum(y) < self.MIN_SAMPLE_SIZE:
                    continue

                r, p_value = spearman_correlation(x, y)

                if p_value <= self.SIGNIFICANCE_THRESHOLD and abs(r) >= 0.3:
                    correlations.append(
                        Correlation(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            correlation_type=corr_type,
                            correlation_value=r,
                            significance=p_value,
                            window_start=window_start,
                            window_end=window_end,
                            window_size_days=window_days,
                            sample_size=len(all_days),
                            description=(
                                f"Cross-domain correlation "
                                f"({source_domain.value} -> {target_domain.value})"
                            ),
                        )
                    )

        logger.info(
            "Detected cross-domain correlations",
            source_domain=source_domain.value,
            target_domain=target_domain.value,
            correlations_found=len(correlations),
        )

        correlations.sort(key=lambda c: abs(c.correlation_value), reverse=True)
        return correlations

    def _get_cross_domain_type(
        self, source: Domain, target: Domain
    ) -> CorrelationType:
        """Get appropriate correlation type for domain pair."""
        pair = frozenset([source, target])
        if pair == frozenset([Domain.TECH, Domain.FINANCE]):
            return CorrelationType.TECH_MARKET
        if pair == frozenset([Domain.FITNESS, Domain.GENERAL]):
            return CorrelationType.FITNESS_SLEEP
        return CorrelationType.CROSS_SOURCE

    # ========================================================================
    # Correlation Analysis
    # ========================================================================

    async def analyze_correlations(
        self,
        window_days: int = 30,
        persist: bool = True,
    ) -> CorrelationResult:
        """
        Run full correlation analysis across the knowledge graph.

        Args:
            window_days: Analysis window
            persist: Whether to save correlations to database

        Returns:
            CorrelationResult with all findings
        """
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=window_days)

        # Find cross-source entities
        cross_source_links = await self.find_cross_source_entities(min_sources=2)

        # Detect mention correlations
        mention_correlations = await self.detect_mention_correlations(
            window_days=window_days
        )

        # Detect cross-domain correlations
        cross_domain_correlations = []
        domain_pairs = [
            (Domain.TECH, Domain.FINANCE),
            (Domain.FITNESS, Domain.GENERAL),
        ]
        for source, target in domain_pairs:
            corrs = await self.detect_cross_domain_correlations(
                source, target, window_days
            )
            cross_domain_correlations.extend(corrs)

        all_correlations = mention_correlations + cross_domain_correlations

        # Persist correlations
        if persist and all_correlations:
            for corr in all_correlations:
                try:
                    await self.repo.upsert(corr)
                except Exception as e:
                    logger.warning("Failed to persist correlation", error=str(e))

        # Count significant
        significant = len([c for c in all_correlations if c.is_significant])

        # Build summary
        summary = self._build_summary(
            cross_source_links, all_correlations, window_days
        )

        logger.info(
            "Correlation analysis complete",
            total_correlations=len(all_correlations),
            significant=significant,
            cross_source_entities=len(cross_source_links),
        )

        return CorrelationResult(
            correlations=all_correlations,
            cross_source_links=cross_source_links,
            analysis_window=(window_start, window_end),
            total_entities_analyzed=len(cross_source_links),
            significant_correlations=significant,
            summary=summary,
        )

    def _build_summary(
        self,
        links: list[CrossSourceLink],
        correlations: list[Correlation],
        window_days: int,
    ) -> str:
        """Build human-readable summary of correlation analysis."""
        parts = [f"Correlation Analysis ({window_days}-day window):"]

        # Cross-source summary
        if links:
            top_links = sorted(
                links, key=lambda x: x.total_mentions, reverse=True
            )[:3]
            link_names = [
                f"{lnk.entity_name} ({len(lnk.sources)} sources)"
                for lnk in top_links
            ]
            parts.append(f"- Top cross-source entities: {', '.join(link_names)}")

        # Correlation summary
        strong = [c for c in correlations if c.strength == CorrelationStrength.STRONG
                  or c.strength == CorrelationStrength.VERY_STRONG]
        if strong:
            parts.append(f"- Found {len(strong)} strong correlations")
            top_corr = sorted(
                strong, key=lambda x: abs(x.correlation_value), reverse=True
            )[:3]
            for c in top_corr:
                parts.append(
                    f"  - Entity {c.source_entity_id} <-> {c.target_entity_id}: "
                    f"r={c.correlation_value:.3f}"
                )

        if not links and not correlations:
            parts.append("- No significant correlations found in this period")

        return "\n".join(parts)

    # ========================================================================
    # Query API
    # ========================================================================

    async def get_correlated_entities(
        self,
        entity_id: int,
        min_strength: float = 0.4,
        max_p_value: float = 0.05,
    ) -> list[tuple[Entity, Correlation]]:
        """
        Get entities correlated with a given entity.

        Args:
            entity_id: Entity to find correlations for
            min_strength: Minimum correlation coefficient
            max_p_value: Maximum p-value

        Returns:
            List of (Entity, Correlation) tuples
        """
        correlations = await self.repo.get_for_entity(
            entity_id, min_significance=max_p_value
        )

        # Filter by strength
        strong_correlations = [
            c for c in correlations if abs(c.correlation_value) >= min_strength
        ]

        # Fetch related entities
        results = []
        for corr in strong_correlations:
            related_id = (
                corr.target_entity_id
                if corr.source_entity_id == entity_id
                else corr.source_entity_id
            )

            entity_row = await self.db.fetchrow(
                """
                SELECT id, external_id, name, entity_type, domain, description,
                       source, source_url, confidence, mention_count,
                       first_seen, last_seen, created_at, updated_at, properties
                FROM entities WHERE id = $1
                """,
                related_id,
            )

            if entity_row:
                entity = Entity(
                    id=entity_row["id"],
                    external_id=entity_row["external_id"],
                    name=entity_row["name"],
                    entity_type=EntityType(entity_row["entity_type"]),
                    domain=Domain(entity_row["domain"]),
                    description=entity_row["description"],
                    source=entity_row["source"],
                    source_url=entity_row["source_url"],
                    confidence=entity_row["confidence"],
                    mention_count=entity_row["mention_count"],
                    first_seen=entity_row["first_seen"],
                    last_seen=entity_row["last_seen"],
                    created_at=entity_row["created_at"],
                    updated_at=entity_row["updated_at"],
                    properties=entity_row["properties"] or {},
                )
                results.append((entity, corr))

        return results

    async def get_strongest_correlations(
        self,
        limit: int = 20,
        correlation_type: CorrelationType | None = None,
    ) -> list[Correlation]:
        """Get the strongest correlations in the system."""
        return await self.repo.get_significant(
            correlation_type=correlation_type,
            min_strength=0.5,
            limit=limit,
        )
