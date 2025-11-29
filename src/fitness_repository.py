"""Repository for fitness data access."""

from datetime import date, datetime
from typing import Any

import asyncpg
import structlog

from src.database import Database
from src.models import Activity, ActivityMetrics

logger = structlog.get_logger()


class ActivityRepository:
    """Repository for activity CRUD operations."""

    def __init__(self, db: Database):
        self.db = db

    async def upsert(self, activity: Activity) -> tuple[int, bool]:
        """
        Insert or update an activity based on external_id.
        Returns (activity_id, is_new).
        """
        query = """
        INSERT INTO activities (
            external_id, name, activity_type, sport_type, start_date,
            distance_meters, moving_time_seconds, elapsed_time_seconds,
            total_elevation_gain, avg_speed, max_speed, avg_hr, max_hr,
            avg_power, max_power, calories, suffer_score, tss
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9,
            $10, $11, $12, $13, $14, $15, $16, $17, $18
        )
        ON CONFLICT (external_id) DO UPDATE SET
            name = EXCLUDED.name,
            tss = EXCLUDED.tss,
            updated_at = NOW()
        RETURNING id, (xmax = 0) as is_new
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                query,
                activity.external_id,
                activity.name,
                activity.activity_type,
                activity.sport_type,
                activity.start_date,
                activity.distance_meters,
                activity.moving_time_seconds,
                activity.elapsed_time_seconds,
                activity.total_elevation_gain,
                activity.avg_speed,
                activity.max_speed,
                activity.avg_hr,
                activity.max_hr,
                activity.avg_power,
                activity.max_power,
                activity.calories,
                activity.suffer_score,
                activity.tss,
            )
            is_new = row["is_new"]
            logger.debug(
                "Activity upserted",
                id=row["id"],
                is_new=is_new,
                name=activity.name[:30],
            )
            return row["id"], is_new

    async def get_by_external_id(self, external_id: str) -> Activity | None:
        """Get activity by Strava ID."""
        query = """
        SELECT id, external_id, name, activity_type, sport_type, start_date,
               distance_meters, moving_time_seconds, elapsed_time_seconds,
               total_elevation_gain, avg_speed, max_speed, avg_hr, max_hr,
               avg_power, max_power, calories, suffer_score, tss
        FROM activities WHERE external_id = $1
        """
        row = await self.db.fetchrow(query, external_id)
        if row is None:
            return None
        return self._row_to_activity(row)

    async def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> list[Activity]:
        """Get activities within date range."""
        if end_date is None:
            end_date = datetime.now()

        query = """
        SELECT id, external_id, name, activity_type, sport_type, start_date,
               distance_meters, moving_time_seconds, elapsed_time_seconds,
               total_elevation_gain, avg_speed, max_speed, avg_hr, max_hr,
               avg_power, max_power, calories, suffer_score, tss
        FROM activities
        WHERE start_date >= $1 AND start_date <= $2
        ORDER BY start_date DESC
        """
        rows = await self.db.fetch(query, start_date, end_date)
        return [self._row_to_activity(row) for row in rows]

    async def get_recent(
        self,
        limit: int = 50,
        activity_type: str | None = None,
    ) -> list[Activity]:
        """Get recent activities."""
        conditions = ["1=1"]
        params: list[Any] = []

        if activity_type:
            conditions.append(f"activity_type = ${len(params) + 1}")
            params.append(activity_type)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, external_id, name, activity_type, sport_type, start_date,
               distance_meters, moving_time_seconds, elapsed_time_seconds,
               total_elevation_gain, avg_speed, max_speed, avg_hr, max_hr,
               avg_power, max_power, calories, suffer_score, tss
        FROM activities
        WHERE {where_clause}
        ORDER BY start_date DESC
        LIMIT ${len(params) + 1}
        """
        params.append(limit)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_activity(row) for row in rows]

    async def count(self, activity_type: str | None = None) -> int:
        """Count activities."""
        if activity_type:
            query = "SELECT COUNT(*) FROM activities WHERE activity_type = $1"
            return await self.db.fetchval(query, activity_type)
        else:
            query = "SELECT COUNT(*) FROM activities"
            return await self.db.fetchval(query)

    async def get_activity_types(self) -> list[str]:
        """Get list of unique activity types."""
        query = """
        SELECT DISTINCT activity_type FROM activities
        ORDER BY activity_type
        """
        rows = await self.db.fetch(query)
        return [row["activity_type"] for row in rows]

    async def get_daily_tss(
        self,
        start_date: date,
        end_date: date | None = None,
    ) -> dict[date, float]:
        """Get daily TSS totals for date range."""
        if end_date is None:
            end_date = date.today()

        query = """
        SELECT DATE(start_date) as day, COALESCE(SUM(tss), 0) as total_tss
        FROM activities
        WHERE DATE(start_date) >= $1 AND DATE(start_date) <= $2
        GROUP BY DATE(start_date)
        ORDER BY day
        """
        rows = await self.db.fetch(query, start_date, end_date)
        return {row["day"]: float(row["total_tss"]) for row in rows}

    def _row_to_activity(self, row: asyncpg.Record) -> Activity:
        """Convert database row to Activity model."""
        return Activity(
            id=row["id"],
            external_id=row["external_id"],
            name=row["name"],
            activity_type=row["activity_type"],
            sport_type=row["sport_type"],
            start_date=row["start_date"],
            distance_meters=row["distance_meters"] or 0,
            moving_time_seconds=row["moving_time_seconds"] or 0,
            elapsed_time_seconds=row["elapsed_time_seconds"] or 0,
            total_elevation_gain=row["total_elevation_gain"] or 0,
            avg_speed=row["avg_speed"],
            max_speed=row["max_speed"],
            avg_hr=row["avg_hr"],
            max_hr=row["max_hr"],
            avg_power=row["avg_power"],
            max_power=row["max_power"],
            calories=row["calories"],
            suffer_score=row["suffer_score"],
            tss=row["tss"],
        )


class FitnessMetricsRepository:
    """Repository for daily fitness metrics."""

    def __init__(self, db: Database):
        self.db = db

    async def upsert(self, metrics: ActivityMetrics) -> int:
        """Insert or update daily metrics."""
        if metrics.date is None:
            raise ValueError("Metrics must have a date")

        query = """
        INSERT INTO fitness_metrics (date, daily_tss, atl, ctl, tsb)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (date) DO UPDATE SET
            daily_tss = EXCLUDED.daily_tss,
            atl = EXCLUDED.atl,
            ctl = EXCLUDED.ctl,
            tsb = EXCLUDED.tsb,
            updated_at = NOW()
        RETURNING id
        """
        return await self.db.fetchval(
            query,
            metrics.date.date() if isinstance(metrics.date, datetime) else metrics.date,
            metrics.daily_tss,
            metrics.atl,
            metrics.ctl,
            metrics.tsb,
        )

    async def get_by_date(self, target_date: date) -> ActivityMetrics | None:
        """Get metrics for a specific date."""
        query = """
        SELECT date, daily_tss, atl, ctl, tsb
        FROM fitness_metrics WHERE date = $1
        """
        row = await self.db.fetchrow(query, target_date)
        if row is None:
            return None
        return self._row_to_metrics(row)

    async def get_latest(self) -> ActivityMetrics | None:
        """Get most recent metrics."""
        query = """
        SELECT date, daily_tss, atl, ctl, tsb
        FROM fitness_metrics
        ORDER BY date DESC LIMIT 1
        """
        row = await self.db.fetchrow(query)
        if row is None:
            return None
        return self._row_to_metrics(row)

    async def get_date_range(
        self,
        start_date: date,
        end_date: date | None = None,
    ) -> list[ActivityMetrics]:
        """Get metrics for date range."""
        if end_date is None:
            end_date = date.today()

        query = """
        SELECT date, daily_tss, atl, ctl, tsb
        FROM fitness_metrics
        WHERE date >= $1 AND date <= $2
        ORDER BY date
        """
        rows = await self.db.fetch(query, start_date, end_date)
        return [self._row_to_metrics(row) for row in rows]

    def _row_to_metrics(self, row: asyncpg.Record) -> ActivityMetrics:
        """Convert database row to ActivityMetrics model."""
        return ActivityMetrics(
            date=datetime.combine(row["date"], datetime.min.time()),
            daily_tss=row["daily_tss"] or 0,
            atl=row["atl"] or 0,
            ctl=row["ctl"] or 0,
            tsb=row["tsb"] or 0,
        )


class StravaTokenRepository:
    """Repository for Strava OAuth tokens."""

    def __init__(self, db: Database):
        self.db = db

    async def save_tokens(
        self,
        athlete_id: str,
        access_token: str,
        refresh_token: str,
        expires_at: int,
    ) -> None:
        """Save or update Strava tokens."""
        query = """
        INSERT INTO strava_tokens (athlete_id, access_token, refresh_token, expires_at)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (athlete_id) DO UPDATE SET
            access_token = EXCLUDED.access_token,
            refresh_token = EXCLUDED.refresh_token,
            expires_at = EXCLUDED.expires_at,
            updated_at = NOW()
        """
        await self.db.execute(
            query, athlete_id, access_token, refresh_token, expires_at
        )
        logger.info("Strava tokens saved", athlete_id=athlete_id)

    async def get_tokens(self, athlete_id: str) -> dict[str, Any] | None:
        """Get stored tokens for athlete."""
        query = """
        SELECT access_token, refresh_token, expires_at
        FROM strava_tokens WHERE athlete_id = $1
        """
        row = await self.db.fetchrow(query, athlete_id)
        if row is None:
            return None
        return {
            "access_token": row["access_token"],
            "refresh_token": row["refresh_token"],
            "expires_at": row["expires_at"],
        }

    async def get_default_tokens(self) -> dict[str, Any] | None:
        """Get tokens for first/default athlete."""
        query = """
        SELECT athlete_id, access_token, refresh_token, expires_at
        FROM strava_tokens
        ORDER BY updated_at DESC LIMIT 1
        """
        row = await self.db.fetchrow(query)
        if row is None:
            return None
        return {
            "athlete_id": row["athlete_id"],
            "access_token": row["access_token"],
            "refresh_token": row["refresh_token"],
            "expires_at": row["expires_at"],
        }
