"""Strava fitness data client for activity ingestion."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
import structlog

from src.models import Activity, ActivityMetrics, FitnessSyncResult

logger = structlog.get_logger()

# Strava API endpoints
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"


class StravaAuth:
    """Handle Strava OAuth2 authentication."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: int = 0

    def get_authorization_url(self, scope: str = "activity:read_all") -> str:
        """Generate OAuth authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": scope,
        }
        return f"{STRAVA_AUTH_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                STRAVA_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                },
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            self._refresh_token = data["refresh_token"]
            self._expires_at = data["expires_at"]

            logger.info("Strava token exchange successful")
            return data

    async def refresh_access_token(self) -> dict[str, Any]:
        """Refresh the access token using refresh token."""
        if not self._refresh_token:
            raise ValueError("No refresh token available")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                STRAVA_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            self._refresh_token = data["refresh_token"]
            self._expires_at = data["expires_at"]

            logger.info("Strava token refresh successful")
            return data

    def set_tokens(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: int,
    ) -> None:
        """Set tokens from stored credentials."""
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at

    @property
    def access_token(self) -> str | None:
        """Get current access token."""
        return self._access_token

    @property
    def is_token_expired(self) -> bool:
        """Check if access token is expired."""
        return time.time() >= self._expires_at

    async def get_valid_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if self._access_token is None:
            raise ValueError("No access token available. Please authenticate first.")

        if self.is_token_expired:
            await self.refresh_access_token()

        return self._access_token


class StravaClient:
    """Client for Strava API operations."""

    def __init__(self, auth: StravaAuth):
        self.auth = auth
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "StravaClient":
        self._client = httpx.AsyncClient(
            base_url=STRAVA_API_BASE,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("StravaClient must be used as async context manager")
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make authenticated request to Strava API."""
        token = await self.auth.get_valid_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await self.client.request(
            method,
            endpoint,
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_athlete(self) -> dict[str, Any]:
        """Get authenticated athlete profile."""
        return await self._request("GET", "/athlete")

    async def get_activities(
        self,
        after: datetime | None = None,
        before: datetime | None = None,
        page: int = 1,
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        """Get athlete activities with pagination."""
        params: dict[str, Any] = {
            "page": page,
            "per_page": per_page,
        }

        if after:
            params["after"] = int(after.timestamp())
        if before:
            params["before"] = int(before.timestamp())

        return await self._request("GET", "/athlete/activities", params)

    async def get_activity(self, activity_id: int) -> dict[str, Any]:
        """Get detailed activity by ID."""
        return await self._request("GET", f"/activities/{activity_id}")

    async def get_all_activities(
        self,
        after: datetime | None = None,
        before: datetime | None = None,
        max_activities: int = 200,
    ) -> list[dict[str, Any]]:
        """Get all activities with automatic pagination."""
        all_activities: list[dict[str, Any]] = []
        page = 1
        per_page = 30

        while len(all_activities) < max_activities:
            activities = await self.get_activities(
                after=after,
                before=before,
                page=page,
                per_page=per_page,
            )

            if not activities:
                break

            all_activities.extend(activities)
            page += 1

            logger.debug(
                "Fetched activities page",
                page=page - 1,
                count=len(activities),
                total=len(all_activities),
            )

        return all_activities[:max_activities]


class ActivityParser:
    """Parse Strava API responses into Activity models."""

    @staticmethod
    def parse_activity(data: dict[str, Any]) -> Activity:
        """Parse Strava activity data into Activity model."""
        # Parse start date
        start_date = datetime.fromisoformat(
            data["start_date"].replace("Z", "+00:00")
        )

        # Extract heart rate data if available
        avg_hr = data.get("average_heartrate")
        max_hr = data.get("max_heartrate")

        # Extract power data if available (cycling)
        avg_power = data.get("average_watts")
        max_power = data.get("max_watts")

        return Activity(
            external_id=str(data["id"]),
            name=data.get("name", "Untitled"),
            activity_type=data["type"],
            sport_type=data.get("sport_type", data["type"]),
            start_date=start_date,
            distance_meters=data.get("distance", 0),
            moving_time_seconds=data.get("moving_time", 0),
            elapsed_time_seconds=data.get("elapsed_time", 0),
            total_elevation_gain=data.get("total_elevation_gain", 0),
            avg_speed=data.get("average_speed"),
            max_speed=data.get("max_speed"),
            avg_hr=int(avg_hr) if avg_hr else None,
            max_hr=int(max_hr) if max_hr else None,
            avg_power=int(avg_power) if avg_power else None,
            max_power=int(max_power) if max_power else None,
            calories=data.get("calories"),
            suffer_score=data.get("suffer_score"),
        )


class TrainingMetricsCalculator:
    """Calculate training metrics from activities."""

    # Exponential decay constants
    ATL_DECAY = 7  # Acute Training Load - 7 day time constant
    CTL_DECAY = 42  # Chronic Training Load - 42 day time constant

    @staticmethod
    def estimate_tss(activity: Activity) -> float:
        """
        Estimate Training Stress Score (TSS) for an activity.

        For activities without power data, we estimate based on:
        - Duration
        - Heart rate intensity (if available)
        - Activity type
        """
        duration_hours = activity.moving_time_seconds / 3600

        # If we have heart rate data, use TRIMP-like calculation
        if activity.avg_hr:
            # Assume max HR of 185 if not available
            max_hr = activity.max_hr or 185
            # Estimate resting HR
            rest_hr = 60
            hr_reserve = max_hr - rest_hr
            intensity = (activity.avg_hr - rest_hr) / hr_reserve if hr_reserve > 0 else 0.5
            intensity = max(0.3, min(1.0, intensity))

            # TSS approximation: duration * intensity^2 * 100
            return duration_hours * (intensity ** 2) * 100

        # For activities without HR, estimate based on type
        intensity_map = {
            "Run": 0.7,
            "Ride": 0.6,
            "Swim": 0.65,
            "Walk": 0.4,
            "Hike": 0.5,
            "WeightTraining": 0.5,
            "Workout": 0.6,
        }
        intensity = intensity_map.get(activity.activity_type, 0.5)

        return duration_hours * (intensity ** 2) * 100

    @classmethod
    def calculate_exponential_moving_average(
        cls,
        values: list[float],
        decay_days: int,
    ) -> list[float]:
        """Calculate exponential moving average with given decay constant."""
        if not values:
            return []

        decay = 1 - (1 / decay_days)
        ema = [values[0]]

        for i in range(1, len(values)):
            ema.append(values[i] + decay * ema[i - 1])

        return ema

    @classmethod
    def calculate_training_load(
        cls,
        daily_tss: list[float],
    ) -> ActivityMetrics:
        """
        Calculate training load metrics from daily TSS values.

        Returns the latest ATL, CTL, and TSB values.
        """
        if not daily_tss:
            return ActivityMetrics(atl=0, ctl=0, tsb=0)

        # Calculate ATL (7-day exponential average)
        atl_values = cls.calculate_exponential_moving_average(
            daily_tss, cls.ATL_DECAY
        )

        # Calculate CTL (42-day exponential average)
        ctl_values = cls.calculate_exponential_moving_average(
            daily_tss, cls.CTL_DECAY
        )

        # Get latest values
        atl = atl_values[-1] if atl_values else 0
        ctl = ctl_values[-1] if ctl_values else 0

        # TSB = CTL - ATL (Training Stress Balance / Form)
        tsb = ctl - atl

        return ActivityMetrics(
            atl=round(atl, 1),
            ctl=round(ctl, 1),
            tsb=round(tsb, 1),
        )


async def sync_strava_activities(
    auth: StravaAuth,
    days: int = 90,
) -> FitnessSyncResult:
    """Sync activities from Strava for the last N days."""
    result = FitnessSyncResult(started_at=datetime.now(UTC))

    try:
        async with StravaClient(auth) as client:
            # Calculate date range
            after = datetime.now(UTC) - timedelta(days=days)

            # Fetch activities
            raw_activities = await client.get_all_activities(
                after=after,
                max_activities=500,
            )

            # Parse activities
            parser = ActivityParser()
            activities = []
            for raw in raw_activities:
                try:
                    activity = parser.parse_activity(raw)
                    activities.append(activity)
                except Exception as e:
                    logger.warning(
                        "Failed to parse activity",
                        activity_id=raw.get("id"),
                        error=str(e),
                    )
                    result.errors.append(str(e))

            result.activities = activities
            result.activities_fetched = len(activities)

            # Calculate TSS for each activity
            calculator = TrainingMetricsCalculator()
            for activity in activities:
                activity.tss = calculator.estimate_tss(activity)

            logger.info(
                "Strava sync completed",
                activities=len(activities),
                days=days,
            )

    except Exception as e:
        result.success = False
        result.errors.append(str(e))
        logger.error("Strava sync failed", error=str(e))

    result.completed_at = datetime.now(UTC)
    return result
