"""Tests for Strava fitness client."""

from datetime import UTC, datetime, timedelta

import pytest

from src.ingestion.fitness_client import (
    ActivityParser,
    StravaAuth,
    TrainingMetricsCalculator,
)
from src.models import Activity, ActivityMetrics


class TestStravaAuth:
    """Tests for Strava OAuth authentication."""

    def test_auth_initialization(self) -> None:
        """Test StravaAuth initialization."""
        auth = StravaAuth(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8080/callback",
        )
        assert auth.client_id == "test_client_id"
        assert auth.client_secret == "test_secret"
        assert auth.redirect_uri == "http://localhost:8080/callback"

    def test_authorization_url(self) -> None:
        """Test authorization URL generation."""
        auth = StravaAuth(
            client_id="12345",
            client_secret="secret",
        )
        url = auth.get_authorization_url()

        assert "https://www.strava.com/oauth/authorize" in url
        assert "client_id=12345" in url
        assert "response_type=code" in url
        assert "scope=activity%3Aread_all" in url

    def test_authorization_url_custom_scope(self) -> None:
        """Test authorization URL with custom scope."""
        auth = StravaAuth(
            client_id="12345",
            client_secret="secret",
        )
        url = auth.get_authorization_url(scope="read_all")

        assert "scope=read_all" in url

    def test_set_tokens(self) -> None:
        """Test setting tokens."""
        auth = StravaAuth(
            client_id="12345",
            client_secret="secret",
        )
        auth.set_tokens(
            access_token="access123",
            refresh_token="refresh456",
            expires_at=9999999999,
        )

        assert auth.access_token == "access123"
        assert auth._refresh_token == "refresh456"
        assert auth._expires_at == 9999999999

    def test_is_token_expired(self) -> None:
        """Test token expiration check."""
        auth = StravaAuth(
            client_id="12345",
            client_secret="secret",
        )

        # Expired token
        auth.set_tokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=0,
        )
        assert auth.is_token_expired is True

        # Valid token
        auth.set_tokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=9999999999,
        )
        assert auth.is_token_expired is False


class TestActivityParser:
    """Tests for Strava activity parsing."""

    def test_parse_activity_basic(self) -> None:
        """Test parsing basic activity data."""
        data = {
            "id": 12345,
            "name": "Morning Run",
            "type": "Run",
            "sport_type": "Run",
            "start_date": "2024-01-15T08:00:00Z",
            "distance": 5000.0,
            "moving_time": 1800,
            "elapsed_time": 2000,
            "total_elevation_gain": 50.0,
            "average_speed": 2.78,
            "max_speed": 3.5,
        }

        activity = ActivityParser.parse_activity(data)

        assert activity.external_id == "12345"
        assert activity.name == "Morning Run"
        assert activity.activity_type == "Run"
        assert activity.sport_type == "Run"
        assert activity.distance_meters == 5000.0
        assert activity.moving_time_seconds == 1800
        assert activity.elapsed_time_seconds == 2000
        assert activity.total_elevation_gain == 50.0
        assert activity.avg_speed == 2.78
        assert activity.max_speed == 3.5

    def test_parse_activity_with_heart_rate(self) -> None:
        """Test parsing activity with heart rate data."""
        data = {
            "id": 12346,
            "name": "Tempo Run",
            "type": "Run",
            "start_date": "2024-01-15T08:00:00Z",
            "distance": 10000.0,
            "moving_time": 3600,
            "elapsed_time": 3700,
            "average_heartrate": 150.5,
            "max_heartrate": 175.0,
        }

        activity = ActivityParser.parse_activity(data)

        assert activity.avg_hr == 150
        assert activity.max_hr == 175

    def test_parse_activity_with_power(self) -> None:
        """Test parsing activity with power data (cycling)."""
        data = {
            "id": 12347,
            "name": "Morning Ride",
            "type": "Ride",
            "start_date": "2024-01-15T07:00:00Z",
            "distance": 50000.0,
            "moving_time": 7200,
            "elapsed_time": 7500,
            "average_watts": 200.0,
            "max_watts": 450.0,
        }

        activity = ActivityParser.parse_activity(data)

        assert activity.avg_power == 200
        assert activity.max_power == 450

    def test_parse_activity_missing_optional_fields(self) -> None:
        """Test parsing activity with missing optional fields."""
        data = {
            "id": 12348,
            "name": "Quick Walk",
            "type": "Walk",
            "start_date": "2024-01-15T12:00:00Z",
        }

        activity = ActivityParser.parse_activity(data)

        assert activity.external_id == "12348"
        assert activity.name == "Quick Walk"
        assert activity.activity_type == "Walk"
        assert activity.distance_meters == 0
        assert activity.moving_time_seconds == 0
        assert activity.avg_hr is None
        assert activity.avg_power is None

    def test_parse_activity_default_name(self) -> None:
        """Test activity gets default name when missing."""
        data = {
            "id": 12349,
            "type": "Workout",
            "start_date": "2024-01-15T12:00:00Z",
        }

        activity = ActivityParser.parse_activity(data)
        assert activity.name == "Untitled"


class TestTrainingMetricsCalculator:
    """Tests for training metrics calculations."""

    def test_estimate_tss_with_heart_rate(self) -> None:
        """Test TSS estimation with heart rate data."""
        activity = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            moving_time_seconds=3600,  # 1 hour
            avg_hr=160,
            max_hr=180,
        )

        tss = TrainingMetricsCalculator.estimate_tss(activity)

        # With high intensity HR, TSS should be significant
        assert tss > 50
        assert tss < 150

    def test_estimate_tss_without_heart_rate(self) -> None:
        """Test TSS estimation without heart rate data."""
        activity = Activity(
            external_id="2",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            moving_time_seconds=3600,  # 1 hour
        )

        tss = TrainingMetricsCalculator.estimate_tss(activity)

        # Should use default intensity for running (0.7)
        # TSS = 1 hour * 0.7^2 * 100 = 49
        assert abs(tss - 49) < 1

    def test_estimate_tss_different_activity_types(self) -> None:
        """Test TSS estimation for different activity types."""
        base_time = 3600  # 1 hour

        run = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            moving_time_seconds=base_time,
        )

        ride = Activity(
            external_id="2",
            name="Ride",
            activity_type="Ride",
            start_date=datetime.now(UTC),
            moving_time_seconds=base_time,
        )

        walk = Activity(
            external_id="3",
            name="Walk",
            activity_type="Walk",
            start_date=datetime.now(UTC),
            moving_time_seconds=base_time,
        )

        run_tss = TrainingMetricsCalculator.estimate_tss(run)
        ride_tss = TrainingMetricsCalculator.estimate_tss(ride)
        walk_tss = TrainingMetricsCalculator.estimate_tss(walk)

        # Run should have higher TSS than ride, walk should be lowest
        assert run_tss > ride_tss
        assert ride_tss > walk_tss

    def test_calculate_exponential_moving_average(self) -> None:
        """Test exponential moving average calculation."""
        values = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 7 days

        ema = TrainingMetricsCalculator.calculate_exponential_moving_average(
            values, decay_days=7
        )

        assert len(ema) == 7
        # First value should be scaled by alpha (1/7)
        assert abs(ema[0] - (100.0 / 7)) < 0.01
        # Values should decay over time as no new TSS is added
        assert ema[-1] < ema[0]

    def test_calculate_exponential_moving_average_empty(self) -> None:
        """Test EMA with empty input."""
        ema = TrainingMetricsCalculator.calculate_exponential_moving_average(
            [], decay_days=7
        )
        assert ema == []

    def test_calculate_training_load(self) -> None:
        """Test training load calculation."""
        # Simulate 42 days of training with consistent 50 TSS per day
        daily_tss = [50.0] * 42

        metrics = TrainingMetricsCalculator.calculate_training_load(daily_tss)

        # With consistent training, both values should be positive
        assert metrics.atl > 0
        assert metrics.ctl > 0
        # After 42 days of consistent training, values should approach 50
        # ATL converges faster (7-day) than CTL (42-day)
        assert 30 < metrics.atl < 55
        assert 15 < metrics.ctl < 55

    def test_calculate_training_load_empty(self) -> None:
        """Test training load with no data."""
        metrics = TrainingMetricsCalculator.calculate_training_load([])

        assert metrics.atl == 0
        assert metrics.ctl == 0
        assert metrics.tsb == 0

    def test_calculate_training_load_high_atl(self) -> None:
        """Test training load with recent high training (high ATL)."""
        # Low training for 35 days, then high for 7 days
        daily_tss = [30.0] * 35 + [100.0] * 7

        metrics = TrainingMetricsCalculator.calculate_training_load(daily_tss)

        # After recent high load, ATL should be elevated
        assert metrics.atl > 50
        # CTL should reflect the recent increase but less responsive
        assert metrics.ctl > 20
        # With recent high training spike, ATL should be higher than CTL
        # leading to negative TSB (fatigued state)
        assert metrics.atl > metrics.ctl


class TestActivityModel:
    """Tests for Activity model properties."""

    def test_distance_conversions(self) -> None:
        """Test distance unit conversions."""
        activity = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            distance_meters=10000,  # 10km
        )

        assert activity.distance_km == 10.0
        assert abs(activity.distance_miles - 6.2137) < 0.01

    def test_pace_calculation(self) -> None:
        """Test pace per km calculation."""
        activity = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            distance_meters=5000,  # 5km
            moving_time_seconds=1500,  # 25 minutes
        )

        # 5 min/km pace
        assert activity.pace_per_km == "5:00"

    def test_pace_calculation_with_seconds(self) -> None:
        """Test pace calculation with non-zero seconds."""
        activity = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            distance_meters=5000,
            moving_time_seconds=1650,  # 27:30 for 5km = 5:30/km
        )

        assert activity.pace_per_km == "5:30"

    def test_pace_no_distance(self) -> None:
        """Test pace returns None with no distance."""
        activity = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            distance_meters=0,
            moving_time_seconds=1500,
        )

        assert activity.pace_per_km is None

    def test_duration_formatted_minutes(self) -> None:
        """Test duration formatting for short activities."""
        activity = Activity(
            external_id="1",
            name="Run",
            activity_type="Run",
            start_date=datetime.now(UTC),
            moving_time_seconds=1830,  # 30:30
        )

        assert activity.duration_formatted == "30:30"

    def test_duration_formatted_hours(self) -> None:
        """Test duration formatting for long activities."""
        activity = Activity(
            external_id="1",
            name="Ride",
            activity_type="Ride",
            start_date=datetime.now(UTC),
            moving_time_seconds=7323,  # 2:02:03
        )

        assert activity.duration_formatted == "2:02:03"


class TestActivityMetricsModel:
    """Tests for ActivityMetrics model properties."""

    def test_form_status_fresh(self) -> None:
        """Test form status when fresh."""
        metrics = ActivityMetrics(tsb=30)
        assert metrics.form_status == "Fresh"

    def test_form_status_recovered(self) -> None:
        """Test form status when recovered."""
        metrics = ActivityMetrics(tsb=10)
        assert metrics.form_status == "Recovered"

    def test_form_status_neutral(self) -> None:
        """Test form status when neutral."""
        metrics = ActivityMetrics(tsb=0)
        assert metrics.form_status == "Neutral"

    def test_form_status_tired(self) -> None:
        """Test form status when tired."""
        metrics = ActivityMetrics(tsb=-15)
        assert metrics.form_status == "Tired"

    def test_form_status_very_fatigued(self) -> None:
        """Test form status when very fatigued."""
        metrics = ActivityMetrics(tsb=-40)
        assert metrics.form_status == "Very Fatigued"

    def test_injury_risk_high(self) -> None:
        """Test injury risk when ATL much higher than CTL."""
        metrics = ActivityMetrics(atl=100, ctl=50)
        assert metrics.injury_risk == "High"

    def test_injury_risk_moderate(self) -> None:
        """Test injury risk when ATL moderately higher than CTL."""
        metrics = ActivityMetrics(atl=70, ctl=50)
        assert metrics.injury_risk == "Moderate"

    def test_injury_risk_low(self) -> None:
        """Test injury risk when ATL slightly higher than CTL."""
        metrics = ActivityMetrics(atl=60, ctl=50)
        assert metrics.injury_risk == "Low"

    def test_injury_risk_very_low(self) -> None:
        """Test injury risk when well balanced."""
        metrics = ActivityMetrics(atl=50, ctl=50)
        assert metrics.injury_risk == "Very Low"

    def test_injury_risk_unknown(self) -> None:
        """Test injury risk when no CTL data."""
        metrics = ActivityMetrics(atl=50, ctl=0)
        assert metrics.injury_risk == "Unknown"
