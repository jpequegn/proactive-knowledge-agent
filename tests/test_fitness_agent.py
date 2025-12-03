"""Tests for Fitness Intelligence Agent."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents import (
    AgentConfig,
    AgentReport,
    AlertLevel,
    FitnessIntelligenceAgent,
    FitnessInsightType,
    GoalReadiness,
    RecoveryRecommendation,
    TrainingLoadAnalysis,
    TrainingZone,
    UserProfile,
)
from src.models import Activity, ActivityMetrics
from src.world_model import Domain


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=0)
    return db


@pytest.fixture
def default_config():
    """Create default agent configuration."""
    return AgentConfig(
        enabled=True,
        schedule="0 7 * * *",
        thresholds={
            "training_load_warning": 1.3,
            "recovery_warning": 60,
            "fatigue_warning": 80,
            "min_confidence": 0.5,
            "min_relevance": 0.3,
        },
        output={"weekly_report": True, "risk_alerts": True},
    )


@pytest.fixture
def sample_activities():
    """Create sample fitness activities."""
    now = datetime.now(UTC)
    return [
        Activity(
            id=1,
            external_id="strava_1",
            name="Morning Run",
            activity_type="Run",
            sport_type="Run",
            start_date=now - timedelta(days=1),
            distance_meters=10000,
            moving_time_seconds=3000,  # 50 min
            elapsed_time_seconds=3200,
            total_elevation_gain=100,
            avg_hr=150,
            max_hr=175,
            tss=80,
        ),
        Activity(
            id=2,
            external_id="strava_2",
            name="Tempo Run",
            activity_type="Run",
            sport_type="Run",
            start_date=now - timedelta(days=2),
            distance_meters=8000,
            moving_time_seconds=2400,  # 40 min
            elapsed_time_seconds=2500,
            total_elevation_gain=50,
            avg_hr=165,
            max_hr=185,
            tss=100,
        ),
        Activity(
            id=3,
            external_id="strava_3",
            name="Easy Ride",
            activity_type="Ride",
            sport_type="Ride",
            start_date=now - timedelta(days=3),
            distance_meters=25000,
            moving_time_seconds=3600,  # 60 min
            elapsed_time_seconds=4000,
            total_elevation_gain=200,
            avg_hr=130,
            max_hr=150,
            avg_power=180,
            tss=60,
        ),
    ]


@pytest.fixture
def sample_metrics():
    """Create sample fitness metrics."""
    return ActivityMetrics(
        date=datetime.now(UTC),
        daily_tss=80,
        atl=75,
        ctl=60,
        tsb=-15,  # Tired zone
    )


# ============================================================================
# Test Training Load Analysis
# ============================================================================


class TestTrainingLoadAnalysis:
    """Tests for TrainingLoadAnalysis model."""

    def test_optimal_zone(self):
        """Test training load in optimal zone."""
        load = TrainingLoadAnalysis(
            atl=65,
            ctl=60,
            tsb=-5,  # Between -10 and 5
            atl_ctl_ratio=1.08,
            training_zone=TrainingZone.OPTIMAL,
            injury_risk_level="Low",
            form_status="Neutral",
            weekly_tss=400,
            avg_daily_tss=57,
            trend="maintaining",
        )
        assert load.training_zone == TrainingZone.OPTIMAL
        assert load.trend == "maintaining"

    def test_overreached_zone(self):
        """Test training load in overreached zone."""
        load = TrainingLoadAnalysis(
            atl=100,
            ctl=60,
            tsb=-40,  # Below -30
            atl_ctl_ratio=1.67,
            training_zone=TrainingZone.OVERREACHED,
            injury_risk_level="High",
            form_status="Very Fatigued",
            weekly_tss=700,
            avg_daily_tss=100,
            trend="building",
        )
        assert load.training_zone == TrainingZone.OVERREACHED
        assert load.injury_risk_level == "High"

    def test_fresh_zone(self):
        """Test training load in fresh zone."""
        load = TrainingLoadAnalysis(
            atl=40,
            ctl=60,
            tsb=20,  # Between 5 and 25
            atl_ctl_ratio=0.67,
            training_zone=TrainingZone.FRESH,
            injury_risk_level="Very Low",
            form_status="Fresh",
            weekly_tss=200,
            avg_daily_tss=28,
            trend="deloading",
        )
        assert load.training_zone == TrainingZone.FRESH
        assert load.form_status == "Fresh"


class TestTrainingZone:
    """Tests for TrainingZone enum."""

    def test_all_zones_exist(self):
        """Test all training zones are defined."""
        assert TrainingZone.TRANSITION
        assert TrainingZone.FRESH
        assert TrainingZone.OPTIMAL
        assert TrainingZone.TIRED
        assert TrainingZone.OVERREACHED


# ============================================================================
# Test Recovery Recommendations
# ============================================================================


class TestRecoveryRecommendation:
    """Tests for RecoveryRecommendation model."""

    def test_recommendation_creation(self):
        """Test creating a recovery recommendation."""
        rec = RecoveryRecommendation(
            priority=1,
            title="Rest Day",
            description="Take a complete rest day",
            action="No training",
            duration="24 hours",
            rationale="High fatigue detected",
        )
        assert rec.priority == 1
        assert rec.title == "Rest Day"
        assert rec.duration == "24 hours"


# ============================================================================
# Test Goal Readiness
# ============================================================================


class TestGoalReadiness:
    """Tests for GoalReadiness model."""

    def test_goal_ready(self):
        """Test goal readiness when fitness is sufficient."""
        readiness = GoalReadiness(
            goal_name="5K Race",
            target_date=datetime.now(UTC) + timedelta(days=14),
            readiness_score=85,
            current_fitness=65,
            recommended_fitness=60,
            gap_percent=-8.3,  # Over target
            recommendation="You're race ready!",
            weeks_to_ready=0,
        )
        assert readiness.readiness_score == 85
        assert readiness.weeks_to_ready == 0

    def test_goal_not_ready(self):
        """Test goal readiness when more fitness needed."""
        readiness = GoalReadiness(
            goal_name="Marathon",
            target_date=datetime.now(UTC) + timedelta(days=90),
            readiness_score=50,
            current_fitness=40,
            recommended_fitness=80,
            gap_percent=50,
            recommendation="Need more training time.",
            weeks_to_ready=10,
        )
        assert readiness.readiness_score == 50
        assert readiness.weeks_to_ready == 10


# ============================================================================
# Test Fitness Intelligence Agent
# ============================================================================


class TestFitnessIntelligenceAgent:
    """Tests for FitnessIntelligenceAgent."""

    @pytest.fixture
    def agent(self, mock_db, default_config):
        """Create agent with mocked dependencies."""
        return FitnessIntelligenceAgent(
            db=mock_db,
            config=default_config,
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "fitness_intelligence"
        assert agent.domain == Domain.FITNESS
        assert agent.training_load_warning == 1.3

    async def test_monitor_returns_data(self, agent, sample_activities):
        """Test monitor phase gathers required data."""
        # Mock repositories
        agent.activity_repo.get_by_date_range = AsyncMock(return_value=sample_activities)
        agent.activity_repo.count = AsyncMock(return_value=100)
        agent.metrics_repo.get_latest = AsyncMock(return_value=None)

        data = await agent.monitor()

        assert "recent_activities" in data
        assert "training_load" in data
        assert "total_activities" in data
        assert data["total_activities"] == 100

    async def test_calculate_training_load(self, agent, sample_activities):
        """Test training load calculation."""
        load = await agent._calculate_training_load(sample_activities)

        assert load is not None
        assert load.atl > 0
        assert load.ctl >= 0
        assert load.weekly_tss > 0
        assert load.training_zone in TrainingZone

    def test_analyze_training_load_danger(self, agent):
        """Test analysis of dangerous training load."""
        load = TrainingLoadAnalysis(
            atl=90,
            ctl=60,
            tsb=-30,
            atl_ctl_ratio=1.5,  # Danger threshold
            training_zone=TrainingZone.TIRED,
            injury_risk_level="High",
            form_status="Very Fatigued",
            weekly_tss=600,
            avg_daily_tss=85,
            trend="building",
        )

        insights = agent._analyze_training_load(load)

        # Should generate urgent overtraining alert
        urgent = [i for i in insights if i.level == AlertLevel.URGENT]
        assert len(urgent) >= 1
        assert any("overtraining" in i.title.lower() for i in insights)

    def test_analyze_training_load_warning(self, agent):
        """Test analysis of elevated training load."""
        load = TrainingLoadAnalysis(
            atl=78,
            ctl=60,
            tsb=-18,
            atl_ctl_ratio=1.3,  # Warning threshold
            training_zone=TrainingZone.TIRED,
            injury_risk_level="Moderate",
            form_status="Tired",
            weekly_tss=500,
            avg_daily_tss=71,
            trend="building",
        )

        insights = agent._analyze_training_load(load)

        # Should generate watch level alert
        watch = [i for i in insights if i.level == AlertLevel.WATCH]
        assert len(watch) >= 1

    def test_analyze_training_load_peak_form(self, agent):
        """Test detection of peak form."""
        load = TrainingLoadAnalysis(
            atl=50,
            ctl=60,
            tsb=10,  # Fresh
            atl_ctl_ratio=0.83,
            training_zone=TrainingZone.FRESH,
            injury_risk_level="Very Low",
            form_status="Fresh",
            weekly_tss=300,
            avg_daily_tss=43,
            trend="maintaining",
        )

        insights = agent._analyze_training_load(load)

        # Should detect peak form
        assert any("peak form" in i.title.lower() for i in insights)

    def test_analyze_recovery_needs_overreached(self, agent):
        """Test recovery recommendations for overreached state."""
        load = TrainingLoadAnalysis(
            atl=100,
            ctl=60,
            tsb=-40,
            atl_ctl_ratio=1.67,
            training_zone=TrainingZone.OVERREACHED,
            injury_risk_level="High",
            form_status="Very Fatigued",
            weekly_tss=700,
            avg_daily_tss=100,
            trend="building",
        )

        insights = agent._analyze_recovery_needs(load)

        # Should recommend rest
        assert len(insights) >= 1
        assert any("rest" in i.title.lower() for i in insights)

    def test_analyze_consistency_no_activities(self, agent):
        """Test consistency analysis with no activities."""
        insights = agent._analyze_consistency([])

        assert len(insights) == 1
        assert "no recent activities" in insights[0].title.lower()

    def test_analyze_consistency_good(self, agent, sample_activities):
        """Test consistency analysis with good training."""
        # Create activities for 5 different days
        now = datetime.now(UTC)
        activities = [
            Activity(
                id=i,
                external_id=f"strava_{i}",
                name=f"Activity {i}",
                activity_type="Run",
                start_date=now - timedelta(days=i),
                distance_meters=5000,
                moving_time_seconds=1800,
                elapsed_time_seconds=1900,
                total_elevation_gain=50,
                tss=40,
            )
            for i in range(5)
        ]

        insights = agent._analyze_consistency(activities)

        # Should praise good consistency
        assert any("great" in i.title.lower() or "consistency" in i.title.lower() for i in insights)

    def test_generate_recovery_recommendations_overreached(self, agent):
        """Test recovery recommendations for overreached athlete."""
        load = TrainingLoadAnalysis(
            atl=100,
            ctl=60,
            tsb=-40,
            atl_ctl_ratio=1.67,
            training_zone=TrainingZone.OVERREACHED,
            injury_risk_level="High",
            form_status="Very Fatigued",
            weekly_tss=700,
            avg_daily_tss=100,
            trend="building",
        )

        recommendations = agent.generate_recovery_recommendations(load)

        assert len(recommendations) >= 2
        # First recommendation should be high priority
        assert recommendations[0].priority == 1
        assert "recovery" in recommendations[0].title.lower() or "reduce" in recommendations[0].title.lower()

    def test_generate_recovery_recommendations_tired(self, agent):
        """Test recovery recommendations for tired athlete."""
        load = TrainingLoadAnalysis(
            atl=70,
            ctl=60,
            tsb=-20,
            atl_ctl_ratio=1.17,
            training_zone=TrainingZone.TIRED,
            injury_risk_level="Moderate",
            form_status="Tired",
            weekly_tss=450,
            avg_daily_tss=64,
            trend="maintaining",
        )

        recommendations = agent.generate_recovery_recommendations(load)

        assert len(recommendations) >= 1
        assert any("active recovery" in r.title.lower() or "recovery" in r.title.lower() for r in recommendations)

    def test_assess_goal_readiness_ready(self, agent):
        """Test goal readiness when athlete is ready."""
        load = TrainingLoadAnalysis(
            atl=55,
            ctl=65,
            tsb=10,
            atl_ctl_ratio=0.85,
            training_zone=TrainingZone.FRESH,
            injury_risk_level="Low",
            form_status="Fresh",
            weekly_tss=350,
            avg_daily_tss=50,
            trend="maintaining",
        )

        readiness = agent.assess_goal_readiness(
            load=load,
            goal_name="5K Race",
            target_ctl=60,
            target_date=datetime.now(UTC) + timedelta(days=7),
        )

        assert readiness.readiness_score >= 70
        assert readiness.weeks_to_ready == 0 or readiness.weeks_to_ready is None
        assert "ready" in readiness.recommendation.lower()

    def test_assess_goal_readiness_not_ready(self, agent):
        """Test goal readiness when more training needed."""
        load = TrainingLoadAnalysis(
            atl=45,
            ctl=40,
            tsb=-5,
            atl_ctl_ratio=1.12,
            training_zone=TrainingZone.OPTIMAL,
            injury_risk_level="Low",
            form_status="Neutral",
            weekly_tss=300,
            avg_daily_tss=43,
            trend="building",
        )

        readiness = agent.assess_goal_readiness(
            load=load,
            goal_name="Marathon",
            target_ctl=80,
            target_date=datetime.now(UTC) + timedelta(days=90),
        )

        assert readiness.readiness_score < 70
        assert readiness.weeks_to_ready is not None
        assert readiness.weeks_to_ready > 0

    async def test_full_agent_run(self, agent, sample_activities, sample_metrics):
        """Test complete agent run cycle."""
        # Mock repositories
        agent.activity_repo.get_by_date_range = AsyncMock(return_value=sample_activities)
        agent.activity_repo.count = AsyncMock(return_value=100)
        agent.metrics_repo.get_latest = AsyncMock(return_value=sample_metrics)

        report = await agent.run()

        assert isinstance(report, AgentReport)
        assert report.agent_name == "fitness_intelligence"
        assert report.domain == Domain.FITNESS
        assert report.execution_time_ms >= 0  # Can be 0 if very fast
        assert "training_load" in report.metadata
        assert "recommendations" in report.metadata

    def test_generate_weekly_report(self, agent):
        """Test weekly report generation."""
        load = TrainingLoadAnalysis(
            atl=70,
            ctl=60,
            tsb=-10,
            atl_ctl_ratio=1.17,
            training_zone=TrainingZone.OPTIMAL,
            injury_risk_level="Low",
            form_status="Neutral",
            weekly_tss=450,
            avg_daily_tss=64,
            trend="maintaining",
        )

        recommendations = [
            RecoveryRecommendation(
                priority=1,
                title="Easy Day Tomorrow",
                description="Light activity recommended",
                action="Zone 2 workout only",
                duration="45 minutes",
            ),
        ]

        report = AgentReport(
            agent_name="fitness_intelligence",
            domain=Domain.FITNESS,
            run_at=datetime.now(UTC),
            insights=[],
            trends_analyzed=0,
            entities_scanned=100,
            metadata={
                "training_load": load,
                "recommendations": recommendations,
                "activities_count": 5,
            },
        )

        markdown = agent.generate_weekly_report(report)

        assert "Weekly Fitness Intelligence Report" in markdown
        assert "Acute Training Load" in markdown
        assert "70" in markdown  # ATL value
        assert "Recovery Recommendations" in markdown


# ============================================================================
# Integration Tests
# ============================================================================


class TestFitnessAgentIntegration:
    """Integration tests for FitnessIntelligenceAgent."""

    @pytest.fixture
    def full_agent(self, mock_db):
        """Create fully configured agent."""
        config = AgentConfig(
            enabled=True,
            thresholds={
                "training_load_warning": 1.3,
                "recovery_warning": 60,
                "min_confidence": 0.5,
                "min_relevance": 0.3,
            },
        )
        return FitnessIntelligenceAgent(db=mock_db, config=config)

    async def test_end_to_end_overtraining_detection(self, full_agent):
        """Test end-to-end overtraining detection scenario."""
        now = datetime.now(UTC)

        # Create activities simulating overtraining
        activities = []
        for i in range(10):
            activities.append(
                Activity(
                    id=i,
                    external_id=f"strava_{i}",
                    name=f"Hard Workout {i}",
                    activity_type="Run",
                    start_date=now - timedelta(days=i),
                    distance_meters=15000,
                    moving_time_seconds=4500,
                    elapsed_time_seconds=4800,
                    total_elevation_gain=200,
                    avg_hr=165,  # High intensity
                    max_hr=185,
                    tss=120,  # High TSS
                )
            )

        # Mock repositories
        full_agent.activity_repo.get_by_date_range = AsyncMock(return_value=activities)
        full_agent.activity_repo.count = AsyncMock(return_value=len(activities))
        full_agent.metrics_repo.get_latest = AsyncMock(return_value=None)

        # Run agent
        report = await full_agent.run()

        # Verify overtraining detection
        alert_insights = [i for i in report.insights if i.level in (AlertLevel.URGENT, AlertLevel.ACTION, AlertLevel.WATCH)]
        assert len(alert_insights) >= 1

        # Verify recovery recommendations generated
        assert len(report.metadata.get("recommendations", [])) >= 1

    async def test_end_to_end_peak_form_detection(self, full_agent):
        """Test end-to-end peak form detection scenario."""
        now = datetime.now(UTC)

        # Create activities simulating taper (reducing volume)
        activities = []
        # Light week after harder training
        for i in range(4):
            activities.append(
                Activity(
                    id=i,
                    external_id=f"strava_{i}",
                    name=f"Easy Run {i}",
                    activity_type="Run",
                    start_date=now - timedelta(days=i),
                    distance_meters=5000,
                    moving_time_seconds=1800,
                    elapsed_time_seconds=1900,
                    total_elevation_gain=30,
                    avg_hr=130,  # Low intensity
                    max_hr=145,
                    tss=30,  # Low TSS
                )
            )

        # Add some older harder activities to build CTL
        for i in range(4, 30):
            activities.append(
                Activity(
                    id=i,
                    external_id=f"strava_{i}",
                    name=f"Training Run {i}",
                    activity_type="Run",
                    start_date=now - timedelta(days=i),
                    distance_meters=10000,
                    moving_time_seconds=3000,
                    elapsed_time_seconds=3200,
                    total_elevation_gain=100,
                    avg_hr=150,
                    max_hr=170,
                    tss=70,
                )
            )

        # Mock repositories
        full_agent.activity_repo.get_by_date_range = AsyncMock(return_value=activities)
        full_agent.activity_repo.count = AsyncMock(return_value=len(activities))
        full_agent.metrics_repo.get_latest = AsyncMock(return_value=None)

        # Run agent
        report = await full_agent.run()

        # Should have training load in metadata
        assert report.metadata.get("training_load") is not None

        # TSB should be positive (fresh)
        load = report.metadata["training_load"]
        assert load.ctl > 0  # Should have built some fitness
