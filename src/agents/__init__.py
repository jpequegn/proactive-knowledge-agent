"""Proactive intelligence agents for tech, fitness, and finance domains."""

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
from src.agents.fitness_agent import (
    FitnessIntelligenceAgent,
    FitnessInsightType,
    GoalReadiness,
    RecoveryRecommendation,
    TrainingLoadAnalysis,
    TrainingZone,
)
from src.agents.finance_agent import (
    FinanceIntelligenceAgent,
    FinanceInsightType,
    MarketRegime,
    MarketRegimeAnalysis,
    PortfolioRiskAssessment,
    RebalancingSignal,
    RiskLevel,
    SectorPerformance,
    VolatilityMetrics,
)
from src.agents.tech_agent import TechIntelligenceAgent

__all__ = [
    # Base classes and models
    "AgentConfig",
    "AgentReport",
    "AlertLevel",
    "BaseAgent",
    "Insight",
    "InsightType",
    "ProjectIdea",
    "UserProfile",
    # Tech Agent
    "TechIntelligenceAgent",
    # Finance Agent
    "FinanceIntelligenceAgent",
    "FinanceInsightType",
    "MarketRegime",
    "MarketRegimeAnalysis",
    "PortfolioRiskAssessment",
    "RebalancingSignal",
    "RiskLevel",
    "SectorPerformance",
    "VolatilityMetrics",
    # Fitness Agent
    "FitnessIntelligenceAgent",
    "FitnessInsightType",
    "GoalReadiness",
    "RecoveryRecommendation",
    "TrainingLoadAnalysis",
    "TrainingZone",
]
