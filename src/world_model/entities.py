"""Knowledge graph entity models for PKA world model."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    TECHNOLOGY = "technology"  # Frameworks, tools, languages, libraries
    COMPANY = "company"  # Startups, enterprises, organizations
    PERSON = "person"  # Founders, researchers, athletes
    CONCEPT = "concept"  # Trends, methodologies, patterns
    METRIC = "metric"  # Fitness numbers, financial indicators
    EVENT = "event"  # Announcements, releases, races


class Domain(str, Enum):
    """Knowledge domains for cross-domain queries."""

    TECH = "tech"
    FITNESS = "fitness"
    FINANCE = "finance"
    GENERAL = "general"


class Entity(BaseModel):
    """Base entity in the knowledge graph with temporal attributes."""

    id: int | None = None
    external_id: str | None = None  # ID from source system
    name: str
    entity_type: EntityType
    domain: Domain = Domain.GENERAL
    description: str | None = None

    # Source tracking
    source: str | None = None  # Where entity was first discovered
    source_url: str | None = None

    # Confidence and relevance
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    mention_count: int = Field(default=1, ge=0, description="Times mentioned")

    # Temporal attributes (required per issue spec)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata for flexible properties
    properties: dict = Field(default_factory=dict)


class Technology(Entity):
    """Technology entity: frameworks, tools, languages, libraries."""

    entity_type: EntityType = EntityType.TECHNOLOGY
    domain: Domain = Domain.TECH

    # Technology-specific attributes
    category: str | None = None  # e.g., "language", "framework", "library", "tool"
    version: str | None = None
    github_url: str | None = None
    documentation_url: str | None = None


class Company(Entity):
    """Company entity: startups, enterprises, organizations."""

    entity_type: EntityType = EntityType.COMPANY

    # Company-specific attributes
    industry: str | None = None
    company_type: str | None = None  # e.g., "startup", "enterprise", "nonprofit"
    stock_symbol: str | None = None
    website: str | None = None
    founded_year: int | None = None
    headquarters: str | None = None


class Person(Entity):
    """Person entity: founders, researchers, athletes."""

    entity_type: EntityType = EntityType.PERSON

    # Person-specific attributes
    role: str | None = None  # e.g., "founder", "researcher", "athlete", "engineer"
    title: str | None = None
    organization: str | None = None
    twitter_handle: str | None = None
    linkedin_url: str | None = None


class Concept(Entity):
    """Concept entity: trends, methodologies, patterns, ideas."""

    entity_type: EntityType = EntityType.CONCEPT

    # Concept-specific attributes
    category: str | None = None  # e.g., "trend", "methodology", "pattern", "paradigm"
    parent_concept: str | None = None  # For hierarchical concepts


class Metric(Entity):
    """Metric entity: fitness numbers, financial indicators, KPIs."""

    entity_type: EntityType = EntityType.METRIC

    # Metric-specific attributes
    unit: str | None = None  # e.g., "watts", "bpm", "USD", "%"
    value: float | None = None
    value_type: str | None = None  # e.g., "current", "average", "peak"
    target_value: float | None = None
    measurement_date: datetime | None = None


class Event(Entity):
    """Event entity: announcements, releases, milestones, races."""

    entity_type: EntityType = EntityType.EVENT

    # Event-specific attributes
    event_type: str | None = None  # e.g., "announcement", "release", "race", "ipo"
    event_date: datetime | None = None
    location: str | None = None
    significance: str | None = None  # e.g., "major", "minor"


# Type alias for any entity type
AnyEntity = Technology | Company | Person | Concept | Metric | Event


def create_entity(entity_type: EntityType, **kwargs) -> AnyEntity:
    """Factory function to create the appropriate entity type."""
    entity_classes = {
        EntityType.TECHNOLOGY: Technology,
        EntityType.COMPANY: Company,
        EntityType.PERSON: Person,
        EntityType.CONCEPT: Concept,
        EntityType.METRIC: Metric,
        EntityType.EVENT: Event,
    }
    entity_class = entity_classes.get(entity_type)
    if entity_class is None:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return entity_class(**kwargs)
