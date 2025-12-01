"""Knowledge graph relationship models for PKA world model."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    # Core relationships from issue spec
    USED_BY = "used_by"  # Technology -> Company
    FOUNDED = "founded"  # Person -> Company
    ANNOUNCED = "announced"  # Company -> Event
    RELATES_TO = "relates_to"  # Technology -> Concept
    IMPACTS = "impacts"  # Event -> Technology|Company|Metric

    # Additional relationships for cross-domain queries
    WORKS_AT = "works_at"  # Person -> Company
    CREATED = "created"  # Person -> Technology
    COMPETES_WITH = "competes_with"  # Company -> Company
    EMPLOYS = "employs"  # Company -> Person
    MEASURES = "measures"  # Metric -> Entity (what is measured)
    TRACKS = "tracks"  # Metric -> Activity/Event
    MENTIONS = "mentions"  # Article/Podcast -> Entity
    SIMILAR_TO = "similar_to"  # Entity -> Entity (semantic similarity)
    PART_OF = "part_of"  # Entity -> Entity (hierarchical)
    PRECEDES = "precedes"  # Event -> Event (temporal)
    FOLLOWS = "follows"  # Event -> Event (temporal)
    CORRELATES_WITH = "correlates_with"  # Cross-domain correlations


class Relationship(BaseModel):
    """A relationship between two entities in the knowledge graph."""

    id: int | None = None
    relationship_type: RelationshipType
    source_entity_id: int  # The entity where relationship originates
    target_entity_id: int  # The entity where relationship points to

    # Relationship strength and confidence
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relationship strength"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )

    # Source tracking
    source: str | None = None  # Where relationship was discovered
    source_url: str | None = None

    # Temporal attributes (required per issue spec)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime | None = None  # When relationship became valid
    valid_until: datetime | None = None  # When relationship ended (None = still valid)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata for flexible properties
    properties: dict = Field(default_factory=dict)


class EntityMention(BaseModel):
    """
    A mention of an entity in a content source.
    Tracks where entities appear for frequency analysis and source linking.
    """

    id: int | None = None
    entity_id: int
    content_type: str  # "article", "podcast", "activity", "market"
    content_id: int  # ID in the respective content table

    # Context of mention
    context: str | None = None  # Surrounding text snippet
    sentiment: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Sentiment score"
    )

    # Temporal
    mentioned_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EntityVersion(BaseModel):
    """
    Version history for entity changes.
    Enables temporal queries like "What changed since yesterday?"
    """

    id: int | None = None
    entity_id: int
    version: int = Field(default=1, ge=1)

    # Snapshot of entity state
    name: str
    description: str | None = None
    properties: dict = Field(default_factory=dict)

    # Change tracking
    change_type: str  # "created", "updated", "merged", "deleted"
    change_source: str | None = None
    changed_fields: list[str] = Field(default_factory=list)

    # Temporal
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None  # None = current version
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Valid relationship combinations based on issue spec
VALID_RELATIONSHIPS = {
    RelationshipType.USED_BY: [("technology", "company")],
    RelationshipType.FOUNDED: [("person", "company")],
    RelationshipType.ANNOUNCED: [("company", "event")],
    RelationshipType.RELATES_TO: [
        ("technology", "concept"),
        ("concept", "concept"),
    ],
    RelationshipType.IMPACTS: [
        ("event", "technology"),
        ("event", "company"),
        ("event", "metric"),
    ],
    RelationshipType.WORKS_AT: [("person", "company")],
    RelationshipType.CREATED: [("person", "technology")],
    RelationshipType.COMPETES_WITH: [("company", "company")],
    RelationshipType.EMPLOYS: [("company", "person")],
    RelationshipType.MEASURES: [
        ("metric", "technology"),
        ("metric", "company"),
        ("metric", "person"),
        ("metric", "event"),
    ],
    RelationshipType.SIMILAR_TO: [
        ("technology", "technology"),
        ("company", "company"),
        ("person", "person"),
        ("concept", "concept"),
    ],
    RelationshipType.PART_OF: [
        ("technology", "technology"),
        ("concept", "concept"),
        ("company", "company"),
    ],
    RelationshipType.PRECEDES: [("event", "event")],
    RelationshipType.FOLLOWS: [("event", "event")],
    RelationshipType.CORRELATES_WITH: [
        ("metric", "metric"),
        ("event", "metric"),
        ("technology", "metric"),
    ],
}


def validate_relationship(
    rel_type: RelationshipType,
    source_entity_type: str,
    target_entity_type: str,
) -> bool:
    """
    Validate if a relationship type is valid between two entity types.

    Args:
        rel_type: The relationship type
        source_entity_type: The source entity type (lowercase)
        target_entity_type: The target entity type (lowercase)

    Returns:
        True if the relationship is valid, False otherwise
    """
    valid_pairs = VALID_RELATIONSHIPS.get(rel_type, [])
    return (source_entity_type.lower(), target_entity_type.lower()) in valid_pairs
