"""Entity extraction pipeline using LLM for knowledge graph population."""

import json
from datetime import datetime

import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from src.world_model.entities import (
    Company,
    Concept,
    Domain,
    Entity,
    EntityType,
    Event,
    Metric,
    Person,
    Technology,
)
from src.world_model.relationships import Relationship, RelationshipType

logger = structlog.get_logger()


# ============================================================================
# Extraction Result Models
# ============================================================================


class ExtractedEntity(BaseModel):
    """An entity extracted from content."""

    name: str
    entity_type: str  # Will be mapped to EntityType
    domain: str = "general"  # Will be mapped to Domain
    description: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # Type-specific attributes (optional)
    attributes: dict = Field(default_factory=dict)


class ExtractedRelationship(BaseModel):
    """A relationship extracted from content."""

    source_entity: str  # Name of source entity
    relationship_type: str  # Will be mapped to RelationshipType
    target_entity: str  # Name of target entity
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Result of entity extraction from a piece of content."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    content_summary: str | None = None
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Metadata about the extraction
    source_type: str | None = None  # article, podcast, etc.
    source_id: int | None = None
    token_count: int = 0


# ============================================================================
# LLM Entity Extractor
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert entity extraction system. Extract structured entities \
and relationships from text.

## Entity Types

1. **technology**: Languages, frameworks, libraries, tools, platforms, APIs
   - Attributes: category, version, github_url

2. **company**: Businesses, startups, enterprises, organizations
   - Attributes: industry, company_type, stock_symbol

3. **person**: Named individuals (founders, researchers, executives, athletes)
   - Attributes: role, title, organization

4. **concept**: Abstract ideas, trends, methodologies, paradigms
   - Attributes: category (trend/methodology/pattern/paradigm)

5. **metric**: Quantitative measures, KPIs, scores, indicators
   - Attributes: unit, value, value_type

6. **event**: Announcements, releases, conferences, races, milestones
   - Attributes: event_type, event_date, location

## Domains
- tech: Technology-related
- fitness: Health, exercise, sports
- finance: Markets, investing
- general: Other

## Relationship Types
- used_by: Technology used by Company
- founded: Person founded Company
- announced: Company announced Event
- relates_to: Technology relates to Concept
- impacts: Event impacts Technology/Company/Metric
- works_at: Person works at Company
- created: Person created Technology
- competes_with: Company competes with Company

## Rules
1. Only extract explicitly mentioned entities
2. Assign confidence 0.0-1.0 based on clarity
3. Disambiguate ambiguous names (e.g., Apple company vs fruit)
4. Extract relationships only with clear evidence
5. Use canonical names (e.g., "OpenAI" not "open ai")
6. For metrics, capture value and unit if present

## Output
Return JSON with: entities, relationships, content_summary"""

EXTRACTION_USER_PROMPT = """\
Extract entities and relationships from:

---
{content}
---

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "name": "string",
      "entity_type": "technology|company|person|concept|metric|event",
      "domain": "tech|fitness|finance|general",
      "description": "string or null",
      "confidence": 0.0-1.0,
      "attributes": {{}}
    }}
  ],
  "relationships": [
    {{
      "source_entity": "entity name",
      "relationship_type": "used_by|founded|announced|relates_to|impacts|\
works_at|created|competes_with",
      "target_entity": "entity name",
      "confidence": 0.0-1.0
    }}
  ],
  "content_summary": "Brief summary"
}}"""


class EntityExtractor:
    """Extract entities from text using Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def extract(
        self,
        content: str,
        source_type: str | None = None,
        source_id: int | None = None,
        domain_hint: str | None = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from content.

        Args:
            content: The text content to extract from
            source_type: Type of content (article, podcast, etc.)
            source_id: ID of the source content
            domain_hint: Optional hint about the content domain

        Returns:
            ExtractionResult with extracted entities and relationships
        """
        if not content or not content.strip():
            return ExtractionResult(source_type=source_type, source_id=source_id)

        # Truncate very long content
        truncated_content = content[:15000]  # ~4k tokens approx

        # Add domain hint to prompt if provided
        prompt = EXTRACTION_USER_PROMPT.format(content=truncated_content)
        if domain_hint:
            prompt = f"Domain hint: Content is about {domain_hint}.\n\n{prompt}"

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response
            response_text = response.content[0].text

            # Parse JSON from response
            result_data = self._parse_json_response(response_text)

            # Convert to ExtractionResult
            result = ExtractionResult(
                entities=[
                    ExtractedEntity(**e) for e in result_data.get("entities", [])
                ],
                relationships=[
                    ExtractedRelationship(**r)
                    for r in result_data.get("relationships", [])
                ],
                content_summary=result_data.get("content_summary"),
                source_type=source_type,
                source_id=source_id,
                token_count=response.usage.input_tokens + response.usage.output_tokens,
            )

            logger.info(
                "Entities extracted",
                entity_count=len(result.entities),
                relationship_count=len(result.relationships),
                tokens=result.token_count,
            )

            return result

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return ExtractionResult(source_type=source_type, source_id=source_id)

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response", error=str(e))
            return {"entities": [], "relationships": []}


# ============================================================================
# Entity Conversion Utilities
# ============================================================================


def extracted_to_entity(
    extracted: ExtractedEntity,
    source: str | None = None,
    source_url: str | None = None,
) -> Entity:
    """Convert an ExtractedEntity to a proper Entity model."""
    # Map string type to EntityType enum
    try:
        entity_type = EntityType(extracted.entity_type.lower())
    except ValueError:
        entity_type = EntityType.CONCEPT  # Default fallback

    # Map string domain to Domain enum
    try:
        domain = Domain(extracted.domain.lower())
    except ValueError:
        domain = Domain.GENERAL

    # Base entity fields
    base_fields = {
        "name": extracted.name,
        "entity_type": entity_type,
        "domain": domain,
        "description": extracted.description,
        "confidence": extracted.confidence,
        "source": source,
        "source_url": source_url,
        "properties": extracted.attributes,
    }

    # Create appropriate entity subclass
    if entity_type == EntityType.TECHNOLOGY:
        return Technology(
            **base_fields,
            category=extracted.attributes.get("category"),
            version=extracted.attributes.get("version"),
            github_url=extracted.attributes.get("github_url"),
        )
    elif entity_type == EntityType.COMPANY:
        return Company(
            **base_fields,
            industry=extracted.attributes.get("industry"),
            company_type=extracted.attributes.get("company_type"),
            stock_symbol=extracted.attributes.get("stock_symbol"),
        )
    elif entity_type == EntityType.PERSON:
        return Person(
            **base_fields,
            role=extracted.attributes.get("role"),
            title=extracted.attributes.get("title"),
            organization=extracted.attributes.get("organization"),
        )
    elif entity_type == EntityType.CONCEPT:
        return Concept(
            **base_fields,
            category=extracted.attributes.get("category"),
        )
    elif entity_type == EntityType.METRIC:
        return Metric(
            **base_fields,
            unit=extracted.attributes.get("unit"),
            value=extracted.attributes.get("value"),
            value_type=extracted.attributes.get("value_type"),
        )
    elif entity_type == EntityType.EVENT:
        return Event(
            **base_fields,
            event_type=extracted.attributes.get("event_type"),
            location=extracted.attributes.get("location"),
        )
    else:
        return Entity(**base_fields)


def extracted_to_relationship(
    extracted: ExtractedRelationship,
    source_entity_id: int,
    target_entity_id: int,
    source: str | None = None,
) -> Relationship:
    """Convert an ExtractedRelationship to a Relationship model."""
    try:
        rel_type = RelationshipType(extracted.relationship_type.lower())
    except ValueError:
        rel_type = RelationshipType.RELATES_TO  # Default fallback

    return Relationship(
        relationship_type=rel_type,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        confidence=extracted.confidence,
        source=source,
    )
