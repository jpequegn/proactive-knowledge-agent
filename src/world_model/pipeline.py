"""Entity extraction pipeline with disambiguation and persistence."""

from datetime import datetime

import structlog

from src.database import Database
from src.ingestion.embeddings import EmbeddingService
from src.world_model.entities import Domain, Entity, EntityType
from src.world_model.extraction import (
    EntityExtractor,
    extracted_to_entity,
    extracted_to_relationship,
)
from src.world_model.relationships import EntityMention
from src.world_model.repository import (
    EntityRepository,
    MentionRepository,
    RelationshipRepository,
)

logger = structlog.get_logger()


# ============================================================================
# Pipeline Result Models
# ============================================================================


class PipelineStats:
    """Statistics from a pipeline run."""

    def __init__(self):
        self.entities_extracted: int = 0
        self.entities_new: int = 0
        self.entities_updated: int = 0
        self.entities_disambiguated: int = 0
        self.relationships_extracted: int = 0
        self.relationships_new: int = 0
        self.mentions_created: int = 0
        self.tokens_used: int = 0
        self.errors: list[str] = []

    def to_dict(self) -> dict:
        return {
            "entities_extracted": self.entities_extracted,
            "entities_new": self.entities_new,
            "entities_updated": self.entities_updated,
            "entities_disambiguated": self.entities_disambiguated,
            "relationships_extracted": self.relationships_extracted,
            "relationships_new": self.relationships_new,
            "mentions_created": self.mentions_created,
            "tokens_used": self.tokens_used,
            "errors": self.errors,
        }


# ============================================================================
# Entity Disambiguation
# ============================================================================


class EntityDisambiguator:
    """
    Disambiguate entities to prevent duplicates and link to existing entities.

    Uses a combination of:
    1. Exact name + type matching
    2. Semantic similarity via embeddings
    3. Context-aware disambiguation for ambiguous names
    """

    # Known ambiguous entities and their disambiguations
    KNOWN_DISAMBIGUATIONS = {
        # Company vs other meanings
        "apple": {
            "hints": ["iphone", "mac", "ios", "tim cook", "cupertino", "tech"],
            "default_type": EntityType.COMPANY,
            "domain": Domain.TECH,
        },
        "amazon": {
            "hints": ["aws", "bezos", "ecommerce", "cloud", "prime"],
            "default_type": EntityType.COMPANY,
            "domain": Domain.TECH,
        },
        "meta": {
            "hints": ["facebook", "zuckerberg", "instagram", "whatsapp"],
            "default_type": EntityType.COMPANY,
            "domain": Domain.TECH,
        },
        "oracle": {
            "hints": ["database", "java", "cloud", "ellison"],
            "default_type": EntityType.COMPANY,
            "domain": Domain.TECH,
        },
        # Technology vs concepts
        "python": {
            "hints": ["programming", "code", "pip", "django", "flask"],
            "default_type": EntityType.TECHNOLOGY,
            "domain": Domain.TECH,
        },
        "rust": {
            "hints": ["programming", "cargo", "memory safe", "systems"],
            "default_type": EntityType.TECHNOLOGY,
            "domain": Domain.TECH,
        },
        "go": {
            "hints": ["golang", "goroutine", "google", "programming"],
            "default_type": EntityType.TECHNOLOGY,
            "domain": Domain.TECH,
        },
    }

    # Similarity threshold for considering entities as duplicates
    SIMILARITY_THRESHOLD = 0.92

    def __init__(
        self,
        entity_repo: EntityRepository,
        embedding_service: EmbeddingService | None = None,
    ):
        self.entity_repo = entity_repo
        self.embedding_service = embedding_service

    async def disambiguate(
        self,
        entity: Entity,
        context: str | None = None,
    ) -> tuple[Entity, int | None, bool]:
        """
        Disambiguate an entity against existing entities.

        Args:
            entity: The entity to disambiguate
            context: Optional context text for disambiguation hints

        Returns:
            Tuple of (entity, existing_id, was_disambiguated):
            - entity: The entity (possibly modified)
            - existing_id: ID of existing entity if match found, None otherwise
            - was_disambiguated: Whether disambiguation logic was applied
        """
        name_lower = entity.name.lower()
        was_disambiguated = False

        # Check for known ambiguous names
        if name_lower in self.KNOWN_DISAMBIGUATIONS:
            disambiguation = self.KNOWN_DISAMBIGUATIONS[name_lower]
            if context and self._context_matches_hints(
                context, disambiguation["hints"]
            ):
                # Apply disambiguation
                if entity.domain == Domain.GENERAL:
                    entity.domain = disambiguation["domain"]
                was_disambiguated = True
                logger.debug(
                    "Applied known disambiguation",
                    entity=entity.name,
                    domain=entity.domain.value,
                )

        # Try to find existing entity by exact name + type match
        existing = await self.entity_repo.get_by_name_and_type(
            entity.name, entity.entity_type
        )
        if existing and existing.id:
            return entity, existing.id, was_disambiguated

        # Try semantic similarity if we have embeddings
        if self.embedding_service and entity.description:
            similar_entities = await self._find_similar_entities(entity)
            if similar_entities:
                best_match, similarity = similar_entities[0]
                if (
                    similarity >= self.SIMILARITY_THRESHOLD
                    and best_match.entity_type == entity.entity_type
                    and best_match.id
                ):
                    logger.info(
                        "Found similar existing entity",
                        new_entity=entity.name,
                        existing_entity=best_match.name,
                        similarity=similarity,
                    )
                    return entity, best_match.id, True

        # No existing match found
        return entity, None, was_disambiguated

    async def _find_similar_entities(
        self,
        entity: Entity,
    ) -> list[tuple[Entity, float]]:
        """Find semantically similar entities using embeddings."""
        if not self.embedding_service or not entity.description:
            return []

        try:
            # Generate embedding for the entity description
            text = f"{entity.name}: {entity.description}"
            embedding = await self.embedding_service.generate(text)

            # Search for similar entities
            similar = await self.entity_repo.find_similar(
                embedding=embedding,
                limit=5,
                threshold=0.8,
                entity_type=entity.entity_type,
            )
            return similar

        except Exception as e:
            logger.warning("Failed to find similar entities", error=str(e))
            return []

    def _context_matches_hints(self, context: str, hints: list[str]) -> bool:
        """Check if context contains any of the disambiguation hints."""
        context_lower = context.lower()
        return any(hint in context_lower for hint in hints)


# ============================================================================
# Extraction Pipeline
# ============================================================================


class ExtractionPipeline:
    """
    Full pipeline for extracting, disambiguating, and persisting entities.

    Workflow:
    1. Extract entities and relationships from content using LLM
    2. Disambiguate entities against existing knowledge graph
    3. Create or update entities in the repository
    4. Create relationships between entities
    5. Record entity mentions for tracking
    """

    def __init__(
        self,
        db: Database,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.db = db
        self.entity_repo = EntityRepository(db)
        self.relationship_repo = RelationshipRepository(db)
        self.mention_repo = MentionRepository(db)

        self.extractor = EntityExtractor(api_key=anthropic_api_key, model=model)

        # Initialize embedding service for disambiguation if OpenAI key provided
        if openai_api_key:
            self.embedding_service = EmbeddingService(api_key=openai_api_key)
        else:
            self.embedding_service = None

        self.disambiguator = EntityDisambiguator(
            entity_repo=self.entity_repo,
            embedding_service=self.embedding_service,
        )

    async def process_content(
        self,
        content: str,
        source_type: str,
        source_id: int,
        source_url: str | None = None,
        domain_hint: str | None = None,
    ) -> PipelineStats:
        """
        Process content through the full extraction pipeline.

        Args:
            content: Text content to process
            source_type: Type of source (article, podcast, etc.)
            source_id: ID of the source in its table
            source_url: Optional URL of the source
            domain_hint: Optional domain hint for extraction

        Returns:
            PipelineStats with extraction statistics
        """
        stats = PipelineStats()

        # Step 1: Extract entities and relationships
        extraction_result = await self.extractor.extract(
            content=content,
            source_type=source_type,
            source_id=source_id,
            domain_hint=domain_hint,
        )
        stats.entities_extracted = len(extraction_result.entities)
        stats.relationships_extracted = len(extraction_result.relationships)
        stats.tokens_used = extraction_result.token_count

        if not extraction_result.entities:
            logger.info("No entities extracted from content")
            return stats

        # Step 2: Process entities with disambiguation
        entity_id_map: dict[str, int] = {}  # Map entity names to IDs

        for extracted in extraction_result.entities:
            try:
                # Convert to Entity model
                entity = extracted_to_entity(
                    extracted,
                    source=source_type,
                    source_url=source_url,
                )

                # Disambiguate
                entity, existing_id, was_disambiguated = (
                    await self.disambiguator.disambiguate(entity, context=content)
                )

                if was_disambiguated:
                    stats.entities_disambiguated += 1

                if existing_id:
                    # Update existing entity (increment mention count)
                    await self.entity_repo.increment_mention_count(existing_id)
                    entity_id_map[extracted.name] = existing_id
                    stats.entities_updated += 1
                else:
                    # Create new entity
                    entity_id, is_new = await self.entity_repo.upsert(entity)
                    entity_id_map[extracted.name] = entity_id
                    if is_new:
                        stats.entities_new += 1
                    else:
                        stats.entities_updated += 1

                # Record mention
                mention = EntityMention(
                    entity_id=entity_id_map[extracted.name],
                    content_type=source_type,
                    content_id=source_id,
                    mentioned_at=datetime.utcnow(),
                )
                await self.mention_repo.upsert(mention)
                stats.mentions_created += 1

            except Exception as e:
                error_msg = f"Failed to process entity {extracted.name}: {e}"
                stats.errors.append(error_msg)
                logger.error(error_msg)

        # Step 3: Process relationships
        for extracted_rel in extraction_result.relationships:
            try:
                source_id_rel = entity_id_map.get(extracted_rel.source_entity)
                target_id_rel = entity_id_map.get(extracted_rel.target_entity)

                if not source_id_rel or not target_id_rel:
                    logger.debug(
                        "Skipping relationship - entities not found",
                        source=extracted_rel.source_entity,
                        target=extracted_rel.target_entity,
                    )
                    continue

                relationship = extracted_to_relationship(
                    extracted_rel,
                    source_entity_id=source_id_rel,
                    target_entity_id=target_id_rel,
                    source=source_type,
                )

                _, is_new = await self.relationship_repo.upsert(relationship)
                if is_new:
                    stats.relationships_new += 1

            except Exception as e:
                error_msg = f"Failed to process relationship: {e}"
                stats.errors.append(error_msg)
                logger.error(error_msg)

        logger.info(
            "Pipeline completed",
            **stats.to_dict(),
        )

        return stats

    async def process_article(
        self,
        article_id: int,
        title: str,
        content: str,
        source: str,
        url: str | None = None,
        category: str | None = None,
    ) -> PipelineStats:
        """Convenience method for processing an article."""
        # Combine title and content for extraction
        full_content = f"Title: {title}\n\n{content}" if content else title

        # Map category to domain hint
        domain_hint = None
        if category:
            category_lower = category.lower()
            if any(t in category_lower for t in ["tech", "ai", "software", "code"]):
                domain_hint = "tech"
            elif any(
                t in category_lower for t in ["fitness", "health", "sports", "training"]
            ):
                domain_hint = "fitness"
            elif any(
                t in category_lower for t in ["finance", "market", "invest", "economy"]
            ):
                domain_hint = "finance"

        return await self.process_content(
            content=full_content,
            source_type="article",
            source_id=article_id,
            source_url=url,
            domain_hint=domain_hint,
        )

    async def process_podcast(
        self,
        episode_id: int,
        title: str,
        summary: str | None,
        podcast_name: str,
        url: str | None = None,
    ) -> PipelineStats:
        """Convenience method for processing a podcast episode."""
        # Combine available text
        parts = [f"Podcast: {podcast_name}", f"Episode: {title}"]
        if summary:
            parts.append(f"Summary: {summary}")
        full_content = "\n\n".join(parts)

        return await self.process_content(
            content=full_content,
            source_type="podcast",
            source_id=episode_id,
            source_url=url,
            domain_hint="tech",  # Podcasts in this system are typically tech
        )
