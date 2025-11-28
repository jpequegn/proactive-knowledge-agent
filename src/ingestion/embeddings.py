"""Embedding generation service for semantic search."""

import asyncio
from collections.abc import Sequence

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()

# Default embedding dimensions for text-embedding-3-small
EMBEDDING_DIMENSIONS = 1536


class EmbeddingService:
    """Generate embeddings using OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    async def generate(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIMENSIONS

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text.strip()[:8000],  # Limit to ~8000 chars
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise

    async def generate_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Filter and clean texts
        cleaned_texts = []
        indices_map: dict[int, int] = {}  # Map cleaned index to original index

        for i, text in enumerate(texts):
            if text and text.strip():
                indices_map[len(cleaned_texts)] = i
                cleaned_texts.append(text.strip()[:8000])

        if not cleaned_texts:
            return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

        # Process in batches
        all_embeddings: list[tuple[int, list[float]]] = []

        for batch_start in range(0, len(cleaned_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(cleaned_texts))
            batch = cleaned_texts[batch_start:batch_end]

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                for j, embedding_data in enumerate(response.data):
                    original_idx = indices_map[batch_start + j]
                    all_embeddings.append((original_idx, embedding_data.embedding))

                logger.debug(
                    "Batch embeddings generated",
                    batch_size=len(batch),
                    batch_start=batch_start,
                )

                # Small delay between batches to avoid rate limiting
                if batch_end < len(cleaned_texts):
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    "Failed to generate batch embeddings",
                    batch_start=batch_start,
                    error=str(e),
                )
                raise

        # Build result list with correct ordering
        results: list[list[float]] = [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
        for original_idx, embedding in all_embeddings:
            results[original_idx] = embedding

        return results

    async def cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimension")

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
