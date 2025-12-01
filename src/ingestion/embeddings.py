"""Embedding generation service for semantic search."""

import asyncio
import hashlib
import json
import random
from collections.abc import Sequence
from typing import TypeVar, Callable, Any

import structlog
from openai import AsyncOpenAI, RateLimitError

from src.cache import Cache

logger = structlog.get_logger()

# Default embedding dimensions for text-embedding-3-small
EMBEDDING_DIMENSIONS = 1536

T = TypeVar("T")


class EmbeddingService:
    """Generate embeddings using OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        cache: Cache | None = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.cache = cache

    def _generate_cache_key(self, text: str) -> str:
        """Generate a deterministic cache key for the text."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{self.model}:{text_hash}"

    async def _get_cached_embedding(self, text: str) -> list[float] | None:
        """Retrieve embedding from cache if available."""
        if not self.cache:
            return None

        try:
            key = self._generate_cache_key(text)
            cached_data = await self.cache.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning("Failed to retrieve from cache", error=str(e))
        
        return None

    async def _cache_embedding(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        if not self.cache:
            return

        try:
            key = self._generate_cache_key(text)
            await self.cache.set(key, json.dumps(embedding), expire=60 * 60 * 24 * 30) # 30 days
        except Exception as e:
            logger.warning("Failed to write to cache", error=str(e))

    async def _with_retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with exponential backoff for rate limits."""
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                logger.warning(
                    "Rate limit hit, retrying",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e)
                )
                await asyncio.sleep(delay)
            except Exception:
                raise

    async def generate(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIMENSIONS

        # Check cache first
        cached = await self._get_cached_embedding(text)
        if cached:
            return cached

        try:
            response = await self._with_retry(
                self.client.embeddings.create,
                model=self.model,
                input=text.strip()[:8000],  # Limit to ~8000 chars
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            await self._cache_embedding(text, embedding)
            
            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise

    async def generate_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Initialize results with correct size
        results: list[list[float] | None] = [None] * len(texts)
        
        # Identify indices needing processing
        to_process_indices: list[int] = []
        to_process_texts: list[str] = []

        # Check cache for all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = [0.0] * EMBEDDING_DIMENSIONS
                continue
            
            cached = await self._get_cached_embedding(text)
            if cached:
                results[i] = cached
            else:
                to_process_indices.append(i)
                to_process_texts.append(text.strip()[:8000])

        if not to_process_texts:
            # All were empty or cached
            return [r if r is not None else [0.0] * EMBEDDING_DIMENSIONS for r in results]

        # Process remaining texts in batches
        for batch_start in range(0, len(to_process_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(to_process_texts))
            batch = to_process_texts[batch_start:batch_end]
            batch_indices = to_process_indices[batch_start:batch_end]

            try:
                response = await self._with_retry(
                    self.client.embeddings.create,
                    model=self.model,
                    input=batch,
                )

                for j, embedding_data in enumerate(response.data):
                    original_idx = batch_indices[j]
                    embedding = embedding_data.embedding
                    results[original_idx] = embedding
                    
                    # Cache the new embedding (fire and forget essentially, but we await it)
                    await self._cache_embedding(texts[original_idx], embedding)

                logger.debug(
                    "Batch embeddings generated",
                    batch_size=len(batch),
                    batch_start=batch_start,
                )

            except Exception as e:
                logger.error(
                    "Failed to generate batch embeddings",
                    batch_start=batch_start,
                    error=str(e),
                )
                raise

        # Ensure no Nones remain (should handle failures by raising, but for type safety)
        final_results: list[list[float]] = []
        for r in results:
            if r is None:
                # This should theoretically not happen if we raise on error
                final_results.append([0.0] * EMBEDDING_DIMENSIONS) 
            else:
                final_results.append(r)

        return final_results

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