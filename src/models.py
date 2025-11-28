"""Data models for PKA."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field
import hashlib


class Article(BaseModel):
    """Represents an article from an RSS feed."""

    id: int | None = None
    title: str
    url: str
    summary: str | None = None
    content: str | None = None
    published: datetime | None = None
    source: str
    category: str
    author: str | None = None

    @computed_field
    @property
    def url_hash(self) -> str:
        """Generate SHA256 hash of URL for deduplication."""
        return hashlib.sha256(self.url.encode()).hexdigest()

    @computed_field
    @property
    def text_for_embedding(self) -> str:
        """Generate text content for embedding generation."""
        parts = [self.title]
        if self.summary:
            parts.append(self.summary)
        if self.content:
            # Limit content length for embedding
            parts.append(self.content[:2000])
        return " ".join(parts)


class ArticleWithEmbedding(Article):
    """Article with embedding vector."""

    embedding: list[float] | None = None


class FetchResult(BaseModel):
    """Result of fetching a feed."""

    feed_name: str
    feed_url: str
    success: bool
    articles_found: int = 0
    articles_new: int = 0
    error: str | None = None
    duration_ms: float = 0


class SyncReport(BaseModel):
    """Report of a sync operation."""

    started_at: datetime
    completed_at: datetime | None = None
    feeds_processed: int = 0
    articles_found: int = 0
    articles_new: int = 0
    articles_updated: int = 0
    errors: list[str] = Field(default_factory=list)
    results: list[FetchResult] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.completed_at is None:
            return 0
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate success rate of feed fetches."""
        if self.feeds_processed == 0:
            return 0
        successful = sum(1 for r in self.results if r.success)
        return successful / self.feeds_processed
