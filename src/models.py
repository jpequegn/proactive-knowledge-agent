"""Data models for PKA."""

import hashlib
from datetime import datetime

from pydantic import BaseModel, Field, computed_field


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


# ============================================================================
# Fitness Models
# ============================================================================


class Activity(BaseModel):
    """Represents a fitness activity from Strava."""

    id: int | None = None
    external_id: str
    name: str
    activity_type: str
    sport_type: str | None = None
    start_date: datetime
    distance_meters: float = 0
    moving_time_seconds: int = 0
    elapsed_time_seconds: int = 0
    total_elevation_gain: float = 0
    avg_speed: float | None = None
    max_speed: float | None = None
    avg_hr: int | None = None
    max_hr: int | None = None
    avg_power: int | None = None
    max_power: int | None = None
    calories: int | None = None
    suffer_score: int | None = None
    tss: float | None = None  # Training Stress Score

    @property
    def distance_km(self) -> float:
        """Get distance in kilometers."""
        return self.distance_meters / 1000

    @property
    def distance_miles(self) -> float:
        """Get distance in miles."""
        return self.distance_meters / 1609.344

    @property
    def pace_per_km(self) -> str | None:
        """Get pace in min/km format for running activities."""
        if self.distance_meters == 0 or self.moving_time_seconds == 0:
            return None
        pace_seconds = self.moving_time_seconds / (self.distance_meters / 1000)
        minutes = int(pace_seconds // 60)
        seconds = int(pace_seconds % 60)
        return f"{minutes}:{seconds:02d}"

    @property
    def duration_formatted(self) -> str:
        """Get duration in HH:MM:SS format."""
        hours = self.moving_time_seconds // 3600
        minutes = (self.moving_time_seconds % 3600) // 60
        seconds = self.moving_time_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


class ActivityMetrics(BaseModel):
    """Training load metrics calculated from activities."""

    date: datetime | None = None
    atl: float = 0  # Acute Training Load (7-day)
    ctl: float = 0  # Chronic Training Load (42-day)
    tsb: float = 0  # Training Stress Balance (form)
    daily_tss: float = 0  # Total TSS for the day

    @property
    def form_status(self) -> str:
        """Get training form status based on TSB."""
        if self.tsb > 25:
            return "Fresh"
        elif self.tsb > 5:
            return "Recovered"
        elif self.tsb > -10:
            return "Neutral"
        elif self.tsb > -30:
            return "Tired"
        else:
            return "Very Fatigued"

    @property
    def injury_risk(self) -> str:
        """Estimate injury risk based on ATL/CTL ratio."""
        if self.ctl == 0:
            return "Unknown"
        ratio = self.atl / self.ctl
        if ratio > 1.5:
            return "High"
        elif ratio > 1.3:
            return "Moderate"
        elif ratio > 1.1:
            return "Low"
        else:
            return "Very Low"


class FitnessSyncResult(BaseModel):
    """Result of syncing fitness data."""

    started_at: datetime
    completed_at: datetime | None = None
    success: bool = True
    activities_fetched: int = 0
    activities_new: int = 0
    activities_updated: int = 0
    errors: list[str] = Field(default_factory=list)
    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.completed_at is None:
            return 0
        return (self.completed_at - self.started_at).total_seconds()


# ============================================================================
# Market Models
# ============================================================================


class MarketOHLCV(BaseModel):
    """Represents a single day of OHLCV data for a symbol."""

    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float | None = None

    @computed_field
    @property
    def change_percent(self) -> float | None:
        """Calculate percentage change from open to close."""
        if self.open == 0:
            return None
        return ((self.close - self.open) / self.open) * 100


class MarketIndicator(BaseModel):
    """Represents a calculated technical indicator."""

    symbol: str
    date: datetime
    name: str  # e.g., "SMA_20", "RSI_14"
    value: float


    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.completed_at is None:
            return 0
        return (self.completed_at - self.started_at).total_seconds()


# ============================================================================
# Podcast Models
# ============================================================================


class PodcastEpisode(BaseModel):
    """Represents a podcast episode."""

    id: str  # Unique identifier from source
    title: str
    podcast_name: str
    published_date: datetime
    summary: str | None = None
    content: str | None = None
    duration_seconds: int | None = None
    url: str | None = None
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def text_for_embedding(self) -> str:
        """Generate text content for embedding generation."""
        parts = [
            f"Podcast: {self.podcast_name}",
            f"Title: {self.title}",
        ]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")

        return "\n".join(parts)


class PodcastSyncResult(BaseModel):
    """Result of syncing podcast data."""

    started_at: datetime
    completed_at: datetime | None = None
    success: bool = True
    episodes_processed: int = 0
    episodes_new: int = 0
    errors: list[str] = Field(default_factory=list)
    episodes: list[PodcastEpisode] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.completed_at is None:
            return 0
        return (self.completed_at - self.started_at).total_seconds()
