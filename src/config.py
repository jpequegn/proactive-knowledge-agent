"""Configuration management for PKA."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class FeedConfig(BaseModel):
    """Configuration for a single RSS feed."""

    name: str
    url: str
    category: str
    priority: str = "medium"


class FeedSettings(BaseModel):
    """Settings for feed processing."""

    sync_interval_minutes: int = 60
    max_articles_per_feed: int = 20
    retention_days: int = 30
    embedding_model: str = "text-embedding-3-small"


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Database
    database_url: str = "postgresql://localhost/pka"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # OpenAI
    openai_api_key: str = ""

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_feeds_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load feeds configuration from YAML file."""
    if config_path is None:
        config_path = Path("config/feeds.yaml")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_all_feeds(config: dict[str, Any]) -> list[FeedConfig]:
    """Extract all feeds from configuration as flat list."""
    feeds = []
    feeds_section = config.get("feeds", {})

    for category, feed_list in feeds_section.items():
        for feed_data in feed_list:
            feeds.append(FeedConfig(**feed_data))

    return feeds


def get_feed_settings(config: dict[str, Any]) -> FeedSettings:
    """Extract feed settings from configuration."""
    settings_data = config.get("settings", {})
    return FeedSettings(**settings_data)
