"""Tests for configuration management."""

import pytest
from pathlib import Path
import tempfile
import yaml

from src.config import (
    FeedConfig,
    FeedSettings,
    get_all_feeds,
    get_feed_settings,
    load_feeds_config,
)


class TestFeedConfig:
    """Tests for FeedConfig model."""

    def test_feed_config_creation(self) -> None:
        """Test FeedConfig creation with required fields."""
        config = FeedConfig(
            name="Test Feed",
            url="https://example.com/feed.xml",
            category="tech",
        )
        assert config.name == "Test Feed"
        assert config.url == "https://example.com/feed.xml"
        assert config.category == "tech"
        assert config.priority == "medium"  # default

    def test_feed_config_with_priority(self) -> None:
        """Test FeedConfig with custom priority."""
        config = FeedConfig(
            name="Test Feed",
            url="https://example.com/feed.xml",
            category="tech",
            priority="high",
        )
        assert config.priority == "high"


class TestFeedSettings:
    """Tests for FeedSettings model."""

    def test_feed_settings_defaults(self) -> None:
        """Test FeedSettings default values."""
        settings = FeedSettings()
        assert settings.sync_interval_minutes == 60
        assert settings.max_articles_per_feed == 20
        assert settings.retention_days == 30
        assert settings.embedding_model == "text-embedding-3-small"

    def test_feed_settings_custom(self) -> None:
        """Test FeedSettings with custom values."""
        settings = FeedSettings(
            sync_interval_minutes=30,
            max_articles_per_feed=50,
            retention_days=90,
            embedding_model="text-embedding-3-large",
        )
        assert settings.sync_interval_minutes == 30
        assert settings.max_articles_per_feed == 50


class TestLoadFeedsConfig:
    """Tests for loading feeds configuration."""

    def test_load_feeds_config_from_file(self) -> None:
        """Test loading config from YAML file."""
        config_content = """
feeds:
  tech:
    - name: "Hacker News"
      url: "https://news.ycombinator.com/rss"
      category: "tech"
      priority: high
    - name: "TechCrunch"
      url: "https://techcrunch.com/feed/"
      category: "tech"
  finance:
    - name: "Bloomberg"
      url: "https://bloomberg.com/feed/"
      category: "finance"
      priority: high

settings:
  sync_interval_minutes: 30
  max_articles_per_feed: 10
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            f.flush()

            config = load_feeds_config(Path(f.name))

            assert "feeds" in config
            assert "tech" in config["feeds"]
            assert len(config["feeds"]["tech"]) == 2
            assert config["settings"]["sync_interval_minutes"] == 30


class TestGetAllFeeds:
    """Tests for extracting feeds from config."""

    def test_get_all_feeds_flat_list(self) -> None:
        """Test extracting all feeds as flat list."""
        config = {
            "feeds": {
                "tech": [
                    {"name": "Feed 1", "url": "https://1.com", "category": "tech"},
                    {"name": "Feed 2", "url": "https://2.com", "category": "tech"},
                ],
                "finance": [
                    {"name": "Feed 3", "url": "https://3.com", "category": "finance"},
                ],
            }
        }

        feeds = get_all_feeds(config)

        assert len(feeds) == 3
        assert all(isinstance(f, FeedConfig) for f in feeds)
        assert feeds[0].name == "Feed 1"

    def test_get_all_feeds_empty_config(self) -> None:
        """Test empty config returns empty list."""
        config: dict = {"feeds": {}}
        feeds = get_all_feeds(config)
        assert feeds == []


class TestGetFeedSettings:
    """Tests for extracting feed settings from config."""

    def test_get_feed_settings_from_config(self) -> None:
        """Test extracting settings from config."""
        config = {
            "settings": {
                "sync_interval_minutes": 120,
                "max_articles_per_feed": 30,
            }
        }

        settings = get_feed_settings(config)

        assert settings.sync_interval_minutes == 120
        assert settings.max_articles_per_feed == 30
        assert settings.retention_days == 30  # default

    def test_get_feed_settings_missing_uses_defaults(self) -> None:
        """Test missing settings use defaults."""
        config: dict = {}

        settings = get_feed_settings(config)

        assert settings.sync_interval_minutes == 60
        assert settings.max_articles_per_feed == 20
