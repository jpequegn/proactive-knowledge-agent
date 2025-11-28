"""Tests for RSS feed processor."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.config import FeedConfig, FeedSettings
from src.ingestion.rss_processor import RSSProcessor
from src.models import Article


# Sample RSS feed content for testing
SAMPLE_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <description>A test RSS feed</description>
    <item>
      <title>Test Article 1</title>
      <link>https://example.com/article-1</link>
      <description>This is the first test article.</description>
      <pubDate>Mon, 25 Nov 2024 10:00:00 GMT</pubDate>
      <author>John Doe</author>
    </item>
    <item>
      <title>Test Article 2</title>
      <link>https://example.com/article-2</link>
      <description>This is the second test article with &lt;b&gt;HTML&lt;/b&gt; tags.</description>
      <pubDate>Tue, 26 Nov 2024 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <link href="https://example.com"/>
  <entry>
    <title>Atom Article 1</title>
    <link href="https://example.com/atom-1"/>
    <summary>An atom feed entry.</summary>
    <updated>2024-11-25T10:00:00Z</updated>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>
"""


@pytest.fixture
def feed_config() -> FeedConfig:
    """Create test feed configuration."""
    return FeedConfig(
        name="Test Feed",
        url="https://example.com/feed.xml",
        category="tech",
        priority="high",
    )


@pytest.fixture
def feed_settings() -> FeedSettings:
    """Create test feed settings."""
    return FeedSettings(
        sync_interval_minutes=60,
        max_articles_per_feed=20,
        retention_days=30,
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create mock HTTP client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


class TestRSSProcessor:
    """Tests for RSSProcessor class."""

    @pytest.mark.asyncio
    async def test_fetch_feed_success(
        self,
        feed_config: FeedConfig,
        feed_settings: FeedSettings,
    ) -> None:
        """Test successful feed fetch."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.text = SAMPLE_RSS_FEED
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        processor = RSSProcessor(
            feeds=[feed_config],
            settings=feed_settings,
            http_client=mock_client,
        )

        articles, result = await processor.fetch_feed(feed_config)

        assert result.success is True
        assert result.feed_name == "Test Feed"
        assert len(articles) == 2
        assert articles[0].title == "Test Article 1"
        assert articles[0].url == "https://example.com/article-1"
        assert articles[0].source == "Test Feed"
        assert articles[0].category == "tech"

    @pytest.mark.asyncio
    async def test_fetch_feed_with_html_cleaning(
        self,
        feed_config: FeedConfig,
        feed_settings: FeedSettings,
    ) -> None:
        """Test HTML tags are cleaned from content."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.text = SAMPLE_RSS_FEED
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        processor = RSSProcessor(
            feeds=[feed_config],
            settings=feed_settings,
            http_client=mock_client,
        )

        articles, _ = await processor.fetch_feed(feed_config)

        # Second article has HTML in description
        assert "<b>" not in articles[1].summary
        assert "HTML" in articles[1].summary

    @pytest.mark.asyncio
    async def test_fetch_feed_with_atom_format(
        self,
        feed_config: FeedConfig,
        feed_settings: FeedSettings,
    ) -> None:
        """Test Atom feed parsing."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.text = SAMPLE_ATOM_FEED
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        processor = RSSProcessor(
            feeds=[feed_config],
            settings=feed_settings,
            http_client=mock_client,
        )

        articles, result = await processor.fetch_feed(feed_config)

        assert result.success is True
        assert len(articles) == 1
        assert articles[0].title == "Atom Article 1"

    @pytest.mark.asyncio
    async def test_fetch_feed_http_error(
        self,
        feed_config: FeedConfig,
        feed_settings: FeedSettings,
    ) -> None:
        """Test handling of HTTP errors."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

        processor = RSSProcessor(
            feeds=[feed_config],
            settings=feed_settings,
            http_client=mock_client,
        )

        articles, result = await processor.fetch_feed(feed_config)

        assert result.success is False
        assert "Connection failed" in result.error
        assert len(articles) == 0

    @pytest.mark.asyncio
    async def test_fetch_feed_max_articles_limit(
        self,
        feed_config: FeedConfig,
    ) -> None:
        """Test max articles per feed limit."""
        settings = FeedSettings(max_articles_per_feed=1)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.text = SAMPLE_RSS_FEED
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        processor = RSSProcessor(
            feeds=[feed_config],
            settings=settings,
            http_client=mock_client,
        )

        articles, _ = await processor.fetch_feed(feed_config)

        assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_fetch_all_feeds_concurrent(
        self,
        feed_settings: FeedSettings,
    ) -> None:
        """Test concurrent fetching of multiple feeds."""
        feeds = [
            FeedConfig(name=f"Feed {i}", url=f"https://example.com/feed{i}.xml", category="tech")
            for i in range(3)
        ]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.text = SAMPLE_RSS_FEED
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        processor = RSSProcessor(
            feeds=feeds,
            settings=feed_settings,
            http_client=mock_client,
        )

        articles, report = await processor.fetch_all_feeds()

        assert report.feeds_processed == 3
        assert report.success_rate == 1.0
        # 2 articles per feed * 3 feeds
        assert len(articles) == 6


class TestArticleModel:
    """Tests for Article model."""

    def test_url_hash_generation(self) -> None:
        """Test URL hash is generated correctly."""
        article = Article(
            title="Test",
            url="https://example.com/article",
            source="Test",
            category="tech",
        )

        assert article.url_hash is not None
        assert len(article.url_hash) == 64  # SHA256 hex digest

    def test_url_hash_consistency(self) -> None:
        """Test same URL produces same hash."""
        article1 = Article(
            title="Test 1",
            url="https://example.com/article",
            source="Source 1",
            category="tech",
        )
        article2 = Article(
            title="Test 2",
            url="https://example.com/article",
            source="Source 2",
            category="finance",
        )

        assert article1.url_hash == article2.url_hash

    def test_text_for_embedding(self) -> None:
        """Test text for embedding generation."""
        article = Article(
            title="Test Title",
            url="https://example.com",
            summary="Test summary",
            content="Test content that is longer",
            source="Test",
            category="tech",
        )

        text = article.text_for_embedding
        assert "Test Title" in text
        assert "Test summary" in text
        assert "Test content" in text


class TestRSSProcessorCleanHtml:
    """Tests for HTML cleaning functionality."""

    def test_clean_html_removes_tags(self) -> None:
        """Test HTML tags are removed."""
        processor = RSSProcessor([], FeedSettings())
        result = processor._clean_html("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_clean_html_decodes_entities(self) -> None:
        """Test HTML entities are decoded."""
        processor = RSSProcessor([], FeedSettings())
        result = processor._clean_html("Hello &amp; goodbye &lt;3")
        assert result == "Hello & goodbye <3"

    def test_clean_html_normalizes_whitespace(self) -> None:
        """Test whitespace is normalized."""
        processor = RSSProcessor([], FeedSettings())
        result = processor._clean_html("Hello   world\n\n\ttest")
        assert result == "Hello world test"
