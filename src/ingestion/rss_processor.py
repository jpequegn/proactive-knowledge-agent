"""RSS Feed Processor for ingesting news and articles."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx
import structlog

from src.config import FeedConfig, FeedSettings, get_all_feeds, get_feed_settings
from src.models import Article, FetchResult, SyncReport

logger = structlog.get_logger()


class RSSProcessor:
    """Process RSS feeds and extract articles."""

    def __init__(
        self,
        feeds: list[FeedConfig],
        settings: FeedSettings,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.feeds = feeds
        self.settings = settings
        self._client = http_client
        self._owns_client = http_client is None

    async def __aenter__(self) -> "RSSProcessor":
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "PKA-RSS-Processor/0.1"},
            )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("RSSProcessor must be used as async context manager")
        return self._client

    async def fetch_feed(self, feed: FeedConfig) -> tuple[list[Article], FetchResult]:
        """Fetch and parse a single RSS feed."""
        start_time = time.monotonic()
        articles: list[Article] = []

        try:
            response = await self.client.get(feed.url)
            response.raise_for_status()

            parsed = feedparser.parse(response.text)

            if parsed.bozo and not parsed.entries:
                raise ValueError(f"Failed to parse feed: {parsed.bozo_exception}")

            for entry in parsed.entries[: self.settings.max_articles_per_feed]:
                article = self._parse_entry(entry, feed)
                if article:
                    articles.append(article)

            duration_ms = (time.monotonic() - start_time) * 1000

            result = FetchResult(
                feed_name=feed.name,
                feed_url=feed.url,
                success=True,
                articles_found=len(articles),
                duration_ms=duration_ms,
            )

            logger.info(
                "Feed fetched successfully",
                feed=feed.name,
                articles=len(articles),
                duration_ms=round(duration_ms, 2),
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            result = FetchResult(
                feed_name=feed.name,
                feed_url=feed.url,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            logger.error(
                "Failed to fetch feed",
                feed=feed.name,
                error=str(e),
            )

        return articles, result

    def _parse_entry(self, entry: Any, feed: FeedConfig) -> Article | None:
        """Parse a feedparser entry into an Article."""
        try:
            # Get title
            title = entry.get("title", "").strip()
            if not title:
                return None

            # Get URL
            url = entry.get("link", "")
            if not url:
                return None

            # Get summary/description
            summary = None
            if "summary" in entry:
                summary = self._clean_html(entry.summary)
            elif "description" in entry:
                summary = self._clean_html(entry.description)

            # Get content if available
            content = None
            if "content" in entry and entry.content:
                content = self._clean_html(entry.content[0].get("value", ""))

            # Get published date
            published = None
            if "published_parsed" in entry and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=UTC)
                except (TypeError, ValueError):
                    pass
            elif "updated_parsed" in entry and entry.updated_parsed:
                try:
                    published = datetime(*entry.updated_parsed[:6], tzinfo=UTC)
                except (TypeError, ValueError):
                    pass

            # Get author
            author = entry.get("author")

            return Article(
                title=title,
                url=url,
                summary=summary,
                content=content,
                published=published,
                source=feed.name,
                category=feed.category,
                author=author,
            )

        except Exception as e:
            logger.warning(
                "Failed to parse entry",
                feed=feed.name,
                error=str(e),
            )
            return None

    def _clean_html(self, html: str) -> str:
        """Remove HTML tags from string (basic implementation)."""
        import re

        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", html)
        # Decode HTML entities
        clean = clean.replace("&nbsp;", " ")
        clean = clean.replace("&amp;", "&")
        clean = clean.replace("&lt;", "<")
        clean = clean.replace("&gt;", ">")
        clean = clean.replace("&quot;", '"')
        clean = clean.replace("&#39;", "'")
        # Normalize whitespace
        clean = " ".join(clean.split())
        return clean.strip()

    async def fetch_all_feeds(self) -> tuple[list[Article], SyncReport]:
        """Fetch all configured feeds."""
        report = SyncReport(started_at=datetime.now(UTC))
        all_articles: list[Article] = []

        # Fetch feeds concurrently with semaphore to limit parallelism
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(
            feed: FeedConfig,
        ) -> tuple[list[Article], FetchResult]:
            async with semaphore:
                return await self.fetch_feed(feed)

        tasks = [fetch_with_semaphore(feed) for feed in self.feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                report.errors.append(str(result))
                continue

            articles, fetch_result = result
            all_articles.extend(articles)
            report.results.append(fetch_result)
            report.feeds_processed += 1
            report.articles_found += fetch_result.articles_found

        report.completed_at = datetime.now(UTC)

        logger.info(
            "Feed sync completed",
            feeds_processed=report.feeds_processed,
            articles_found=report.articles_found,
            duration_seconds=round(report.duration_seconds, 2),
            success_rate=round(report.success_rate * 100, 1),
        )

        return all_articles, report


async def sync_feeds(
    config_path: str | None = None,
) -> tuple[list[Article], SyncReport]:
    """Convenience function to sync all feeds from config."""
    from pathlib import Path

    from src.config import load_feeds_config

    config = load_feeds_config(Path(config_path) if config_path else None)
    feeds = get_all_feeds(config)
    settings = get_feed_settings(config)

    async with RSSProcessor(feeds, settings) as processor:
        return await processor.fetch_all_feeds()
