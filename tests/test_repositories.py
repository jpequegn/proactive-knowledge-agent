"""Tests for data repositories."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database import Database
from src.models import MarketOHLCV, PodcastEpisode
from src.repositories import MarketRepository, PodcastRepository


class TestMarketRepository:
    """Tests for MarketRepository."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.transaction = MagicMock()
        # Setup async context manager for transaction
        txn = AsyncMock()
        db.transaction.return_value.__aenter__.return_value = txn
        db.transaction.return_value.__aexit__.return_value = None
        
        db.fetch = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_upsert_batch(self, mock_db):
        """Test batch upsert of market data."""
        repo = MarketRepository(mock_db)
        
        data = [
            MarketOHLCV(
                symbol="AAPL",
                date=datetime.now(UTC),
                open=100, high=110, low=90, close=105, volume=1000,
                adjusted_close=105
            ),
            MarketOHLCV(
                symbol="GOOGL",
                date=datetime.now(UTC),
                open=200, high=210, low=190, close=205, volume=2000,
                adjusted_close=205
            ),
        ]

        count = await repo.upsert_batch(data)

        assert count == 2
        
        # Verify transaction was used
        mock_db.transaction.assert_called_once()
        txn = mock_db.transaction.return_value.__aenter__.return_value
        txn.executemany.assert_called_once()
        
        # Verify arguments passed to executemany
        call_args = txn.executemany.call_args
        assert call_args[0][0].strip().startswith("INSERT INTO market_ohlcv")
        assert len(call_args[0][1]) == 2

    @pytest.mark.asyncio
    async def test_upsert_batch_empty(self, mock_db):
        """Test batch upsert with empty list."""
        repo = MarketRepository(mock_db)
        count = await repo.upsert_batch([])
        assert count == 0
        mock_db.transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_history(self, mock_db):
        """Test fetching market history."""
        repo = MarketRepository(mock_db)
        
        # Mock DB returns rows (descending date)
        date1 = datetime(2024, 1, 2, tzinfo=UTC)
        date2 = datetime(2024, 1, 1, tzinfo=UTC)
        
        mock_rows = [
            {
                "symbol": "AAPL", "date": date1,
                "open": 102, "high": 112, "low": 92, "close": 107, "volume": 1100,
                "adjusted_close": 107
            },
            {
                "symbol": "AAPL", "date": date2,
                "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000,
                "adjusted_close": 105
            },
        ]
        mock_db.fetch.return_value = mock_rows

        history = await repo.get_history("AAPL")

        assert len(history) == 2
        # Repository should reverse the order (oldest first)
        assert history[0].date == date2
        assert history[1].date == date1
        assert history[0].close == 105
        
        mock_db.fetch.assert_called_once()


class TestPodcastRepository:
    """Tests for PodcastRepository."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.acquire = MagicMock()
        # Setup async context manager for acquire
        conn = AsyncMock()
        db.acquire.return_value.__aenter__.return_value = conn
        db.acquire.return_value.__aexit__.return_value = None
        
        db.fetch = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_upsert(self, mock_db):
        """Test upserting a podcast episode."""
        repo = PodcastRepository(mock_db)
        
        episode = PodcastEpisode(
            id="123",
            title="Test Episode",
            podcast_name="Test Pod",
            published_date=datetime.now(UTC),
            duration_seconds=3600,
            topics=["AI", "Code"]
        )
        embedding = [0.1, 0.2, 0.3]

        conn = mock_db.acquire.return_value.__aenter__.return_value
        conn.fetchval.return_value = 1  # Return ID 1

        ep_id = await repo.upsert(episode, embedding)

        assert ep_id == 1
        conn.fetchval.assert_called_once()
        
        # Check query contains ON CONFLICT
        call_args = conn.fetchval.call_args
        assert "ON CONFLICT (external_id)" in call_args[0][0]
        # Check params (topics passed as list)
        assert call_args[0][9] == ["AI", "Code"]

    @pytest.mark.asyncio
    async def test_get_recent(self, mock_db):
        """Test fetching recent episodes."""
        repo = PodcastRepository(mock_db)
        
        mock_rows = [
            {
                "external_id": "1", "title": "Ep 1", "podcast_name": "Pod A",
                "published": datetime.now(UTC), "summary": "Sum", "content": "Content",
                "duration_seconds": 100, "url": "http://a.com", "topics": ["A"], "entities": []
            }
        ]
        mock_db.fetch.return_value = mock_rows

        episodes = await repo.get_recent(limit=5)

        assert len(episodes) == 1
        assert episodes[0].id == "1"
        assert episodes[0].title == "Ep 1"
        
        mock_db.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar(self, mock_db):
        """Test searching similar episodes."""
        repo = PodcastRepository(mock_db)
        
        mock_rows = [
            {
                "external_id": "1", "title": "Ep 1", "podcast_name": "Pod A",
                "published": datetime.now(UTC), "summary": "Sum", "content": "Content",
                "duration_seconds": 100, "url": "http://a.com", "topics": [], "entities": [],
                "similarity": 0.85
            }
        ]
        mock_db.fetch.return_value = mock_rows

        results = await repo.search_similar([0.1, 0.2])

        assert len(results) == 1
        episode, score = results[0]
        assert episode.id == "1"
        assert score == 0.85
        
        mock_db.fetch.assert_called_once()
