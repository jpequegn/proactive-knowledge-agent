"""Tests for background jobs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.jobs import sync_fitness, sync_market, sync_podcasts, sync_rss_feeds


class TestJobs:
    """Tests for background jobs."""

    @pytest.fixture
    def mock_settings(self):
        with patch("src.daemon.jobs.Settings") as MockSettings:
            mock_instance = MockSettings.return_value
            mock_instance.database_url = "postgresql://test"
            mock_instance.openai_api_key = "test_key"
            mock_instance.redis_url = "redis://localhost"
            mock_instance.p3_duckdb_path = "test.db"
            yield mock_instance

    @pytest.fixture
    def mock_db(self):
        with patch("src.daemon.jobs.Database") as MockDB:
            mock_instance = MockDB.return_value
            mock_instance.connect = AsyncMock()
            mock_instance.close = AsyncMock()
            yield mock_instance

    @pytest.fixture
    def mock_cache(self):
        with patch("src.daemon.jobs.Cache") as MockCache:
            mock_instance = MockCache.return_value
            mock_instance.connect = AsyncMock()
            mock_instance.close = AsyncMock()
            yield mock_instance

    @pytest.mark.asyncio
    async def test_sync_rss_feeds(self, mock_settings, mock_db, mock_cache):
        """Test RSS sync job."""
        # Mock dependencies
        with patch("src.daemon.jobs.load_feeds_config") as mock_load, \
             patch("src.daemon.jobs.get_all_feeds") as mock_get_feeds, \
             patch("src.daemon.jobs.get_feed_settings") as mock_get_settings, \
             patch("src.daemon.jobs.RSSProcessor") as MockProcessor, \
             patch("src.daemon.jobs.ArticleRepository") as MockRepo, \
             patch("src.daemon.jobs.EmbeddingService") as MockEmbService:
            
            # Setup mocks
            mock_processor = MockProcessor.return_value
            mock_processor.__aenter__.return_value = mock_processor
            
            mock_article = MagicMock()
            mock_article.text_for_embedding = "text"
            mock_report = MagicMock()
            mock_report.articles_found = 1
            
            mock_processor.fetch_all_feeds = AsyncMock(return_value=([mock_article], mock_report))
            
            mock_emb_service = MockEmbService.return_value
            mock_emb_service.generate_batch = AsyncMock(return_value=[[0.1, 0.2]])
            
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.upsert = AsyncMock(return_value=(None, True))

            # Run job
            await sync_rss_feeds()
            
            # Verify
            mock_processor.fetch_all_feeds.assert_called_once()
            mock_db.connect.assert_called_once()
            mock_repo_instance.upsert.assert_called_once()
            mock_emb_service.generate_batch.assert_called_once()
            mock_cache.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_market(self, mock_settings, mock_db):
        """Test Market sync job."""
        with patch("src.daemon.jobs.sync_market_data") as mock_sync_data, \
             patch("src.daemon.jobs.MarketRepository") as MockRepo:
            
            # Setup mocks
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.data = [{"symbol": "AAPL"}]
            mock_sync_data.return_value = mock_result
            
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.upsert_batch = AsyncMock(return_value=1)

            # Run job
            await sync_market()
            
            # Verify
            mock_sync_data.assert_called_once()
            mock_db.connect.assert_called_once()
            mock_repo_instance.upsert_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_podcasts(self, mock_settings, mock_db, mock_cache):
        """Test Podcast sync job."""
        with patch("src.daemon.jobs.sync_podcast_data") as mock_sync_data, \
             patch("src.daemon.jobs.PodcastRepository") as MockRepo, \
             patch("src.daemon.jobs.EmbeddingService") as MockEmbService:
            
            # Setup mocks
            mock_episode = MagicMock()
            mock_episode.text_for_embedding = "text"
            
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.episodes = [mock_episode]
            mock_sync_data.return_value = mock_result
            
            mock_emb_service = MockEmbService.return_value
            mock_emb_service.generate_batch = AsyncMock(return_value=[[0.1]])
            
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.upsert = AsyncMock()

            # Run job
            await sync_podcasts()
            
            # Verify
            mock_sync_data.assert_called_once()
            mock_db.connect.assert_called_once()
            mock_emb_service.generate_batch.assert_called_once()
            mock_repo_instance.upsert.assert_called_once()
            mock_cache.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_fitness(self, mock_settings, mock_db):
        """Test Fitness sync job."""
        with patch("src.daemon.jobs.StravaTokenRepository") as MockTokenRepo, \
             patch("src.daemon.jobs.StravaAuth") as MockAuth, \
             patch("src.daemon.jobs.sync_strava_activities") as mock_sync_activities, \
             patch("src.daemon.jobs.ActivityRepository") as MockActivityRepo, \
             patch("src.daemon.jobs.FitnessMetricsRepository") as MockMetricsRepo, \
             patch("src.daemon.jobs.TrainingMetricsCalculator") as MockCalc:
            
            # Setup mocks
            mock_token_repo = MockTokenRepo.return_value
            mock_token_repo.get_default_tokens = AsyncMock(return_value={
                "access_token": "at",
                "refresh_token": "rt",
                "expires_at": 123,
                "athlete_id": 1
            })
            
            mock_auth = MockAuth.return_value
            mock_auth.access_token = "at"
            
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.activities = [MagicMock()]
            mock_sync_activities.return_value = mock_result
            
            mock_activity_repo = MockActivityRepo.return_value
            mock_activity_repo.upsert = AsyncMock(return_value=(None, True))
            from datetime import date
            mock_activity_repo.get_daily_tss = AsyncMock(return_value={date.today(): 100})
            
            mock_metrics_repo = MockMetricsRepo.return_value
            mock_metrics_repo.upsert = AsyncMock()

            # Run job
            await sync_fitness()
            
            # Verify
            mock_token_repo.get_default_tokens.assert_called_once()
            mock_sync_activities.assert_called_once()
            mock_activity_repo.upsert.assert_called_once()
            mock_metrics_repo.upsert.assert_called_once()
