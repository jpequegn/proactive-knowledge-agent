"""Tests for CLI commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_init_command(self, runner):
        """Test init command."""
        with patch("src.database.Database") as MockDB, \
             patch("src.database.init_schema", new_callable=AsyncMock) as mock_init:
            
            mock_db_instance = MockDB.return_value
            mock_db_instance.connect = AsyncMock()
            mock_db_instance.close = AsyncMock()

            result = runner.invoke(main, ["init"])
            
            assert result.exit_code == 0
            assert "Database schema initialized successfully" in result.output
            mock_init.assert_called_once()

    def test_status_command(self, runner):
        """Test status command."""
        with patch("src.database.Database") as MockDB, \
             patch("src.repositories.ArticleRepository") as MockArtRepo, \
             patch("src.repositories.MarketRepository") as MockMktRepo, \
             patch("src.repositories.PodcastRepository") as MockPodRepo, \
             patch("src.config.load_feeds_config") as mock_load_config, \
             patch("src.config.get_all_feeds") as mock_get_feeds:

            # Setup mocks
            mock_load_config.return_value = {}
            mock_get_feeds.return_value = []

            mock_db_instance = MockDB.return_value
            mock_db_instance.connect = AsyncMock()
            mock_db_instance.close = AsyncMock()

            # Mock Repos
            art_repo = MockArtRepo.return_value
            art_repo.count = AsyncMock(return_value=10)
            art_repo.get_sources = AsyncMock(return_value=["TechCrunch"])
            art_repo.get_categories = AsyncMock(return_value=["Tech"])

            mkt_repo = MockMktRepo.return_value
            mkt_repo.count = AsyncMock(return_value=100)
            mkt_repo.get_symbols = AsyncMock(return_value=["AAPL"])

            pod_repo = MockPodRepo.return_value
            pod_repo.count = AsyncMock(return_value=5)
            pod_repo.get_podcasts = AsyncMock(return_value=["Pod A"])

            result = runner.invoke(main, ["status"])

            assert result.exit_code == 0
            assert "Proactive Knowledge Agent Status" in result.output
            assert "RSS" in result.output
            assert "Market" in result.output
            assert "Podcast" in result.output
            assert "Total Articles" in result.output

    def test_sync_command_dry_run(self, runner):
        """Test sync command in dry run."""
        with patch("src.ingestion.rss_processor.RSSProcessor") as MockProcessor, \
             patch("src.config.load_feeds_config"), \
             patch("src.config.get_all_feeds"), \
             patch("src.config.get_feed_settings"):

            mock_proc = MockProcessor.return_value
            mock_proc.__aenter__.return_value = mock_proc
            mock_proc.__aexit__.return_value = None
            
            # Return empty articles and dummy report
            report = MagicMock()
            report.results = []
            report.articles_found = 0
            report.feeds_processed = 0
            report.duration_seconds = 0
            report.success_rate = 0
            
            mock_proc.fetch_all_feeds = AsyncMock(return_value=([], report))

            result = runner.invoke(main, ["sync", "--dry-run", "--rss"])

            assert result.exit_code == 0
            assert "Syncing" in result.output
            assert "Dry run" in result.output

    def test_search_command(self, runner):
        """Test search command."""
        with patch("src.database.Database") as MockDB, \
             patch("src.repositories.ArticleRepository") as MockArtRepo, \
             patch("src.repositories.PodcastRepository") as MockPodRepo, \
             patch("src.config.load_feeds_config"), \
             patch("src.config.get_feed_settings"):

            mock_db_instance = MockDB.return_value
            mock_db_instance.connect = AsyncMock()
            mock_db_instance.close = AsyncMock()

            art_repo = MockArtRepo.return_value
            art_repo.search_by_text = AsyncMock(return_value=[
                MagicMock(title="AI Article", source="Tech", category="News", published=None)
            ])
            
            pod_repo = MockPodRepo.return_value
            pod_repo.search_by_text = AsyncMock(return_value=[])

            result = runner.invoke(main, ["search", "AI"])

            assert result.exit_code == 0
            assert "Search Results for 'AI'" in result.output
            assert "AI Article" in result.output
