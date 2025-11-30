"""Tests for background scheduler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.daemon.scheduler import SchedulerService


class TestSchedulerService:
    """Tests for SchedulerService."""

    @pytest.fixture
    def mock_scheduler(self):
        with patch("src.daemon.scheduler.AsyncIOScheduler") as MockScheduler:
            mock_instance = MockScheduler.return_value
            yield mock_instance

    def test_init(self, mock_scheduler):
        """Test initialization and job setup."""
        service = SchedulerService()
        
        assert mock_scheduler.add_job.call_count == 4
        
        # Verify specific jobs
        jobs = [call[1]["id"] for call in mock_scheduler.add_job.call_args_list]
        assert "sync_rss" in jobs
        assert "sync_market" in jobs
        assert "sync_podcasts" in jobs
        assert "sync_fitness" in jobs

    def test_start_stop(self, mock_scheduler):
        """Test start and stop methods."""
        service = SchedulerService()
        
        service.start()
        mock_scheduler.start.assert_called_once()
        
        service.stop()
        mock_scheduler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_forever(self, mock_scheduler):
        """Test run_forever loop."""
        service = SchedulerService()
        
        # Mock asyncio.sleep to raise CancelledError to exit the loop
        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
            with patch("src.daemon.scheduler.sync_rss_feeds"), \
                 patch("src.daemon.scheduler.sync_market"), \
                 patch("src.daemon.scheduler.sync_podcasts"), \
                 patch("src.daemon.scheduler.sync_fitness"):
                
                await service.run_forever()
        
        mock_scheduler.start.assert_called_once()
        mock_scheduler.shutdown.assert_called_once()
