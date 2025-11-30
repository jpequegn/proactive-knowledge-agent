"""Background scheduler for PKA."""

import asyncio
from typing import Any

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.daemon.jobs import (
    sync_fitness,
    sync_market,
    sync_podcasts,
    sync_rss_feeds,
)

logger = structlog.get_logger()


class SchedulerService:
    """Service to manage background jobs."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self._setup_jobs()

    def _setup_jobs(self) -> None:
        """Configure scheduled jobs."""
        # RSS Feeds - Every hour
        self.scheduler.add_job(
            sync_rss_feeds,
            trigger=IntervalTrigger(minutes=60),
            id="sync_rss",
            name="Sync RSS Feeds",
            replace_existing=True,
        )

        # Market Data - Every 4 hours (market hours vary, but this is simple)
        self.scheduler.add_job(
            sync_market,
            trigger=IntervalTrigger(minutes=240),
            id="sync_market",
            name="Sync Market Data",
            replace_existing=True,
        )

        # Podcasts - Every 6 hours
        self.scheduler.add_job(
            sync_podcasts,
            trigger=IntervalTrigger(minutes=360),
            id="sync_podcasts",
            name="Sync Podcasts",
            replace_existing=True,
        )

        # Fitness - Every 2 hours
        self.scheduler.add_job(
            sync_fitness,
            trigger=IntervalTrigger(minutes=120),
            id="sync_fitness",
            name="Sync Fitness Data",
            replace_existing=True,
        )

    def start(self) -> None:
        """Start the scheduler."""
        logger.info("Starting background scheduler")
        self.scheduler.start()

    def stop(self) -> None:
        """Stop the scheduler."""
        logger.info("Stopping background scheduler")
        self.scheduler.shutdown()

    async def run_forever(self) -> None:
        """Run scheduler until interrupted."""
        self.start()
        
        # Run initial jobs immediately (asyncio.gather could be used but let's keep it simple)
        logger.info("Running initial jobs...")
        # We trigger them once on startup for immediate feedback, 
        # but wrapped in tasks so they don't block startup
        asyncio.create_task(sync_rss_feeds())
        asyncio.create_task(sync_market())
        asyncio.create_task(sync_podcasts())
        asyncio.create_task(sync_fitness())

        try:
            # Keep the loop alive
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.stop()
