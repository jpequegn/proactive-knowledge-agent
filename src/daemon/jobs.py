"""Background jobs for the scheduler."""

from datetime import date, timedelta

import structlog

from src.cache import Cache
from src.config import Settings, get_all_feeds, get_feed_settings, load_feeds_config
from src.database import Database
from src.fitness_repository import (
    ActivityRepository,
    FitnessMetricsRepository,
    StravaTokenRepository,
)
from src.ingestion.embeddings import EmbeddingService
from src.ingestion.fitness_client import (
    StravaAuth,
    TrainingMetricsCalculator,
    sync_strava_activities,
)
from src.ingestion.market_client import sync_market_data
from src.ingestion.podcast_bridge import sync_podcast_data
from src.ingestion.rss_processor import RSSProcessor
from src.repositories import (
    ArticleRepository,
    MarketRepository,
    PodcastRepository,
)

logger = structlog.get_logger()


async def sync_rss_feeds() -> None:
    """Job: Sync RSS feeds."""
    logger.info("Starting RSS sync job")
    settings = Settings()

    try:
        config = load_feeds_config(settings.config_dir / "feeds.yaml")
        feeds = get_all_feeds(config)
        feed_settings = get_feed_settings(config)

        # Fetch feeds
        async with RSSProcessor(feeds, feed_settings) as processor:
            articles, report = await processor.fetch_all_feeds()

        logger.info(
            "RSS fetch completed",
            articles=report.articles_found,
            feeds=report.feeds_processed
        )

        if not articles:
            return

        # Store articles
        db = Database(settings.database_url)
        await db.connect()

        cache = None
        if settings.redis_url:
            cache = Cache(settings.redis_url)
            try:
                await cache.connect()
            except Exception as e:
                logger.warning("Redis connection failed", error=str(e))
                cache = None

        try:
            repo = ArticleRepository(db)

            # Generate embeddings
            embeddings = [None] * len(articles)
            if settings.openai_api_key:
                logger.info("Generating embeddings for articles")
                embedding_service = EmbeddingService(
                    api_key=settings.openai_api_key,
                    model=feed_settings.embedding_model,
                    cache=cache,
                )
                texts = [a.text_for_embedding for a in articles]
                embeddings = await embedding_service.generate_batch(texts)

            new_count = 0
            for article, embedding in zip(articles, embeddings):
                _, is_new = await repo.upsert(article, embedding)
                if is_new:
                    new_count += 1

            logger.info("RSS storage completed", new_articles=new_count)

        finally:
            await db.close()
            if cache:
                await cache.close()

    except Exception as e:
        logger.error("RSS sync job failed", error=str(e))


async def sync_market() -> None:
    """Job: Sync Market data."""
    logger.info("Starting Market sync job")
    settings = Settings()

    try:
        # TODO: Load symbols from config
        symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
            "SPY", "QQQ", "BTC-USD", "ETH-USD",
        ]

        market_result = await sync_market_data(symbols)

        if not market_result.success:
            logger.error("Market fetch failed", errors=market_result.errors)
            return

        if not market_result.data:
            logger.info("No new market data found")
            return

        db = Database(settings.database_url)
        await db.connect()

        try:
            repo = MarketRepository(db)
            count = await repo.upsert_batch(market_result.data)
            logger.info("Market storage completed", records=count)
        finally:
            await db.close()

    except Exception as e:
        logger.error("Market sync job failed", error=str(e))


async def sync_podcasts() -> None:
    """Job: Sync Podcasts."""
    logger.info("Starting Podcast sync job")
    settings = Settings()

    if not settings.p3_duckdb_path:
        logger.warning("Skipping podcast sync: p3_duckdb_path not set")
        return

    try:
        pod_result = await sync_podcast_data(settings.p3_duckdb_path)

        if not pod_result.success:
            logger.error("Podcast fetch failed", errors=pod_result.errors)
            return

        if not pod_result.episodes:
            logger.info("No new podcast episodes found")
            return

        db = Database(settings.database_url)
        await db.connect()

        cache = None
        if settings.redis_url:
            cache = Cache(settings.redis_url)
            try:
                await cache.connect()
            except Exception as e:
                logger.warning("Redis connection failed", error=str(e))
                cache = None

        try:
            repo = PodcastRepository(db)

            # Embeddings
            embeddings = [None] * len(pod_result.episodes)
            if settings.openai_api_key:
                logger.info("Generating embeddings for podcasts")
                embedding_service = EmbeddingService(
                    api_key=settings.openai_api_key,
                    cache=cache,
                )
                texts = [e.text_for_embedding for e in pod_result.episodes]
                embeddings = await embedding_service.generate_batch(texts)

            count = 0
            for episode, embedding in zip(pod_result.episodes, embeddings):
                await repo.upsert(episode, embedding)
                count += 1

            logger.info("Podcast storage completed", episodes=count)

        finally:
            await db.close()
            if cache:
                await cache.close()

    except Exception as e:
        logger.error("Podcast sync job failed", error=str(e))


async def sync_fitness() -> None:
    """Job: Sync Fitness data."""
    logger.info("Starting Fitness sync job")
    settings = Settings()

    try:
        db = Database(settings.database_url)
        await db.connect()

        try:
            # Check auth
            token_repo = StravaTokenRepository(db)
            tokens = await token_repo.get_default_tokens()

            if not tokens:
                logger.warning("Skipping fitness sync: No tokens found")
                return

            auth = StravaAuth(
                client_id=settings.strava_client_id or "",
                client_secret=settings.strava_client_secret or "",
            )
            auth.set_tokens(
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
            )

            # Fetch (short period for recurring job)
            result = await sync_strava_activities(auth, days=7)

            if not result.success:
                logger.error("Fitness fetch failed", errors=result.errors)
                return

            # Store
            activity_repo = ActivityRepository(db)
            metrics_repo = FitnessMetricsRepository(db)

            new_count = 0
            for activity in result.activities:
                _, is_new = await activity_repo.upsert(activity)
                if is_new:
                    new_count += 1

            # Update tokens if needed
            if auth.access_token != tokens["access_token"]:
                await token_repo.save_tokens(
                    athlete_id=tokens["athlete_id"],
                    access_token=auth.access_token or "",
                    refresh_token=auth._refresh_token or "",
                    expires_at=auth._expires_at,
                )

            # Calculate metrics if new data
            if new_count > 0:
                start_date = date.today() - timedelta(days=42)
                daily_tss = await activity_repo.get_daily_tss(start_date)

                if daily_tss:
                    all_days = []
                    current = start_date
                    end = date.today()
                    while current <= end:
                        all_days.append(daily_tss.get(current, 0))
                        current += timedelta(days=1)

                    calculator = TrainingMetricsCalculator()
                    metrics = calculator.calculate_training_load(all_days)
                    metrics.date = end
                    metrics.daily_tss = daily_tss.get(end, 0)

                    await metrics_repo.upsert(metrics)

            logger.info("Fitness sync completed", new_activities=new_count)

        finally:
            await db.close()

    except Exception as e:
        logger.error("Fitness sync job failed", error=str(e))


async def generate_weekly_report() -> None:
    """Job: Generate weekly intelligence report."""

    from src.agents import (
        AgentConfig,
        AgentReport,
        FinanceIntelligenceAgent,
        FitnessIntelligenceAgent,
        TechIntelligenceAgent,
        UserProfile,
    )
    from src.outputs import ReportConfig, WeeklyReportGenerator
    from src.world_model import Domain

    logger.info("Starting weekly report generation job")
    settings = Settings()

    try:
        db = Database(settings.database_url)
        await db.connect()

        try:
            # Initialize user profile (could be loaded from config)
            user_profile = UserProfile(
                role="developer",
                experience_level="senior",
                interests=["AI/ML", "Python", "Cloud"],
                known_technologies=["Python", "TypeScript", "React"],
                learning_goals=["Rust", "Machine Learning"],
            )

            # Initialize agents
            config = AgentConfig(enabled=True)
            agent_reports: dict[Domain, AgentReport] = {}

            # Run Tech Intelligence Agent
            try:
                tech_agent = TechIntelligenceAgent(
                    db=db,
                    config=config,
                    user_profile=user_profile,
                )
                agent_reports[Domain.TECH] = await tech_agent.run()
                logger.info(
                    "Tech agent completed",
                    insights=len(agent_reports[Domain.TECH].insights),
                )
            except Exception as e:
                logger.warning("Tech agent failed", error=str(e))

            # Run Fitness Intelligence Agent
            try:
                fitness_agent = FitnessIntelligenceAgent(
                    db=db,
                    config=config,
                    user_profile=user_profile,
                )
                agent_reports[Domain.FITNESS] = await fitness_agent.run()
                logger.info(
                    "Fitness agent completed",
                    insights=len(agent_reports[Domain.FITNESS].insights),
                )
            except Exception as e:
                logger.warning("Fitness agent failed", error=str(e))

            # Run Finance Intelligence Agent
            try:
                finance_agent = FinanceIntelligenceAgent(
                    db=db,
                    config=config,
                    user_profile=user_profile,
                )
                agent_reports[Domain.FINANCE] = await finance_agent.run()
                logger.info(
                    "Finance agent completed",
                    insights=len(agent_reports[Domain.FINANCE].insights),
                )
            except Exception as e:
                logger.warning("Finance agent failed", error=str(e))

            # Try to run Synthesis Agent if available
            try:
                from src.agents.synthesis_agent import SynthesisAgent

                synthesis_agent = SynthesisAgent(
                    db=db,
                    config=config,
                    user_profile=user_profile,
                    domain_insights={
                        domain: report.insights
                        for domain, report in agent_reports.items()
                    },
                )
                agent_reports[Domain.GENERAL] = await synthesis_agent.run()
                logger.info(
                    "Synthesis agent completed",
                    insights=len(agent_reports[Domain.GENERAL].insights),
                )
            except ImportError:
                logger.info("Synthesis agent not available, skipping")
            except Exception as e:
                logger.warning("Synthesis agent failed", error=str(e))

            # Generate and save report
            if agent_reports:
                report_config = ReportConfig(
                    output_dir=settings.data_dir / "reports",
                    archive_dir=settings.data_dir / "reports" / "archive",
                )
                generator = WeeklyReportGenerator(report_config)
                report = generator.generate_report(agent_reports)
                output_path, archive_path = generator.save_and_archive(report)

                logger.info(
                    "Weekly report generated",
                    report_id=report.report_id,
                    output_path=str(output_path),
                    total_insights=report.total_insights,
                    urgent=report.urgent_count,
                    action=report.action_count,
                )
            else:
                logger.warning("No agent reports generated, skipping report")

        finally:
            await db.close()

    except Exception as e:
        logger.error("Weekly report generation failed", error=str(e))
