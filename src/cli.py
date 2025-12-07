"""CLI for Proactive Knowledge Agent.

Complete CLI for PKA with commands for:
- Database initialization
- Daemon management
- Data synchronization
- Search and queries
- Reports and alerts
- Entity exploration
- Fitness tracking
"""

from __future__ import annotations

import asyncio
import webbrowser
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from src.models import SyncReport

console = Console()


# ============================================================================
# Rich Formatting Helpers
# ============================================================================


def format_level_badge(level: str) -> Text:
    """Format alert level as colored badge."""
    colors = {
        "info": "blue",
        "watch": "yellow",
        "action": "magenta",
        "urgent": "red bold",
    }
    color = colors.get(level.lower(), "white")
    return Text(f"[{level.upper()}]", style=color)


def format_domain_badge(domain: str) -> Text:
    """Format domain as colored badge."""
    colors = {
        "tech": "cyan",
        "fitness": "green",
        "finance": "yellow",
        "general": "white",
    }
    color = colors.get(domain.lower(), "white")
    return Text(domain.upper(), style=color)


def format_trend_arrow(direction: str) -> str:
    """Format trend direction as arrow."""
    arrows = {
        "rising": "[green]↑[/green]",
        "falling": "[red]↓[/red]",
        "stable": "[yellow]→[/yellow]",
        "new": "[cyan]★[/cyan]",
    }
    return arrows.get(direction.lower(), "?")


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="config/feeds.yaml",
    help="Path to feeds configuration file",
)
@click.pass_context
def main(ctx: click.Context, config: Path) -> None:
    """Proactive Knowledge Agent - Ambient AI for Personal Intelligence."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize database schema."""
    from src.config import Settings
    from src.database import Database, init_schema

    async def _init() -> None:
        settings = Settings()
        db = Database(settings.database_url)
        await db.connect()
        try:
            await init_schema(db)
            console.print("[green]Database schema initialized successfully.[/green]")
        finally:
            await db.close()

    asyncio.run(_init())


@main.group()
def daemon() -> None:
    """Manage the background daemon process."""


def _get_pid_file() -> Path:
    """Get the path to the daemon PID file."""
    return Path.home() / ".pka" / "daemon.pid"


@daemon.command("start")
@click.option("--background", "-b", is_flag=True, help="Run daemon in background")
def daemon_start(background: bool) -> None:
    """Start the background scheduler.

    The daemon runs continuously, syncing data and generating insights.

    Examples:

        pka daemon start          # Run in foreground

        pka daemon start -b       # Run in background
    """
    import os
    import signal
    import sys

    from src.daemon.scheduler import SchedulerService

    pid_file = _get_pid_file()

    # Check if already running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is actually running
            os.kill(pid, 0)
            console.print(f"[yellow]Daemon already running (PID: {pid})[/yellow]")
            console.print("Use 'pka daemon stop' to stop it first.")
            return
        except (ProcessLookupError, ValueError):
            # Stale PID file, remove it
            pid_file.unlink()

    if background:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            # Parent process
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(pid))
            console.print(f"[green]Daemon started in background (PID: {pid})[/green]")
            console.print("Use 'pka daemon status' to check status.")
            console.print("Use 'pka daemon stop' to stop.")
            return
        else:
            # Child process - detach from terminal
            os.setsid()
            sys.stdin.close()
            sys.stdout.close()
            sys.stderr.close()
    else:
        # Write PID for foreground process too
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))
        console.print("[blue]Starting daemon (Ctrl+C to stop)...[/blue]")

    service = SchedulerService()

    def cleanup(signum: int, frame: object) -> None:
        if pid_file.exists():
            pid_file.unlink()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        asyncio.run(service.run_forever())
    except KeyboardInterrupt:
        pass
    finally:
        if pid_file.exists():
            pid_file.unlink()
        if not background:
            console.print("\n[yellow]Daemon stopped.[/yellow]")


@daemon.command("stop")
def daemon_stop() -> None:
    """Stop the running daemon.

    Sends a termination signal to the background daemon process.

    Examples:

        pka daemon stop
    """
    import os
    import signal

    pid_file = _get_pid_file()

    if not pid_file.exists():
        console.print("[yellow]No daemon running (PID file not found)[/yellow]")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Daemon stopped (PID: {pid})[/green]")
        pid_file.unlink()
    except ProcessLookupError:
        console.print("[yellow]Daemon not running (stale PID file removed)[/yellow]")
        pid_file.unlink()
    except ValueError:
        console.print("[red]Invalid PID file[/red]")
        pid_file.unlink()
    except PermissionError:
        console.print("[red]Permission denied - cannot stop daemon[/red]")


@daemon.command("status")
def daemon_status() -> None:
    """Check daemon status.

    Shows whether the daemon is running and its process ID.

    Examples:

        pka daemon status
    """
    import os

    pid_file = _get_pid_file()

    if not pid_file.exists():
        console.print("[yellow]Daemon not running[/yellow]")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        console.print(f"[green]Daemon running (PID: {pid})[/green]")
    except ProcessLookupError:
        console.print("[yellow]Daemon not running (stale PID file)[/yellow]")
        pid_file.unlink()
    except ValueError:
        console.print("[red]Invalid PID file[/red]")


@main.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Fetch feeds but don't store to database",
)
@click.option(
    "--no-embeddings",
    is_flag=True,
    help="Skip embedding generation",
)
@click.option("--rss", is_flag=True, help="Sync only RSS feeds")
@click.option("--market", is_flag=True, help="Sync only Market data")
@click.option("--podcast", is_flag=True, help="Sync only Podcast data")
@click.pass_context
def sync(
    ctx: click.Context,
    dry_run: bool,
    no_embeddings: bool,
    rss: bool,
    market: bool,
    podcast: bool,
) -> None:
    """Sync data from enabled sources."""
    from src.cache import Cache
    from src.config import Settings, get_all_feeds, get_feed_settings, load_feeds_config
    from src.database import Database
    from src.ingestion.embeddings import EmbeddingService
    from src.ingestion.market_client import sync_market_data
    from src.ingestion.podcast_bridge import sync_podcast_data
    from src.ingestion.rss_processor import RSSProcessor
    from src.repositories import (
        ArticleRepository,
        MarketRepository,
        PodcastRepository,
    )

    config_path = ctx.obj["config_path"]

    # If no specific source is selected, sync all
    sync_all = not (rss or market or podcast)
    if sync_all:
        rss = market = podcast = True

    async def _sync() -> None:
        settings = Settings()
        db = None
        cache = None

        if dry_run:
            console.print("[yellow]Dry run enabled - no data will be stored[/yellow]")

        # Initialize DB if not dry-run
        if not dry_run:
            db = Database(settings.database_url)
            await db.connect()

            if settings.redis_url:
                cache = Cache(settings.redis_url)
                try:
                    await cache.connect()
                except Exception as e:
                    console.print(
                        f"[yellow]Redis connection failed: {e}. "
                        "Caching disabled.[/yellow]"
                    )
                    cache = None

        try:
            # --- RSS Sync ---
            if rss:
                config = load_feeds_config(config_path)
                feeds = get_all_feeds(config)
                feed_settings = get_feed_settings(config)

                console.print(f"\n[blue]Syncing {len(feeds)} RSS feeds...[/blue]")
                async with RSSProcessor(feeds, feed_settings) as processor:
                    articles, report = await processor.fetch_all_feeds()
                _display_sync_report(report)

                if not dry_run and articles and db:
                    repo = ArticleRepository(db)
                    # Embeddings
                    embeddings = [None] * len(articles)
                    if not no_embeddings and settings.openai_api_key:
                        console.print("[blue]Generating article embeddings...[/blue]")
                        embedding_service = EmbeddingService(
                            api_key=settings.openai_api_key,
                            model=feed_settings.embedding_model,
                            cache=cache,
                        )
                        texts = [a.text_for_embedding for a in articles]
                        embeddings = await embedding_service.generate_batch(texts)

                    with console.status("[blue]Storing articles...[/blue]"):
                        new_c, upd_c = 0, 0
                        for art, emb in zip(articles, embeddings):
                            _, is_new = await repo.upsert(art, emb)
                            if is_new:
                                new_c += 1
                            else:
                                upd_c += 1
                        console.print(
                            f"[green]Articles: {new_c} new, {upd_c} updated[/green]"
                        )

            # --- Market Sync ---
            if market:
                # TODO: Load symbols from config
                symbols = [
                    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
                    "SPY", "QQQ", "BTC-USD", "ETH-USD",
                ]
                console.print(
                    f"\n[blue]Syncing Market data for {len(symbols)} symbols...[/blue]"
                )

                market_result = await sync_market_data(symbols)
                if market_result.success:
                    console.print(
                        f"[green]Fetched {len(market_result.data)} data points[/green]"
                    )
                    if not dry_run and market_result.data and db:
                        market_repo = MarketRepository(db)
                        with console.status("[blue]Storing market data...[/blue]"):
                            count = await market_repo.upsert_batch(market_result.data)
                            console.print(f"[green]Stored {count} records[/green]")
                else:
                    console.print(
                        f"[red]Market sync failed: {market_result.errors}[/red]"
                    )

            # --- Podcast Sync ---
            if podcast:
                if settings.p3_duckdb_path:
                    console.print("\n[blue]Syncing Podcasts from P³...[/blue]")
                    pod_result = await sync_podcast_data(settings.p3_duckdb_path)

                    if pod_result.success:
                        console.print(f"[green]Fetched {len(pod_result.episodes)} episodes[/green]")
                        if not dry_run and pod_result.episodes and db:
                            pod_repo = PodcastRepository(db)
                            # Embeddings (optional, if model supports it and not disabled)
                            # For now assuming embeddings might be generated here or skipped
                            # Let's generate if API key present
                            pod_embeddings = [None] * len(pod_result.episodes)
                            if not no_embeddings and settings.openai_api_key:
                                 console.print("[blue]Generating podcast embeddings...[/blue]")
                                 emb_svc = EmbeddingService(settings.openai_api_key, cache=cache)
                                 texts = [e.text_for_embedding for e in pod_result.episodes]
                                 pod_embeddings = await emb_svc.generate_batch(texts)

                            with console.status("[blue]Storing episodes...[/blue]"):
                                count = 0
                                for ep, emb in zip(pod_result.episodes, pod_embeddings):
                                    await pod_repo.upsert(ep, emb)
                                    count += 1
                                console.print(f"[green]Stored {count} episodes[/green]")
                    else:
                         console.print(f"[red]Podcast sync failed: {pod_result.errors}[/red]")
                else:
                    console.print("[yellow]Skipping Podcast sync: p3_duckdb_path not set[/yellow]")

        finally:
            if db:
                await db.close()

    asyncio.run(_sync())


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    from src.config import Settings, get_all_feeds, load_feeds_config
    from src.database import Database
    from src.repositories import (
        ArticleRepository,
        MarketRepository,
        PodcastRepository,
    )

    config_path = ctx.obj["config_path"]

    async def _status() -> None:
        # Load configuration
        config = load_feeds_config(config_path)
        feeds = get_all_feeds(config)
        settings = Settings()

        console.print("[bold]Proactive Knowledge Agent Status[/bold]\n")

        # Feed configuration
        table = Table(title="Configured Feeds")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Priority")

        for feed in feeds:
            table.add_row(feed.name, feed.category, feed.priority)

        console.print(table)
        console.print()

        # Database statistics
        try:
            db = Database(settings.database_url)
            await db.connect()

            try:
                # RSS Stats
                article_repo = ArticleRepository(db)
                total_articles = await article_repo.count()
                sources = await article_repo.get_sources()
                categories = await article_repo.get_categories()

                # Market Stats
                market_repo = MarketRepository(db)
                total_market = await market_repo.count()
                symbols = await market_repo.get_symbols()

                # Podcast Stats
                podcast_repo = PodcastRepository(db)
                total_episodes = await podcast_repo.count()
                podcasts = await podcast_repo.get_podcasts()

                stats_table = Table(title="Database Statistics")
                stats_table.add_column("Domain")
                stats_table.add_column("Metric")
                stats_table.add_column("Value")

                # RSS Rows
                stats_table.add_row("RSS", "Total Articles", str(total_articles))
                stats_table.add_row("RSS", "Sources", str(len(sources)))
                stats_table.add_row("RSS", "Categories", str(len(categories)))

                # Market Rows
                stats_table.add_row("Market", "Data Points", str(total_market))
                stats_table.add_row("Market", "Symbols", ", ".join(symbols) or "None")

                # Podcast Rows
                stats_table.add_row("Podcast", "Episodes", str(total_episodes))
                stats_table.add_row("Podcast", "Shows", str(len(podcasts)))

                console.print(stats_table)

                # Per-source breakdown (RSS)
                if sources:
                    console.print()
                    source_table = Table(title="Articles by Source")
                    source_table.add_column("Source")
                    source_table.add_column("Count")

                    for source in sources:
                        count = await article_repo.count(source=source)
                        source_table.add_row(source, str(count))

                    console.print(source_table)

            finally:
                await db.close()

        except Exception as e:
            console.print(f"[red]Database not available: {e}[/red]")

    asyncio.run(_status())


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results to return")
@click.option("--category", "-c", help="Filter by category")
@click.option("--semantic", "-s", is_flag=True, help="Use semantic search")
@click.option("--podcasts", "-p", is_flag=True, help="Search podcasts only")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    category: str | None,
    semantic: bool,
    podcasts: bool,
) -> None:
    """Search articles and podcasts in the knowledge base."""
    from src.config import Settings, get_feed_settings, load_feeds_config
    from src.database import Database
    from src.ingestion.embeddings import EmbeddingService
    from src.repositories import ArticleRepository, PodcastRepository

    config_path = ctx.obj["config_path"]

    async def _search() -> None:
        config = load_feeds_config(config_path)
        feed_settings = get_feed_settings(config)
        settings = Settings()

        db = Database(settings.database_url)
        await db.connect()

        try:
            article_repo = ArticleRepository(db)
            podcast_repo = PodcastRepository(db)

            results_table = Table(title=f"Search Results for '{query}'")
            results_table.add_column("Type", style="cyan")
            results_table.add_column("Title", max_width=40)
            results_table.add_column("Source", style="green")
            results_table.add_column("Published")
            results_table.add_column("Score", style="magenta")

            if semantic and settings.openai_api_key:
                console.print("[blue]Performing semantic search...[/blue]")
                embedding_service = EmbeddingService(
                    api_key=settings.openai_api_key,
                    model=feed_settings.embedding_model,
                )
                query_embedding = await embedding_service.generate(query)

                # Search Articles
                if not podcasts:
                    art_results = await article_repo.find_similar(
                        embedding=query_embedding,
                        limit=limit,
                        threshold=0.5,
                    )
                    for art, score in art_results:
                         results_table.add_row(
                            "Article",
                            art.title[:40],
                            art.source,
                            art.published.strftime("%Y-%m-%d") if art.published else "?",
                            f"{score:.2%}"
                        )

                # Search Podcasts
                if podcasts or not category: # If category is set, skip podcasts (no category field)
                    pod_results = await podcast_repo.search_similar(
                        embedding=query_embedding,
                        limit=limit,
                        threshold=0.5
                    )
                    for pod, score in pod_results:
                        results_table.add_row(
                            "Podcast",
                            pod.title[:40],
                            pod.podcast_name,
                            pod.published_date.strftime("%Y-%m-%d") if pod.published_date else "?",
                            f"{score:.2%}"
                        )

            else:
                # Text search
                # Articles
                if not podcasts:
                    articles = await article_repo.search_by_text(
                        search_text=query,
                        limit=limit,
                        category=category,
                    )
                    for art in articles:
                        results_table.add_row(
                            "Article",
                            art.title[:40],
                            art.source,
                            art.published.strftime("%Y-%m-%d") if art.published else "?",
                            "-"
                        )

                # Podcasts
                if (podcasts or not category):
                    episodes = await podcast_repo.search_by_text(
                        search_text=query,
                        limit=limit
                    )
                    for ep in episodes:
                        results_table.add_row(
                            "Podcast",
                            ep.title[:40],
                            ep.podcast_name,
                            ep.published_date.strftime("%Y-%m-%d") if ep.published_date else "?",
                            "-"
                        )

            if results_table.row_count > 0:
                console.print(results_table)
            else:
                console.print("[yellow]No results found.[/yellow]")

        finally:
            await db.close()

    asyncio.run(_search())


# =============================================================================
# Report Commands
# =============================================================================


@main.command()
@click.argument("report_type", default="weekly", type=click.Choice(["weekly", "daily"]))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--no-archive", is_flag=True, help="Don't archive the report")
@click.option("--view", is_flag=True, help="Open report in browser after generation")
@click.pass_context
def report(
    ctx: click.Context,
    report_type: str,
    output: Path | None,
    no_archive: bool,
    view: bool,
) -> None:
    """Generate an intelligence report.

    Generates a comprehensive report combining insights from all domain agents.
    Reports are saved in Markdown format.

    Examples:

        pka report              # Generate weekly report

        pka report daily        # Generate daily report

        pka report --view       # Generate and open in browser
    """
    from src.agents import AgentReport, AlertLevel, Insight, InsightType
    from src.config import Settings
    from src.database import Database
    from src.outputs.reports import ReportConfig, WeeklyReportGenerator
    from src.world_model import Domain, EntityRepository, TemporalReasoningService

    async def _generate_report() -> None:
        settings = Settings()
        db = Database(settings.database_url)
        await db.connect()

        try:
            console.print(f"[blue]Generating {report_type} report...[/blue]")

            # Set up report configuration
            config = ReportConfig()
            if output:
                config.output_dir = output
                config.archive_dir = output / "archive"

            # Get data from services
            entity_repo = EntityRepository(db)
            temporal_service = TemporalReasoningService(db)

            # Build agent reports for each domain
            agent_reports: dict[Domain, AgentReport] = {}

            for domain in [Domain.TECH, Domain.FITNESS, Domain.FINANCE]:
                console.print(f"  Analyzing {domain.value} domain...")

                # Get trends
                trends = await temporal_service.analyze_trends(
                    domain=domain,
                    period_days=7 if report_type == "weekly" else 1,
                    limit=10,
                )

                # Get entities
                entities = await entity_repo.find_by_domain(domain, limit=20)

                # Generate insights from trends
                insights = []
                for trend in trends[:5]:
                    insight = Insight(
                        insight_type=InsightType.TREND_CHANGE,
                        title=f"{trend.entity_name} is {trend.direction.value}",
                        description=(
                            f"{trend.entity_name} has "
                            f"{'increased' if trend.change_ratio > 0 else 'decreased'} "
                            f"by {abs(trend.change_ratio) * 100:.0f}% "
                            f"over the past {trend.period_days} days."
                        ),
                        level=AlertLevel.WATCH,
                        confidence=trend.confidence,
                        relevance_score=0.8,
                        entity_names=[trend.entity_name],
                    )
                    insights.append(insight)

                agent_reports[domain] = AgentReport(
                    domain=domain,
                    insights=insights,
                    trends_analyzed=len(trends),
                    entities_scanned=len(entities),
                    alerts_generated=len([i for i in insights if i.level != AlertLevel.INFO]),
                )

            # Generate report
            generator = WeeklyReportGenerator(config)
            weekly_report = generator.generate_report(
                agent_reports,
                period_days=7 if report_type == "weekly" else 1,
            )

            # Save report
            if no_archive:
                output_path = generator.save_report(weekly_report)
            else:
                output_path, _ = generator.save_and_archive(weekly_report)

            console.print(f"\n[green]Report generated:[/green] {output_path}")

            # Display summary
            console.print(Panel(
                weekly_report.executive_summary,
                title="Executive Summary",
                border_style="blue",
            ))

            console.print("\n[bold]Statistics:[/bold]")
            console.print(f"  Total Insights: {weekly_report.total_insights}")
            console.print(f"  Urgent: {weekly_report.urgent_count}")
            console.print(f"  Action Items: {weekly_report.action_count}")

            # Open in browser if requested
            if view:
                import webbrowser
                webbrowser.open(f"file://{output_path.absolute()}")

        finally:
            await db.close()

    asyncio.run(_generate_report())


# =============================================================================
# Alerts Commands
# =============================================================================


@main.command()
@click.option(
    "--level", "-l",
    type=click.Choice(["info", "watch", "action", "urgent"]),
    help="Filter by minimum alert level",
)
@click.option("--limit", "-n", default=20, help="Maximum alerts to show")
@click.option("--acknowledge", "-a", type=str, help="Acknowledge alert by ID")
@click.option("--snooze", "-s", type=str, help="Snooze alert by ID (hours)")
@click.pass_context
def alerts(
    ctx: click.Context,
    level: str | None,
    limit: int,
    acknowledge: str | None,
    snooze: str | None,
) -> None:
    """Show and manage pending alerts.

    Displays alerts from the PKA system that require attention.
    Supports filtering by level and acknowledging/snoozing alerts.

    Examples:

        pka alerts                  # Show all pending alerts

        pka alerts -l action        # Show only ACTION and URGENT

        pka alerts -a alert_123     # Acknowledge an alert

        pka alerts -s alert_123:2   # Snooze for 2 hours
    """
    from src.agents import AlertLevel
    from src.outputs.alerts import AlertManager, AlertManagerConfig

    # Create alert manager (in-memory for now)
    manager = AlertManager(AlertManagerConfig())

    # Handle acknowledge
    if acknowledge:
        success = manager.acknowledge_alert(acknowledge)
        if success:
            console.print(f"[green]Alert {acknowledge} acknowledged[/green]")
        else:
            console.print(f"[red]Alert {acknowledge} not found[/red]")
        return

    # Handle snooze
    if snooze:
        if ":" in snooze:
            alert_id, hours = snooze.split(":", 1)
            hours_int = int(hours)
        else:
            alert_id = snooze
            hours_int = 1

        success = manager.snooze_alert(alert_id, hours=hours_int)
        if success:
            console.print(f"[green]Alert {alert_id} snoozed for {hours_int} hour(s)[/green]")
        else:
            console.print(f"[red]Alert {alert_id} not found[/red]")
        return

    # Get pending alerts
    pending_alerts = manager.get_pending_alerts()

    # Filter by level if specified
    if level:
        level_priority = {
            "info": 0, "watch": 1, "action": 2, "urgent": 3,
        }
        min_priority = level_priority.get(level, 0)
        pending_alerts = [
            a for a in pending_alerts
            if level_priority.get(a.level.value, 0) >= min_priority
        ]

    # Apply limit
    pending_alerts = pending_alerts[:limit]

    if not pending_alerts:
        console.print("[green]No pending alerts.[/green]")
        return

    # Display alerts
    console.print(f"\n[bold]Pending Alerts ({len(pending_alerts)})[/bold]\n")

    for alert in pending_alerts:
        level_color = {
            AlertLevel.INFO: "blue",
            AlertLevel.WATCH: "yellow",
            AlertLevel.ACTION: "magenta",
            AlertLevel.URGENT: "red",
        }.get(alert.level, "white")

        console.print(Panel(
            f"[bold]{alert.title}[/bold]\n\n"
            f"{alert.description}\n\n"
            f"[dim]Source: {alert.source} | Created: {alert.created_at.strftime('%Y-%m-%d %H:%M')}[/dim]",
            title=f"[{level_color}][{alert.level.value.upper()}][/{level_color}] {alert.alert_id}",
            border_style=level_color,
        ))

    # Show stats
    stats = manager.get_stats()
    console.print(f"\n[dim]Total: {stats['total_alerts']} | "
                  f"Pending: {stats['pending_count']} | "
                  f"Actionable: {stats['actionable_count']}[/dim]")


# =============================================================================
# Entity Commands
# =============================================================================


@main.command()
@click.argument("name")
@click.option(
    "--type", "-t", "entity_type",
    type=click.Choice(["technology", "company", "person", "concept", "metric", "event"]),
    help="Filter by entity type",
)
@click.option("--trends", is_flag=True, help="Show trend analysis")
@click.option("--related", is_flag=True, help="Show related entities")
@click.pass_context
def entity(
    ctx: click.Context,
    name: str,
    entity_type: str | None,
    trends: bool,
    related: bool,
) -> None:
    """Show details about a knowledge graph entity.

    Displays comprehensive information about an entity including
    metadata, trends, and related entities.

    Examples:

        pka entity Python               # Look up "Python"

        pka entity "Claude AI" -t technology   # Specific type

        pka entity Rust --trends        # Include trend data
    """
    from src.config import Settings
    from src.database import Database
    from src.world_model import (
        EntityRepository,
        EntityType,
        RelationshipRepository,
        TemporalReasoningService,
    )

    async def _show_entity() -> None:
        settings = Settings()
        db = Database(settings.database_url)
        await db.connect()

        try:
            entity_repo = EntityRepository(db)
            temporal_service = TemporalReasoningService(db)

            # Search for entity
            etype = EntityType(entity_type) if entity_type else None
            entities = await entity_repo.search_by_name(
                search_text=name,
                limit=5,
                entity_type=etype,
            )

            if not entities:
                console.print(f"[yellow]Entity not found: {name}[/yellow]")
                return

            # Get best match
            ent = next(
                (e for e in entities if e.name.lower() == name.lower()),
                entities[0],
            )

            # Display entity info
            domain_color = {
                "tech": "cyan",
                "fitness": "green",
                "finance": "yellow",
            }.get(ent.domain.value, "white")

            console.print(Panel(
                f"[bold]{ent.name}[/bold]\n\n"
                f"[dim]Type:[/dim] {ent.entity_type.value}\n"
                f"[dim]Domain:[/dim] [{domain_color}]{ent.domain.value.upper()}[/{domain_color}]\n"
                f"[dim]Description:[/dim] {ent.description or 'N/A'}\n\n"
                f"[dim]Confidence:[/dim] {ent.confidence:.0%}\n"
                f"[dim]Mentions:[/dim] {ent.mention_count}\n"
                f"[dim]First Seen:[/dim] {ent.first_seen.strftime('%Y-%m-%d') if ent.first_seen else 'N/A'}\n"
                f"[dim]Last Seen:[/dim] {ent.last_seen.strftime('%Y-%m-%d') if ent.last_seen else 'N/A'}",
                title=f"Entity: {ent.name}",
                border_style=domain_color,
            ))

            # Show properties if any
            if ent.properties:
                props_table = Table(title="Properties")
                props_table.add_column("Key")
                props_table.add_column("Value")
                for key, value in ent.properties.items():
                    props_table.add_row(str(key), str(value))
                console.print(props_table)

            # Show trends if requested
            if trends and ent.id:
                console.print("\n[bold]Trend Analysis[/bold]")
                trend_results = await temporal_service.analyze_trends(
                    domain=ent.domain,
                    period_days=7,
                    limit=20,
                )

                entity_trend = next(
                    (t for t in trend_results if t.entity_id == ent.id),
                    None,
                )

                if entity_trend:
                    arrow = format_trend_arrow(entity_trend.direction.value)
                    console.print(
                        f"  Direction: {arrow} {entity_trend.direction.value}\n"
                        f"  Change: {entity_trend.change_ratio * 100:+.1f}%\n"
                        f"  Current: {entity_trend.current_count} mentions\n"
                        f"  Previous: {entity_trend.previous_count} mentions"
                    )
                else:
                    console.print("  [dim]No trend data available[/dim]")

            # Show related entities if requested
            if related and ent.id:
                console.print("\n[bold]Related Entities[/bold]")
                rel_repo = RelationshipRepository(db)
                relationships = await rel_repo.get_relationships_for_entity(ent.id, limit=10)

                if relationships:
                    rel_table = Table()
                    rel_table.add_column("Relationship")
                    rel_table.add_column("Entity")
                    rel_table.add_column("Strength")

                    for rel in relationships:
                        # Get related entity name
                        related_id = (
                            rel.target_entity_id
                            if rel.source_entity_id == ent.id
                            else rel.source_entity_id
                        )
                        related_ent = await entity_repo.get_by_id(related_id)
                        if related_ent:
                            rel_table.add_row(
                                rel.relationship_type.value,
                                related_ent.name,
                                f"{rel.strength:.0%}",
                            )

                    console.print(rel_table)
                else:
                    console.print("  [dim]No relationships found[/dim]")

            # Show similar entities
            if len(entities) > 1:
                console.print("\n[dim]Similar entities:[/dim]")
                for similar in entities[1:4]:
                    console.print(f"  - {similar.name} ({similar.entity_type.value})")

        finally:
            await db.close()

    asyncio.run(_show_entity())


def _display_sync_report(report: SyncReport) -> None:
    """Display sync report as rich table."""
    table = Table(title="Feed Sync Results")
    table.add_column("Feed")
    table.add_column("Status")
    table.add_column("Articles")
    table.add_column("Time (ms)")

    for result in report.results:
        status = "[green]OK[/green]" if result.success else f"[red]{result.error}[/red]"
        table.add_row(
            result.feed_name,
            status,
            str(result.articles_found),
            f"{result.duration_ms:.0f}",
        )

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {report.articles_found} articles from "
        f"{report.feeds_processed} feeds in {report.duration_seconds:.2f}s "
        f"({report.success_rate * 100:.0f}% success rate)"
    )


# =============================================================================
# Fitness Commands
# =============================================================================


@main.group()
def fitness() -> None:
    """Fitness tracking commands (Strava integration)."""


@fitness.command("auth")
@click.option("--port", default=8080, help="Local server port for OAuth callback")
def fitness_auth(port: int) -> None:
    """Authenticate with Strava using OAuth2."""
    from src.config import Settings
    from src.database import Database
    from src.fitness_repository import StravaTokenRepository
    from src.ingestion.fitness_client import StravaAuth

    settings = Settings()

    if not settings.strava_client_id or not settings.strava_client_secret:
        console.print(
            "[red]Error: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set[/red]"
        )
        console.print(
            "\nGet your credentials at: https://www.strava.com/settings/api"
        )
        return

    auth = StravaAuth(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        redirect_uri=f"http://localhost:{port}/callback",
    )

    # Container for authorization code
    auth_code: dict[str, str | None] = {"code": None}

    class OAuthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/callback":
                query = parse_qs(parsed.query)
                if "code" in query:
                    auth_code["code"] = query["code"][0]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>Authorization successful!</h1>"
                        b"<p>You can close this window.</p></body></html>"
                    )
                else:
                    error = query.get("error", ["Unknown error"])[0]
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        f"<html><body><h1>Error: {error}</h1></body></html>".encode()
                    )
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass  # Suppress logging

    # Generate authorization URL
    auth_url = auth.get_authorization_url(scope="activity:read_all")

    console.print("\n[bold]Strava OAuth Authorization[/bold]\n")
    console.print("Opening browser for Strava authorization...")
    console.print(f"\nIf browser doesn't open, visit:\n[blue]{auth_url}[/blue]\n")

    # Open browser
    webbrowser.open(auth_url)

    # Start local server
    server = HTTPServer(("localhost", port), OAuthHandler)
    console.print(f"Waiting for authorization on port {port}...")

    # Handle single request
    server.handle_request()

    if not auth_code["code"]:
        console.print("[red]Authorization failed - no code received[/red]")
        return

    # Exchange code for tokens
    async def _exchange_and_save() -> None:
        try:
            tokens = await auth.exchange_code(auth_code["code"])  # type: ignore

            # Save tokens to database
            db = Database(settings.database_url)
            await db.connect()

            try:
                token_repo = StravaTokenRepository(db)
                await token_repo.save_tokens(
                    athlete_id=str(tokens["athlete"]["id"]),
                    access_token=tokens["access_token"],
                    refresh_token=tokens["refresh_token"],
                    expires_at=tokens["expires_at"],
                )

                athlete = tokens["athlete"]
                firstname = athlete.get("firstname", "")
                lastname = athlete.get("lastname", "")
                console.print(
                    f"\n[green]Successfully authenticated as "
                    f"{firstname} {lastname}![/green]"
                )
                console.print(
                    f"Athlete ID: {athlete['id']}"
                )
            finally:
                await db.close()

        except Exception as e:
            console.print(f"[red]Error exchanging authorization code: {e}[/red]")

    asyncio.run(_exchange_and_save())


@fitness.command("sync")
@click.option("--days", default=90, help="Number of days to sync (default: 90)")
@click.option("--dry-run", is_flag=True, help="Fetch activities but don't store")
def fitness_sync(days: int, dry_run: bool) -> None:
    """Sync activities from Strava."""
    from src.config import Settings
    from src.database import Database
    from src.fitness_repository import (
        ActivityRepository,
        FitnessMetricsRepository,
        StravaTokenRepository,
    )
    from src.ingestion.fitness_client import (
        StravaAuth,
        TrainingMetricsCalculator,
        sync_strava_activities,
    )

    settings = Settings()

    async def _sync() -> None:
        db = Database(settings.database_url)
        await db.connect()

        try:
            # Get stored tokens
            token_repo = StravaTokenRepository(db)
            tokens = await token_repo.get_default_tokens()

            if not tokens:
                console.print(
                    "[red]No Strava credentials found. "
                    "Run 'pka fitness auth' first.[/red]"
                )
                return

            # Initialize auth with stored tokens
            auth = StravaAuth(
                client_id=settings.strava_client_id or "",
                client_secret=settings.strava_client_secret or "",
            )
            auth.set_tokens(
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
            )

            console.print(f"[blue]Syncing activities for last {days} days...[/blue]")

            # Fetch activities
            result = await sync_strava_activities(auth, days=days)

            if not result.success:
                console.print(f"[red]Sync failed: {result.errors}[/red]")
                return

            console.print(
                f"[green]Fetched {result.activities_fetched} activities[/green]"
            )

            if dry_run:
                console.print("[yellow]Dry run - skipping database storage[/yellow]")
                _display_activities_table(result.activities[:10])
                return

            # Store activities
            activity_repo = ActivityRepository(db)
            metrics_repo = FitnessMetricsRepository(db)

            new_count = 0
            updated_count = 0

            with console.status("[blue]Storing activities...[/blue]"):
                for activity in result.activities:
                    _, is_new = await activity_repo.upsert(activity)
                    if is_new:
                        new_count += 1
                    else:
                        updated_count += 1

            console.print(
                f"[green]Stored {new_count} new activities, "
                f"updated {updated_count} existing[/green]"
            )

            # Update tokens if refreshed
            if auth.access_token != tokens["access_token"]:
                await token_repo.save_tokens(
                    athlete_id=tokens["athlete_id"],
                    access_token=auth.access_token or "",
                    refresh_token=auth._refresh_token or "",
                    expires_at=auth._expires_at,
                )
                console.print("[blue]Tokens refreshed and saved[/blue]")

            # Calculate and store daily metrics
            console.print("[blue]Calculating training metrics...[/blue]")

            # Get daily TSS for the period
            start_date = date.today() - timedelta(days=days + 42)  # Extra for CTL
            daily_tss = await activity_repo.get_daily_tss(start_date)

            if daily_tss:
                # Fill in missing days with 0 TSS
                all_days: list[float] = []
                current = start_date
                end = date.today()

                while current <= end:
                    all_days.append(daily_tss.get(current, 0))
                    current += timedelta(days=1)

                # Calculate metrics
                calculator = TrainingMetricsCalculator()
                metrics = calculator.calculate_training_load(all_days)
                metrics.date = end  # type: ignore
                metrics.daily_tss = daily_tss.get(end, 0)

                await metrics_repo.upsert(metrics)

                console.print(
                    f"\n[bold]Current Training Status:[/bold]\n"
                    f"  CTL (Fitness): {metrics.ctl:.1f}\n"
                    f"  ATL (Fatigue): {metrics.atl:.1f}\n"
                    f"  TSB (Form): {metrics.tsb:.1f} ({metrics.form_status})\n"
                    f"  Injury Risk: {metrics.injury_risk}"
                )

        finally:
            await db.close()

    asyncio.run(_sync())


@fitness.command("status")
def fitness_status() -> None:
    """Show fitness tracking status and metrics."""
    from src.config import Settings
    from src.database import Database
    from src.fitness_repository import (
        ActivityRepository,
        FitnessMetricsRepository,
        StravaTokenRepository,
    )

    settings = Settings()

    async def _status() -> None:
        db = Database(settings.database_url)
        await db.connect()

        try:
            token_repo = StravaTokenRepository(db)
            activity_repo = ActivityRepository(db)
            metrics_repo = FitnessMetricsRepository(db)

            console.print("[bold]Fitness Tracking Status[/bold]\n")

            # Check authentication
            tokens = await token_repo.get_default_tokens()
            if tokens:
                athlete_id = tokens["athlete_id"]
                console.print(
                    f"[green]✓[/green] Authenticated (Athlete ID: {athlete_id})"
                )
            else:
                console.print(
                    "[yellow]✗[/yellow] Not authenticated - "
                    "run 'pka fitness auth' to connect Strava"
                )
                return

            # Activity statistics
            console.print()
            total_activities = await activity_repo.count()
            activity_types = await activity_repo.get_activity_types()

            stats_table = Table(title="Activity Statistics")
            stats_table.add_column("Metric")
            stats_table.add_column("Value")

            stats_table.add_row("Total Activities", str(total_activities))
            stats_table.add_row("Activity Types", ", ".join(activity_types) or "None")

            for activity_type in activity_types[:5]:
                count = await activity_repo.count(activity_type=activity_type)
                stats_table.add_row(f"  {activity_type}", str(count))

            console.print(stats_table)

            # Recent activities
            recent = await activity_repo.get_recent(limit=5)
            if recent:
                console.print()
                _display_activities_table(recent)

            # Current metrics
            latest_metrics = await metrics_repo.get_latest()
            if latest_metrics:
                console.print()
                metrics_table = Table(title="Training Metrics")
                metrics_table.add_column("Metric")
                metrics_table.add_column("Value")
                metrics_table.add_column("Status")

                metrics_table.add_row(
                    "CTL (Fitness)",
                    f"{latest_metrics.ctl:.1f}",
                    "",
                )
                metrics_table.add_row(
                    "ATL (Fatigue)",
                    f"{latest_metrics.atl:.1f}",
                    "",
                )
                metrics_table.add_row(
                    "TSB (Form)",
                    f"{latest_metrics.tsb:.1f}",
                    latest_metrics.form_status,
                )
                metrics_table.add_row(
                    "Injury Risk",
                    "",
                    latest_metrics.injury_risk,
                )

                console.print(metrics_table)

        finally:
            await db.close()

    asyncio.run(_status())


def _display_activities_table(activities: list) -> None:
    """Display activities as a rich table."""
    from src.models import Activity

    table = Table(title="Recent Activities")
    table.add_column("Date")
    table.add_column("Name", max_width=30)
    table.add_column("Type")
    table.add_column("Distance")
    table.add_column("Duration")
    table.add_column("TSS")

    for activity in activities:
        if isinstance(activity, Activity):
            table.add_row(
                activity.start_date.strftime("%Y-%m-%d"),
                activity.name[:30],
                activity.activity_type,
                f"{activity.distance_km:.1f} km",
                activity.duration_formatted,
                f"{activity.tss:.0f}" if activity.tss else "-",
            )

    console.print(table)


# =============================================================================
# Shell Completion
# =============================================================================


@main.command()
@click.argument(
    "shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    required=False,
)
def completion(shell: str | None) -> None:
    """Generate shell completion script.

    Outputs a shell completion script that can be sourced to enable
    tab completion for all PKA commands.

    Examples:

        pka completion bash >> ~/.bashrc   # Bash

        pka completion zsh >> ~/.zshrc     # Zsh

        pka completion fish > ~/.config/fish/completions/pka.fish

    \b
    Without argument, auto-detects current shell:

        eval "$(pka completion)"
    """
    import os

    # Auto-detect shell if not specified
    if shell is None:
        shell_path = os.environ.get("SHELL", "")
        if "zsh" in shell_path:
            shell = "zsh"
        elif "fish" in shell_path:
            shell = "fish"
        else:
            shell = "bash"

    # Get the completion script using click's built-in support
    env_var = "_PKA_COMPLETE"

    if shell == "bash":
        script = f'''
# PKA Bash completion
eval "$({env_var}=bash_source pka)"
'''
    elif shell == "zsh":
        script = f'''
# PKA Zsh completion
eval "$({env_var}=zsh_source pka)"
'''
    elif shell == "fish":
        script = f'''
# PKA Fish completion
eval ({env_var}=fish_source pka)
'''
    else:
        script = ""

    click.echo(script.strip())


@main.command()
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def version(format: str) -> None:
    """Show PKA version and system information.

    Examples:

        pka version           # Human-readable output

        pka version -f json   # JSON output for scripts
    """
    import platform
    import sys
    from importlib.metadata import version as pkg_version

    try:
        pka_version = pkg_version("proactive-knowledge-agent")
    except Exception:
        pka_version = "development"

    info = {
        "version": pka_version,
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "architecture": platform.machine(),
    }

    if format == "json":
        import json
        click.echo(json.dumps(info, indent=2))
    else:
        console.print(f"[bold]Proactive Knowledge Agent[/bold] v{info['version']}")
        console.print(f"  Python: {info['python']}")
        console.print(f"  Platform: {info['platform']} ({info['architecture']})")


if __name__ == "__main__":
    main()
