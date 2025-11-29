"""CLI for Proactive Knowledge Agent."""

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
from rich.table import Table

if TYPE_CHECKING:
    from src.models import SyncReport

console = Console()


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
@click.pass_context
def sync(ctx: click.Context, dry_run: bool, no_embeddings: bool) -> None:
    """Sync RSS feeds and store articles."""
    from src.config import Settings, get_all_feeds, get_feed_settings, load_feeds_config
    from src.database import Database
    from src.ingestion.embeddings import EmbeddingService
    from src.ingestion.rss_processor import RSSProcessor
    from src.repositories import ArticleRepository

    config_path = ctx.obj["config_path"]

    async def _sync() -> None:
        # Load configuration
        config = load_feeds_config(config_path)
        feeds = get_all_feeds(config)
        feed_settings = get_feed_settings(config)
        settings = Settings()

        console.print(f"[blue]Syncing {len(feeds)} feeds...[/blue]")

        # Fetch feeds
        async with RSSProcessor(feeds, feed_settings) as processor:
            articles, report = await processor.fetch_all_feeds()

        # Display fetch results
        _display_sync_report(report)

        if dry_run:
            console.print("[yellow]Dry run - skipping database storage[/yellow]")
            return

        if not articles:
            console.print("[yellow]No articles to process[/yellow]")
            return

        # Initialize database
        db = Database(settings.database_url)
        await db.connect()

        try:
            repo = ArticleRepository(db)

            # Generate embeddings if requested
            embeddings: list[list[float] | None] = [None] * len(articles)

            if not no_embeddings and settings.openai_api_key:
                console.print("[blue]Generating embeddings...[/blue]")
                embedding_service = EmbeddingService(
                    api_key=settings.openai_api_key,
                    model=feed_settings.embedding_model,
                )
                texts = [a.text_for_embedding for a in articles]
                embeddings = await embedding_service.generate_batch(texts)
                console.print(f"[green]Generated {len(embeddings)} embeddings[/green]")

            # Store articles
            new_count = 0
            updated_count = 0

            with console.status("[blue]Storing articles...[/blue]"):
                for article, embedding in zip(articles, embeddings):
                    _, is_new = await repo.upsert(article, embedding)
                    if is_new:
                        new_count += 1
                    else:
                        updated_count += 1

            console.print(
                f"[green]Stored {new_count} new articles, "
                f"updated {updated_count} existing[/green]"
            )

        finally:
            await db.close()

    asyncio.run(_sync())


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status."""
    from src.config import Settings, get_all_feeds, load_feeds_config
    from src.database import Database
    from src.repositories import ArticleRepository

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
                repo = ArticleRepository(db)
                total_count = await repo.count()
                sources = await repo.get_sources()
                categories = await repo.get_categories()

                stats_table = Table(title="Database Statistics")
                stats_table.add_column("Metric")
                stats_table.add_column("Value")

                stats_table.add_row("Total Articles", str(total_count))
                stats_table.add_row("Sources", str(len(sources)))
                stats_table.add_row("Categories", ", ".join(categories) or "None")

                console.print(stats_table)

                # Per-source breakdown
                if sources:
                    console.print()
                    source_table = Table(title="Articles by Source")
                    source_table.add_column("Source")
                    source_table.add_column("Count")

                    for source in sources:
                        count = await repo.count(source=source)
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
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    category: str | None,
    semantic: bool,
) -> None:
    """Search articles in the knowledge base."""
    from src.config import Settings, get_feed_settings, load_feeds_config
    from src.database import Database
    from src.ingestion.embeddings import EmbeddingService
    from src.repositories import ArticleRepository

    config_path = ctx.obj["config_path"]

    async def _search() -> None:
        config = load_feeds_config(config_path)
        feed_settings = get_feed_settings(config)
        settings = Settings()

        db = Database(settings.database_url)
        await db.connect()

        try:
            repo = ArticleRepository(db)

            if semantic and settings.openai_api_key:
                # Semantic search using embeddings
                console.print("[blue]Performing semantic search...[/blue]")
                embedding_service = EmbeddingService(
                    api_key=settings.openai_api_key,
                    model=feed_settings.embedding_model,
                )
                query_embedding = await embedding_service.generate(query)
                results = await repo.find_similar(
                    embedding=query_embedding,
                    limit=limit,
                    threshold=0.5,
                )

                if not results:
                    console.print("[yellow]No matching articles found[/yellow]")
                    return

                table = Table(title=f"Semantic Search Results for '{query}'")
                table.add_column("Title", max_width=50)
                table.add_column("Source")
                table.add_column("Similarity")
                table.add_column("Published")

                for article, similarity in results:
                    published = (
                        article.published.strftime("%Y-%m-%d")
                        if article.published
                        else "Unknown"
                    )
                    table.add_row(
                        article.title[:50],
                        article.source,
                        f"{similarity:.2%}",
                        published,
                    )

                console.print(table)

            else:
                # Text search
                articles = await repo.search_by_text(
                    search_text=query,
                    limit=limit,
                    category=category,
                )

                if not articles:
                    console.print("[yellow]No matching articles found[/yellow]")
                    return

                table = Table(title=f"Search Results for '{query}'")
                table.add_column("Title", max_width=50)
                table.add_column("Source")
                table.add_column("Category")
                table.add_column("Published")

                for article in articles:
                    published = (
                        article.published.strftime("%Y-%m-%d")
                        if article.published
                        else "Unknown"
                    )
                    table.add_row(
                        article.title[:50],
                        article.source,
                        article.category or "",
                        published,
                    )

                console.print(table)

        finally:
            await db.close()

    asyncio.run(_search())


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


if __name__ == "__main__":
    main()
