"""CLI for Proactive Knowledge Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

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


if __name__ == "__main__":
    main()
