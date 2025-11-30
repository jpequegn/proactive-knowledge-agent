"""Bridge for importing podcast data from P³ DuckDB."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import structlog
from pydantic import BaseModel

from src.models import PodcastEpisode, PodcastSyncResult

logger = structlog.get_logger()


class P3Bridge:
    """Bridge to read data from P³ (Podcast Processing Pipeline) DuckDB."""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        """Establish connection to DuckDB."""
        try:
            self._conn = duckdb.connect(self.db_path, read_only=True)
            logger.info("Connected to P³ DuckDB", path=self.db_path)
        except Exception as e:
            logger.error("Failed to connect to DuckDB", path=self.db_path, error=str(e))
            raise

    def close(self) -> None:
        """Close DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_recent_episodes(self, days: int = 7) -> list[PodcastEpisode]:
        """
        Fetch episodes from the last N days.

        Assumes a schema compatible with P³:
        - episodes (id, title, podcast, published, url, duration)
        - summaries (episode_id, content)
        - topics (episode_id, topic)
        """
        if not self._conn:
            self.connect()

        # Calculate cutoff date in Python to avoid SQL interval syntax issues
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        query = """
            SELECT
                e.id,
                e.title,
                e.podcast,
                e.published,
                e.url,
                e.duration,
                s.content as summary
            FROM episodes e
            LEFT JOIN summaries s ON e.id = s.episode_id
            WHERE e.published >= ?
            ORDER BY e.published DESC
        """

        try:
            if self._conn is None: # Should be connected by connect() above
                 raise ConnectionError("Not connected to database")

            results = self._conn.execute(query, [cutoff_date]).fetchall()
            episodes = []

            for row in results:
                episode_id = str(row[0])
                
                # Fetch topics
                topics_query = "SELECT topic FROM topics WHERE episode_id = ?"
                topics_rows = self._conn.execute(topics_query, [episode_id]).fetchall()
                topics = [r[0] for r in topics_rows]

                # Fetch entities (if available)
                # Assuming a simple entities table or reusing topics logic
                # For now, we'll just use topics as entities placeholder or add strict check
                entity_list = [] 
                # Check if entities table exists
                try:
                    entities_query = "SELECT name FROM entities WHERE episode_id = ?"
                    entities_rows = self._conn.execute(entities_query, [episode_id]).fetchall()
                    entity_list = [r[0] for r in entities_rows]
                except duckdb.CatalogException:
                    pass # Entities table might not exist
                except duckdb.InvalidInputException:
                    pass # Table might not exist and variable name conflict might cause this

                # Parse date - duckdb returns datetime objects or strings depending on version/driver
                published = row[3]
                if isinstance(published, str):
                    published = datetime.fromisoformat(published)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=UTC)

                episode = PodcastEpisode(
                    id=episode_id,
                    title=row[1],
                    podcast_name=row[2],
                    published_date=published,
                    url=row[4],
                    duration_seconds=row[5],
                    summary=row[6],
                    topics=topics,
                    entities=entity_list
                )
                episodes.append(episode)

            return episodes

        except Exception as e:
            logger.error("Failed to fetch episodes", error=str(e))
            raise


async def sync_podcast_data(
    db_path: Path | str,
    days: int = 7
) -> PodcastSyncResult:
    """Sync podcast data from P³."""
    result = PodcastSyncResult(started_at=datetime.now(UTC))
    bridge = P3Bridge(db_path)

    try:
        # DuckDB is fast and local, so synchronous execution is usually fine,
        # but we could wrap in run_in_executor if needed for very large datasets.
        episodes = bridge.get_recent_episodes(days=days)
        
        result.episodes = episodes
        result.episodes_processed = len(episodes)
        result.episodes_new = len(episodes) # In a real sync, we'd check against DB
        
        logger.info(
            "Podcast sync completed",
            episodes=len(episodes),
            days=days
        )

    except Exception as e:
        result.success = False
        result.errors.append(str(e))
        logger.error("Podcast sync failed", error=str(e))

    finally:
        bridge.close()

    result.completed_at = datetime.now(UTC)
    return result
