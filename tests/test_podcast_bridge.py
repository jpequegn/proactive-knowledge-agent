"""Tests for P³ Podcast Bridge."""

import tempfile
from datetime import datetime, timedelta, UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from src.ingestion.podcast_bridge import P3Bridge, sync_podcast_data
from src.models import PodcastEpisode


class TestP3Bridge:
    """Tests for P3Bridge."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary DuckDB database with schema and data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test.duckdb")
        
            conn = duckdb.connect(db_path)
            
            # Create tables
            conn.execute("""
                CREATE TABLE episodes (
                    id VARCHAR PRIMARY KEY,
                    title VARCHAR,
                    podcast VARCHAR,
                    published TIMESTAMP,
                    url VARCHAR,
                    duration INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE summaries (
                    episode_id VARCHAR,
                    content VARCHAR
                )
            """)
            
            conn.execute("""
                CREATE TABLE topics (
                    episode_id VARCHAR,
                    topic VARCHAR
                )
            """)

            conn.execute("""
                CREATE TABLE entities (
                    episode_id VARCHAR,
                    name VARCHAR
                )
            """)
            
            # Insert data
            base_date = datetime.now()
            conn.execute("""
                INSERT INTO episodes VALUES 
                ('1', 'AI Trends', 'Tech Talk', ?, 'http://example.com/1', 3600),
                ('2', 'Market Crash', 'Finance Weekly', ?, 'http://example.com/2', 1800)
            """, [base_date, base_date - timedelta(days=2)])
            
            conn.execute("""
                INSERT INTO summaries VALUES 
                ('1', 'Summary regarding AI'),
                ('2', 'Summary regarding Markets')
            """)
            
            conn.execute("""
                INSERT INTO topics VALUES 
                ('1', 'AI'), ('1', 'ML'),
                ('2', 'Stocks')
            """)

            conn.execute("""
                INSERT INTO entities VALUES
                ('1', 'OpenAI'),
                ('2', 'NYSE')
            """)
            
            conn.close()
            
            yield db_path

    def test_connect_success(self, temp_db):
        """Test successful connection."""
        bridge = P3Bridge(temp_db)
        bridge.connect()
        assert bridge._conn is not None
        bridge.close()

    def test_connect_failure(self):
        """Test connection failure with invalid path."""
        bridge = P3Bridge("/nonexistent/path.duckdb")
        # DuckDB might create a new file if it doesn't exist if not read_only, 
        # but we set read_only=True in code.
        with pytest.raises(Exception):
             bridge.connect()

    def test_get_recent_episodes(self, temp_db):
        """Test fetching episodes."""
        bridge = P3Bridge(temp_db)
        episodes = bridge.get_recent_episodes(days=7)
        
        assert len(episodes) == 2
        
        # Check first episode (ordered by date desc, so '1' is newer or '2' depending on my insert?
        # Actually, I inserted '1' as base_date and '2' as base_date - 2.
        # Order DESC means '1' should be first.
        
        ep1 = episodes[0]
        assert ep1.id == '1'
        assert ep1.title == 'AI Trends'
        assert 'AI' in ep1.topics
        assert 'OpenAI' in ep1.entities
        assert ep1.summary == 'Summary regarding AI'

    def test_get_recent_episodes_filter(self, temp_db):
        """Test filtering by days."""
        bridge = P3Bridge(temp_db)
        # Only 1 day, should exclude the 2-day old episode
        episodes = bridge.get_recent_episodes(days=1)
        
        assert len(episodes) == 1
        assert episodes[0].id == '1'

    def test_missing_entities_table(self, temp_db):
        """Test resilience when entities table is missing."""
        # Drop entities table
        conn = duckdb.connect(temp_db)
        conn.execute("DROP TABLE entities")
        conn.close()
        
        bridge = P3Bridge(temp_db)
        episodes = bridge.get_recent_episodes(days=7)
        
        assert len(episodes) == 2
        assert episodes[0].entities == []


@pytest.mark.asyncio
async def test_sync_podcast_data_success():
    """Test sync function success."""
    with patch("src.ingestion.podcast_bridge.P3Bridge") as MockBridge:
        mock_instance = MockBridge.return_value
        
        ep = PodcastEpisode(
            id="1", 
            title="Test", 
            podcast_name="P", 
            published_date=datetime.now(UTC)
        )
        mock_instance.get_recent_episodes.return_value = [ep]
        
        result = await sync_podcast_data("dummy.db")
        
        assert result.success is True
        assert result.episodes_processed == 1
        assert len(result.episodes) == 1

@pytest.mark.asyncio
async def test_sync_podcast_data_failure():
    """Test sync function failure."""
    with patch("src.ingestion.podcast_bridge.P3Bridge") as MockBridge:
        mock_instance = MockBridge.return_value
        mock_instance.get_recent_episodes.side_effect = Exception("DB Error")
        
        result = await sync_podcast_data("dummy.db")
        
        assert result.success is False
        assert len(result.errors) == 1
        assert "DB Error" in result.errors[0]
