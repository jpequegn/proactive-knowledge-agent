# Phase 1 Implementation Guide: Multi-Source Ingestion

**Duration**: Weeks 1-2 (~25 hours)
**Goal**: Build unified data ingestion layer for 4+ information sources

---

## Overview

Phase 1 establishes the foundation for the Proactive Knowledge Agent by creating:
1. RSS feed processor for news and tech content
2. Fitness data client (Strava integration)
3. Market data client (financial signals)
4. P³ bridge (podcast processing integration)
5. Unified storage layer (PostgreSQL + pgvector)

---

## Prerequisites

### System Requirements
- macOS with Apple Silicon (M1/M2/M3) recommended
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+ (for event streaming)

### Installation

```bash
# Clone and setup
cd /Users/julienpequegnot/Code/proactive-knowledge-agent
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,fitness,market]"

# PostgreSQL with pgvector
brew install postgresql@15
brew services start postgresql@15
psql -c "CREATE DATABASE pka;"
psql -d pka -c "CREATE EXTENSION vector;"

# Redis
brew install redis
brew services start redis
```

### Environment Setup

Create `.env` file:
```bash
# Database
DATABASE_URL=postgresql://localhost/pka

# Redis
REDIS_URL=redis://localhost:6379

# LLM (for entity extraction)
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here

# Fitness APIs
STRAVA_CLIENT_ID=your_id
STRAVA_CLIENT_SECRET=your_secret

# Market Data
ALPHA_VANTAGE_API_KEY=your_key
```

---

## Task 1.1: RSS Feed Processor

**Time Estimate**: 4 hours
**File**: `src/ingestion/rss_processor.py`

### Requirements
- Parse RSS/Atom feeds from configured sources
- Extract article content (title, summary, link, published date)
- Generate embeddings for semantic search
- Store with metadata in PostgreSQL
- Handle deduplication via URL hash

### Implementation Steps

1. **Create RSS parser class** (1 hour)
   ```python
   # src/ingestion/rss_processor.py
   import feedparser
   import httpx
   from datetime import datetime
   from dataclasses import dataclass

   @dataclass
   class Article:
       title: str
       url: str
       summary: str
       published: datetime
       source: str
       category: str
       content_hash: str

   class RSSProcessor:
       def __init__(self, config: dict):
           self.config = config
           self.client = httpx.AsyncClient()

       async def fetch_feed(self, feed_url: str) -> list[Article]:
           """Fetch and parse RSS feed."""
           pass

       async def process_feeds(self) -> list[Article]:
           """Process all configured feeds."""
           pass
   ```

2. **Add embedding generation** (1 hour)
   ```python
   from openai import OpenAI

   class EmbeddingGenerator:
       def __init__(self, model: str = "text-embedding-3-small"):
           self.client = OpenAI()
           self.model = model

       async def generate(self, text: str) -> list[float]:
           """Generate embedding for text."""
           response = self.client.embeddings.create(
               model=self.model,
               input=text
           )
           return response.data[0].embedding
   ```

3. **Implement deduplication** (1 hour)
   - Hash URL for primary deduplication
   - Semantic similarity check for near-duplicates
   - Merge metadata for duplicate stories

4. **Add database storage** (1 hour)
   ```python
   async def store_article(self, article: Article, embedding: list[float]):
       """Store article with embedding in PostgreSQL."""
       query = """
       INSERT INTO articles (title, url, summary, published, source, category, embedding)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (url_hash) DO UPDATE SET
           title = EXCLUDED.title,
           summary = EXCLUDED.summary
       RETURNING id
       """
   ```

### Success Criteria
- [ ] 10+ RSS feeds configured in `config/feeds.yaml`
- [ ] New articles ingested hourly via scheduler
- [ ] Embeddings generated for all content
- [ ] Duplicates detected and merged (same story, multiple sources)
- [ ] Unit tests passing with >80% coverage

### Testing

```bash
# Run RSS processor tests
pytest tests/test_rss_processor.py -v

# Manual test
python -m src.ingestion.rss_processor --fetch --verbose
```

---

## Task 1.2: Fitness Data Client (Strava)

**Time Estimate**: 6 hours
**File**: `src/ingestion/fitness_client.py`

### Requirements
- OAuth2 authentication with Strava API
- Pull activities (runs, rides, workouts)
- Extract metrics (distance, pace, HR, power zones)
- Calculate derived metrics (training load, fatigue)
- Store time-series data

### Implementation Steps

1. **OAuth2 flow** (2 hours)
   ```python
   # src/ingestion/fitness_client.py
   from stravalib import Client

   class StravaClient:
       def __init__(self, client_id: str, client_secret: str):
           self.client = Client()
           self.client_id = client_id
           self.client_secret = client_secret

       def get_auth_url(self) -> str:
           """Generate OAuth authorization URL."""
           return self.client.authorization_url(
               client_id=self.client_id,
               redirect_uri='http://localhost:8080/callback',
               scope=['activity:read_all', 'profile:read_all']
           )

       async def exchange_token(self, code: str) -> dict:
           """Exchange authorization code for access token."""
           pass
   ```

2. **Activity fetching** (2 hours)
   ```python
   async def get_activities(self, after: datetime = None) -> list[Activity]:
       """Fetch activities since date."""
       activities = self.client.get_activities(after=after)
       return [self._parse_activity(a) for a in activities]

   def _parse_activity(self, activity) -> Activity:
       """Parse Strava activity into our schema."""
       return Activity(
           id=activity.id,
           type=activity.type,
           start_date=activity.start_date,
           distance=activity.distance.num,
           duration=activity.moving_time.seconds,
           avg_hr=activity.average_heartrate,
           max_hr=activity.max_heartrate,
           # ... more fields
       )
   ```

3. **Derived metrics calculation** (1.5 hours)
   ```python
   class TrainingMetrics:
       def calculate_tss(self, activity: Activity) -> float:
           """Calculate Training Stress Score."""
           # Simplified TSS = (duration * IF^2) / 36
           pass

       def calculate_atl(self, activities: list[Activity]) -> float:
           """Acute Training Load (7-day rolling)."""
           pass

       def calculate_ctl(self, activities: list[Activity]) -> float:
           """Chronic Training Load (42-day rolling)."""
           pass

       def calculate_tsb(self, atl: float, ctl: float) -> float:
           """Training Stress Balance (freshness)."""
           return ctl - atl
   ```

4. **Database storage** (0.5 hours)
   - Store raw activities
   - Store daily metric snapshots
   - Enable time-series queries

### Success Criteria
- [ ] Strava OAuth flow working (auth URL → callback → token)
- [ ] Historical activities imported (last 90 days)
- [ ] New activities sync automatically (webhook or polling)
- [ ] Training metrics calculated (TSS, ATL, CTL, TSB)
- [ ] Tests passing

### Testing

```bash
# Test Strava integration
pytest tests/test_fitness_client.py -v

# Manual OAuth test
python -m src.ingestion.fitness_client --auth
```

---

## Task 1.3: Market Data Client

**Time Estimate**: 4 hours
**File**: `src/ingestion/market_client.py`

### Requirements
- Connect to free market data API (Alpha Vantage or yfinance)
- Pull daily OHLCV data for watchlist
- Calculate technical indicators (SMA, RSI, VIX)
- Store historical data

### Implementation Steps

1. **API client setup** (1 hour)
   ```python
   # src/ingestion/market_client.py
   import yfinance as yf

   class MarketDataClient:
       def __init__(self, watchlist: list[str]):
           self.watchlist = watchlist

       async def fetch_daily(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
           """Fetch daily OHLCV data."""
           ticker = yf.Ticker(symbol)
           return ticker.history(period=period)
   ```

2. **Technical indicators** (1.5 hours)
   ```python
   class TechnicalIndicators:
       @staticmethod
       def sma(prices: pd.Series, window: int) -> pd.Series:
           """Simple Moving Average."""
           return prices.rolling(window=window).mean()

       @staticmethod
       def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
           """Relative Strength Index."""
           delta = prices.diff()
           gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
           loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
           rs = gain / loss
           return 100 - (100 / (1 + rs))

       @staticmethod
       def volatility(prices: pd.Series, window: int = 20) -> pd.Series:
           """Historical volatility (annualized)."""
           returns = prices.pct_change()
           return returns.rolling(window=window).std() * (252 ** 0.5)
   ```

3. **Watchlist management** (1 hour)
   ```python
   async def update_watchlist(self):
       """Update all watchlist symbols."""
       for symbol in self.watchlist:
           data = await self.fetch_daily(symbol)
           indicators = self._calculate_indicators(data)
           await self._store_data(symbol, data, indicators)
   ```

4. **Database storage** (0.5 hours)

### Success Criteria
- [ ] Market data API connected (yfinance or Alpha Vantage)
- [ ] 10+ symbols tracked (SPY, QQQ, VIX, sector ETFs)
- [ ] Daily data updated automatically
- [ ] Technical indicators calculated (SMA, RSI, volatility)

---

## Task 1.4: P³ Bridge

**Time Estimate**: 3 hours
**File**: `src/ingestion/podcast_bridge.py`

### Requirements
- Read from existing P³ DuckDB database
- Extract summaries, topics, companies, quotes
- Sync new episodes automatically
- Maintain entity references for knowledge graph

### Implementation Steps

1. **DuckDB connection** (1 hour)
   ```python
   # src/ingestion/podcast_bridge.py
   import duckdb

   class P3Bridge:
       def __init__(self, db_path: str = "~/.p3/data/p3.duckdb"):
           self.db_path = Path(db_path).expanduser()
           self.conn = duckdb.connect(str(self.db_path), read_only=True)

       def get_recent_episodes(self, days: int = 7) -> list[Episode]:
           """Get episodes from last N days."""
           query = """
           SELECT e.id, e.title, p.title as podcast, e.date,
                  s.key_topics, s.themes, s.quotes, s.startups, s.key_takeaways
           FROM episodes e
           JOIN podcasts p ON e.podcast_id = p.id
           LEFT JOIN summaries s ON e.id = s.episode_id
           WHERE e.date >= current_date - interval ? day
           """
           return self.conn.execute(query, [days]).fetchall()
   ```

2. **Entity extraction** (1.5 hours)
   ```python
   def extract_entities(self, episode: Episode) -> list[Entity]:
       """Extract entities from episode summary."""
       entities = []

       # Companies mentioned
       for startup in episode.startups:
           entities.append(Entity(
               type="Company",
               name=startup,
               source="podcast",
               episode_id=episode.id
           ))

       # Topics as concepts
       for topic in episode.key_topics:
           entities.append(Entity(
               type="Concept",
               name=topic,
               source="podcast",
               episode_id=episode.id
           ))

       return entities
   ```

3. **Sync mechanism** (0.5 hours)

### Success Criteria
- [ ] P³ data accessible from PKA
- [ ] New episodes detected and imported
- [ ] Topics and entities extracted
- [ ] Cross-reference capability with other sources

---

## Task 1.5: Unified Storage Layer

**Time Estimate**: 4 hours
**Files**: `src/database.py`, `src/models.py`

### Requirements
- PostgreSQL schema for all entity types
- pgvector for embedding storage
- Redis for event queue and caching
- Unified query layer

### Database Schema

```sql
-- src/schema.sql

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Articles from RSS feeds
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    url_hash TEXT UNIQUE NOT NULL,
    summary TEXT,
    content TEXT,
    published TIMESTAMPTZ,
    source TEXT NOT NULL,
    category TEXT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX articles_embedding_idx ON articles
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Fitness activities
CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    external_id TEXT UNIQUE NOT NULL,
    activity_type TEXT NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    distance_meters FLOAT,
    duration_seconds INT,
    avg_hr INT,
    max_hr INT,
    avg_power INT,
    tss FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily fitness metrics
CREATE TABLE fitness_metrics (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    atl FLOAT,
    ctl FLOAT,
    tsb FLOAT,
    hrv FLOAT,
    sleep_hours FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market data
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    sma_20 FLOAT,
    sma_50 FLOAT,
    rsi_14 FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, date)
);

-- Podcast episodes (synced from P³)
CREATE TABLE podcast_episodes (
    id SERIAL PRIMARY KEY,
    external_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    podcast_name TEXT,
    published DATE,
    summary TEXT,
    key_topics JSONB,
    themes JSONB,
    companies JSONB,
    quotes JSONB,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge graph entities (Phase 2 foundation)
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases JSONB DEFAULT '[]',
    properties JSONB DEFAULT '{}',
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    mention_count INT DEFAULT 1,
    embedding vector(1536),
    UNIQUE(type, name)
);

-- Entity mentions (source tracking)
CREATE TABLE entity_mentions (
    id SERIAL PRIMARY KEY,
    entity_id INT REFERENCES entities(id),
    source_type TEXT NOT NULL,
    source_id INT NOT NULL,
    context TEXT,
    mentioned_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX entity_mentions_entity_idx ON entity_mentions(entity_id);
CREATE INDEX entity_mentions_source_idx ON entity_mentions(source_type, source_id);
```

### Implementation Steps

1. **Database connection manager** (1 hour)
   ```python
   # src/database.py
   import asyncpg
   from contextlib import asynccontextmanager

   class Database:
       def __init__(self, database_url: str):
           self.database_url = database_url
           self.pool = None

       async def connect(self):
           self.pool = await asyncpg.create_pool(self.database_url)

       @asynccontextmanager
       async def transaction(self):
           async with self.pool.acquire() as conn:
               async with conn.transaction():
                   yield conn
   ```

2. **Pydantic models** (1 hour)
   ```python
   # src/models.py
   from pydantic import BaseModel
   from datetime import datetime

   class Article(BaseModel):
       id: int | None = None
       title: str
       url: str
       summary: str | None = None
       published: datetime | None = None
       source: str
       category: str | None = None

   class Activity(BaseModel):
       id: int | None = None
       external_id: str
       activity_type: str
       start_date: datetime
       distance_meters: float | None = None
       duration_seconds: int | None = None
       avg_hr: int | None = None
       tss: float | None = None
   ```

3. **Repository pattern** (1.5 hours)
   ```python
   class ArticleRepository:
       def __init__(self, db: Database):
           self.db = db

       async def create(self, article: Article, embedding: list[float]) -> int:
           pass

       async def find_similar(self, embedding: list[float], limit: int = 10) -> list[Article]:
           """Find semantically similar articles."""
           query = """
           SELECT id, title, url, summary, published, source, category,
                  1 - (embedding <=> $1) as similarity
           FROM articles
           ORDER BY embedding <=> $1
           LIMIT $2
           """
           pass
   ```

4. **Redis event queue** (0.5 hours)
   ```python
   import redis.asyncio as redis

   class EventQueue:
       def __init__(self, redis_url: str):
           self.redis = redis.from_url(redis_url)

       async def publish(self, event_type: str, data: dict):
           """Publish event to stream."""
           await self.redis.xadd(
               f"pka:events:{event_type}",
               {"data": json.dumps(data)}
           )
   ```

### Success Criteria
- [ ] PostgreSQL schema created and migrated
- [ ] pgvector extension configured
- [ ] All data sources writing to unified store
- [ ] Semantic search queries working
- [ ] Redis event queue operational

---

## CLI Commands

**File**: `src/cli.py`

```python
import click
from rich.console import Console

console = Console()

@click.group()
def main():
    """Proactive Knowledge Agent CLI."""
    pass

@main.command()
def init():
    """Initialize database and configuration."""
    pass

@main.command()
@click.option('--source', type=click.Choice(['rss', 'fitness', 'market', 'podcast', 'all']))
def sync(source: str):
    """Sync data from sources."""
    pass

@main.command()
def status():
    """Show system status."""
    pass

@main.command()
@click.argument('query')
def search(query: str):
    """Search knowledge base."""
    pass

if __name__ == '__main__':
    main()
```

---

## Testing Strategy

### Unit Tests
- Each ingestion client has isolated tests
- Mock external APIs
- Test data transformation logic

### Integration Tests
- Database operations
- End-to-end sync flow
- Event queue publishing

### Test Data
Create fixtures in `tests/fixtures/`:
- `sample_rss_feed.xml`
- `sample_strava_activity.json`
- `sample_market_data.csv`

---

## Phase 1 Deliverables Checklist

- [ ] RSS Feed Processor
  - [ ] Feed parsing
  - [ ] Embedding generation
  - [ ] Deduplication
  - [ ] Database storage
  - [ ] Unit tests

- [ ] Fitness Data Client
  - [ ] OAuth2 flow
  - [ ] Activity fetching
  - [ ] Metrics calculation
  - [ ] Database storage
  - [ ] Unit tests

- [ ] Market Data Client
  - [ ] API connection
  - [ ] Data fetching
  - [ ] Technical indicators
  - [ ] Database storage
  - [ ] Unit tests

- [ ] P³ Bridge
  - [ ] DuckDB connection
  - [ ] Entity extraction
  - [ ] Sync mechanism
  - [ ] Unit tests

- [ ] Unified Storage
  - [ ] PostgreSQL schema
  - [ ] pgvector setup
  - [ ] Repository layer
  - [ ] Redis queue
  - [ ] Unit tests

- [ ] CLI
  - [ ] init command
  - [ ] sync command
  - [ ] status command
  - [ ] search command

---

## Next Steps (Phase 2 Preview)

After Phase 1 is complete, Phase 2 will build on this foundation:
- Knowledge graph schema for entities and relationships
- Entity extraction pipeline using LLM
- Temporal reasoning for change detection
- Cross-source correlation algorithms

---

**Questions?** Open a GitHub issue with the `phase-1` label.
