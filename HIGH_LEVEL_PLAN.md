# Proactive Knowledge Agent - Full Project Roadmap

**Duration**: 7 weeks
**Effort**: ~80-100 hours
**Goal**: Build an ambient AI system that proactively surfaces insights across technology, fitness, and financial dimensions.

---

## Project Vision

### The Problem
You consume vast amounts of information (podcasts, news, fitness data, market signals) but:
- Information is siloed across different sources
- Insights require manual synthesis
- Emerging opportunities get missed
- No persistent memory across sessions

### The Solution
An **ambient AI agent system** that:
- Runs continuously in the background
- Monitors multiple information streams
- Builds a **structured world model** (persistent knowledge)
- **Proactively** surfaces relevant insights
- Generates automatic reports and alerts

### Why This Matters
This project is **meta** - it improves your ability to learn and stay ahead:
- Tech trends identified → project ideas generated
- Fitness risks detected → training adjusted
- Financial signals spotted → risk reduced
- Cross-dimensional insights → compound advantages

---

## Timeline Overview

```
Week 1-2: Multi-Source Ingestion (Phase 1)
Week 3-4: Structured World Model (Phase 2)
Week 5-6: Proactive Agents (Phase 3)
Week 7:   Integration & Reports (Phase 4)
```

---

## Phase 1: Multi-Source Ingestion (Weeks 1-2)

**Goal**: Build unified data ingestion layer for 4+ information sources.

### Why This Phase First
- Foundation for everything else
- Proves the concept (data flowing)
- Relatively straightforward implementation
- Immediate value (unified data access)

### Components

#### 1.1 RSS Feed Processor
**Time**: 4 hours
**Purpose**: Ingest tech, finance, and health news from RSS feeds

```python
# Key capabilities:
- Parse RSS/Atom feeds
- Extract article content (title, summary, full text)
- Generate embeddings for semantic search
- Store with metadata (source, date, category)
- Deduplication (same story from multiple sources)
```

**Success Criteria**:
- [ ] 10+ RSS feeds configured (tech, finance, health)
- [ ] New articles ingested every hour
- [ ] Embeddings generated for all content
- [ ] Duplicates detected and merged

#### 1.2 Fitness Data Client (Strava/Garmin)
**Time**: 6 hours
**Purpose**: Pull training data, metrics, and activities

```python
# Key capabilities:
- OAuth2 authentication with Strava
- Pull activities (runs, rides, workouts)
- Extract metrics (distance, pace, HR, power)
- Calculate derived metrics (training load, fatigue)
- Store time-series data
```

**Success Criteria**:
- [ ] Strava OAuth flow working
- [ ] Historical activities imported (last 90 days)
- [ ] New activities sync automatically
- [ ] Training metrics calculated

#### 1.3 Market Data Client
**Time**: 4 hours
**Purpose**: Pull financial market data and indicators

```python
# Key capabilities:
- Connect to free API (Alpha Vantage, Yahoo Finance)
- Pull price data (daily OHLCV)
- Calculate indicators (SMA, RSI, VIX)
- Monitor watchlist symbols
- Store historical data
```

**Success Criteria**:
- [ ] Market data API connected
- [ ] 10+ symbols tracked
- [ ] Daily data updated automatically
- [ ] Basic indicators calculated

#### 1.4 P³ Bridge
**Time**: 3 hours
**Purpose**: Integrate existing podcast processing pipeline

```python
# Key capabilities:
- Read from P³ DuckDB database
- Extract summaries, topics, companies
- Sync new episodes automatically
- Maintain entity references
```

**Success Criteria**:
- [ ] P³ data accessible from PKA
- [ ] New episodes detected and imported
- [ ] Topics and entities extracted
- [ ] Cross-reference with other sources

#### 1.5 Unified Storage Layer
**Time**: 4 hours
**Purpose**: Store all data in unified schema

```python
# Storage architecture:
- PostgreSQL: Structured data, knowledge graph
- Vector DB (pgvector): Embeddings for similarity
- Redis: Event queue, caching
- File storage: Raw content backup
```

**Success Criteria**:
- [ ] PostgreSQL schema designed
- [ ] pgvector extension configured
- [ ] All sources writing to unified store
- [ ] Query layer for cross-source access

### Phase 1 Deliverables
- ✅ 4 data source integrations
- ✅ Unified storage layer
- ✅ Hourly sync running
- ✅ Basic CLI for status/queries
- ✅ All unit tests passing

---

## Phase 2: Structured World Model (Weeks 3-4)

**Goal**: Build persistent knowledge representation with temporal reasoning.

### Why This Matters
From the AI Breakdown podcast on Cosmos (Edison Scientific):
> "Structured world models are critical for AI agents to effectively process and retain information over extended periods."

This is the **key innovation** - instead of treating each interaction as isolated, we build persistent memory.

### Components

#### 2.1 Knowledge Graph Schema
**Time**: 5 hours
**Purpose**: Define entity types and relationships

```
Entities:
├── Technology (frameworks, tools, languages)
├── Company (startups, enterprises)
├── Person (founders, researchers, athletes)
├── Concept (trends, methodologies)
├── Metric (fitness numbers, financial indicators)
└── Event (announcements, releases, races)

Relationships:
├── Technology -[USED_BY]-> Company
├── Person -[FOUNDED]-> Company
├── Company -[ANNOUNCED]-> Event
├── Technology -[RELATES_TO]-> Concept
├── Metric -[MEASURED_ON]-> Date
└── Event -[IMPACTS]-> Technology|Company|Metric
```

**Success Criteria**:
- [ ] Schema supports all 3 dimensions (tech, fitness, finance)
- [ ] Relationships enable cross-domain queries
- [ ] Temporal attributes on all entities

#### 2.2 Entity Extraction
**Time**: 6 hours
**Purpose**: Extract entities from ingested content

```python
# Extraction pipeline:
1. Named Entity Recognition (NER) for companies, people
2. Topic modeling for technologies, concepts
3. Numeric extraction for metrics
4. Date/event detection
5. LLM-assisted disambiguation
```

**Success Criteria**:
- [ ] 80%+ entity extraction accuracy
- [ ] Disambiguation working (Apple company vs fruit)
- [ ] New entities auto-created
- [ ] Existing entities linked

#### 2.3 Temporal Reasoning
**Time**: 5 hours
**Purpose**: Track changes over time, detect trends

```python
# Temporal capabilities:
- Version history for all entities
- Change detection ("X mentioned 3x more this week")
- Trend analysis ("growing interest in Y")
- Anomaly detection ("unusual activity in Z")
- Decay functions (relevance decreases over time)
```

**Success Criteria**:
- [ ] Entity history tracked
- [ ] Change detection working
- [ ] Trend queries functional
- [ ] "What's new since yesterday?" answerable

#### 2.4 Cross-Source Correlation
**Time**: 4 hours
**Purpose**: Link related information across sources

```python
# Correlation types:
- Same entity mentioned in podcast AND news
- Technology trend correlates with stock movement
- Training load correlates with sleep quality
- Market volatility correlates with tech mentions
```

**Success Criteria**:
- [ ] Cross-source entity linking
- [ ] Correlation detection algorithms
- [ ] Significance scoring
- [ ] Query API for correlations

### Phase 2 Deliverables
- ✅ Knowledge graph with 1000+ entities
- ✅ Entity extraction pipeline
- ✅ Temporal queries working
- ✅ Cross-source correlations detected
- ✅ Query API documented

---

## Phase 3: Proactive Agents (Weeks 5-6)

**Goal**: Build specialized agents that proactively surface insights.

### Agent Architecture

Each agent follows the same pattern:
1. **Monitor**: Watch for relevant changes in world model
2. **Analyze**: Apply domain-specific reasoning
3. **Decide**: Determine if insight is actionable
4. **Alert**: Surface to user if significance threshold met

### Agents

#### 3.1 Tech Intelligence Agent
**Time**: 8 hours
**Purpose**: Surface emerging technologies and learning opportunities

```python
# Monitoring for:
- New technologies gaining mentions
- Tools with accelerating adoption
- Skills gaps relative to trends
- Project opportunities

# Outputs:
- Weekly "Emerging Tech Report"
- Real-time alerts for significant announcements
- Project ideas with learning paths
- Relevance scores based on your profile
```

**Success Criteria**:
- [ ] Detects emerging tech from multiple sources
- [ ] Generates project ideas with rationale
- [ ] Ranks by relevance to your role
- [ ] < 24hr latency from announcement to alert

#### 3.2 Fitness Intelligence Agent
**Time**: 6 hours
**Purpose**: Monitor training, detect risks, optimize performance

```python
# Monitoring for:
- Training load trends (acute vs chronic)
- Recovery indicators (HRV, sleep)
- Performance progression (pace, power)
- Injury risk patterns

# Outputs:
- Weekly "Fitness Report"
- Alerts for overtraining risk
- Recovery recommendations
- Goal readiness assessments
```

**Success Criteria**:
- [ ] Calculates training stress scores
- [ ] Detects overtraining patterns
- [ ] Integrates sleep/recovery data
- [ ] Generates actionable recommendations

#### 3.3 Finance Intelligence Agent
**Time**: 6 hours
**Purpose**: Monitor markets, track portfolio risk

```python
# Monitoring for:
- Market regime changes
- Sector rotations
- Risk indicators (VIX, yield curve)
- News sentiment on holdings

# Outputs:
- Weekly "Market Report"
- Alerts for significant market events
- Portfolio risk assessment
- Rebalancing triggers
```

**Success Criteria**:
- [ ] Tracks market indicators
- [ ] Sentiment analysis on holdings
- [ ] Risk scoring for portfolio
- [ ] Alerts on significant changes

#### 3.4 Synthesis Agent
**Time**: 5 hours
**Purpose**: Generate cross-dimensional insights

```python
# Cross-dimensional analysis:
- "High work stress + increasing training load → injury risk"
- "Tech trend in finance → learning opportunity"
- "Market volatility + fitness goals → adjust risk"
- "Podcast topic + news trend → emerging opportunity"
```

**Success Criteria**:
- [ ] Detects cross-dimensional patterns
- [ ] Generates compound insights
- [ ] Prioritizes by impact
- [ ] Explains reasoning

### Phase 3 Deliverables
- ✅ 4 specialized agents operational
- ✅ Proactive alerts flowing
- ✅ Weekly reports auto-generated
- ✅ Cross-dimensional insights working
- ✅ Agent configuration UI

---

## Phase 4: Integration & Reports (Week 7)

**Goal**: Polish the system, integrate with Claude Code, enable daily use.

### Components

#### 4.1 Report Generation
**Time**: 4 hours
**Purpose**: Auto-generate weekly reports

```markdown
# Weekly Intelligence Report - 2025-11-26

## Technology
- 3 emerging technologies identified
- Top opportunity: [X] with project idea
- Skills gap: [Y] trending but not in your profile

## Fitness
- Training load: 85% of target
- Recovery score: 72 (below optimal)
- Recommendation: Reduce intensity this week

## Finance
- Market regime: Risk-on
- Portfolio exposure: 15% above target volatility
- Action: Consider rebalancing [sector]

## Cross-Dimensional
- Insight: [X] trend correlates with [Y] - investigate
```

**Success Criteria**:
- [ ] Reports generated automatically
- [ ] Markdown format (readable anywhere)
- [ ] Emailed or saved to designated location
- [ ] Historical reports archived

#### 4.2 Proactive Alert System
**Time**: 4 hours
**Purpose**: Push alerts for significant events

```python
# Alert channels:
- Console (for testing)
- Slack webhook
- Email
- Push notification (future)

# Alert levels:
- INFO: FYI, no action needed
- WATCH: Monitor this situation
- ACTION: Do something about this
- URGENT: Immediate attention required
```

**Success Criteria**:
- [ ] Multiple alert channels
- [ ] Configurable thresholds
- [ ] Alert batching (no spam)
- [ ] Acknowledge/snooze functionality

#### 4.3 MCP Integration
**Time**: 5 hours
**Purpose**: Expose knowledge to Claude Code

```python
# MCP Tools:
- pka_search(query) → Search knowledge graph
- pka_trends(domain, period) → Get trends
- pka_alerts() → Get pending alerts
- pka_report(type, period) → Generate report
- pka_entity(name) → Get entity details
```

**Success Criteria**:
- [ ] MCP server running
- [ ] All tools documented
- [ ] Claude Code can query knowledge
- [ ] Natural language queries work

#### 4.4 CLI Polish
**Time**: 3 hours
**Purpose**: Complete command-line interface

```bash
# Commands:
pka init              # Initialize database
pka daemon start/stop # Control background process
pka sync              # Force sync all sources
pka status            # Show system status
pka report [type]     # Generate report
pka search [query]    # Search knowledge
pka alerts            # Show pending alerts
pka entity [name]     # Show entity details
```

**Success Criteria**:
- [ ] All commands implemented
- [ ] Help text complete
- [ ] Tab completion
- [ ] Rich output formatting

### Phase 4 Deliverables
- ✅ Weekly reports auto-generating
- ✅ Alert system operational
- ✅ MCP integration complete
- ✅ CLI polished
- ✅ Documentation complete

---

## Success Criteria (Project-Wide)

### Technical
- [ ] 4+ data sources integrated and syncing
- [ ] Knowledge graph with 1000+ entities
- [ ] 3+ specialized agents operational
- [ ] < 5 minute latency from event to insight
- [ ] 99%+ uptime for daemon

### Learning (Karpathy Method)
- [ ] Understand event-driven architecture deeply
- [ ] Can explain knowledge graph design decisions
- [ ] Can modify/extend agents confidently
- [ ] Comfortable with MCP protocol

### Personal Value
- [ ] Weekly reports save 2+ hours of manual research
- [ ] Proactive alerts surface 1+ actionable opportunity per week
- [ ] Cross-dimensional insights reveal hidden connections
- [ ] System becomes indispensable for decision-making

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.11+ | Ecosystem, async support |
| Database | PostgreSQL + pgvector | Structured + vector storage |
| Queue | Redis Streams | Event-driven processing |
| LLM | Claude API | Entity extraction, reasoning |
| Embeddings | OpenAI/local | Semantic similarity |
| Fitness API | Strava | Comprehensive activity data |
| Market API | Alpha Vantage | Free tier sufficient |
| MCP | Claude SDK | IDE integration |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Caching, batch requests, multiple sources |
| Data quality | Validation, deduplication, confidence scores |
| Alert fatigue | Configurable thresholds, batching, snooze |
| Scope creep | Strict phase boundaries, MVP first |
| Complexity | Start simple, iterate |

---

## Git Workflow

```
main          - Stable releases
├── develop   - Integration branch
├── phase-1   - Multi-source ingestion
├── phase-2   - World model
├── phase-3   - Agents
└── phase-4   - Integration
```

Releases:
- `v0.1.0` - Phase 1 complete
- `v0.2.0` - Phase 2 complete
- `v0.3.0` - Phase 3 complete
- `v1.0.0` - Production ready

---

## Resource Allocation

**Estimated Total**: 80-100 hours
- Phase 1: 25 hours
- Phase 2: 25 hours
- Phase 3: 30 hours
- Phase 4: 20 hours

**Pace**: 10-15 hours/week → 6-8 weeks

---

## Next Steps

1. **Read** this plan thoroughly
2. **Review** GitHub issues for Phase 1 tasks
3. **Start** with Issue #1: RSS Feed Processor
4. **Track** progress via GitHub project board
5. **Document** learnings as you build

---

**Start Date**: November 26, 2025
**Target Completion**: January 15, 2026
**Current Status**: Phase 1 - Ready to Begin

---

**Let's build an AI that makes us smarter every day.**
