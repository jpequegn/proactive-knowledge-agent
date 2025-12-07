# Proactive Knowledge Agent

An ambient AI system for personal intelligence that monitors multiple information streams and generates actionable insights across technology, fitness, and financial dimensions.

## Vision

Build an **ambient AI agent system** that runs continuously in the background, processing information from multiple sources and proactively surfacing relevant insights - without requiring explicit user requests.

Inspired by:
- **Scouts** (TWiML) - Proactive browser agents operating without user input
- **Cosmos** (Edison Scientific) - Structured world models for AI research agents
- **Karpathy Method** - Learn by building from first principles

## Key Features

- **Multi-Source Ingestion**: Podcasts, news feeds, fitness data, market signals
- **Structured World Model**: Persistent knowledge graph across sessions
- **Proactive Agents**: Specialized agents for tech, fitness, and finance
- **Ambient Operation**: Background processing with proactive alerts
- **Claude Code Integration**: MCP tools for querying knowledge

## Architecture

```
Information Sources
├── P³ Podcasts (existing integration)
├── RSS News Feeds (tech, finance, health)
├── Strava/Garmin APIs (fitness metrics)
├── Market Data APIs (financial signals)
└── Hacker News/GitHub Trends

         ↓
    Ingestion Layer (Event-Driven)
├── Stream processing
├── Content extraction
├── Embedding generation
└── Vector DB storage

         ↓
    Structured World Model
├── Knowledge graph (entities, relationships)
├── Temporal reasoning (change detection)
├── Cross-source correlation
└── Relevance scoring

         ↓
    Proactive Agent Layer
├── Tech Intelligence Agent
├── Fitness Intelligence Agent
├── Finance Intelligence Agent
└── Synthesis Agent

         ↓
    Output Layer
├── Weekly auto-reports
├── Proactive alerts
├── MCP query interface
└── Dashboard (future)
```

## Project Structure

```
proactive-knowledge-agent/
├── README.md
├── HIGH_LEVEL_PLAN.md           # Full project roadmap (7 weeks)
├── docs/
│   ├── IMPLEMENTATION_GUIDE.md   # Phase 1 detailed guide
│   ├── ARCHITECTURE.md           # System design details
│   └── phases/
│       ├── PHASE_1_INGESTION.md
│       ├── PHASE_2_WORLD_MODEL.md
│       ├── PHASE_3_AGENTS.md
│       └── PHASE_4_INTEGRATION.md
├── src/
│   ├── __init__.py
│   ├── ingestion/               # Multi-source data ingestion
│   │   ├── __init__.py
│   │   ├── rss_processor.py     # RSS/news feed processing
│   │   ├── fitness_client.py    # Strava/Garmin integration
│   │   ├── market_client.py     # Financial data APIs
│   │   └── podcast_bridge.py    # P³ integration
│   ├── world_model/             # Structured knowledge representation
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py   # Entity/relationship storage
│   │   ├── temporal.py          # Change detection
│   │   └── embeddings.py        # Vector similarity
│   ├── agents/                  # Proactive intelligence agents
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Base agent class
│   │   ├── tech_agent.py        # Technology intelligence
│   │   ├── fitness_agent.py     # Fitness intelligence
│   │   ├── finance_agent.py     # Financial intelligence
│   │   └── synthesis_agent.py   # Cross-dimensional insights
│   ├── outputs/                 # Report and alert generation
│   │   ├── __init__.py
│   │   ├── reports.py           # Weekly report generation
│   │   ├── alerts.py            # Proactive alert system
│   │   └── mcp_server.py        # Claude Code integration
│   └── daemon/                  # Background processing
│       ├── __init__.py
│       └── runner.py            # Main daemon loop
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_world_model.py
│   └── test_agents.py
├── config/
│   ├── feeds.yaml               # RSS feed configuration
│   ├── agents.yaml              # Agent configuration
│   └── alerts.yaml              # Alert rules
├── data/
│   └── .gitkeep
├── .gitignore
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (for knowledge graph)
- Redis (for event streaming)
- Existing P³ installation (optional but recommended)

### Installation

```bash
git clone https://github.com/yourusername/proactive-knowledge-agent.git
cd proactive-knowledge-agent
python -m venv venv
source venv/bin/activate
pip install -e .

# Initialize database
pka init

# Start daemon
pka daemon start
```

### Configuration

1. Copy config templates:
```bash
cp config/feeds.yaml.example config/feeds.yaml
cp config/agents.yaml.example config/agents.yaml
```

2. Configure your data sources in `config/feeds.yaml`
3. Set up API keys (Strava, market data) in `.env`

## CLI Reference

PKA provides a comprehensive command-line interface for managing your personal knowledge system.

### Core Commands

```bash
# Initialize database schema
pka init

# Check system status
pka status

# Sync all data sources
pka sync

# Sync specific sources only
pka sync --rss          # RSS feeds only
pka sync --market       # Market data only
pka sync --podcast      # Podcasts only
pka sync --dry-run      # Preview without storing
```

### Daemon Management

```bash
# Start the background daemon
pka daemon start            # Run in foreground
pka daemon start -b         # Run in background

# Check daemon status
pka daemon status

# Stop the daemon
pka daemon stop
```

### Search & Exploration

```bash
# Search the knowledge base
pka search "machine learning"
pka search "AI startups" --semantic    # Use vector similarity
pka search "fitness" --podcasts        # Search podcasts only
pka search "rust" --category tech      # Filter by category

# Explore entities in the knowledge graph
pka entity "OpenAI"
pka entity "Claude" -t technology      # Filter by type
pka entity "Python" --trends           # Include trend analysis
pka entity "Rust" --related            # Show related entities
```

### Reports & Alerts

```bash
# Generate intelligence reports
pka report                 # Weekly report (default)
pka report daily           # Daily report
pka report --view          # Open in browser after generation
pka report -o ./reports    # Custom output directory

# Manage alerts
pka alerts                      # Show pending alerts
pka alerts -l action            # Filter by level (info/watch/action/urgent)
pka alerts -a alert_123         # Acknowledge an alert
pka alerts -s alert_123:2       # Snooze for 2 hours
```

### Fitness Tracking (Strava)

```bash
# Authenticate with Strava
pka fitness auth

# Sync activities
pka fitness sync              # Sync last 90 days
pka fitness sync --days 30    # Sync last 30 days
pka fitness sync --dry-run    # Preview without storing

# Check fitness status
pka fitness status
```

### Shell Completion

Enable tab completion for all PKA commands:

```bash
# Bash
pka completion bash >> ~/.bashrc

# Zsh
pka completion zsh >> ~/.zshrc

# Fish
pka completion fish > ~/.config/fish/completions/pka.fish

# Auto-detect shell
eval "$(pka completion)"
```

### Global Options

```bash
pka --config /path/to/feeds.yaml <command>   # Use custom config
pka version                                   # Show version info
pka version -f json                           # JSON output
```

## Development Phases

### Phase 1: Multi-Source Ingestion (Weeks 1-2)
- RSS feed processing
- Strava/Garmin integration
- Market data API integration
- P³ bridge for podcast data

### Phase 2: Structured World Model (Weeks 3-4)
- Knowledge graph implementation
- Entity extraction and linking
- Temporal reasoning (change detection)
- Cross-source correlation

### Phase 3: Proactive Agents (Weeks 5-6)
- Tech Intelligence Agent
- Fitness Intelligence Agent
- Finance Intelligence Agent
- Synthesis Agent for cross-dimensional insights

### Phase 4: Integration & Reports (Week 7)
- Auto-generated weekly reports
- Proactive alert system
- MCP integration for Claude Code
- Dashboard (optional)

## Learning Goals (Karpathy Method)

By building this project from scratch, you'll deeply understand:

| Skill | How You'll Learn It |
|-------|---------------------|
| Ambient AI agents | Build background daemon that monitors without prompting |
| Structured world models | Implement persistent knowledge graph across sessions |
| Multi-agent orchestration | Coordinate 3+ specialized agents |
| Event-driven architecture | Process streams of information |
| Vector databases | Store embeddings for semantic search |
| MCP integration | Expose agent knowledge to Claude Code |
| API integration | Connect Strava, market data, RSS feeds |

## Documentation

- **[HIGH_LEVEL_PLAN.md](./HIGH_LEVEL_PLAN.md)** - Full 7-week project roadmap
- **[docs/IMPLEMENTATION_GUIDE.md](./docs/IMPLEMENTATION_GUIDE.md)** - Phase 1 detailed guide
- **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** - System design details

## GitHub Issues

All tasks are tracked as GitHub issues:
- Phase 1: Issues #1-8 (Multi-Source Ingestion)
- Phase 2: Issues #9-14 (World Model)
- Phase 3: Issues #15-20 (Agents)
- Phase 4: Issues #21-25 (Integration)

## Success Metrics

### Technical
- [ ] 4+ data sources integrated
- [ ] Knowledge graph with 1000+ entities
- [ ] 3 specialized agents operational
- [ ] < 5 minute latency from event to insight

### Personal Value
- [ ] Weekly reports generated automatically
- [ ] Proactive alerts surface actionable opportunities
- [ ] Cross-dimensional insights reveal hidden connections
- [ ] Time saved on manual information processing

## Integration with Personal Transformation System

This project directly supports the 3-dimensional transformation framework:

1. **Technology Growth** - Tech agent surfaces emerging tech + project ideas
2. **Fitness Optimization** - Fitness agent monitors training, alerts on risk
3. **Financial Stability** - Finance agent monitors markets, portfolio risk

## License

MIT

## Contact

Questions? Open a GitHub issue.

---

**Let's build something that makes us smarter every day.**
