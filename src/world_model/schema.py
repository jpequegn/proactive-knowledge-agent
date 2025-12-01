"""SQL schema definitions for the knowledge graph."""

# Knowledge Graph Schema for PostgreSQL
# Supports entities, relationships, mentions, and version history

KNOWLEDGE_GRAPH_SCHEMA = """
-- ============================================================================
-- Knowledge Graph Entities
-- ============================================================================

-- Entity types enum (stored as text for flexibility)
-- Types: technology, company, person, concept, metric, event

-- Domain types enum
-- Domains: tech, fitness, finance, general

CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    external_id TEXT,                           -- ID from source system
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,                  -- EntityType enum value
    domain TEXT DEFAULT 'general',              -- Domain enum value
    description TEXT,

    -- Source tracking
    source TEXT,                                -- Where entity was first discovered
    source_url TEXT,

    -- Confidence and relevance
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    mention_count INT DEFAULT 1 CHECK (mention_count >= 0),

    -- Temporal attributes (required per issue spec)
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Entity embedding for similarity search
    embedding vector(1536),

    -- Flexible properties as JSONB
    properties JSONB DEFAULT '{}'::jsonb,

    -- Prevent duplicate entities of same type with same name
    UNIQUE(name, entity_type)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS entities_type_idx ON entities (entity_type);
CREATE INDEX IF NOT EXISTS entities_domain_idx ON entities (domain);
CREATE INDEX IF NOT EXISTS entities_name_idx ON entities (name);
CREATE INDEX IF NOT EXISTS entities_external_id_idx ON entities (external_id);
CREATE INDEX IF NOT EXISTS entities_first_seen_idx ON entities (first_seen DESC);
CREATE INDEX IF NOT EXISTS entities_last_seen_idx ON entities (last_seen DESC);
CREATE INDEX IF NOT EXISTS entities_mention_count_idx ON entities (mention_count DESC);
CREATE INDEX IF NOT EXISTS entities_properties_idx ON entities USING gin (properties);

-- Vector similarity search index
CREATE INDEX IF NOT EXISTS entities_embedding_idx ON entities
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


-- ============================================================================
-- Knowledge Graph Relationships
-- ============================================================================

-- Relationship types:
-- used_by, founded, announced, relates_to, impacts,
-- works_at, created, competes_with, employs, measures, tracks,
-- mentions, similar_to, part_of, precedes, follows, correlates_with

CREATE TABLE IF NOT EXISTS relationships (
    id SERIAL PRIMARY KEY,
    relationship_type TEXT NOT NULL,
    source_entity_id INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,

    -- Relationship strength and confidence
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Source tracking
    source TEXT,
    source_url TEXT,

    -- Temporal attributes
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    valid_from TIMESTAMPTZ,      -- When relationship became valid
    valid_until TIMESTAMPTZ,     -- When relationship ended (NULL=still valid)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Flexible properties
    properties JSONB DEFAULT '{}'::jsonb,

    -- Prevent duplicate relationships
    UNIQUE(relationship_type, source_entity_id, target_entity_id)
);

-- Indexes for graph traversal
CREATE INDEX IF NOT EXISTS relationships_type_idx ON relationships (relationship_type);
CREATE INDEX IF NOT EXISTS relationships_source_idx ON relationships (source_entity_id);
CREATE INDEX IF NOT EXISTS relationships_target_idx ON relationships (target_entity_id);
CREATE INDEX IF NOT EXISTS relationships_valid_idx
ON relationships (valid_from, valid_until);
CREATE INDEX IF NOT EXISTS relationships_temporal_idx
ON relationships (first_seen DESC, last_seen DESC);


-- ============================================================================
-- Entity Mentions (tracks where entities appear)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_mentions (
    id SERIAL PRIMARY KEY,
    entity_id INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    content_type TEXT NOT NULL,                 -- article, podcast, activity, market
    content_id INT NOT NULL,                    -- ID in respective content table

    -- Context of mention
    context TEXT,                               -- Surrounding text snippet
    sentiment FLOAT CHECK (sentiment >= -1 AND sentiment <= 1),

    -- Temporal
    mentioned_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate mentions
    UNIQUE(entity_id, content_type, content_id)
);

-- Indexes for mention lookups
CREATE INDEX IF NOT EXISTS mentions_entity_idx ON entity_mentions (entity_id);
CREATE INDEX IF NOT EXISTS mentions_content_idx
ON entity_mentions (content_type, content_id);
CREATE INDEX IF NOT EXISTS mentions_temporal_idx
ON entity_mentions (mentioned_at DESC);


-- ============================================================================
-- Entity Version History (for temporal reasoning)
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_versions (
    id SERIAL PRIMARY KEY,
    entity_id INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    version INT NOT NULL DEFAULT 1 CHECK (version >= 1),

    -- Snapshot of entity state
    name TEXT NOT NULL,
    description TEXT,
    properties JSONB DEFAULT '{}'::jsonb,

    -- Change tracking
    change_type TEXT NOT NULL,                  -- created, updated, merged, deleted
    change_source TEXT,
    changed_fields TEXT[],

    -- Temporal
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,                    -- NULL = current version
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Each entity can only have one version number
    UNIQUE(entity_id, version)
);

-- Indexes for version queries
CREATE INDEX IF NOT EXISTS versions_entity_idx ON entity_versions (entity_id);
CREATE INDEX IF NOT EXISTS versions_temporal_idx
ON entity_versions (valid_from, valid_until);
CREATE INDEX IF NOT EXISTS versions_change_type_idx
ON entity_versions (change_type);


-- ============================================================================
-- Cross-Domain Correlations
-- ============================================================================

CREATE TABLE IF NOT EXISTS correlations (
    id SERIAL PRIMARY KEY,
    source_entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INT REFERENCES entities(id) ON DELETE CASCADE,

    -- Correlation metrics
    correlation_type TEXT NOT NULL,   -- e.g., "mention_frequency"
    correlation_value FLOAT NOT NULL, -- -1 to 1 correlation coefficient
    significance FLOAT,               -- p-value or confidence

    -- Time window for correlation
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_size_days INT NOT NULL,

    -- Metadata
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_entity_id, target_entity_id, correlation_type, window_start)
);

-- Indexes for correlation queries
CREATE INDEX IF NOT EXISTS correlations_source_idx
ON correlations (source_entity_id);
CREATE INDEX IF NOT EXISTS correlations_target_idx
ON correlations (target_entity_id);
CREATE INDEX IF NOT EXISTS correlations_type_idx
ON correlations (correlation_type);
CREATE INDEX IF NOT EXISTS correlations_window_idx
ON correlations (window_start, window_end);
"""

# Helper function to get schema
def get_knowledge_graph_schema() -> str:
    """Return the complete knowledge graph schema SQL."""
    return KNOWLEDGE_GRAPH_SCHEMA
