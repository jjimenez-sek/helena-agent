CREATE EXTENSION IF NOT EXISTS vector;

-- RAG document store
CREATE TABLE IF NOT EXISTS documents (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content    TEXT NOT NULL,
    metadata   JSONB NOT NULL DEFAULT '{}',
    embedding  vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Interrupt audit log
CREATE TABLE IF NOT EXISTS approval_requests (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id   TEXT NOT NULL,
    payload     JSONB NOT NULL,
    status      TEXT DEFAULT 'pending',  -- pending | approved | rejected
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS approval_requests_thread_status_idx
    ON approval_requests (thread_id, status);

-- LangGraph checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes,
-- checkpoint_migrations) are created by AsyncPostgresSaver.setup() at agent
-- startup. Do NOT duplicate the DDL here — setup() uses checkpoint_migrations
-- to track its own schema versions, and pre-existing tables with an empty
-- tracker cause a silent rollback.
