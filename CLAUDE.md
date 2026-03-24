# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the service (dev, with hot reload)
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_graph.py -v

# Run with Docker Compose
docker compose up agent --build

# LangGraph Studio (visual graph debugging, in-memory checkpointer)
docker compose up langgraph-studio --build
# Opens at http://localhost:2024
```

## Architecture

Helena Agent is a **LangGraph-based multi-agent service** for security operations. A NestJS backend sends chat messages via HTTP; this service processes them through a state machine and streams responses as Server-Sent Events (SSE). State is persisted per conversation in PostgreSQL using LangGraph's `AsyncPostgresSaver`.

### Request Flow

```
NestJS → POST /chat/stream (JWT + message body)
  → FastAPI (src/main.py): validate JWT, scope thread_id as "{keycloak_sub}:{client_thread_id}"
  → LangGraph graph.astream() → PostgreSQL checkpointer
  → SSE events streamed back: token | rfc_step_progress | closed_questions | execute_workflow | node_update | interrupt_detected | done | error
```

### Graph State Machine (`src/graphs/main_graph.py`)

```
START → triage → rfc_open_questions → rfc_closed_questions → rfc_summary_confirm → rfc_execute → END
                ↘ fallback_response → END
```

- **triage**: Silent LLM intent classification — `rfc | incident | knowledge | escalation | unknown`. Skips re-classification if RFC flow is already in progress.
- **rfc_open_questions**: 5 conversational steps collecting structured fields via two LLM calls (extraction + natural reply). Emits `rfc_step_progress` SSE.
- **rfc_closed_questions**: 8 hardcoded single-select questions (in Spanish). Emits `closed_questions` SSE with form schema for the frontend.
- **rfc_summary_confirm**: Formats a professional RFC document and waits for keyword confirmation (`confirm`, `yes`, `sí`, `ok`, `approve`).
- **rfc_execute**: Emits `execute_workflow` SSE for NestJS to trigger N8N webhooks. Sets `rfc_execute_confirmed=True` to prevent re-execution.
- **fallback**: Explains capabilities and lists upcoming features.

**Auto-chaining**: When a node completes all its sub-steps, the next node runs in the same turn — no extra user message needed. This is implemented via conditional edges and state flags (`rfc_open_complete`, `rfc_closed_complete`, `rfc_confirmed`).

### Key State (`src/graphs/state.py`)

`AgentState` is a `TypedDict` persisted per `thread_id`:
- `messages`: full conversation history (LangGraph `add_messages` reducer)
- `intent`: current classified intent
- `rfc_step`: 0–7 tracking progress through the RFC flow
- `rfc_data`: accumulated RFC fields dict
- `project_workflows`: list of `{workflow_id, name, description}` from NestJS

### Authentication (`src/auth/keycloak.py`)

RS256 JWT validation via JWKS (cached 5 min). Thread IDs are scoped per user (`{sub}:{thread_id}`) to prevent cross-user state leaks. API keys from the request body are stripped before emitting `node_update` SSE events.

### Database (`src/db/postgres.py`, `migrations/init.sql`)

asyncpg pool + LangGraph's `AsyncPostgresSaver`. Schema runs idempotently on startup. Tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` (LangGraph), `documents` (pgvector for RAG), `approval_requests` (interrupt audit log).

### Observability (`src/observability.py`)

- Prometheus metrics at `GET /metrics` — request counts, node invocations, RFC funnel, errors
- Optional Langfuse LLM tracing — gracefully disabled if keys not configured

## Environment

Copy `.env.example` to `.env`. Required vars:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `KEYCLOAK_ISSUER` | Keycloak issuer URL |
| `KEYCLOAK_JWKS_URL` | JWKS endpoint for RS256 validation |

Optional: `OPENAI_API_KEY` (fallback; per-request key preferred), `LANGFUSE_*`, `KEYCLOAK_TLS_VERIFY`.

## Testing

Tests use `MemorySaver` (no database needed). Key patterns: mock agent nodes, patch LLM calls, stream the graph, inspect SSE chunks. The test suite verifies routing logic, API key security (no leaks into state/interrupts), and interrupt/resume flows.

## Studio Graph

`src/graphs/studio_graph.py` is the LangGraph Studio entrypoint — uses an in-memory checkpointer and is not used in production.
