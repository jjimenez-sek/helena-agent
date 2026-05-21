"""
Emergency script: creates LangGraph checkpoint tables if they don't exist.
Run with: kubectl exec -n frontend <agent-pod> -- python scripts/fix_checkpoints.py
"""
import asyncio
import os
import sys

import asyncpg


async def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    print(f"Connected to database")

    async with pool.acquire() as conn:
        # LangGraph AsyncPostgresSaver tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_migrations (
                v INTEGER PRIMARY KEY
            )
        """)
        print("checkpoint_migrations: OK")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id            TEXT NOT NULL,
                checkpoint_ns        TEXT NOT NULL DEFAULT '',
                checkpoint_id        TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type                 TEXT,
                checkpoint           JSONB NOT NULL,
                metadata             JSONB NOT NULL DEFAULT '{}',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        print("checkpoints: OK")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id     TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                channel       TEXT NOT NULL,
                version       TEXT NOT NULL,
                type          TEXT NOT NULL,
                blob          BYTEA,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            )
        """)
        print("checkpoint_blobs: OK")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id     TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id       TEXT NOT NULL,
                idx           INTEGER NOT NULL,
                channel       TEXT NOT NULL,
                type          TEXT,
                blob          BYTEA NOT NULL,
                task_path     TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)
        print("checkpoint_writes: OK")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx
                ON checkpoints (thread_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx
                ON checkpoint_blobs (thread_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx
                ON checkpoint_writes (thread_id)
        """)
        print("indexes: OK")

        # Also create application tables if missing
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS approval_requests (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                thread_id   TEXT NOT NULL,
                payload     JSONB NOT NULL,
                status      TEXT DEFAULT 'pending',
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                resolved_at TIMESTAMPTZ
            )
        """)
        print("approval_requests: OK")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS approval_requests_thread_status_idx
                ON approval_requests (thread_id, status)
        """)

    await pool.close()
    print("\nAll checkpoint tables created successfully.")
    print("Restart the agent pods to apply changes:")
    print("  kubectl rollout restart deployment/helena-agent -n frontend")


if __name__ == "__main__":
    asyncio.run(main())
