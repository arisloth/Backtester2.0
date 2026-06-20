"""
db/ — SQLite persistence layer for backtester runs.

The database is the system of record for every run: config, metrics, trades,
equity curve, and optimizer / walk-forward output. It is additive — the CLI
keeps writing its per-run folders under results/; the DB is populated in
parallel and powers the web dashboard.

Quick start:
    from db import init_db, session_scope
    from db.ingest import ingest_backtest

    init_db()                       # create tables (idempotent)
    with session_scope() as s:
        run_id = ingest_backtest(cfg, metrics, eq, trades, session=s)
"""

from db.engine import engine, SessionLocal, init_db, session_scope, DB_PATH

__all__ = ["engine", "SessionLocal", "init_db", "session_scope", "DB_PATH"]
