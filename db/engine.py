"""
db/engine.py — SQLAlchemy engine, session factory, and schema bootstrap.

The database is a single local SQLite file (backtester.db) in the project
root. SQLite needs no server and travels with the repo's working tree; the
schema is intentionally portable so it can move to Postgres later with minimal
changes (JSON columns, no SQLite-only types).
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Project root = parent of this db/ package.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allow override (e.g. tests use a temp file / :memory:) via env var.
DB_PATH = os.getenv("BACKTESTER_DB_PATH", os.path.join(_PROJECT_ROOT, "backtester.db"))

_DB_URL = os.getenv("BACKTESTER_DB_URL", f"sqlite:///{DB_PATH}")

# check_same_thread=False so the FastAPI background worker thread can share the
# engine; sessions are still used one-per-unit-of-work via session_scope().
engine = create_engine(
    _DB_URL,
    future=True,
    connect_args={"check_same_thread": False} if _DB_URL.startswith("sqlite") else {},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_fks(dbapi_conn, _):
    """SQLite ignores ON DELETE CASCADE unless foreign keys are switched on
    per-connection. Enable them so deleting a run cascades to its children."""
    try:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        pass


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they don't exist. Idempotent."""
    # Import here so models register on Base before create_all.
    from db.models import Base
    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    """Transactional session scope: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
