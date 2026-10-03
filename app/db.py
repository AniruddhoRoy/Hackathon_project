"""SQLite for app metadata: indexed books and ingestion jobs."""

import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    source     TEXT PRIMARY KEY,   -- file name, e.g. ICT_ben_class7.pdf
    subject    TEXT NOT NULL,
    class_num  INTEGER NOT NULL,
    language   TEXT NOT NULL,      -- bn | en
    chunks     INTEGER NOT NULL DEFAULT 0,
    origin     TEXT NOT NULL,      -- hf-dataset | upload
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS jobs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    status     TEXT NOT NULL,      -- queued | running | done | failed
    message    TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def upsert_book(source: str, subject: str, class_num: int, language: str, chunks: int, origin: str) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO books (source, subject, class_num, language, chunks, origin)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(source) DO UPDATE SET
                 subject=excluded.subject, class_num=excluded.class_num,
                 language=excluded.language, chunks=excluded.chunks, origin=excluded.origin""",
            (source, subject, class_num, language, chunks, origin),
        )


def list_books() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM books ORDER BY class_num, subject, language").fetchall()


def delete_book(source: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM books WHERE source = ?", (source,))


def filter_options() -> dict[str, list]:
    with connect() as conn:
        classes = [r[0] for r in conn.execute("SELECT DISTINCT class_num FROM books ORDER BY class_num")]
        subjects = [r[0] for r in conn.execute("SELECT DISTINCT subject FROM books ORDER BY subject")]
    return {"classes": classes, "subjects": subjects}


def create_job(source: str) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO jobs (source, status) VALUES (?, 'queued')", (source,))
        return cur.lastrowid


def update_job(job_id: int, status: str, message: str | None = None) -> None:
    with connect() as conn:
        conn.execute("UPDATE jobs SET status = ?, message = ? WHERE id = ?", (status, message, job_id))


def recent_jobs(limit: int = 20) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
