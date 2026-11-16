"""Small SQLite catalog for document lifecycle and chunk previews."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


class DocumentRepository:
    def __init__(self, path: Path):
        self.path = path
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, chunk_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL, error TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, position INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM documents ORDER BY created_at DESC")]

    def get(self, doc_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def add(self, doc_id, filename, path):
        with self.connection() as db:
            db.execute(
                "INSERT INTO documents(id,filename,path,status,stage,created_at) "
                "VALUES(?,?,?,'queued','等待处理',?)",
                (doc_id, filename, str(path), datetime.now(timezone.utc).isoformat()),
            )

    def update(self, doc_id, status, stage, error="", count=0):
        with self.connection() as db:
            db.execute(
                "UPDATE documents SET status=?,stage=?,error=?,chunk_count=? WHERE id=?",
                (status, stage, error, count, doc_id),
            )

    def save_chunks(self, doc_id, chunks):
        with self.connection() as db:
            db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            db.executemany(
                "INSERT INTO chunks VALUES(?,?,?,?)",
                [
                    (c.id, doc_id, c.index, json.dumps(asdict(c), ensure_ascii=False))
                    for c in chunks
                ],
            )

    def chunks(self, doc_id, offset=0, limit=100):
        with self.connection() as db:
            return [
                json.loads(r["payload"])
                for r in db.execute(
                    "SELECT payload FROM chunks WHERE document_id=? "
                    "ORDER BY position LIMIT ? OFFSET ?",
                    (doc_id, limit, offset),
                )
            ]

    def delete(self, doc_id):
        with self.connection() as db:
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
