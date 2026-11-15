"""Tiny SQLite key/value cache with per-read TTL."""
from __future__ import annotations

import sqlite3
import time
import zlib
from pathlib import Path
from typing import Optional, Tuple

from src import config


class Store:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or config.CACHE_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                "key TEXT PRIMARY KEY, value BLOB NOT NULL, fetched_at REAL NOT NULL)"
            )

    def _ensure_series(self, c: sqlite3.Connection) -> None:
        c.execute("CREATE TABLE IF NOT EXISTS series ("
                  "kind TEXT NOT NULL, key TEXT NOT NULL, date TEXT NOT NULL, value REAL NOT NULL, "
                  "fetched_at REAL NOT NULL, PRIMARY KEY (kind, key, date))")

    def upsert_series(self, kind: str, key: str, rows) -> None:
        """Append-only history: add (date, value) points, replacing a date only with a newer
        pull of that same date. Nothing is ever expired."""
        now = time.time()
        with self._conn() as c:
            self._ensure_series(c)
            c.executemany("INSERT OR REPLACE INTO series (kind, key, date, value, fetched_at) "
                          "VALUES (?, ?, ?, ?, ?)", [(kind, key, d, float(v), now) for d, v in rows])

    def read_series(self, kind: str, key: str):
        """All stored (date, value) points for a series, oldest first."""
        with self._conn() as c:
            self._ensure_series(c)
            return c.execute("SELECT date, value FROM series WHERE kind = ? AND key = ? "
                             "ORDER BY date", (kind, key)).fetchall()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str, ttl: Optional[float] = None) -> Optional[str]:
        """Return the cached text, or None if missing or older than ttl seconds."""
        hit = self.get_with_age(key)
        if hit is None:
            return None
        text, age = hit
        return text if ttl is None or age <= ttl else None

    def get_with_age(self, key: str) -> Optional[Tuple[str, float]]:
        with self._conn() as c:
            row = c.execute("SELECT value, fetched_at FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return zlib.decompress(row[0]).decode("utf-8"), time.time() - row[1]

    def put(self, key: str, text: str) -> None:
        blob = zlib.compress(text.encode("utf-8"))
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO kv (key, value, fetched_at) VALUES (?, ?, ?)",
                (key, blob, time.time()),
            )
