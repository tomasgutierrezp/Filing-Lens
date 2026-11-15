"""Throttled, cached HTTP access to SEC EDGAR."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

import requests

from src import config
from src.cache.store import Store


class EdgarError(RuntimeError):
    """A request to EDGAR failed and no cached copy was available."""


class EdgarClient:
    def __init__(self, store: Optional[Store] = None, user_agent: Optional[str] = None):
        self.store = store or Store()
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent or config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )
        self._lock = threading.Lock()
        self._min_interval = 1.0 / config.SEC_MAX_REQUESTS_PER_SECOND
        self._last_request = 0.0

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get_json(self, url: str, ttl: float = config.TTL_EDGAR) -> Any:
        """Fetch JSON, using the cache when fresh and falling back to a stale copy on failure."""
        fresh = self.store.get(url, ttl)
        if fresh is not None:
            return json.loads(fresh)
        try:
            self._throttle()
            resp = self.session.get(url, timeout=config.HTTP_TIMEOUT_SECONDS)
            if resp.status_code == 403:
                raise EdgarError(
                    "SEC refused the request (403). Set SEC_USER_AGENT in .env to "
                    '"Your Name your@email.com" — the SEC requires a contact address.'
                )
            if resp.status_code == 404:
                raise EdgarError(f"Not found on EDGAR: {url}")
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError, EdgarError) as exc:
            stale = self.store.get(url)
            if stale is not None:
                return json.loads(stale)
            if isinstance(exc, EdgarError):
                raise
            raise EdgarError(f"Could not reach EDGAR ({url}): {exc}") from exc
        self.store.put(url, resp.text)
        return data


_default: Optional[EdgarClient] = None


def default_client() -> EdgarClient:
    global _default
    if _default is None:
        _default = EdgarClient()
    return _default
