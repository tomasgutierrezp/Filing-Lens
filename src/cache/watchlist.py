"""A saved list of tickers, kept in the local SQLite cache."""
from __future__ import annotations

import json
from typing import List, Optional

from src.cache.store import Store

_KEY = "watchlist"
MAX_ITEMS = 30


def load(store: Optional[Store] = None) -> List[str]:
    raw = (store or Store()).get(_KEY)
    try:
        items = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [str(t) for t in items if isinstance(t, str)]


def add(ticker: str, store: Optional[Store] = None) -> List[str]:
    store = store or Store()
    t = ticker.strip().upper()
    items = load(store)
    if t and t not in items:
        items = (items + [t])[-MAX_ITEMS:]
        store.put(_KEY, json.dumps(items))
    return items


def remove(ticker: str, store: Optional[Store] = None) -> List[str]:
    store = store or Store()
    items = [t for t in load(store) if t != ticker.strip().upper()]
    store.put(_KEY, json.dumps(items))
    return items
