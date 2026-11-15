"""News coverage volume and tone from the GDELT DOC 2.0 API (free, no key).

Context only: nothing here feeds a KPI. The API reaches back roughly three months, so
every pull is stored permanently (append-only) and history builds up over time.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src import config
from src.cache.store import Store

MODES = {"volume": "timelinevol", "tone": "timelinetone"}
_SUFFIXES = re.compile(
    r"[\s,]+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|n\.?v|s\.?a|ag|se|"
    r"holdings?|group|the)\.?$", re.IGNORECASE)


class GdeltError(RuntimeError):
    """GDELT could not be reached or refused the request."""


def default_query(company_name: str) -> str:
    """'UNILEVER PLC' -> '"Unilever"', 'Apple Inc.' -> '"Apple"'. A starting point the user
    can edit: a short name can match unrelated stories."""
    name = re.sub(r"\s*/[A-Za-z]{2,3}/?\s*$", "", company_name.strip())   # 'XYZ CORP /DE/'
    while True:
        shorter = _SUFFIXES.sub("", name).strip(" ,.")
        if shorter == name or not shorter:
            break
        name = shorter
    name = name.title() if name.isupper() else name
    return f'"{name}"' if name else ""


def parse_timeline(payload: dict) -> List[Tuple[str, float]]:
    """GDELT timeline JSON -> [(YYYY-MM-DD, value)], one point per day (finer-grained points
    are averaged). An empty or malformed payload gives an empty list."""
    try:
        data = payload["timeline"][0]["data"]
    except (KeyError, IndexError, TypeError):
        return []
    by_day: Dict[str, List[float]] = {}
    for point in data:
        try:
            raw, value = str(point["date"]), float(point["value"])
        except (KeyError, TypeError, ValueError):
            continue
        digits = re.sub(r"\D", "", raw)[:8]
        if len(digits) == 8:
            by_day.setdefault(f"{digits[:4]}-{digits[4:6]}-{digits[6:]}", []).append(value)
    return sorted((day, sum(vals) / len(vals)) for day, vals in by_day.items())


def fetch_mode(query: str, mode: str, session=None,
               timespan: str = config.GDELT_TIMESPANS[0]) -> List[Tuple[str, float]]:
    params = {"query": query, "mode": MODES[mode], "format": "json", "timespan": timespan}
    try:
        resp = (session or requests).get(config.URL_GDELT_DOC, params=params,
                                         timeout=config.HTTP_TIMEOUT_SECONDS,
                                         headers={"User-Agent": config.GENERIC_USER_AGENT})
    except requests.RequestException as exc:
        raise GdeltError(f"GDELT could not be reached ({exc}).") from exc
    if resp.status_code == 429:
        raise GdeltError("GDELT is limiting requests right now (it allows one every 5 seconds "
                         "and sometimes blocks an address for a while). Try again later.")
    if resp.status_code != 200:
        raise GdeltError(f"GDELT answered with status {resp.status_code}.")
    try:
        return parse_timeline(resp.json())
    except ValueError as exc:   # GDELT reports query errors as plain text
        raise GdeltError(f"GDELT did not return data: {resp.text[:120].strip()}") from exc


def pull_state(query: str, store: Optional[Store] = None) -> Dict[str, str]:
    """Whether a pull is worth making now, without touching the network:
    'fresh' (pulled recently), 'waiting' (a recent pull failed; message says why) or 'due'."""
    store = store or Store()
    if store.get(f"gdelt:ok:{query}", config.GDELT_REFRESH_SECONDS) is not None:
        return {"status": "fresh", "message": ""}
    failed = store.get(f"gdelt:try:{query}", config.GDELT_RETRY_SECONDS)
    if failed is not None:
        return {"status": "waiting", "message": failed}
    return {"status": "due", "message": ""}


def refresh(query: str, store: Optional[Store] = None, session=None,
            sleep: Callable[[float], None] = time.sleep,
            now: Callable[[], float] = time.time) -> Dict[str, str]:
    """Pull both timelines if a pull is due, and append them to the stored history.

    Returns {'status': 'fresh' | 'updated' | 'waiting' | 'failed', 'message': ...}. Never
    raises: the panel shows the message and whatever history is already stored.
    """
    store = store or Store()
    if not query.strip():
        return {"status": "failed", "message": "Enter a search phrase for news coverage."}
    ok_key, try_key = f"gdelt:ok:{query}", f"gdelt:try:{query}"
    if store.get(ok_key, config.GDELT_REFRESH_SECONDS) is not None:
        return {"status": "fresh", "message": ""}
    failed = store.get(try_key, config.GDELT_RETRY_SECONDS)
    if failed is not None:
        return {"status": "waiting", "message": failed}
    try:
        first = True
        for mode in MODES:
            for timespan in config.GDELT_TIMESPANS:   # longest window first; shorter if empty
                if not first:
                    sleep(config.GDELT_MIN_GAP_SECONDS)
                first = False
                points = fetch_mode(query, mode, session, timespan)
                if points:
                    store.upsert_series(f"gdelt:{mode}", query, points)
                    break
    except GdeltError as exc:
        store.put(try_key, str(exc))
        return {"status": "failed", "message": str(exc)}
    store.put(ok_key, str(now()))
    return {"status": "updated", "message": ""}


def history(query: str, store: Optional[Store] = None) -> pd.DataFrame:
    """Everything stored for a query: date, volume, tone (NaN where a mode has no point)."""
    store = store or Store()
    frames = [pd.DataFrame(store.read_series(f"gdelt:{mode}", query), columns=["date", mode])
              .set_index("date") for mode in MODES]
    out = frames[0].join(frames[1], how="outer").sort_index().reset_index()
    return out.rename(columns={"index": "date"})
