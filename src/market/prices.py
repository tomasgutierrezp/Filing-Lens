"""Daily prices: yfinance first, Stooq CSV as fallback, cached in SQLite.

yfinance is an unofficial Yahoo Finance client, so every failure is expected and handled:
callers get a PriceError with a plain message and the filing views keep working.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd
import requests

from src import config
from src.cache.store import Store

COLUMNS = ["date", "close", "volume"]
Fetcher = Callable[[str], Tuple[pd.DataFrame, str, str]]   # -> (frame, currency, source name)


class PriceError(RuntimeError):
    """No price source worked and nothing usable was cached."""


@dataclass
class PriceHistory:
    ticker: str
    currency: str
    source: str
    fetched_at: str        # ISO timestamp (UTC)
    frame: pd.DataFrame    # date (ISO string), close (adjusted), volume
    stale: bool = False    # True when served from an expired cache after a failed refresh

    @property
    def latest(self) -> Tuple[str, float]:
        row = self.frame.iloc[-1]
        return str(row["date"]), float(row["close"])


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[COLUMNS].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("no price rows")
    return frame


def fetch_yfinance(ticker: str) -> Tuple[pd.DataFrame, str, str]:
    import yfinance as yf  # imported lazily: optional at import time, slow to load

    t = yf.Ticker(ticker.replace(".", "-"))
    hist = t.history(period=config.PRICE_HISTORY_PERIOD, auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError("Yahoo Finance returned no data")
    try:
        currency = str(t.fast_info["currency"] or "USD")
    except Exception:
        currency = "USD"
    frame = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in hist.index],
                          "close": hist["Close"].astype(float).values,
                          "volume": hist["Volume"].astype(float).values})
    return _clean(frame), currency, "Yahoo Finance (yfinance)"


def parse_stooq_csv(text: str) -> pd.DataFrame:
    """Stooq daily CSV -> frame. Raises ValueError for anything that is not that CSV
    (Stooq sometimes answers with an HTML verification page or 'No data')."""
    if not text.lstrip().startswith("Date,"):
        raise ValueError("Stooq did not return CSV data")
    raw = pd.read_csv(io.StringIO(text))
    if "Close" not in raw or raw.empty:
        raise ValueError("Stooq CSV has no prices")
    volume = raw["Volume"] if "Volume" in raw else float("nan")
    return _clean(pd.DataFrame({"date": raw["Date"].astype(str), "close": raw["Close"].astype(float),
                                "volume": volume}))


def fetch_stooq(ticker: str) -> Tuple[pd.DataFrame, str, str]:
    symbol = ticker.lower().replace(".", "-") + ".us"
    resp = requests.get(config.URL_STOOQ.format(symbol=symbol), timeout=config.HTTP_TIMEOUT_SECONDS,
                        headers={"User-Agent": config.GENERIC_USER_AGENT})
    resp.raise_for_status()
    return parse_stooq_csv(resp.text), "USD", "Stooq"


def _to_json(h: PriceHistory) -> str:
    return json.dumps({"ticker": h.ticker, "currency": h.currency, "source": h.source,
                       "fetched_at": h.fetched_at, "rows": h.frame[COLUMNS].values.tolist()})


def _from_json(text: str, stale: bool = False) -> PriceHistory:
    d = json.loads(text)
    return PriceHistory(d["ticker"], d["currency"], d["source"], d["fetched_at"],
                        pd.DataFrame(d["rows"], columns=COLUMNS), stale)


def get_history(ticker: str, store: Optional[Store] = None, ttl: float = config.TTL_PRICES,
                fetchers: Optional[Sequence[Fetcher]] = None) -> PriceHistory:
    """Cached daily history. Tries each fetcher in order; on total failure returns the last
    cached copy (flagged stale) or raises PriceError."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise PriceError("No ticker to look up prices for.")
    store = store or Store()
    key = f"prices:{ticker}"
    fresh = store.get(key, ttl)
    if fresh is not None:
        return _from_json(fresh)
    errors = []
    for fetch in (fetchers if fetchers is not None else (fetch_yfinance, fetch_stooq)):
        try:
            frame, currency, source = fetch(ticker)
        except Exception as exc:   # any failure of an unofficial source is survivable
            errors.append(f"{getattr(fetch, '__name__', 'source')}: {exc}")
            continue
        history = PriceHistory(ticker, currency, source,
                               datetime.now(timezone.utc).isoformat(timespec="seconds"), frame)
        store.put(key, _to_json(history))
        return history
    old = store.get(key)
    if old is not None:
        return _from_json(old, stale=True)
    raise PriceError(f"Prices for {ticker} are unavailable right now ({'; '.join(errors)}).")
