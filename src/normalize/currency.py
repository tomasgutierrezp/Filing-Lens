"""FX conversion with ECB reference rates from the Frankfurter API (free, no key)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import requests

from src import config
from src.cache.store import Store


class FxError(RuntimeError):
    """The exchange rate could not be fetched and none was cached."""


@dataclass(frozen=True)
class FxRate:
    base: str
    quote: str
    rate: float      # 1 unit of `base` = `rate` units of `quote`
    date: str        # date the reference rate applies to
    source: str = "ECB reference rate via Frankfurter"


def convert(amount: float, rate: FxRate) -> float:
    return amount * rate.rate


def get_rate(base: str, quote: str, store: Optional[Store] = None,
             session: Optional[requests.Session] = None, ttl: float = config.TTL_PRICES) -> FxRate:
    base, quote = base.upper(), quote.upper()
    if base == quote:
        return FxRate(base, quote, 1.0, "", "same currency")
    store = store or Store()
    key = f"fx:{base}:{quote}"
    cached = store.get(key, ttl)
    if cached is None:
        try:
            resp = (session or requests).get(
                config.URL_FRANKFURTER.format(base=base, quote=quote),
                timeout=config.HTTP_TIMEOUT_SECONDS, headers={"User-Agent": config.GENERIC_USER_AGENT})
            resp.raise_for_status()
            data = resp.json()
            float(data["rates"][quote])   # validate before caching
            cached = json.dumps(data)
            store.put(key, cached)
        except Exception as exc:
            cached = store.get(key)       # stale rate beats no rate
            if cached is None:
                raise FxError(f"Exchange rate {base}/{quote} is unavailable right now ({exc}).") from exc
    data = json.loads(cached)
    return FxRate(base, quote, float(data["rates"][quote]), str(data.get("date", "")))
