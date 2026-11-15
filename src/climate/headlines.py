"""Headline lists from Google News RSS: titles, links, dates and outlet names only.
Article bodies are never fetched."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import List, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from src import config
from src.cache.store import Store

HEADLINE_COLUMNS = ["date", "title", "source", "link"]
_MAX_BYTES = 3_000_000


class HeadlinesError(RuntimeError):
    """The headline feed could not be fetched and nothing was cached."""


def build_url(query: str, start: str, end: str) -> str:
    """Search feed for `query` between two ISO dates (both inclusive)."""
    until = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    params = {"q": f"{query} after:{start} before:{until}", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    return f"{config.URL_GOOGLE_NEWS_RSS}?{urlencode(params)}"


def parse_rss(xml_text: str) -> List[dict]:
    """RSS -> [{'date', 'title', 'source', 'link'}], newest first. Bad items are skipped."""
    if "<!DOCTYPE" in xml_text[:2000].upper() or "<!ENTITY" in xml_text.upper():
        raise ValueError("unexpected document type")     # a feed has no DTD; refuse entities
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        title, link = (item.findtext("title") or "").strip(), (item.findtext("link") or "").strip()
        if not title or not link.startswith("http"):
            continue
        try:
            date = parsedate_to_datetime(item.findtext("pubDate") or "").strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            date = ""
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        items.append({"date": date, "title": title, "source": source, "link": link})
    return sorted(items, key=lambda i: i["date"], reverse=True)


def get_headlines(query: str, start: str, end: str, store: Optional[Store] = None,
                  session=None) -> pd.DataFrame:
    store = store or Store()
    url = build_url(query, start, end)
    cached = store.get(url, config.TTL_HEADLINES)
    if cached is None:
        try:
            resp = (session or requests).get(url, timeout=config.HTTP_TIMEOUT_SECONDS,
                                             headers={"User-Agent": config.GENERIC_USER_AGENT})
            resp.raise_for_status()
            if len(resp.content) > _MAX_BYTES:
                raise ValueError("feed too large")
            cached = json.dumps(parse_rss(resp.text))
            store.put(url, cached)
        except Exception as exc:
            cached = store.get(url)
            if cached is None:
                raise HeadlinesError(f"Headlines are unavailable right now ({exc}).") from exc
    frame = pd.DataFrame(json.loads(cached), columns=HEADLINE_COLUMNS)
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)
