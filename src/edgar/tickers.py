"""Ticker -> CIK lookup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src import config
from src.edgar.client import EdgarClient, EdgarError, default_client


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str


def lookup(ticker: str, client: Optional[EdgarClient] = None) -> Company:
    """Resolve a ticker (or a numeric CIK) to a Company."""
    client = client or default_client()
    query = ticker.strip().upper()
    if not query:
        raise EdgarError("Enter a ticker.")
    if query.isdigit():
        sub = client.get_json(config.URL_SUBMISSIONS.format(cik=int(query)))
        tickers = sub.get("tickers") or [""]
        return Company(int(query), tickers[0], sub.get("name", ""))
    table = client.get_json(config.URL_TICKERS)
    wanted = {query, query.replace(".", "-")}
    for row in table.values():
        if str(row["ticker"]).upper() in wanted:
            return Company(int(row["cik_str"]), str(row["ticker"]).upper(), row["title"])
    raise EdgarError(f"Ticker '{query}' was not found in the SEC ticker list.")
