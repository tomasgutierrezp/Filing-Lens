"""Company lookup: ticker, SEC company number (CIK) or company name."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from src import config
from src.edgar.client import EdgarClient, EdgarError, default_client

# Everyday names that differ from the registered company name or collide with another ticker.
ALIASES: Dict[str, str] = {
    "google": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "instagram": "META",
    "ford": "F", "coke": "KO", "coca cola": "KO", "jp morgan": "JPM", "chase": "JPM",
    "berkshire": "BRK-B", "exxon": "XOM", "mcdonalds": "MCD", "pepsi": "PEP", "ibm": "IBM",
    "p g": "PG", "procter and gamble": "PG", "johnson and johnson": "JNJ", "j j": "JNJ",
    "att": "T", "ge": "GE", "3m": "MMM", "ups": "UPS", "gm": "GM", "disney": "DIS",
}


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str


def _norm(text: str) -> str:
    """Lower-case, punctuation to spaces: 'AT&T Inc.' -> 'at t inc'."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def display_name(name: str) -> str:
    """SEC names are often all capitals; show those in title case."""
    return name.title() if name.isupper() else name


def option_label(company: Company) -> str:
    return f"{display_name(company.name)} ({company.ticker})"


def company_list(client: Optional[EdgarClient] = None) -> List[Company]:
    """Every company in the SEC ticker list, in the SEC's order (largest first)."""
    table = (client or default_client()).get_json(config.URL_TICKERS)
    return [Company(int(r["cik_str"]), str(r["ticker"]).upper(), str(r["title"])) for r in table.values()]


def search(query: str, companies: Iterable[Company], limit: int = 8) -> List[Company]:
    """Best matches for a ticker or (part of) a company name. Exact ticker first, then names
    that equal, start with, contain a word starting with, or merely contain the text; ties
    keep the SEC's order, which puts larger companies first."""
    q = _norm(query)
    if not q:
        return []
    ticker_q = query.strip().upper().replace(".", "-")
    scored = []
    for rank, c in enumerate(companies):
        name = _norm(c.name)
        if c.ticker == ticker_q:
            tier = 0
        elif name == q:
            tier = 1
        elif name.startswith(q + " ") or name.startswith(q):
            tier = 2
        elif f" {q}" in f" {name}":
            tier = 3
        elif q in name:
            tier = 4
        else:
            continue
        scored.append((tier, rank, c))
    return [c for _, _, c in sorted(scored, key=lambda t: t[:2])[:limit]]


def lookup(query: str, client: Optional[EdgarClient] = None) -> Company:
    """Resolve a ticker, a numeric CIK or a company name ('apple') to a Company."""
    client = client or default_client()
    text = query.strip()
    if not text:
        raise EdgarError("Enter a company name or ticker.")
    if text.isdigit():
        sub = client.get_json(config.URL_SUBMISSIONS.format(cik=int(text)))
        tickers = sub.get("tickers") or [""]
        return Company(int(text), tickers[0], sub.get("name", ""))
    companies = company_list(client)
    by_ticker = {c.ticker: c for c in reversed(companies)}
    alias = ALIASES.get(_norm(text))
    if alias and alias in by_ticker:
        return by_ticker[alias]
    ticker = text.upper().replace(".", "-")
    if ticker in by_ticker:
        return by_ticker[ticker]
    hits = search(text, companies, limit=1)
    if hits:
        return hits[0]
    raise EdgarError(f"No company matching '{text}' was found in the SEC list. "
                     "Try the ticker symbol or the registered company name.")
