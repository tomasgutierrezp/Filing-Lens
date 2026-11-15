"""Central configuration: .env loading, constants, rate limits."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("FILING_LENS_DATA_DIR", ROOT / "data"))
CACHE_DB = DATA_DIR / "cache.db"

# SEC asks automated clients to identify themselves with a contact address.
# www.sec.gov rejects requests without one, so set SEC_USER_AGENT in .env.
_DEFAULT_UA = "FilingLens/0.1 (personal research tool)"
SEC_USER_AGENT = (os.getenv("SEC_USER_AGENT") or "").strip().strip('"') or _DEFAULT_UA
SEC_USER_AGENT_IS_DEFAULT = SEC_USER_AGENT == _DEFAULT_UA

SEC_MAX_REQUESTS_PER_SECOND = 5  # SEC limit is 10; stay well under it
HTTP_TIMEOUT_SECONDS = 30

HOUR = 3600
TTL_EDGAR = 24 * HOUR
TTL_PRICES = 12 * HOUR

URL_TICKERS = "https://www.sec.gov/files/company_tickers.json"
URL_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
URL_SUBMISSIONS_FILE = "https://data.sec.gov/submissions/{name}"
URL_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
URL_FILING_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{accn}-index.htm"
URL_FILING_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{doc}"

ANNUAL_FORMS = ("10-K", "20-F", "40-F")
INTERIM_FORMS = ("10-Q", "6-K")
EVENT_FORMS = ("8-K", "6-K")
SUPPORTED_FORMS = ANNUAL_FORMS + ("10-Q", "6-K", "8-K")


def base_form(form: str) -> str:
    """'10-K/A' -> '10-K'."""
    return form[:-2] if form.endswith("/A") else form

# ---- market data (Tier 3) --------------------------------------------------------
PRICE_HISTORY_PERIOD = "5y"
URL_STOOQ = "https://stooq.com/q/d/l/?s={symbol}&i=d"
URL_FRANKFURTER = "https://api.frankfurter.dev/v1/latest?base={base}&symbols={quote}"
GENERIC_USER_AGENT = "FilingLens/0.1"   # non-SEC services never receive the SEC contact

# Ordinary shares represented by one US-listed ADR, per ticker. Check each value against
# the company's 20-F before relying on it; tickers not listed here are asked for in the UI.
ADR_ORDINARY_PER_ADR = {"UL": 1.0}

# ---- news climate (context only, never used in a KPI) ----------------------------------
URL_GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
# The DOC 2.0 API reaches back about three months, but long windows sometimes come back
# empty (seen 2026-11-15: "3months" gave {}, "1w" gave data), so shorter ones are tried next.
GDELT_TIMESPANS = ("3months", "1month", "1w")
GDELT_MIN_GAP_SECONDS = 8           # GDELT allows one request every 5 seconds; stay clear of it
GDELT_REFRESH_SECONDS = 12 * HOUR   # pull new points at most twice a day
GDELT_RETRY_SECONDS = 15 * 60       # after a failed pull, wait before trying again
URL_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
TTL_HEADLINES = HOUR
