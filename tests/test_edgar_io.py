import json

import pytest
import requests

from src import config
from src.cache.store import Store
from src.edgar import filings, tickers
from src.edgar.client import EdgarClient, EdgarError


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def get(self, url, timeout=None):
        self.calls.append(url)
        r = self.routes[url]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def client(tmp_path):
    c = EdgarClient(store=Store(tmp_path / "cache.db"), user_agent="test")
    c._min_interval = 0
    return c


def test_store_ttl(tmp_path):
    s = Store(tmp_path / "c.db")
    s.put("k", "hello")
    assert s.get("k", ttl=60) == "hello"
    assert s.get("k", ttl=-1) is None      # expired
    assert s.get("k") == "hello"           # still readable as a stale fallback
    assert s.get("missing") is None


def test_client_caches_and_falls_back_to_stale(client):
    client.session = FakeSession({"u": FakeResp({"a": 1})})
    assert client.get_json("u") == {"a": 1}
    assert client.get_json("u") == {"a": 1}
    assert client.session.calls == ["u"]                       # second read came from cache
    client.session = FakeSession({"u": requests.ConnectionError("down")})
    assert client.get_json("u", ttl=-1) == {"a": 1}            # stale copy beats an outage


def test_client_errors_are_clear(client):
    client.session = FakeSession({"u": requests.ConnectionError("down"), "f": FakeResp({}, 403)})
    with pytest.raises(EdgarError, match="Could not reach EDGAR"):
        client.get_json("u")
    with pytest.raises(EdgarError, match="SEC_USER_AGENT"):
        client.get_json("f")


def test_ticker_lookup(client):
    table = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
             "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"}}
    client.session = FakeSession({config.URL_TICKERS: FakeResp(table)})
    assert tickers.lookup(" aapl ", client).cik == 320193
    assert tickers.lookup("BRK.B", client).cik == 1067983
    with pytest.raises(EdgarError, match="not found"):
        tickers.lookup("NOPE", client)


def test_list_filings(client):
    sub = {"name": "Test Co", "tickers": ["TST"], "fiscalYearEnd": "1231", "filings": {"recent": {
        "accessionNumber": ["0000000001-25-000002", "0000000001-25-000001", "0000000001-24-000009"],
        "form": ["8-K", "10-K/A", "10-K"], "filingDate": ["2025-05-01", "2025-04-01", "2025-02-15"],
        "reportDate": ["2025-04-30", "2024-12-31", "2024-12-31"],
        "primaryDocument": ["ev.htm", "", "k.htm"], "primaryDocDescription": ["8-K", "10-K/A", "10-K"],
        "items": ["5.02", "", ""]}, "files": []}}
    client.session = FakeSession({config.URL_SUBMISSIONS.format(cik=1): FakeResp(sub)})
    df = filings.list_filings(1, client=client)
    assert list(df["form"]) == ["8-K", "10-K/A", "10-K"]       # newest first
    annual = filings.list_filings(1, forms=["10-K"], client=client)
    assert list(annual["form"]) == ["10-K/A", "10-K"] and list(annual["is_amendment"]) == [True, False]
    assert annual.iloc[1]["doc_url"] == "https://www.sec.gov/Archives/edgar/data/1/000000000124000009/k.htm"
    assert annual.iloc[0]["doc_url"].endswith("0000000001-25-000001-index.htm")
    assert filings.company_profile(1, client)["fiscal_year_end"] == "1231"


def test_watchlist(tmp_path):
    from src.cache import watchlist
    s = Store(tmp_path / "c.db")
    assert watchlist.load(s) == []
    assert watchlist.add(" aapl ", s) == ["AAPL"] and watchlist.add("UL", s) == ["AAPL", "UL"]
    assert watchlist.add("AAPL", s) == ["AAPL", "UL"] and watchlist.add("  ", s) == ["AAPL", "UL"]
    assert watchlist.remove("aapl", s) == ["UL"] and watchlist.load(s) == ["UL"]
    s.put("watchlist", "not json")
    assert watchlist.load(s) == []
