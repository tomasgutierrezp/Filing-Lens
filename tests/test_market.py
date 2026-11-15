import json

import pandas as pd
import pytest
import requests

from src.cache.store import Store
from src.edgar.facts import facts_to_frame, shares_outstanding
from src.explain import rules
from src.market import prices
from src.metrics import market
from src.normalize import currency

NAN = float("nan")


# ---- prices ---------------------------------------------------------------------------
def _frame(close=10.0):
    return pd.DataFrame({"date": ["2026-11-12", "2026-11-13"], "close": [close - 1, close],
                         "volume": [100.0, 200.0]})


def good(ticker):
    return _frame(), "USD", "Primary"


def backup(ticker):
    return _frame(20.0), "USD", "Backup"


def broken(ticker):
    raise RuntimeError("blocked")


def test_primary_source_and_cache(tmp_path):
    store = Store(tmp_path / "c.db")
    h = prices.get_history(" aapl ", store=store, fetchers=[good, backup])
    assert (h.ticker, h.source, h.currency, h.stale) == ("AAPL", "Primary", "USD", False)
    assert h.latest == ("2026-11-13", 10.0)
    again = prices.get_history("AAPL", store=store, fetchers=[broken])   # served from cache
    assert again.latest == ("2026-11-13", 10.0) and not again.stale


def test_fallback_then_stale_then_error(tmp_path):
    store = Store(tmp_path / "c.db")
    assert prices.get_history("AAPL", store=store, fetchers=[broken, backup]).source == "Backup"
    stale = prices.get_history("AAPL", store=store, ttl=-1, fetchers=[broken])
    assert stale.stale and stale.latest[1] == 20.0
    with pytest.raises(prices.PriceError, match="unavailable right now"):
        prices.get_history("MSFT", store=store, fetchers=[broken, broken])
    with pytest.raises(prices.PriceError):
        prices.get_history("  ", store=store, fetchers=[good])


def test_parse_stooq_csv():
    text = "Date,Open,High,Low,Close,Volume\n2026-11-13,1,2,0.5,1.5,900\n2026-11-12,1,2,0.5,1.4,800\n"
    f = prices.parse_stooq_csv(text)
    assert list(f["date"]) == ["2026-11-12", "2026-11-13"] and list(f["close"]) == [1.4, 1.5]
    for bad in ("<!DOCTYPE html><html>verify</html>", "No data", "Date,Open,High,Low,Close,Volume\n"):
        with pytest.raises(ValueError):
            prices.parse_stooq_csv(bad)


# ---- FX -------------------------------------------------------------------------------
class FakeSession:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def get(self, url, timeout=None, headers=None):
        self.calls += 1
        if self.error:
            raise self.error
        resp = requests.Response()
        resp.status_code, resp._content = 200, json.dumps(self.payload).encode()
        return resp


def test_fx_rate_cache_and_fallback(tmp_path):
    store = Store(tmp_path / "c.db")
    session = FakeSession({"base": "USD", "date": "2026-11-13", "rates": {"EUR": 0.9}})
    r = currency.get_rate("usd", "eur", store=store, session=session)
    assert (r.base, r.quote, r.rate, r.date) == ("USD", "EUR", 0.9, "2026-11-13")
    assert currency.convert(100, r) == 90
    currency.get_rate("USD", "EUR", store=store, session=session)
    assert session.calls == 1                                          # cached
    down = FakeSession(error=requests.ConnectionError("down"))
    assert currency.get_rate("USD", "EUR", store=store, session=down, ttl=-1).rate == 0.9   # stale
    with pytest.raises(currency.FxError):
        currency.get_rate("USD", "GBP", store=store, session=down)
    assert currency.get_rate("USD", "USD", store=store, session=down).rate == 1.0           # no call needed


# ---- shares outstanding ------------------------------------------------------------------
def test_shares_outstanding_picks_latest_cover_value():
    facts = facts_to_frame({"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2024-10-18", "val": 100, "accn": "A", "form": "10-K", "filed": "2024-11-01"},
        {"end": "2025-10-17", "val": 95, "accn": "B", "form": "10-K", "filed": "2025-10-31"}]}}}}})
    s = shares_outstanding(facts)
    assert (s["value"], s["end"], s["form"], s["accn"]) == (95, "2025-10-17", "10-K", "B")
    assert shares_outstanding(facts_to_frame({"facts": {}})) is None


# ---- multiples ---------------------------------------------------------------------------
def test_price_in_reporting_currency():
    assert market.price_in_reporting_currency(60.0) == 60.0
    assert market.price_in_reporting_currency(60.0, fx_rate=0.9) == pytest.approx(54.0)
    assert market.price_in_reporting_currency(60.0, fx_rate=0.9, ordinary_per_adr=2) == pytest.approx(27.0)
    with pytest.raises(ValueError):
        market.price_in_reporting_currency(60.0, ordinary_per_adr=0)


def test_market_multiples():
    m = market.market_multiples(50.0, 10.0, net_income=25.0, net_debt=100.0, ebitda=60.0, fcf=20.0)
    assert m == {"market_cap": 500.0, "pe": 20.0, "enterprise_value": 600.0, "ev_ebitda": 10.0,
                 "fcf_yield": 0.04}
    cash_rich = market.market_multiples(50.0, 10.0, net_debt=-100.0, ebitda=40.0)
    assert cash_rich["enterprise_value"] == 400.0 and cash_rich["ev_ebitda"] == 10.0


def test_market_multiples_not_meaningful_or_missing():
    m = market.market_multiples(50.0, 10.0, net_income=-5.0, net_debt=NAN, ebitda=-1.0, fcf=0.0)
    assert m["market_cap"] == 500.0
    assert m["pe"] is None and m["ev_ebitda"] is None and m["fcf_yield"] is None
    assert m["enterprise_value"] is None
    assert set(market.market_multiples(None, 10.0).values()) == {None}
    assert set(market.market_multiples(50.0, NAN, net_income=5.0).values()) == {None}


def test_market_sentences():
    m = market.market_multiples(50.0, 10e9, net_income=25e9, net_debt=100e9, ebitda=60e9, fcf=20e9)
    args = dict(year=2025, currency="USD", net_income=25e9, ebitda=60e9, fcf=20e9)
    assert rules.market_sentence("market_cap", m, **args) == \
        "At today's price the market values the company's shares at USD 500.0bn."
    assert rules.market_sentence("pe", m, **args) == \
        "The market value is 20.0 times FY2025 net income (USD 25.0bn)."
    assert rules.market_sentence("ev_ebitda", m, **args) == (
        "Enterprise value (market value plus net debt, USD 600.0bn) is 10.0 times FY2025 EBITDA (USD 60.0bn).")
    assert rules.market_sentence("fcf_yield", m, **args) == \
        "FY2025 free cash flow (USD 20.0bn) equals 4.0% of the market value."
    assert set(market.MARKET_KPIS) == {"market_cap", "pe", "ev_ebitda", "fcf_yield"}


def test_market_sentences_when_not_meaningful():
    m = market.market_multiples(50.0, 10e9, net_income=-5e9, net_debt=1e9, ebitda=-1e9, fcf=-2e9)
    args = dict(year=2025, currency="USD", net_income=-5e9, ebitda=-1e9, fcf=-2e9)
    assert "P/E ratio is not meaningful" in rules.market_sentence("pe", m, **args)
    assert "EV / EBITDA is not meaningful" in rules.market_sentence("ev_ebitda", m, **args)
    assert "FCF yield is not meaningful" in rules.market_sentence("fcf_yield", m, **args)
    assert rules.market_sentence("pe", m, 2025, "USD") is None                 # inputs not given
    assert rules.market_sentence("market_cap", market.market_multiples(None, None), 2025, "USD") is None
