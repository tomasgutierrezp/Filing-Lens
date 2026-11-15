import json

import pandas as pd
import pytest
import requests

from src import config
from src.cache.store import Store
from src.climate import events, gdelt, headlines


class Resp:
    def __init__(self, payload=None, status=200, text=None):
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def timeline(points):
    return {"timeline": [{"series": "s", "data": [{"date": d, "value": v} for d, v in points]}]}


# ---- GDELT ------------------------------------------------------------------------------
def test_default_query():
    assert gdelt.default_query("UNILEVER PLC") == '"Unilever"'
    assert gdelt.default_query("Apple Inc.") == '"Apple"'
    assert gdelt.default_query("BERKSHIRE HATHAWAY INC") == '"Berkshire Hathaway"'
    assert gdelt.default_query("Alphabet Inc. /DE/") == '"Alphabet"'
    assert gdelt.default_query("Coca-Cola Co") == '"Coca-Cola"'
    assert gdelt.default_query("  ") == ""


def test_parse_timeline_daily_and_averaged():
    p = timeline([("20261101T000000Z", 1.0), ("20261102T000000Z", 2.0), ("20261102T120000Z", 4.0),
                  ("bad", 1.0), ("20261103T000000Z", "x")])
    assert gdelt.parse_timeline(p) == [("2026-11-01", 1.0), ("2026-11-02", 3.0)]
    assert gdelt.parse_timeline({}) == [] and gdelt.parse_timeline({"timeline": []}) == []


def test_refresh_appends_history_and_respects_schedule(tmp_path):
    store, sleeps = Store(tmp_path / "c.db"), []
    s = Session([Resp(timeline([("20261101T000000Z", 1.0), ("20261102T000000Z", 2.0)])),
                 Resp(timeline([("20261101T000000Z", -0.5)]))])
    assert gdelt.pull_state('"Acme"', store)["status"] == "due"
    out = gdelt.refresh('"Acme"', store, s, sleep=sleeps.append)
    assert out["status"] == "updated" and sleeps == [config.GDELT_MIN_GAP_SECONDS]
    assert gdelt.pull_state('"Acme"', store)["status"] == "fresh"
    assert [c[1]["mode"] for c in s.calls] == ["timelinevol", "timelinetone"]
    h = gdelt.history('"Acme"', store)
    assert list(h["date"]) == ["2026-11-01", "2026-11-02"]
    assert list(h["volume"]) == [1.0, 2.0] and h["tone"].iloc[0] == -0.5 and pd.isna(h["tone"].iloc[1])
    assert gdelt.refresh('"Acme"', store, Session([]), sleep=sleeps.append)["status"] == "fresh"   # no call

    # A later pull adds new days and keeps the old ones (history accumulates).
    store.put('gdelt:ok:"Acme"', "0")
    with store._conn() as c:
        c.execute("UPDATE kv SET fetched_at = 0 WHERE key = ?", ('gdelt:ok:"Acme"',))
    s2 = Session([Resp(timeline([("20261102T000000Z", 2.5), ("20261103T000000Z", 3.0)]))]
                 + [Resp({})] * len(config.GDELT_TIMESPANS))
    assert gdelt.refresh('"Acme"', store, s2, sleep=sleeps.append)["status"] == "updated"
    h = gdelt.history('"Acme"', store)
    assert list(h["date"]) == ["2026-11-01", "2026-11-02", "2026-11-03"]
    assert list(h["volume"]) == [1.0, 2.5, 3.0]


def test_refresh_falls_back_to_shorter_windows_when_empty(tmp_path):
    store, sleeps = Store(tmp_path / "c.db"), []
    s = Session([Resp({}), Resp({}), Resp(timeline([("20261109T000000Z", 0.0), ("20261109T010000Z", 0.06)])),
                 Resp(timeline([("20261109T000000Z", -1.0)]))])
    assert gdelt.refresh('"Acme"', store, s, sleep=sleeps.append)["status"] == "updated"
    assert [c[1]["timespan"] for c in s.calls] == list(config.GDELT_TIMESPANS) + [config.GDELT_TIMESPANS[0]]
    assert len(sleeps) == 3                                   # a pause before every request but the first
    h = gdelt.history('"Acme"', store)
    assert list(h["date"]) == ["2026-11-09"] and h["volume"].iloc[0] == pytest.approx(0.03)


def test_refresh_failure_is_reported_and_not_retried_immediately(tmp_path):
    store = Store(tmp_path / "c.db")
    out = gdelt.refresh('"Acme"', store, Session([Resp(status=429, text="Please limit requests")]),
                        sleep=lambda s: None)
    assert out["status"] == "failed" and "limiting requests" in out["message"]
    assert gdelt.pull_state('"Acme"', store)["status"] == "waiting"
    again = gdelt.refresh('"Acme"', store, Session([]), sleep=lambda s: None)
    assert again["status"] == "waiting" and "limiting requests" in again["message"]
    assert gdelt.history('"Acme"', store).empty
    assert gdelt.refresh("  ", store)["status"] == "failed"
    down = gdelt.refresh('"Other"', store, Session([requests.ConnectionError("x")]), sleep=lambda s: None)
    assert down["status"] == "failed" and "could not be reached" in down["message"]
    text = gdelt.refresh('"Third"', store, Session([Resp(text="Your query was too short")]), sleep=lambda s: None)
    assert text["status"] == "failed" and "too short" in text["message"]


# ---- events -------------------------------------------------------------------------------
def test_classify_8k():
    assert events.classify_8k("2.02,9.01") == ("Results", "Results of operations")
    assert events.classify_8k("5.02") == ("Leadership", "Director or officer change")
    assert events.classify_8k("9.01,2.01") == ("Deals", "Acquisition or disposal completed")
    assert events.classify_8k("5.07,9.01")[0] == "Shareholder matters"
    assert events.classify_8k("8.01,9.01") == ("Other", "Other events")
    assert events.classify_8k("") == ("Other", "Current report")


def test_classify_6k():
    assert events.classify_6k("DIRECTORATE CHANGE") == ("Leadership", "Directorate change")
    assert events.classify_6k("DIRECTOR/PDMR SHAREHOLDING") == ("Routine", "Director/pdmr shareholding")
    assert events.classify_6k("TOTAL VOTING RIGHTS")[0] == "Routine"
    assert events.classify_6k("2025 FULL YEAR RESULTS")[0] == "Results"
    assert events.classify_6k("Q1 2025 TRADING STATEMENT")[0] == "Results"
    assert events.classify_6k("SHARE BUY-BACK PROGRAMME")[0] == "Shareholder matters"
    assert events.classify_6k("RESULT OF AGM")[0] == "Shareholder matters"
    assert events.classify_6k("RESPONSE TO MEDIA SPECULATION")[0] == "Disclosure"
    assert events.classify_6k("6-K") == ("Other", "Report of foreign issuer")
    assert events.classify_6k("Something unusual") == ("Other", "Something unusual")


def test_build_events():
    f = pd.DataFrame({
        "filed": ["2026-11-03", "2026-10-30", "2026-10-01", "2026-09-01"],
        "form": ["8-K", "8-K/A", "10-Q", "6-K"], "base_form": ["8-K", "8-K", "10-Q", "6-K"],
        "is_amendment": [False, True, False, False], "items": ["2.02,9.01", "5.02", "", ""],
        "description": ["8-K", "8-K/A", "10-Q", "DIRECTORATE CHANGE"], "doc_url": ["u1", "u2", "u3", "u4"]})
    e = events.build_events(f)
    assert list(e.columns) == events.EVENT_COLUMNS and len(e) == 3      # the 10-Q is not an event
    assert list(e["category"]) == ["Results", "Leadership", "Leadership"]
    assert e.iloc[1]["label"] == "Director or officer change (amendment)" and e.iloc[2]["url"] == "u4"
    assert set(events.DEFAULT_CATEGORIES) < set(events.CATEGORIES)


# ---- headlines ------------------------------------------------------------------------------
RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>x</title>
<item><title>Acme profit rises - Daily News</title><link>https://news.example/a</link>
<pubDate>Mon, 09 Nov 2026 10:00:00 GMT</pubDate><source url="https://d.example">Daily News</source></item>
<item><title>Acme names new chief</title><link>https://news.example/b</link>
<pubDate>Tue, 10 Nov 2026 08:00:00 GMT</pubDate><source url="https://w.example">Wire</source></item>
<item><title>Old story</title><link>https://news.example/c</link>
<pubDate>Sun, 01 Nov 2026 08:00:00 GMT</pubDate></item>
<item><title></title><link>https://news.example/d</link></item>
<item><title>No link</title><link>javascript:alert(1)</link></item>
</channel></rss>"""


def test_build_url_covers_inclusive_range():
    url = headlines.build_url('"Acme"', "2026-11-05", "2026-11-10")
    assert url.startswith(config.URL_GOOGLE_NEWS_RSS + "?q=%22Acme%22+after%3A2026-11-05+before%3A2026-11-11")


def test_parse_rss():
    items = headlines.parse_rss(RSS)
    assert [i["title"] for i in items] == ["Acme names new chief", "Acme profit rises", "Old story"]
    assert items[1] == {"date": "2026-11-09", "title": "Acme profit rises", "source": "Daily News",
                        "link": "https://news.example/a"}
    with pytest.raises(ValueError):
        headlines.parse_rss('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><rss/>')


def test_get_headlines_filters_range_caches_and_fails_clearly(tmp_path):
    store = Store(tmp_path / "c.db")
    s = Session([Resp(text=RSS)])
    h = headlines.get_headlines('"Acme"', "2026-11-05", "2026-11-10", store, s)
    assert list(h["title"]) == ["Acme names new chief", "Acme profit rises"]      # 1 Nov is out of range
    assert list(h.columns) == headlines.HEADLINE_COLUMNS
    headlines.get_headlines('"Acme"', "2026-11-05", "2026-11-10", store, Session([]))   # cached: no call
    with pytest.raises(headlines.HeadlinesError):
        headlines.get_headlines('"Acme"', "2026-10-01", "2026-10-02", store,
                                Session([requests.ConnectionError("down")]))


def test_series_store_is_append_only(tmp_path):
    store = Store(tmp_path / "c.db")
    store.upsert_series("k", "q", [("2026-01-02", 2.0), ("2026-01-01", 1.0)])
    store.upsert_series("k", "q", [("2026-01-02", 2.5), ("2026-01-03", 3.0)])
    assert store.read_series("k", "q") == [("2026-01-01", 1.0), ("2026-01-02", 2.5), ("2026-01-03", 3.0)]
    assert store.read_series("k", "other") == []
