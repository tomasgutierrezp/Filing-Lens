import pandas as pd
import pytest

from src.explain import rules
from src.metrics import verdict as V


def test_band_and_decide():
    assert V._band(70, 65, 50) == 1 and V._band(55, 65, 50) == 0 and V._band(40, 65, 50) == -1
    assert V._band(0.05, 0.10, 0.30, higher_is_better=False) == 1
    assert V._band(0.30, 0.10, 0.30, higher_is_better=False) == -1 and V._band(None, 1, 0) is None
    s = lambda score: V.Signal("k", "L", score, "d")     # noqa: E731
    assert V.decide("long", [s(1), s(1), s(0)]).label == "Favourable"
    assert V.decide("long", [s(1), s(0), s(-1)]).label == "Mixed"
    assert V.decide("long", [s(-1), s(-1), s(0), s(None)]).label == "Unfavourable"
    assert V.decide("long", [s(1), s(0), s(0)]).label == "Mixed"            # 0.33 is not past the cut
    lonely = V.decide("short", [s(1), s(None)])
    assert lonely.label == "Not enough data" and lonely.average is None


def test_long_term_signals():
    assert V.quality_signal(90, 80).score == 1 and V.quality_signal(60, 50).score == 0
    assert V.quality_signal(40, None).score == -1 and V.quality_signal(None, None).score is None
    assert V.direction_signal(45).score == -1 and "45 out of 100" in V.direction_signal(45).detail
    assert V.health_signal(1, 12).score == 1 and V.health_signal(2, 10).score == 0
    assert V.health_signal(4, 10).score == -1 and V.health_signal(0, 3).score is None
    assert V.piotroski_signal(8, 9).score == 1 and V.piotroski_signal(5, 9).score == 0
    assert V.piotroski_signal(3, 9).score == -1 and V.piotroski_signal(2, 4).score is None
    assert V.expectations_signal(0.05, 0.08).score == 1
    assert V.expectations_signal(0.10, 0.08).score == 0 and V.expectations_signal(0.25, 0.08).score == -1
    e = V.expectations_signal(0.249, 0.079)
    assert e.detail == ("The price assumes free cash flow grows 24.9% a year; over the last five "
                        "years it grew 7.9% a year.")
    assert V.expectations_signal(None, 0.08).score is None


def test_long_term_verdict():
    good = V.long_term(90, 85, 80, 0, 12, 8, 9, 0.04, 0.08)
    assert good.label == "Favourable" and good.average == 1.0 and len(good.signals) == 5
    mixed = V.long_term(98, 79, 98, 1, 10, 9, 9, 0.249, 0.079)      # strong business, demanding price
    assert [s.score for s in mixed.signals] == [1, 1, 1, 1, -1] and mixed.label == "Favourable"
    bad = V.long_term(40, 45, 30, 5, 10, 2, 9, 0.30, 0.02)
    assert bad.label == "Unfavourable"
    assert V.long_term(None, None, None, 0, 0, None, 0, None, None).label == "Not enough data"


def _prices(values):
    return pd.DataFrame({"date": [f"d{i}" for i in range(len(values))], "close": values})


def test_price_trend_and_signals():
    rising = V.price_trend(_prices([100 + i * 0.5 for i in range(260)]))
    assert rising["price"] == pytest.approx(229.5) and rising["ma50"] > rising["ma200"]
    assert rising["return_3m"] == pytest.approx(229.5 / 198.0 - 1)
    assert V.trend_signal(rising).score == 1 and V.momentum_signal(rising).score == 1
    falling = V.price_trend(_prices([300 - i * 0.5 for i in range(260)]))
    assert V.trend_signal(falling).score == -1 and V.momentum_signal(falling).score == -1
    flat = V.price_trend(_prices([100.0] * 199 + [101.0] * 61))
    assert V.momentum_signal(flat).score == 0
    short = V.price_trend(_prices([100.0] * 60))
    assert short["ma200"] is None and V.trend_signal(short).score is None
    assert short["return_3m"] is None and V.momentum_signal(short).score is None
    assert V.price_trend(None) == {"price": None, "ma50": None, "ma200": None, "return_3m": None}
    mixed = {"price": 90.0, "ma50": 105.0, "ma200": 100.0}
    assert V.trend_signal(mixed).score == 0 and "below its 200-day average" in V.trend_signal(mixed).detail


def test_latest_period_growth_and_results_signal():
    interim = pd.DataFrame({"fiscal_year": [2025, 2025, 2026, 2026], "period": ["Q2", "Q3", "Q2", "Q3"],
                            "value": [100.0, 110.0, 105.0, 121.0]})
    assert V.latest_period_growth(interim) == ("Q3 FY2026", pytest.approx(0.10))
    assert V.latest_period_growth(interim.iloc[[0, 3]]) is None           # no year-earlier Q3
    assert V.latest_period_growth(interim.assign(value=[100.0, -5.0, 105.0, 121.0])) is None
    assert V.latest_period_growth(interim.iloc[0:0]) is None
    up = V.results_signal(("Q3 FY2026", 0.10), ("Q3 FY2026", 0.20))
    assert up.score == 1 and up.detail == ("In Q3 FY2026, revenue changed 10.0% and net income 20.0% "
                                            "on a year earlier.")
    assert V.results_signal(("Q3 FY2026", -0.1), ("Q3 FY2026", -0.2)).score == -1
    assert V.results_signal(("Q3 FY2026", 0.1), ("Q3 FY2026", -0.2)).score == 0
    assert V.results_signal(None, ("Q3 FY2026", 0.1)).score is None


def test_tone_and_short_term_verdict():
    assert V.tone_signal(80).score == 1 and V.tone_signal(50).score == 0 and V.tone_signal(20).score == -1
    assert V.tone_signal(None).score is None and "Climate tab" in V.tone_signal(None).detail
    rising = V.price_trend(_prices([100 + i * 0.5 for i in range(260)]))
    v = V.short_term(rising, ("Q3 FY2026", 0.1), ("Q3 FY2026", 0.2), None)
    assert v.label == "Favourable" and [s.score for s in v.signals] == [1, 1, 1, None]
    falling = V.price_trend(_prices([300 - i * 0.5 for i in range(260)]))
    assert V.short_term(falling, ("Q3", -0.1), ("Q3", -0.2), 20).label == "Unfavourable"
    assert V.short_term(V.price_trend(None), None, None, None).label == "Not enough data"


def test_verdict_sentence():
    v = V.long_term(98, 79, 98, 1, 10, 9, 9, 0.249, 0.079)
    assert rules.verdict_sentence(v) == "4 of 5 signals are favourable, none neutral and 1 unfavourable."
    one = V.short_term(V.price_trend(_prices([100.0] * 260)), ("Q3", 0.1), ("Q3", 0.2), None)
    # a perfectly flat price is not above its averages (unfavourable trend) and has no momentum (neutral)
    assert rules.verdict_sentence(one) == ("1 of 3 signals is favourable, 1 neutral and 1 unfavourable. "
                                           "1 signal has no data.")
    assert rules.verdict_sentence(V.short_term(V.price_trend(None), None, None, None)) == \
        "Fewer than 2 signals have data, so there is no verdict."
