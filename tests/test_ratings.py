import numpy as np
import pandas as pd
import pytest

from src.metrics import kpis, models, ratings

NAN = float("nan")


def strong_company(years=7):
    """Steadily growing, high-margin, net-cash business."""
    idx = list(range(2026 - years + 1, 2027))
    rev = np.array([100.0 * 1.10 ** i for i in range(years)])
    w = pd.DataFrame({
        "revenue": rev, "cost_of_revenue": rev * 0.5, "gross_profit": rev * 0.5, "rnd": rev * 0.08,
        "sga": rev * 0.12, "operating_income": rev * 0.30, "pretax_income": rev * 0.30,
        "income_tax": rev * 0.06, "net_income": rev * 0.24, "cfo": rev * 0.32, "capex": rev * 0.05,
        "d_and_a": rev * 0.04, "interest_expense": rev * 0.005, "cash": rev * 0.5,
        "total_debt": rev * 0.2, "total_equity": rev * 1.0, "total_assets": rev * 2.0,
        "shares_diluted": [100.0 * 0.98 ** i for i in range(years)], "dividends_paid": rev * 0.05,
        "buybacks": rev * 0.10, "acquisitions": 0.0, "debt_repaid": 0.0, "debt_issued": 0.0}, index=idx)
    return w.reindex(columns=list(models.MODEL_CONCEPTS)), kpis.compute(w)


def weak_company(years=7):
    """Shrinking, thin-margin, indebted business that pays out more than it earns."""
    idx = list(range(2026 - years + 1, 2027))
    rev = np.array([100.0 * 0.97 ** i for i in range(years)])
    margin = np.linspace(0.10, 0.02, years)
    w = pd.DataFrame({
        "revenue": rev, "cost_of_revenue": rev * 0.8, "gross_profit": rev * 0.2, "rnd": NAN,
        "sga": rev * 0.1, "operating_income": rev * margin, "pretax_income": rev * margin * 0.6,
        "income_tax": rev * margin * 0.15, "net_income": rev * margin * 0.45, "cfo": rev * 0.03,
        "capex": rev * 0.025, "d_and_a": rev * 0.03, "interest_expense": rev * 0.03, "cash": rev * 0.02,
        "total_debt": rev * 0.6, "total_equity": rev * 0.1, "total_assets": rev * 1.0,
        "shares_diluted": [100.0 * 1.05 ** i for i in range(years)], "dividends_paid": rev * 0.03,
        "buybacks": 0.0, "acquisitions": 0.0, "debt_repaid": 0.0, "debt_issued": rev * 0.02}, index=idx)
    return w.reindex(columns=list(models.MODEL_CONCEPTS)), kpis.compute(w)


def test_scale_and_grades():
    assert ratings.scale(0.15, 0.05, 0.25) == pytest.approx(50)
    assert ratings.scale(0.40, 0.05, 0.25) == 100 and ratings.scale(-1, 0.05, 0.25) == 0
    assert ratings.scale(2.25, 4.0, 0.5) == pytest.approx(50)          # lower is better
    assert ratings.scale(0.2, 4.0, 0.5) == 100 and ratings.scale(9, 4.0, 0.5) == 0
    assert ratings.scale(NAN, 0, 1) is None and ratings.scale(None, 0, 1) is None
    assert ratings.grade_for(80) == "A" and ratings.grade_for(79.9) == "B" and ratings.grade_for(65) == "B"
    assert ratings.grade_for(64.9) == "C" and ratings.grade_for(50) == "C" and ratings.grade_for(49.9) == "D"
    assert ratings.grade_for(35) == "D" and ratings.grade_for(34.9) == "E" and ratings.grade_for(None) == "Not rated"


def test_every_metric_has_thresholds_and_combine_weights():
    a = ratings.metric("roe", "ROE", 0.25, "pct")                       # 100
    b = ratings.metric("roic", "ROIC", 0.05, "pct", weight=3)            # 0
    r = ratings.combine("x", "X", [a, b])
    assert r.score == 25.0 and r.grade == "E" and (a.poor, a.strong) == ratings.THRESHOLDS["roe"]
    missing = ratings.metric("roic", "ROIC", NAN, "pct", weight=3)
    assert ratings.combine("x", "X", [a, missing]).grade == "Not rated"     # only 25% of weight has data
    assert ratings.combine("x", "X", [a, ratings.metric("roic", "ROIC", None, "pct")]).score == 100.0
    assert ratings.combine("x", "X", []).grade == "Not rated"


def test_model_ratings_strong_vs_weak():
    ws, ks = strong_company()
    ww, kw = weak_company()
    strong = ratings.model_ratings(ks, ws, share_price=1.0)
    weak = ratings.model_ratings(kw, ww, share_price=1e9)
    assert set(strong) == {"common_size", "bridge", "dupont", "capital_allocation", "shares", "dcf"}
    for key in strong:
        assert strong[key].score is not None and weak[key].score is not None, key
        assert strong[key].score > weak[key].score, key
    assert strong["common_size"].grade == "A" and weak["common_size"].grade == "E"
    assert strong["dcf"].score == 100 and weak["dcf"].score == 0
    assert strong["shares"].metrics[0].value == pytest.approx(-0.02)
    assert weak["shares"].metrics[0].value == pytest.approx(0.05)


def test_individual_model_inputs():
    w, k = strong_company()
    p = {m.key: m for m in ratings.rate_profitability(k).metrics}
    assert p["operating_margin"].value == pytest.approx(0.30) and p["margin_change_3y"].value == pytest.approx(0.0)
    c = {m.key: m for m in ratings.rate_cash_conversion(k).metrics}
    assert c["fcf_to_net_income"].value == pytest.approx(0.27 / 0.24)
    a = {m.key: m for m in ratings.rate_capital_allocation(w).metrics}
    assert a["payouts_to_fcf_5y"].value == pytest.approx(0.15 / 0.27)
    assert a["uses_to_cfo_5y"].value == pytest.approx(0.20 / 0.32)
    d = {m.key: m for m in ratings.rate_returns(k, models.dupont(w)).metrics}
    assert d["equity_multiplier"].value == pytest.approx(2.0)


def test_dcf_rating_needs_price_and_positive_fcf():
    w, k = strong_company()
    assert ratings.rate_dcf(k, None).grade == "Not rated"
    assert ratings.rate_dcf(k.assign(fcf=-1.0), 10.0).grade == "Not rated"
    mid = ratings.rate_dcf(k, 1.0).metrics[0]
    assert mid.value == 1.0 and mid.kind == "share"


def test_overall_ratings_strong_vs_weak():
    ws, ks = strong_company()
    ww, kw = weak_company()
    strong, weak = ratings.overall_ratings(ks, ws), ratings.overall_ratings(kw, ww)
    assert set(strong) == {"current", "historical", "future"}
    for key in strong:
        assert strong[key].score > weak[key].score, key
        assert strong[key].grade in ("A", "B"), (key, strong[key])
    # The weak company still had positive free cash flow every year, so its history is a C.
    assert [weak[k].grade for k in ("current", "historical", "future")] == ["E", "C", "E"]
    hist = {m.key: m for m in strong["historical"].metrics}
    assert hist["fcf_positive_years"].value == 1.0 and hist["growth_years"].value == 1.0
    assert hist["margin_volatility"].value == pytest.approx(0.0)
    fut = {m.key: m for m in strong["future"].metrics}
    assert fut["reinvestment_rate"].value == pytest.approx(0.13)
    assert fut["dividends_to_fcf"].value == pytest.approx(0.05 / 0.27)
    assert fut["revenue_cagr_3y"].value == pytest.approx(0.10)


def test_ratings_with_little_or_no_data():
    k = kpis.compute(pd.DataFrame({"revenue": [100.0, 110.0]}, index=[2025, 2026]))
    w = pd.DataFrame({"revenue": [100.0, 110.0]}, index=[2025, 2026])
    for r in list(ratings.model_ratings(k, w).values()) + list(ratings.overall_ratings(k, w).values()):
        assert r.grade == "Not rated" and r.score is None, r.key
    empty = kpis.compute(pd.DataFrame({"revenue": []}))
    assert ratings.rate_current(empty).grade == "Not rated"


def test_climate_rating():
    days = pd.date_range("2026-08-18", "2026-11-15").strftime("%Y-%m-%d")
    tone = [-3.0] * 60 + [1.0] * 30                                  # improved in the last 30 days
    r = ratings.rate_climate(pd.DataFrame({"date": days, "tone": tone}), "2026-11-15")
    m = {x.key: x for x in r.metrics}
    assert m["tone_30d"].value == pytest.approx(1.0) and m["tone_trend"].value == pytest.approx(4.0)
    assert m["tone_trend"].score == 100 and r.grade == "A"
    worse = ratings.rate_climate(pd.DataFrame({"date": days, "tone": [1.0] * 60 + [-5.0] * 30}), "2026-11-15")
    assert worse.grade == "E"
    week = ratings.rate_climate(pd.DataFrame({"date": days[-8:], "tone": [0.0] * 8}), "2026-11-15")
    assert week.score == pytest.approx(ratings.scale(0.0, -4.0, 2.0), abs=0.1)      # level only, no trend yet
    assert week.metrics[1].value is None
    assert ratings.rate_climate(pd.DataFrame({"date": days[-3:], "tone": [0.0] * 3}), "2026-11-15").grade == "Not rated"
    assert ratings.rate_climate(pd.DataFrame(columns=["date", "volume", "tone"]), "2026-11-15").grade == "Not rated"


def test_first_restated_year():
    long = pd.DataFrame({"concept": ["revenue"] * 3 + ["cfo"], "fiscal_year": [2022, 2023, 2024, 2021],
                         "restated": [False, True, True, True]})
    assert ratings.first_restated_year(long) == 2023
    assert ratings.first_restated_year(long, "cfo") == 2021
    assert ratings.first_restated_year(long[long.concept == "nope"]) is None
    blocks = pd.DataFrame({"concept": "revenue", "fiscal_year": [2010, 2011, 2019, 2023, 2024],
                           "restated": True})
    assert ratings.first_restated_year(blocks) == 2023          # only the latest run matters


def test_growth_across_a_change_of_basis_is_left_out():
    w, k = strong_company()                      # years 2020-2026
    plain = ratings.overall_ratings(k, w)
    assert plain["future"].note == "" and plain["historical"].note == ""
    cut = ratings.overall_ratings(k, w, restated_from=2024)
    fut = {m.key: m for m in cut["future"].metrics}
    hist = {m.key: m for m in cut["historical"].metrics}
    assert fut["revenue_cagr_3y"].value is None and hist["revenue_cagr_5y"].value is None
    assert "FY2024" in cut["future"].note and "FY2024" in cut["historical"].note
    assert cut["future"].score is not None and fut["margin_change_3y"].value is not None
    old = ratings.overall_ratings(k, w, restated_from=2021)     # long ago: recent spans are clean
    assert {m.key: m for m in old["future"].metrics}["revenue_cagr_3y"].value == pytest.approx(0.10)
    assert old["future"].note == ""


def test_grade_history_uses_only_years_up_to_each_point():
    w, k = strong_company()                       # 2020-2026
    h = ratings.grade_history(k, w)
    assert list(h.columns) == ["current", "historical", "future"] and list(h.index) == list(range(2020, 2027))
    assert h.loc[2026, "current"] == ratings.rate_current(k).score
    assert h.loc[2023, "current"] == ratings.rate_current(k.loc[:2023]).score
    assert pd.isna(h.loc[2020, "historical"])                    # one year is not a track record
    assert ratings.grade_history(k, w, years=3).index.tolist() == [2024, 2025, 2026]
    assert ratings.grade_history(kpis.compute(pd.DataFrame({"revenue": []})), w).empty


def test_comparison_table():
    ws, ks = strong_company()
    ww, kw = weak_company()
    c = ratings.comparison({"Strong": ks, "Weak": kw, "Empty": kpis.compute(pd.DataFrame({"revenue": []}))})
    assert list(c.columns) == ["Strong", "Weak", "Empty"]
    assert c.loc["fiscal_year", "Strong"] == 2026 and c.loc["operating_margin", "Strong"] == pytest.approx(0.30)
    assert c.loc["operating_margin", "Weak"] == pytest.approx(0.02) and c["Empty"].isna().all()
    assert list(c.index) == ["fiscal_year"] + [k_ for k_, _ in ratings.COMPARE_ROWS]
