import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import kpis


def S(*vals, start=2020):
    return pd.Series(vals, index=range(start, start + len(vals)), dtype=float)


def test_to_wide_fills_year_gaps_and_missing_concepts():
    long = pd.DataFrame({"fiscal_year": [2020, 2022, 2020], "concept": ["revenue", "revenue", "cfo"],
                         "value": [100, 120, 30]})
    w = kpis.to_wide(long)
    assert list(w.index) == [2020, 2021, 2022]
    assert math.isnan(w.loc[2021, "revenue"]) and w.loc[2022, "revenue"] == 120
    assert "total_debt" in w.columns and w["total_debt"].isna().all()


def test_safe_div_guards_bad_denominators():
    r = kpis.safe_div(S(10, 10, 10, 10), S(5, 0, -2, np.nan))
    assert r.iloc[0] == 2 and r.iloc[1:].isna().all()


def test_average_with_prior():
    a = kpis.average_with_prior(S(10, 20, 40))
    assert math.isnan(a[2020]) and a[2021] == 15 and a[2022] == 30


def test_margin_and_fcf():
    assert kpis.margin(S(30), S(120))[2020] == 0.25
    assert kpis.free_cash_flow(S(50), S(20))[2020] == 30


def test_yoy_change():
    g = kpis.yoy_change(S(100, 110, np.nan, 120))
    assert math.isnan(g[2020]) and g[2021] == pytest.approx(0.10)
    assert math.isnan(g[2022]) and math.isnan(g[2023])      # no growth across a missing year


def test_cagr():
    c = kpis.cagr(S(100, 0, 0, 133.1), 3)
    assert c[2023] == pytest.approx(0.10) and c.loc[:2022].isna().all()
    assert kpis.cagr(S(-5, 10), 1).isna().all()             # undefined from a negative base


def test_roe_uses_average_equity():
    r = kpis.roe(S(0, 30), S(100, 200))
    assert math.isnan(r[2020]) and r[2021] == pytest.approx(0.20)
    assert kpis.roe(S(0, 30), S(-100, -200)).isna().all()   # negative equity: not meaningful


def test_effective_tax_rate_is_clipped():
    t = kpis.effective_tax_rate(S(20, -5, 300, 10), S(100, 100, 100, -50))
    assert list(t.loc[:2022]) == [0.20, 0.0, 1.0] and math.isnan(t[2023])


def test_roic():
    r = kpis.roic(S(0, 100), S(0.25, 0.25), equity=S(300, 500), debt=S(200, 200), cash=S(100, 100))
    assert r[2021] == pytest.approx(75 / 500)


def test_leverage_and_coverage():
    assert kpis.net_debt(S(80), S(30))[2020] == 50
    assert kpis.ebitda(S(40), S(10))[2020] == 50
    assert kpis.net_debt_to_ebitda(S(80), S(30), S(40), S(10))[2020] == 1.0
    assert kpis.net_debt_to_ebitda(S(10), S(30), S(40), S(10))[2020] == -0.4   # net cash
    assert kpis.interest_coverage(S(40), S(8))[2020] == 5
    assert kpis.interest_coverage(S(40), S(0)).isna().all()


def test_compute_end_to_end():
    w = pd.DataFrame({"revenue": [100.0, 120.0], "gross_profit": [40.0, 54.0],
                      "operating_income": [20.0, 30.0], "pretax_income": [20.0, 30.0],
                      "income_tax": [5.0, 6.0], "net_income": [15.0, 24.0], "cfo": [25.0, 35.0],
                      "capex": [5.0, 10.0], "d_and_a": [5.0, 6.0], "interest_expense": [2.0, 3.0],
                      "cash": [10.0, 20.0], "total_debt": [50.0, 40.0], "total_equity": [60.0, 100.0],
                      "shares_diluted": [10.0, 9.0]}, index=[2023, 2024])
    k = kpis.compute(w).loc[2024]
    assert k["gross_margin"] == pytest.approx(0.45) and k["operating_margin"] == 0.25
    assert k["net_margin"] == 0.2 and k["fcf"] == 25 and k["fcf_margin"] == pytest.approx(25 / 120)
    assert k["revenue_growth"] == pytest.approx(0.2) and math.isnan(k["revenue_cagr_3y"])
    assert k["roe"] == pytest.approx(24 / 80)
    assert k["roic"] == pytest.approx(30 * 0.8 / (80 + 45 - 15))
    assert k["net_debt"] == 20 and k["net_debt_to_ebitda"] == pytest.approx(20 / 36)
    assert k["interest_coverage"] == 10 and k["share_change"] == pytest.approx(-0.1)


def test_compute_tolerates_missing_inputs():
    k = kpis.compute(pd.DataFrame({"revenue": [100.0, 110.0]}, index=[2023, 2024]))
    assert k.loc[2024, "revenue_growth"] == pytest.approx(0.1)
    assert math.isnan(k.loc[2024, "roic"]) and math.isnan(k.loc[2024, "fcf"])


def test_every_kpi_is_computed_and_cards_exist():
    cols = kpis.compute(pd.DataFrame({"revenue": [1.0]}, index=[2024])).columns
    assert set(kpis.KPIS) <= set(cols)
    assert set(kpis.DASHBOARD_CARDS) <= set(kpis.KPIS)
