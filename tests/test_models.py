import math

import pandas as pd
import pytest

from src.metrics import kpis, models


def W():
    return pd.DataFrame({
        "revenue": [100.0, 120.0], "cost_of_revenue": [60.0, 66.0], "gross_profit": [40.0, 54.0],
        "rnd": [5.0, 6.0], "sga": [10.0, 12.0], "operating_income": [20.0, 30.0],
        "pretax_income": [19.0, 29.0], "income_tax": [4.0, 6.0], "net_income": [15.0, 24.0],
        "cfo": [25.0, 35.0], "capex": [5.0, 10.0], "d_and_a": [5.0, 6.0],
        "interest_expense": [2.0, 3.0], "cash": [10.0, 20.0], "total_debt": [50.0, 40.0],
        "total_equity": [60.0, 100.0], "total_assets": [200.0, 240.0], "shares_diluted": [10.0, 9.0],
        "dividends_paid": [3.0, 4.0], "buybacks": [6.0, 14.0], "acquisitions": [0.0, 2.0],
        "debt_repaid": [8.0, 4.0], "debt_issued": [2.0, 9.0]}, index=[2023, 2024])


def test_common_size_adds_up_to_revenue():
    cs = models.common_size(W())
    assert list(cs.columns) == list(models.COMMON_SIZE_LABELS)
    assert cs.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    r = cs.loc[2024]
    assert r["cost_of_revenue"] == pytest.approx(0.55) and r["net_income"] == pytest.approx(0.2)
    assert r["other_operating"] == pytest.approx(6 / 120)      # 120 - 30 - 66 - 6 - 12
    assert r["other_items"] == pytest.approx(0.0)               # 30 - 6 - 24


def test_common_size_folds_unreported_lines_into_other():
    w = W().drop(columns=["cost_of_revenue", "gross_profit", "rnd", "income_tax"])
    cs = models.common_size(w).loc[2024]
    assert math.isnan(cs["cost_of_revenue"]) and math.isnan(cs["rnd"]) and math.isnan(cs["income_tax"])
    assert cs["other_operating"] == pytest.approx((120 - 30 - 12) / 120)
    assert cs["other_items"] == pytest.approx(6 / 120)
    assert cs.sum() == pytest.approx(1.0)
    assert models.common_size(W().drop(columns=["net_income"])).empty


def test_fcf_bridge_reconciles():
    steps = models.fcf_bridge(W(), 2024)
    assert [s["label"] for s in steps] == [
        "Revenue", "Cost of revenue", "Operating expenses", "Operating income", "Income tax",
        "Interest & other", "Net income", "Depreciation & amortization",
        "Working capital & other", "Cash from operations", "Capital expenditure", "Free cash flow"]
    running = 0.0
    for s in steps:                       # every total equals the running sum of what came before
        if s["kind"] == "total":
            assert s["value"] == pytest.approx(running) or s["label"] == "Revenue"
            running = s["value"]
        else:
            running += s["value"]
    assert steps[-1]["value"] == 25


def test_fcf_bridge_with_missing_detail_and_missing_essentials():
    w = W().drop(columns=["cost_of_revenue", "gross_profit", "income_tax", "d_and_a"])
    labels = [s["label"] for s in models.fcf_bridge(w, 2024)]
    assert labels == ["Revenue", "Operating costs", "Operating income", "Interest, tax & other",
                      "Net income", "Non-cash items & working capital", "Cash from operations",
                      "Capital expenditure", "Free cash flow"]
    assert models.fcf_bridge(W().drop(columns=["capex"]), 2024) is None
    assert models.fcf_bridge(W(), 1999) is None


def test_dupont_components_multiply_to_roe():
    d = models.dupont(W()).loc[2024]
    assert d["net_margin"] == pytest.approx(0.2)
    assert d["asset_turnover"] == pytest.approx(120 / 220)
    assert d["equity_multiplier"] == pytest.approx(220 / 80)
    assert d["roe"] == pytest.approx(24 / 80)
    assert d["roe"] == pytest.approx(kpis.compute(W()).loc[2024, "roe"])


def test_dupont_driver():
    a = pd.Series({"net_margin": 0.10, "asset_turnover": 1.0, "equity_multiplier": 2.0})
    b = pd.Series({"net_margin": 0.11, "asset_turnover": 1.0, "equity_multiplier": 3.0})
    assert models.dupont_driver(a, b) == "equity_multiplier"
    assert models.dupont_driver(a, b.replace(0.11, 0.2)) == "net_margin"
    assert models.dupont_driver(a, b.replace(0.11, -0.1)) is None


def test_capital_allocation():
    uses = models.capital_uses(W())
    assert uses.loc[2023, "debt_paydown"] == 6 and uses.loc[2024, "debt_paydown"] == 0   # net borrower
    ca = models.capital_allocation(W()).loc[2024]
    assert ca["buybacks"] == pytest.approx(14 / 35) and ca["capex"] == pytest.approx(10 / 35)
    assert list(ca.index) == list(models.ALLOCATION_LABELS)


def test_allocation_summary():
    s = models.allocation_summary(W())
    assert s["years"] == 2 and s["cfo"] == 60
    assert s["top_use"] == "buybacks" and s["top_share"] == pytest.approx(20 / 60)
    assert s["total_share"] == pytest.approx((7 + 20 + 15 + 2 + 6) / 60)
    assert models.allocation_summary(W().assign(cfo=[-1.0, -2.0])) is None
    assert models.allocation_summary(W()[["cfo"]]) is None


def test_dcf_equity_value():
    # One year at 0% growth, 10% discount, 0% terminal growth: 100/1.1 + (100/0.1)/1.1 = 1000
    assert models.dcf_equity_value(100, 0.0, 0.10, 0.0, years=1) == pytest.approx(1000)
    assert models.dcf_equity_value(100, 0.0, 0.10, 0.0, years=5) == pytest.approx(1000)
    assert models.dcf_equity_value(100, 0.0, 0.10, 0.0, years=5, net_debt=200) == pytest.approx(800)
    assert models.dcf_equity_value(100, 0.05, 0.10, 0.02) > models.dcf_equity_value(100, 0.0, 0.10, 0.02)
    assert math.isnan(models.dcf_equity_value(100, 0.05, 0.02, 0.03))   # discount <= terminal growth


def test_dcf_grid():
    g = models.dcf_grid(100, [0.0, 0.05], [0.08, 0.10], 0.0, shares=10)
    assert g.shape == (2, 2) and g.loc[0.10, 0.0] == pytest.approx(100)
    assert g.loc[0.08, 0.05] == g.values.max() and g.loc[0.10, 0.0] == g.values.min()


def test_dcf_defaults():
    idx = range(2019, 2025)
    k = pd.DataFrame({"fcf": [100.0, 0, 0, 0, 0, 161.051], "shares_diluted": 10.0,
                      "net_debt": 30.0}, index=idx)
    d = models.dcf_defaults(k)
    assert d["year"] == 2024 and d["base_fcf"] == pytest.approx(161.051)
    assert d["growth"] == pytest.approx(0.10) and d["net_debt"] == 30 and d["shares"] == 10
    high = models.dcf_defaults(k.assign(fcf=[10.0, 0, 0, 0, 0, 500.0]))
    assert high["growth"] == 0.15                                   # capped
    none = models.dcf_defaults(k.assign(fcf=[-5.0, 0, 0, -1, 0, 50.0]))
    assert none["growth"] == 0.03 and "placeholder" in none["growth_basis"]
    assert models.dcf_defaults(pd.DataFrame({"fcf": [float("nan")]}, index=[2024])) is None


def test_implied_growth_inverts_the_dcf():
    value = models.dcf_equity_value(100.0, 0.07, 0.09, 0.025, net_debt=50.0) / 10.0
    g = models.implied_growth(value, 100.0, 0.09, 0.025, shares=10.0, net_debt=50.0)
    assert g == pytest.approx(0.07, abs=1e-6)
    assert models.implied_growth(value * 0.5, 100.0, 0.09, 0.025, 10.0, 50.0) < g
    assert models.implied_growth(1e9, 100.0, 0.09, 0.025, 10.0) is None       # no rate up to 100% reaches it
    assert models.implied_growth(1e-6, 100.0, 0.09, 0.025, 10.0, net_debt=1e6) is None
    assert models.implied_growth(10.0, -5.0, 0.09, 0.025, 10.0) is None        # negative FCF
    assert models.implied_growth(None, 100.0, 0.09, 0.025, 10.0) is None
