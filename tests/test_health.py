import pandas as pd
import pytest

from src.metrics import health, kpis

NAN = float("nan")


def W(**over):
    base = {
        "revenue": [80.0, 90.0, 95.0, 100.0], "gross_profit": [32.0, 37.0, 40.0, 43.0],
        "operating_income": [16.0, 18.5, 20.0, 22.0], "pretax_income": [15.0, 17.5, 19.0, 21.0],
        "income_tax": [3.0, 3.5, 4.0, 4.0], "net_income": [12.0, 14.0, 15.0, 17.0],
        "cfo": [15.0, 17.0, 19.0, 21.0], "capex": [4.0, 4.0, 5.0, 5.0], "d_and_a": [3.0, 3.0, 4.0, 4.0],
        "interest_expense": [1.0, 1.0, 1.0, 1.0], "cash": [10.0, 12.0, 14.0, 18.0],
        "total_debt": [30.0, 28.0, 26.0, 24.0], "total_equity": [50.0, 56.0, 62.0, 70.0],
        "total_assets": [120.0, 124.0, 128.0, 134.0], "total_liabilities": [70.0, 68.0, 66.0, 64.0],
        "current_assets": [40.0, 42.0, 45.0, 50.0], "current_liabilities": [25.0, 25.0, 25.0, 25.0],
        "receivables": [10.0, 11.0, 11.5, 12.0], "inventory": [8.0, 9.0, 9.5, 10.0],
        "goodwill": [20.0, 20.0, 20.0, 20.0], "retained_earnings": [30.0, 38.0, 46.0, 55.0],
        "sbc": [1.0, 1.0, 1.0, 1.0], "shares_diluted": [10.0, 10.0, 9.9, 9.8],
        "dividends_paid": [5.0, 5.0, 6.0, 6.0]}
    base.update(over)
    w = pd.DataFrame(base, index=[2023, 2024, 2025, 2026]).reindex(columns=list(health.HEALTH_CONCEPTS))
    return w, kpis.compute(w)


def flags(w, k, **kw):
    return {c.key: c for c in health.red_flags(w, k, **kw)}


def test_healthy_company_has_no_flags():
    w, k = W()
    f = flags(w, k)
    assert all(c.status == "ok" for c in f.values()), {k_: c.status for k_, c in f.items()}
    assert len(f) == 11 and "auditor_events" not in f
    assert "of average assets" in f["accruals"].detail and f["accruals"].why


def test_each_flag_fires():
    cases = {
        "accruals": dict(cfo=[15.0, 17.0, 19.0, 5.0]),
        "receivables_gap": dict(receivables=[10.0, 11.0, 11.5, 15.0]),
        "inventory_gap": dict(inventory=[8.0, 9.0, 9.5, 12.0]),
        "sbc_to_revenue": dict(sbc=[1.0, 1.0, 1.0, 12.0]),
        "goodwill_to_assets": dict(goodwill=[20.0, 20.0, 20.0, 70.0]),
        "net_debt_to_ebitda": dict(total_debt=[30.0, 28.0, 26.0, 150.0]),
        "interest_coverage": dict(interest_expense=[1.0, 1.0, 1.0, 10.0]),
        "current_ratio": dict(current_liabilities=[25.0, 25.0, 25.0, 80.0]),
        "dividends_to_fcf": dict(dividends_paid=[5.0, 5.0, 6.0, 20.0]),
        "share_growth": dict(shares_diluted=[10.0, 10.0, 10.5, 11.0]),
        "margin_decline": dict(operating_income=[20.0, 20.0, 19.0, 18.0]),
    }
    for key, change in cases.items():
        w, k = W(**change)
        assert flags(w, k)[key].status == "flag", key


def test_missing_data_and_special_cases():
    w, k = W(inventory=[NAN] * 4, sbc=[NAN] * 4)
    f = flags(w, k)
    assert f["inventory_gap"].status == "no data" and f["sbc_to_revenue"].status == "no data"
    w, k = W(cfo=[15.0, 17.0, 19.0, 4.0])                # FCF not positive but dividends paid
    assert "not positive" in flags(w, k)["dividends_to_fcf"].detail
    w, k = W(operating_income=[16.0, 18.5, 20.0, -10.0], d_and_a=[3.0, 3.0, 4.0, 4.0])
    assert flags(w, k)["net_debt_to_ebitda"].status == "flag"      # negative EBITDA
    assert health.red_flags(pd.DataFrame({"revenue": []}), pd.DataFrame()) == []


def test_auditor_events():
    w, k = W()
    filings = pd.DataFrame({"base_form": ["8-K", "8-K", "10-K", "8-K"],
                            "filed": ["2026-03-01", "2025-06-01", "2026-02-01", "2020-01-01"],
                            "items": ["4.01,9.01", "2.02", "", "4.02"]})
    c = flags(w, k, filings=filings, today="2026-11-15")["auditor_events"]
    assert c.status == "flag" and "2026-03-01 (item 4.01)" in c.detail and "2020" not in c.detail
    clean = flags(w, k, filings=filings.iloc[1:3], today="2026-11-15")["auditor_events"]
    assert clean.status == "ok" and "No such 8-K" in clean.detail


def test_piotroski():
    w, _ = W()
    p = health.piotroski(w)
    assert p["year"] == 2026 and p["available"] == 9 and p["score"] == 9
    worse, _ = W(net_income=[12.0, 14.0, 15.0, -2.0], shares_diluted=[10.0, 10.0, 9.9, 10.5])
    q = health.piotroski(worse)
    results = dict(q["tests"])
    assert results["Positive net income"] is False and results["No increase in share count"] is False
    assert q["score"] == 6
    partial, _ = W(gross_profit=[NAN] * 4, current_assets=[NAN] * 4)
    r = health.piotroski(partial)
    assert r["available"] == 7 and dict(r["tests"])["Gross margin higher than last year"] is None
    assert health.piotroski(pd.DataFrame({"revenue": []}))["score"] is None


def test_altman_z():
    w, _ = W()
    a = health.altman_z(w, market_cap=300.0)
    expected = 1.2 * 25 / 134 + 1.4 * 55 / 134 + 3.3 * 22 / 134 + 0.6 * 300 / 64 + 1.0 * 100 / 134
    assert a["z"] == pytest.approx(expected) and a["zone"] == "Safe zone" and len(a["parts"]) == 5
    assert health.altman_z(w, market_cap=20.0)["zone"] == "Grey zone"
    weak, _ = W(retained_earnings=[0, 0, 0, -60.0], operating_income=[16.0, 18.5, 20.0, -5.0])
    assert health.altman_z(weak, market_cap=5.0)["zone"] == "Distress zone"
    assert health.altman_z(w, None)["z"] is None and health.altman_z(w, None)["zone"] == "Not available"
    no_liab, _ = W(total_liabilities=[NAN] * 4)
    assert health.altman_z(no_liab, 300.0)["z"] is None
