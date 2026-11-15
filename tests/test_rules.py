import pandas as pd

from src.explain import rules
from src.metrics import kpis

NAN = float("nan")


def test_formatters():
    assert rules.money(478_834e6, "USD") == "USD 478.8bn"
    assert rules.money(836e6, "EUR") == "EUR 836m"
    assert rules.money(-1.2e9, "USD") == "−USD 1.2bn"
    assert rules.money(2.5e12, "USD") == "USD 2.50tn"
    assert rules.count(14_720e6) == "14.72bn" and rules.count(2_195e6) == "2,195m"
    assert rules.pct(0.3301) == "33.0%" and rules.pct(-0.05) == "−5.0%"
    assert rules.multiple(1.234) == "1.2×"
    assert rules.format_kpi("roe", 0.25, "USD") == "25.0%"
    assert rules.format_kpi("fcf", 1e9, "USD") == "USD 1.0bn"
    assert rules.format_kpi("net_debt_to_ebitda", 2.04, "USD") == "2.0×"
    assert rules.format_kpi("shares_diluted", 2_195e6, "EUR") == "2,195m"
    assert rules.format_kpi("roe", NAN, "USD") == "Not available"


def test_growth():
    assert rules.growth("Revenue", 100e9, 115e9, "USD") == "Revenue rose 15.0% to USD 115.0bn from USD 100.0bn."
    assert rules.growth("Revenue", 100e9, 80e9, "USD") == "Revenue fell 20.0% to USD 80.0bn from USD 100.0bn."
    assert rules.growth("Revenue", 100e9, 102e9, "USD") == "Revenue was little changed at USD 102.0bn (2.0% on the year before)."
    assert rules.growth("Net income", -2e9, 3e9, "USD") == "Net income was USD 3.0bn, compared with −USD 2.0bn the year before."
    assert rules.growth("Revenue", NAN, 1e9, "USD") is None


def test_cagr_note():
    assert rules.cagr_note("Revenue", 0.055, 5) == "Over 5 years revenue grew 5.5% a year on average."
    assert rules.cagr_note("Revenue", -0.02, 3) == "Over 3 years revenue shrank 2.0% a year on average."
    assert rules.cagr_note("Revenue", NAN, 5) is None


def test_points_change():
    assert rules.points_change("Return on equity", 0.20, 0.25) == "Return on equity rose 5.0 pts to 25.0% from 20.0%."
    assert rules.points_change("Gross margin", 0.45, 0.42) == "Gross margin fell 3.0 pts to 42.0% from 45.0%."
    assert rules.points_change("Gross margin", 0.450, 0.455) == "Gross margin was steady at 45.5% (45.0% the year before)."
    assert rules.points_change("Gross margin", NAN, 0.4) is None


def test_margin_change():
    s = rules.margin_change("Operating margin", 0.30, 0.27, 0.05, 0.095)
    assert s == "Operating margin fell from 30.0% to 27.0%. Revenue grew 5.0% while operating costs grew 9.5%."
    s = rules.margin_change("Operating margin", 0.20, 0.25, -0.02, -0.08)
    assert s == "Operating margin rose from 20.0% to 25.0%. Revenue fell 2.0% while operating costs fell 8.0%."
    assert rules.margin_change("Operating margin", 0.20, 0.25, NAN, NAN) == "Operating margin rose from 20.0% to 25.0%."
    assert rules.margin_change("Operating margin", 0.200, 0.205, NAN, NAN).startswith("Operating margin was steady at 20.5%")
    assert rules.margin_change("Operating margin", NAN, 0.25, 0.1, 0.1) is None


def test_fcf_vs_net_income():
    # FCF 130 vs NI 100: D&A +20, working capital +30, capex -20 -> working capital leads
    s = rules.fcf_vs_net_income(130e9, 100e9, cfo=150e9, capex=20e9, d_and_a=20e9, currency="USD")
    assert s == ("Free cash flow was 1.3× net income — more of reported profit turned into cash, "
                 "mainly due to working capital and other non-cash items (USD 30.0bn).")
    # FCF 60 vs NI 100: D&A +10, working capital 0, capex -50 -> capex leads
    s = rules.fcf_vs_net_income(60e9, 100e9, cfo=110e9, capex=50e9, d_and_a=10e9, currency="USD")
    assert s == ("Free cash flow was 0.6× net income — less of reported profit turned into cash, "
                 "mainly due to capital expenditure (USD 50.0bn).")
    assert rules.fcf_vs_net_income(101e9, 100e9, 110e9, 9e9, 10e9, "USD") == \
        "Free cash flow was 1.0× net income — reported profit and cash generated were close."
    assert rules.fcf_vs_net_income(130e9, 100e9, NAN, NAN, NAN, "USD") == \
        "Free cash flow was 1.3× net income — more of reported profit turned into cash."
    assert "not meaningful" in rules.fcf_vs_net_income(5e9, -1e9, 6e9, 1e9, 1e9, "USD")
    assert rules.fcf_vs_net_income(NAN, 1e9, 1e9, 1e9, 1e9, "USD") is None


def test_dilution():
    assert rules.dilution(100, 90, 5) == "Diluted share count fell 10.0% over 5 years, so each share owns more of the company."
    assert rules.dilution(100, 104, 1) == "Diluted share count rose 4.0% over 1 year, so each share owns less of the company."
    assert rules.dilution(100, 100, 3) == "Diluted share count was unchanged over 3 years."
    assert rules.dilution(NAN, 100, 3) is None


def test_leverage():
    assert rules.leverage(20e9, 10e9, "EUR") == "Net debt of EUR 20.0bn equals 2.0 years of EBITDA (EUR 10.0bn)."
    assert rules.leverage(-5e9, 10e9, "USD") == "Cash exceeds debt by USD 5.0bn (a net cash position)."
    assert "not meaningful" in rules.leverage(5e9, -1e9, "USD")
    assert rules.leverage(NAN, 1e9, "USD") is None


def test_coverage():
    assert rules.coverage(90e9, 10e9) == "Operating income covers interest expense 9.0 times."
    assert rules.coverage(90e9, 10e9, 12.0) == "Operating income covers interest expense 9.0 times, down from 12.0 the year before."
    assert rules.coverage(90e9, 10e9, 9.05) == "Operating income covers interest expense 9.0 times."
    assert rules.coverage(90e9, 0) is None and rules.coverage(NAN, 1e9) is None


def _k():
    w = pd.DataFrame({"revenue": [100e9, 120e9], "gross_profit": [40e9, 54e9],
                      "operating_income": [20e9, 30e9], "pretax_income": [20e9, 30e9],
                      "income_tax": [5e9, 6e9], "net_income": [15e9, 24e9], "cfo": [25e9, 35e9],
                      "capex": [5e9, 10e9], "d_and_a": [5e9, 6e9], "interest_expense": [2e9, 3e9],
                      "cash": [10e9, 20e9], "total_debt": [50e9, 40e9], "total_equity": [60e9, 100e9],
                      "shares_diluted": [10e9, 9e9]}, index=[2023, 2024])
    return kpis.compute(w)


def test_every_dashboard_card_gets_a_sentence():
    k = _k()
    sentences = {key: rules.card_sentence(key, k, 2024, "USD") for key in kpis.DASHBOARD_CARDS}
    assert all(sentences.values()), sentences
    assert sentences["revenue"] == "Revenue rose 20.0% to USD 120.0bn from USD 100.0bn."
    assert sentences["operating_margin"] == ("Operating margin rose from 20.0% to 25.0%. "
                                             "Revenue grew 20.0% while operating costs grew 12.5%.")
    assert sentences["shares_diluted"].startswith("Diluted share count fell 10.0% over 1 year")
    assert sentences["net_debt_to_ebitda"].startswith("Net debt of USD 20.0bn equals 0.6 years")


def test_card_sentence_is_none_without_data():
    k = kpis.compute(pd.DataFrame({"revenue": [100e9]}, index=[2024]))
    assert rules.card_sentence("roic", k, 2024, "USD") is None
    assert rules.card_sentence("revenue", k, 2024, "USD") == "Revenue was USD 100.0bn."
    assert rules.card_sentence("roe", _k(), 2024, "USD") == "Return on equity was 30.0%."


# ---- model explanations ----------------------------------------------------------------
from src.metrics import models  # noqa: E402


def test_common_size_note():
    cur = pd.Series({"cost_of_revenue": 0.55, "rnd": 0.05, "sga": 0.10, "other_operating": 0.05,
                     "income_tax": 0.05, "other_items": 0.0, "net_income": 0.20})
    pri = cur.copy()
    pri["net_income"] = 0.15
    s = rules.common_size_note(cur, pri, 2024, models.COMMON_SIZE_LABELS)
    assert s == ("In FY2024, of every 100 of revenue, 55.0 went to cost of revenue, 10.0 went to "
                 "selling, general & admin, 5.0 went to research & development, 5.0 went to other "
                 "operating costs, 5.0 went to income tax, leaving 20.0 as net income (15.0 the year before).")
    cur["other_items"] = -0.02
    s = rules.common_size_note(cur, None, 2024, models.COMMON_SIZE_LABELS)
    assert s.endswith("leaving 20.0 as net income. Interest & other items added 2.0.")
    assert rules.common_size_note(pd.Series({"net_income": NAN}), None, 2024, {}) is None


def test_bridge_note():
    assert rules.bridge_note(120e9, 25e9, "USD") == "Of USD 120.0bn of revenue, USD 25.0bn (20.8%) ended up as free cash flow."
    assert rules.bridge_note(120e9, -5e9, "USD") == "Revenue of USD 120.0bn did not turn into free cash flow: it was −USD 5.0bn."
    assert rules.bridge_note(NAN, 1e9, "USD") is None


def test_dupont_note():
    a = pd.Series({"net_margin": 0.10, "asset_turnover": 1.0, "equity_multiplier": 2.0, "roe": 0.20})
    b = pd.Series({"net_margin": 0.10, "asset_turnover": 1.0, "equity_multiplier": 3.0, "roe": 0.30})
    L = models.DUPONT_LABELS
    assert rules.dupont_note(b, a, "equity_multiplier", L) == \
        "ROE changed by +10.0 pts (20.0% to 30.0%), driven mostly by equity multiplier (2.00 to 3.00)."
    assert rules.dupont_note(a, b, "net_margin", L) == \
        "ROE changed by −10.0 pts (30.0% to 20.0%), driven mostly by net margin (10.0% to 10.0%)."
    assert rules.dupont_note(b, a, None, L) == "ROE changed by +10.0 pts (20.0% to 30.0%)."
    assert rules.dupont_note(a, None, None, L) == \
        "ROE of 20.0% = net margin 10.0% × asset turnover 1.00 × equity multiplier 2.00."
    assert rules.dupont_note(a, a, None, L).startswith("ROE was steady at 20.0% (20.0% the year before)")
    assert rules.dupont_note(pd.Series({"roe": NAN}), a, None, L) is None


def test_capital_allocation_note():
    L = models.ALLOCATION_LABELS
    s = {"years": 5, "top_use": "buybacks", "top_share": 0.62, "total_share": 1.08}
    assert rules.capital_allocation_note(s, L) == (
        "Over the last 5 years, the largest use of operating cash was share repurchases (62%). "
        "The uses shown add up to 108% of operating cash, so part was funded from cash on hand, "
        "asset sales or borrowing.")
    s = {"years": 1, "top_use": "capex", "top_share": 0.4, "total_share": 0.7}
    assert rules.capital_allocation_note(s, L) == (
        "Over the last year, the largest use of operating cash was capital expenditure (40%). "
        "The uses shown add up to 70% of operating cash.")
    assert rules.capital_allocation_note(None, L) is None


def test_dcf_note():
    s = rules.dcf_note(120.5, 310.25, 190.0, "USD")
    assert s.startswith("Across this range of assumptions the model gives USD 120.50 to USD 310.25 "
                        "per share (USD 190.00 at your inputs).")
    assert "not a forecast or a price target" in s
    assert rules.dcf_note(NAN, 1.0, 1.0, "USD") is None


# ---- ratings --------------------------------------------------------------------------------
from src.metrics import ratings  # noqa: E402


def test_format_metric():
    assert rules.format_metric(0.123, "pct") == "12.3%"
    assert rules.format_metric(-0.015, "pts") == "−1.5 pts" and rules.format_metric(0.02, "pts") == "2.0 pts"
    assert rules.format_metric(1.24, "multiple") == "1.2×" and rules.format_metric(2.1, "ratio") == "2.10"
    assert rules.format_metric(0.8, "share") == "80%"
    assert rules.format_metric(0.42, "score") == "+0.42" and rules.format_metric(-1.5, "score") == "−1.50"
    assert rules.format_metric(None, "pct") == "Not reported" and rules.format_metric(NAN, "pct") == "Not reported"


def test_rating_sentence():
    a = ratings.metric("roe", "Return on equity", 0.25, "pct")
    b = ratings.metric("equity_multiplier", "Equity multiplier (leverage)", 4.4, "ratio")
    r = ratings.combine("dupont", "Returns", [a, b])
    assert rules.rating_sentence(r) == ("Grade C (60 out of 100). Strongest input: return on equity (25.0%). "
                                        "Weakest: equity multiplier (leverage) (4.40).")
    one = ratings.combine("shares", "Shares", [ratings.metric("share_change_per_year", "Share change per year", -0.02, "pct")])
    assert rules.rating_sentence(one) == "Grade A (100 out of 100). Share change per year: −2.0%."
    same = ratings.combine("x", "X", [a, ratings.metric("roic", "ROIC", 0.5, "pct")])
    assert rules.rating_sentence(same) == "Grade A (100 out of 100). All inputs score about the same."
    assert rules.rating_sentence(r, include_grade=False).startswith("Strongest input: return on equity")
    caps = ratings.combine("x", "X", [a, ratings.metric("roic", "ROIC", 0.06, "pct")])
    assert "Weakest: ROIC (6.0%)" in rules.rating_sentence(caps)        # acronyms keep their capitals
    none = ratings.combine("x", "X", [ratings.metric("roe", "Return on equity", NAN, "pct")])
    assert rules.rating_sentence(none) == "Not rated: the data needed is missing (return on equity)."


def test_reverse_dcf_note():
    assert rules.reverse_dcf_note(0.12, 0.079, 0.09, 0.025) == (
        "Today's price matches this model if free cash flow grows 12.0% a year for 5 years and 2.5% "
        "after that, at a 9.0% discount rate. Over the last 5 years free cash flow actually grew 7.9% a year.")
    assert rules.reverse_dcf_note(-0.03, NAN, 0.09, 0.025).endswith("at a 9.0% discount rate.")
    assert rules.reverse_dcf_note(None, 0.05, 0.09, 0.025) is None
