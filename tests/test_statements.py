import math

from src.normalize.periods import FactIndex
from src.normalize.statements import STATEMENTS, _pct_change, available_years, build_statement
from src.edgar.concept_map import CONCEPTS


def test_every_statement_line_is_a_known_concept():
    for spec in STATEMENTS.values():
        assert all(k in CONCEPTS for k in spec["lines"])


def test_pct_change():
    assert _pct_change(110, 100) == 110 / 100 - 1
    assert _pct_change(50, 0) is None and _pct_change(50, -10) is None
    assert _pct_change(None, 10) is None and _pct_change(10, None) is None


def test_income_statement_current_vs_prior(synthetic_facts):
    ix = FactIndex(synthetic_facts, cik=1)
    assert available_years(ix) == [2024, 2023]
    t = build_statement(ix, "income", 2024).set_index("concept")
    rev = t.loc["revenue"]
    assert (rev["current"], rev["prior"], rev["change"]) == (460, 390, 70)
    assert math.isclose(rev["change_pct"], 460 / 390 - 1)
    assert not rev["restated"] and rev["prior_restated"]          # FY2023 was restated
    assert rev["tags"] == "Revenues" and rev["form"] == "10-K" and rev["url"]
    assert bool(t.loc["net_income", "restated"])                   # 10-K/A amended FY2024
    assert math.isnan(t.loc["gross_profit", "current"])            # not reported -> empty
    assert t.loc["gross_profit", "tags"] is None


def test_oldest_year_has_no_prior(synthetic_facts):
    t = build_statement(FactIndex(synthetic_facts, cik=1), "cashflow", 2023).set_index("concept")
    assert t.loc["cfo", "current"] == 90 and math.isnan(t.loc["cfo", "prior"])
    assert math.isnan(t.loc["cfo", "change"])


def test_balance_sheet_uses_year_end_points(synthetic_facts):
    t = build_statement(FactIndex(synthetic_facts, cik=1), "balance", 2024).set_index("concept")
    assert (t.loc["total_assets", "current"], t.loc["total_assets", "prior"]) == (1000, 900)
    assert (t.loc["total_debt", "current"], t.loc["total_debt", "prior"]) == (58, 40)
