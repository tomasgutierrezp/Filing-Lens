from src.normalize.periods import FactIndex, dedupe, fiscal_year_label, period_class


def _ix(facts):
    return FactIndex(facts, cik=1)


def _vals(series):
    return {(r.period, r.fiscal_year): r.value for r in series.itertuples()}


def test_period_class():
    assert period_class("", "2024-12-31") == "I"
    assert period_class("2024-01-01", "2024-03-31") == "Q"
    assert period_class("2024-01-01", "2024-06-30") == "H"
    assert period_class("2024-01-01", "2024-09-30") == "9M"
    assert period_class("2023-10-01", "2024-09-28") == "FY"
    assert period_class("2024-01-01", "2024-01-31") == ""


def test_fiscal_year_label_handles_53_week_spillover():
    assert fiscal_year_label("2024-09-28") == 2024
    assert fiscal_year_label("2021-01-02") == 2020
    assert fiscal_year_label("2025-01-31") == 2025


def test_taxonomy_currency_and_fiscal_years(synthetic_facts):
    ix = _ix(synthetic_facts)
    assert (ix.taxonomy, ix.currency) == ("us-gaap", "USD")
    assert list(ix.fiscal_years["fiscal_year"]) == [2023, 2024]


def test_dedupe_keeps_latest_and_flags_restatement(synthetic_facts):
    d = dedupe(synthetic_facts)
    row = d[(d.tag == "Revenues") & (d.start == "2023-01-01") & (d.end == "2023-12-31")].iloc[0]
    assert (row.val, row.original_val, bool(row.restated), row.accn) == (390, 400, True, "A-24")


def test_annual_series_with_provenance(synthetic_facts):
    a = _ix(synthetic_facts).annual("revenue").set_index("fiscal_year")
    assert a.loc[2023, "value"] == 390 and a.loc[2023, "original_value"] == 400
    assert bool(a.loc[2023, "restated"]) and not bool(a.loc[2024, "restated"])
    r = a.loc[2024]
    assert (r["value"], r["tags"], r["form"], r["filed"], r["unit"]) == (460, "Revenues", "10-K", "2025-02-15", "USD")
    assert r["url"].endswith("/Archives/edgar/data/1/A24/A-24-index.htm")


def test_amendment_supersedes_original(synthetic_facts):
    a = _ix(synthetic_facts).annual("net_income").set_index("fiscal_year")
    assert a.loc[2024, "value"] == 75 and a.loc[2024, "form"] == "10-K/A"


def test_not_reported_is_empty_not_guessed(synthetic_facts):
    assert _ix(synthetic_facts).annual("buybacks").empty


def test_total_debt_sums_components_once(synthetic_facts):
    a = _ix(synthetic_facts).annual("total_debt").set_index("fiscal_year")
    # noncurrent 50 + current 5 + first available short-term tag 3 (commercial paper not double counted)
    assert a.loc[2024, "value"] == 58
    assert a.loc[2024, "tags"] == "LongTermDebtNoncurrent + LongTermDebtCurrent + ShortTermBorrowings"
    assert a.loc[2023, "value"] == 40 and a.loc[2023, "tags"] == "LongTermDebt"


def test_interim_uses_discrete_then_derives_from_ytd(synthetic_facts):
    q = _ix(synthetic_facts).interim("revenue")
    # Q2 is reported discretely; Q3 = 9M - 6M. No Q4: the 10-K restated the prior year, so
    # its full-year figure may be on a different basis from the 9M figure filed earlier.
    assert _vals(q) == {("Q1", 2024): 100, ("Q2", 2024): 110, ("Q3", 2024): 120}
    derived = {r.period: bool(r.derived) for r in q.itertuples()}
    assert derived == {"Q1": False, "Q2": False, "Q3": True}


def test_interim_cash_flow_from_ytd_only(synthetic_facts):
    q = _ix(synthetic_facts).interim("cfo")
    assert _vals(q) == {("Q1", 2024): 20, ("Q2", 2024): 25, ("Q3", 2024): 30, ("Q4", 2024): 35}


def test_no_derivation_across_restatement(synthetic_facts):
    q = _ix(synthetic_facts).interim("net_income")
    assert _vals(q) == {("Q1", 2024): 15, ("Q2", 2024): 20, ("Q3", 2024): 25}  # no Q4


def test_interim_stock_uses_points(synthetic_facts):
    q = _ix(synthetic_facts).interim("total_assets")
    assert _vals(q) == {("Q1", 2024): 950, ("Q4", 2024): 1000}


def test_ttm(synthetic_facts):
    ix = _ix(synthetic_facts)
    t = ix.ttm("cfo")
    assert t["value"] == 110 and t["start"] == "2024-01-01" and t["end"] == "2024-12-31"
    assert t["periods"] == ["Q1 FY2024", "Q2 FY2024", "Q3 FY2024", "Q4 FY2024"]
    assert ix.ttm("revenue") is None               # Q4 not derivable (see above)
    assert ix.ttm("net_income") is None            # only three quarters available
    assert ix.ttm("total_assets")["value"] == 1000  # balance sheet: latest point
    assert ix.ttm("buybacks") is None
