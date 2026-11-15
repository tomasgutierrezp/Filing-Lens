"""Tag coverage on companies that use different XBRL tags from the golden pair."""
import json
from pathlib import Path

import pytest

from src.edgar.client import EdgarError
from src.edgar.facts import load_facts
from src.normalize.periods import FactIndex

SPEC = json.loads((Path(__file__).parent / "golden" / "coverage.json").read_text())
pytestmark = pytest.mark.network


@pytest.mark.parametrize("company", SPEC["companies"], ids=lambda c: c["ticker"])
def test_key_metrics_resolve_for_latest_year(company):
    try:
        facts = load_facts(company["cik"])
    except EdgarError as exc:
        pytest.skip(f"EDGAR unavailable and no cached copy: {exc}")
    ix = FactIndex(facts, company["cik"])
    assert not ix.fiscal_years.empty
    fy = ix.fiscal_years.iloc[-1]
    missing = [k for k in company["must_have"] if ix.resolve(k, fy["start"], fy["end"]) is None]
    assert not missing, f"{company['ticker']} FY{fy['fiscal_year']} missing: {missing}"
