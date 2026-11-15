"""Golden tests: parsed EDGAR numbers must equal the figures printed in the filings."""
import json
from pathlib import Path

import pytest

from src.edgar.client import EdgarError
from src.edgar.facts import load_facts
from src.normalize.periods import FactIndex

GOLDEN = [p for p in sorted((Path(__file__).parent / "golden").glob("*.json")) if p.stem != "coverage"]
pytestmark = pytest.mark.network


@pytest.fixture(scope="module", params=GOLDEN, ids=lambda p: p.stem)
def case(request):
    spec = json.loads(request.param.read_text())
    try:
        facts = load_facts(spec["cik"])
    except EdgarError as exc:
        pytest.skip(f"EDGAR unavailable and no cached copy: {exc}")
    return spec, FactIndex(facts, spec["cik"])


def test_taxonomy_and_currency(case):
    spec, ix = case
    assert ix.taxonomy == spec["taxonomy"]
    assert ix.currency == spec["currency"]


def test_values_match_filings(case):
    spec, ix = case
    scale = spec["scale"]
    mismatches = []
    for metric in spec["metrics"]:
        series = ix.annual(metric).set_index("fiscal_year")
        for year, expected in spec["years"].items():
            for field, column in (("as_first_reported", "original_value"), ("latest", "value")):
                want = expected.get(field, {}).get(metric)
                if want is None:
                    continue
                got = series[column].get(int(year))
                if got is None or round(got / scale) != want:
                    mismatches.append((metric, year, field, want, got))
    assert not mismatches, mismatches


def test_restatement_flags(case):
    spec, ix = case
    for year, expected in spec["years"].items():
        for metric in expected.get("restated", []):
            row = ix.annual(metric).set_index("fiscal_year").loc[int(year)]
            assert bool(row["restated"]), (metric, year)
            assert row["form"] in ("10-K", "20-F", "10-K/A", "20-F/A")
