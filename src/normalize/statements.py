"""Income statement / balance sheet / cash flow tables for a fiscal year vs the prior year."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.edgar.concept_map import CONCEPTS
from src.normalize.periods import FactIndex

STATEMENTS: Dict[str, Dict[str, object]] = {
    "income": {"title": "Income statement", "lines": [
        "revenue", "cost_of_revenue", "gross_profit", "rnd", "sga", "operating_income",
        "interest_expense", "pretax_income", "income_tax", "net_income", "eps_diluted",
        "shares_diluted"]},
    "balance": {"title": "Balance sheet", "lines": [
        "cash", "receivables", "inventory", "current_assets", "ppe", "total_assets",
        "current_liabilities", "total_debt", "total_liabilities", "total_equity"]},
    "cashflow": {"title": "Cash flow statement", "lines": [
        "cfo", "d_and_a", "capex", "acquisitions", "cfi", "dividends_paid", "buybacks", "cff"]},
}

STATEMENT_COLUMNS = ["concept", "label", "unit", "current", "prior", "change", "change_pct",
                     "restated", "prior_restated", "tags", "form", "filed", "end", "url"]


def available_years(ix: FactIndex) -> List[int]:
    """Fiscal years with annual data, newest first."""
    return sorted((int(y) for y in ix.fiscal_years["fiscal_year"]), reverse=True)


def _pct_change(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Year-over-year change as a fraction; undefined when the base is zero or negative."""
    if current is None or prior is None or prior <= 0:
        return None
    return current / prior - 1


def build_statement(ix: FactIndex, statement: str, fiscal_year: int) -> pd.DataFrame:
    """One row per line item. Missing values stay None so the UI can show 'Not reported'."""
    fy = ix.fiscal_years.set_index("fiscal_year")
    periods = {y: (fy.loc[y, "start"], fy.loc[y, "end"]) for y in (fiscal_year, fiscal_year - 1)
               if y in fy.index}
    rows = []
    for key in STATEMENTS[statement]["lines"]:
        cur = ix.resolve(key, *periods[fiscal_year]) if fiscal_year in periods else None
        pri = ix.resolve(key, *periods[fiscal_year - 1]) if fiscal_year - 1 in periods else None
        cv = cur["value"] if cur else None
        pv = pri["value"] if pri else None
        src = cur or pri or {}
        rows.append({
            "concept": key, "label": CONCEPTS[key].label, "unit": CONCEPTS[key].unit,
            "current": cv, "prior": pv,
            "change": cv - pv if cv is not None and pv is not None else None,
            "change_pct": _pct_change(cv, pv),
            "restated": bool(cur and cur["restated"]),
            "prior_restated": bool(pri and pri["restated"]),
            "tags": src.get("tags"), "form": src.get("form"), "filed": src.get("filed"),
            "end": src.get("end"), "url": src.get("url"),
        })
    return pd.DataFrame(rows, columns=STATEMENT_COLUMNS).astype(
        {"current": "float64", "prior": "float64", "change": "float64", "change_pct": "float64"})
