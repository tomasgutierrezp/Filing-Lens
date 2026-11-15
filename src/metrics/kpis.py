"""KPI formulas. Pure functions: DataFrame / Series in, DataFrame / Series out, no network.

All series are indexed by consecutive fiscal years, so `.shift(1)` is the prior year.
A KPI is NaN whenever an input is missing or a denominator is zero or negative —
it is shown as "Not available", never estimated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

INPUT_CONCEPTS = ("revenue", "gross_profit", "operating_income", "pretax_income", "income_tax",
                  "net_income", "cfo", "capex", "d_and_a", "interest_expense", "cash",
                  "total_debt", "total_equity", "shares_diluted")

TIER_LABELS = {1: "Tier 1 · Reported", 2: "Tier 2 · Derived from filings",
               3: "Tier 3 · Depends on today's price"}


@dataclass(frozen=True)
class KPI:
    key: str
    label: str
    tier: int
    kind: str                 # "currency", "shares", "pct" or "multiple"
    inputs: Tuple[str, ...]   # canonical concepts the value is built from
    formula: str


KPIS: Dict[str, KPI] = {k.key: k for k in [
    KPI("revenue", "Revenue", 1, "currency", ("revenue",), "As reported"),
    KPI("operating_income", "Operating income", 1, "currency", ("operating_income",), "As reported"),
    KPI("net_income", "Net income", 1, "currency", ("net_income",), "As reported"),
    KPI("cfo", "Cash from operations", 1, "currency", ("cfo",), "As reported"),
    KPI("capex", "Capital expenditure", 1, "currency", ("capex",), "As reported"),
    KPI("cash", "Cash and equivalents", 1, "currency", ("cash",), "As reported"),
    KPI("total_debt", "Total debt", 1, "currency", ("total_debt",), "As reported"),
    KPI("shares_diluted", "Diluted shares", 1, "shares", ("shares_diluted",), "As reported"),
    KPI("gross_margin", "Gross margin", 2, "pct", ("gross_profit", "revenue"),
        "Gross profit ÷ revenue"),
    KPI("operating_margin", "Operating margin", 2, "pct", ("operating_income", "revenue"),
        "Operating income ÷ revenue"),
    KPI("net_margin", "Net margin", 2, "pct", ("net_income", "revenue"), "Net income ÷ revenue"),
    KPI("fcf", "Free cash flow", 2, "currency", ("cfo", "capex"),
        "Cash from operations − capital expenditure"),
    KPI("fcf_margin", "FCF margin", 2, "pct", ("cfo", "capex", "revenue"),
        "Free cash flow ÷ revenue"),
    KPI("revenue_growth", "Revenue growth", 2, "pct", ("revenue",), "Revenue ÷ prior-year revenue − 1"),
    KPI("revenue_cagr_3y", "Revenue CAGR (3 yr)", 2, "pct", ("revenue",), "(Revenue ÷ revenue 3 years ago)^(1/3) − 1"),
    KPI("revenue_cagr_5y", "Revenue CAGR (5 yr)", 2, "pct", ("revenue",), "(Revenue ÷ revenue 5 years ago)^(1/5) − 1"),
    KPI("revenue_cagr_10y", "Revenue CAGR (10 yr)", 2, "pct", ("revenue",), "(Revenue ÷ revenue 10 years ago)^(1/10) − 1"),
    KPI("roe", "Return on equity", 2, "pct", ("net_income", "total_equity"),
        "Net income ÷ average shareholders' equity"),
    KPI("roic", "Return on invested capital", 2, "pct",
        ("operating_income", "income_tax", "pretax_income", "total_equity", "total_debt", "cash"),
        "≈ Operating income × (1 − effective tax rate) ÷ average (equity + debt − cash)"),
    KPI("net_debt", "Net debt", 2, "currency", ("total_debt", "cash"), "Total debt − cash"),
    KPI("net_debt_to_ebitda", "Net debt / EBITDA", 2, "multiple",
        ("total_debt", "cash", "operating_income", "d_and_a"),
        "(Total debt − cash) ÷ (operating income + D&A)"),
    KPI("interest_coverage", "Interest coverage", 2, "multiple",
        ("operating_income", "interest_expense"), "Operating income ÷ interest expense"),
    KPI("share_change", "Share count change", 2, "pct", ("shares_diluted",),
        "Diluted shares ÷ prior-year diluted shares − 1"),
]}

DASHBOARD_CARDS = ("revenue", "operating_income", "net_income", "cfo", "fcf", "gross_margin",
                   "operating_margin", "roe", "roic", "net_debt_to_ebitda", "interest_coverage",
                   "shares_diluted")


# ---- building blocks ------------------------------------------------------------
def to_wide(annual_long: pd.DataFrame) -> pd.DataFrame:
    """Long annual series (one row per concept and year) -> years x concepts, with every
    year between the first and last present so that shifts line up with calendar gaps."""
    wide = annual_long.pivot(index="fiscal_year", columns="concept", values="value").astype(float)
    if len(wide):
        wide = wide.reindex(range(int(wide.index.min()), int(wide.index.max()) + 1))
    wide.index.name, wide.columns.name = "fiscal_year", None
    return wide.reindex(columns=list(dict.fromkeys(list(INPUT_CONCEPTS) + list(wide.columns))))


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that is NaN where the denominator is missing, zero or negative."""
    return num / den.where(den > 0)


def average_with_prior(s: pd.Series) -> pd.Series:
    return (s + s.shift(1)) / 2


def margin(part: pd.Series, revenue: pd.Series) -> pd.Series:
    return safe_div(part, revenue)


def free_cash_flow(cfo: pd.Series, capex: pd.Series) -> pd.Series:
    return cfo - capex


def yoy_change(s: pd.Series) -> pd.Series:
    return safe_div(s, s.shift(1)) - 1


def cagr(s: pd.Series, years: int) -> pd.Series:
    start = s.shift(years)
    ratio = safe_div(s.where(s > 0), start)
    return ratio ** (1 / years) - 1


def roe(net_income: pd.Series, equity: pd.Series) -> pd.Series:
    return safe_div(net_income, average_with_prior(equity))


def effective_tax_rate(income_tax: pd.Series, pretax_income: pd.Series) -> pd.Series:
    return safe_div(income_tax, pretax_income).clip(0, 1)


def roic(operating_income: pd.Series, tax_rate: pd.Series, equity: pd.Series,
         debt: pd.Series, cash: pd.Series) -> pd.Series:
    invested = average_with_prior(equity) + average_with_prior(debt) - average_with_prior(cash)
    return safe_div(operating_income * (1 - tax_rate), invested)


def net_debt(debt: pd.Series, cash: pd.Series) -> pd.Series:
    return debt - cash


def ebitda(operating_income: pd.Series, d_and_a: pd.Series) -> pd.Series:
    return operating_income + d_and_a


def net_debt_to_ebitda(debt: pd.Series, cash: pd.Series, operating_income: pd.Series,
                       d_and_a: pd.Series) -> pd.Series:
    return safe_div(net_debt(debt, cash), ebitda(operating_income, d_and_a))


def interest_coverage(operating_income: pd.Series, interest_expense: pd.Series) -> pd.Series:
    return safe_div(operating_income, interest_expense)


# ---- everything at once -----------------------------------------------------------
def compute(wide: pd.DataFrame) -> pd.DataFrame:
    """Inputs plus every Tier 2 KPI, by fiscal year."""
    w = wide.reindex(columns=list(dict.fromkeys(list(INPUT_CONCEPTS) + list(wide.columns))))
    out = w.copy()
    out["gross_margin"] = margin(w["gross_profit"], w["revenue"])
    out["operating_margin"] = margin(w["operating_income"], w["revenue"])
    out["net_margin"] = margin(w["net_income"], w["revenue"])
    out["fcf"] = free_cash_flow(w["cfo"], w["capex"])
    out["fcf_margin"] = margin(out["fcf"], w["revenue"])
    out["revenue_growth"] = yoy_change(w["revenue"])
    for n in (3, 5, 10):
        out[f"revenue_cagr_{n}y"] = cagr(w["revenue"], n)
    out["roe"] = roe(w["net_income"], w["total_equity"])
    out["effective_tax_rate"] = effective_tax_rate(w["income_tax"], w["pretax_income"])
    out["roic"] = roic(w["operating_income"], out["effective_tax_rate"], w["total_equity"],
                       w["total_debt"], w["cash"])
    out["net_debt"] = net_debt(w["total_debt"], w["cash"])
    out["ebitda"] = ebitda(w["operating_income"], w["d_and_a"])
    out["net_debt_to_ebitda"] = net_debt_to_ebitda(w["total_debt"], w["cash"],
                                                   w["operating_income"], w["d_and_a"])
    out["interest_coverage"] = interest_coverage(w["operating_income"], w["interest_expense"])
    out["share_change"] = yoy_change(w["shares_diluted"])
    return out.replace([np.inf, -np.inf], np.nan)
