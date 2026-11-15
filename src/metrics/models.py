"""Simple financial models built only from filed numbers. Pure functions, no network.

Every function takes the wide annual table (fiscal years x canonical concepts) or plain
numbers, and returns a DataFrame / list / float. Missing inputs give NaN or None.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.metrics.kpis import INPUT_CONCEPTS, average_with_prior, safe_div

MODEL_CONCEPTS = tuple(dict.fromkeys(INPUT_CONCEPTS + (
    "cost_of_revenue", "rnd", "sga", "total_assets", "dividends_paid", "buybacks",
    "acquisitions", "debt_repaid", "debt_issued")))

COMMON_SIZE_LABELS = {
    "cost_of_revenue": "Cost of revenue", "rnd": "Research & development",
    "sga": "Selling, general & admin", "other_operating": "Other operating costs",
    "income_tax": "Income tax", "other_items": "Interest & other items",
    "net_income": "Net income"}
ALLOCATION_LABELS = {
    "dividends_paid": "Dividends", "buybacks": "Share repurchases",
    "capex": "Capital expenditure", "acquisitions": "Acquisitions",
    "debt_paydown": "Net debt repayment"}
DUPONT_LABELS = {"net_margin": "Net margin", "asset_turnover": "Asset turnover",
                 "equity_multiplier": "Equity multiplier"}


def _cols(wide: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    return wide.reindex(columns=list(names))


def _num(v) -> Optional[float]:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


# ---- 1. common-size income statement ----------------------------------------------
def common_size(wide: pd.DataFrame) -> pd.DataFrame:
    """Where each unit of revenue goes, as fractions of revenue that add up to 1.

    Years without revenue, operating income and net income are dropped. Pieces the company
    does not report separately are folded into 'other' lines rather than guessed.
    """
    w = _cols(wide, ["revenue", "cost_of_revenue", "gross_profit", "rnd", "sga",
                     "operating_income", "income_tax", "net_income"])
    w = w.dropna(subset=["revenue", "operating_income", "net_income"])
    w = w[w["revenue"] > 0]
    cost = w["cost_of_revenue"].fillna(w["revenue"] - w["gross_profit"])
    out = pd.DataFrame(index=w.index)
    out["cost_of_revenue"] = cost
    out["rnd"] = w["rnd"]
    out["sga"] = w["sga"]
    out["other_operating"] = (w["revenue"] - w["operating_income"]
                              - out[["cost_of_revenue", "rnd", "sga"]].fillna(0).sum(axis=1))
    out["income_tax"] = w["income_tax"]
    out["other_items"] = w["operating_income"] - w["income_tax"].fillna(0) - w["net_income"]
    out["net_income"] = w["net_income"]
    return out.div(w["revenue"], axis=0)


# ---- 2. revenue -> free cash flow bridge -------------------------------------------
def fcf_bridge(wide: pd.DataFrame, year: int) -> Optional[List[Dict[str, object]]]:
    """Waterfall steps from revenue to free cash flow for one year.

    Each step is {'label', 'value', 'kind'} with kind 'total' (a level) or 'delta'
    (a signed move). Returns None when a required number is missing.
    """
    if year not in wide.index:
        return None
    r = _cols(wide, ["revenue", "cost_of_revenue", "gross_profit", "operating_income",
                     "income_tax", "net_income", "d_and_a", "cfo", "capex"]).loc[year]
    rev, oi, ni, cfo, capex = (_num(r[c]) for c in ("revenue", "operating_income", "net_income",
                                                     "cfo", "capex"))
    if None in (rev, oi, ni, cfo, capex):
        return None
    cost = _num(r["cost_of_revenue"])
    if cost is None and _num(r["gross_profit"]) is not None:
        cost = rev - float(r["gross_profit"])
    tax, da = _num(r["income_tax"]), _num(r["d_and_a"])

    def total(label, v):
        return {"label": label, "value": v, "kind": "total"}

    def delta(label, v):
        return {"label": label, "value": v, "kind": "delta"}

    steps = [total("Revenue", rev)]
    if cost is None:
        steps.append(delta("Operating costs", oi - rev))
    else:
        steps += [delta("Cost of revenue", -cost), delta("Operating expenses", oi - (rev - cost))]
    steps.append(total("Operating income", oi))
    if tax is None:
        steps.append(delta("Interest, tax & other", ni - oi))
    else:
        steps += [delta("Income tax", -tax), delta("Interest & other", ni - oi + tax)]
    steps.append(total("Net income", ni))
    if da is None:
        steps.append(delta("Non-cash items & working capital", cfo - ni))
    else:
        steps += [delta("Depreciation & amortization", da),
                  delta("Working capital & other", cfo - ni - da)]
    steps += [total("Cash from operations", cfo), delta("Capital expenditure", -capex),
              total("Free cash flow", cfo - capex)]
    return steps


# ---- 3. DuPont ROE -------------------------------------------------------------------
def dupont(wide: pd.DataFrame) -> pd.DataFrame:
    """ROE = net margin x asset turnover x equity multiplier (average balances)."""
    w = _cols(wide, ["net_income", "revenue", "total_assets", "total_equity"])
    assets, equity = average_with_prior(w["total_assets"]), average_with_prior(w["total_equity"])
    out = pd.DataFrame(index=w.index)
    out["net_margin"] = safe_div(w["net_income"], w["revenue"])
    out["asset_turnover"] = safe_div(w["revenue"], assets)
    out["equity_multiplier"] = safe_div(assets, equity)
    out["roe"] = out["net_margin"] * out["asset_turnover"] * out["equity_multiplier"]
    return out


def dupont_driver(prior: pd.Series, current: pd.Series) -> Optional[str]:
    """Which component moved ROE most between two years (largest log change).
    None unless all components are positive in both years."""
    moves = {}
    for key in DUPONT_LABELS:
        a, b = _num(prior.get(key)), _num(current.get(key))
        if a is None or b is None or a <= 0 or b <= 0:
            return None
        moves[key] = abs(math.log(b / a))
    return max(moves, key=moves.get)


# ---- 4. capital allocation -----------------------------------------------------------
def capital_uses(wide: pd.DataFrame) -> pd.DataFrame:
    """Cash uses by year in currency. Net debt repayment counts only when repayments exceed
    new borrowing; net borrowing is a source of cash, not a use, and shows as 0."""
    w = _cols(wide, ["dividends_paid", "buybacks", "capex", "acquisitions", "debt_repaid",
                     "debt_issued", "cfo"])
    out = w[["dividends_paid", "buybacks", "capex", "acquisitions"]].copy()
    out["debt_paydown"] = (w["debt_repaid"] - w["debt_issued"].fillna(0)).clip(lower=0)
    out["cfo"] = w["cfo"]
    return out


def capital_allocation(wide: pd.DataFrame) -> pd.DataFrame:
    """Each use as a fraction of cash from operations (NaN when CFO is not positive)."""
    uses = capital_uses(wide)
    return uses[list(ALLOCATION_LABELS)].apply(lambda col: safe_div(col, uses["cfo"]))


def allocation_summary(wide: pd.DataFrame, years: int = 5) -> Optional[Dict[str, object]]:
    """Totals over the last `years` fiscal years with cash-flow data."""
    uses = capital_uses(wide).dropna(subset=["cfo"]).tail(years)
    if uses.empty or uses["cfo"].sum() <= 0:
        return None
    totals = uses[list(ALLOCATION_LABELS)].sum(min_count=1).dropna()
    totals = totals[totals > 0]
    if totals.empty:
        return None
    cfo = float(uses["cfo"].sum())
    return {"years": len(uses), "cfo": cfo, "top_use": totals.idxmax(),
            "top_share": float(totals.max() / cfo), "total_share": float(totals.sum() / cfo),
            "shares": {k: float(v / cfo) for k, v in totals.items()}}


# ---- 6. simple DCF ---------------------------------------------------------------------
def dcf_equity_value(base_fcf: float, growth: float, discount: float, terminal_growth: float,
                     years: int = 5, net_debt: float = 0.0) -> float:
    """Present value of `years` of FCF growing at `growth`, plus a Gordon-growth terminal
    value, minus net debt. NaN unless discount > terminal growth and discount > -100%."""
    if discount <= terminal_growth or discount <= -1 or years < 1:
        return float("nan")
    value, fcf = 0.0, base_fcf
    for t in range(1, years + 1):
        fcf *= 1 + growth
        value += fcf / (1 + discount) ** t
    terminal = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    return value + terminal / (1 + discount) ** years - net_debt


def dcf_grid(base_fcf: float, growths: Sequence[float], discounts: Sequence[float],
             terminal_growth: float, shares: float, years: int = 5,
             net_debt: float = 0.0) -> pd.DataFrame:
    """Value per share for every growth x discount-rate pair (rows: discount, cols: growth)."""
    data = [[dcf_equity_value(base_fcf, g, d, terminal_growth, years, net_debt) / shares
             for g in growths] for d in discounts]
    return pd.DataFrame(data, index=list(discounts), columns=list(growths))


def dcf_defaults(k: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Starting inputs taken from history: latest FCF, net debt and diluted shares, and a
    growth rate equal to the 5-year (else 3-year) FCF CAGR, kept within 0%-15%.
    `k` is the output of kpis.compute. None when FCF or share count is missing."""
    fcf = k["fcf"].dropna() if "fcf" in k else pd.Series(dtype=float)
    shares = k["shares_diluted"].dropna() if "shares_diluted" in k else pd.Series(dtype=float)
    if fcf.empty or shares.empty:
        return None
    year = int(fcf.index[-1])
    growth, basis = 0.03, "no usable FCF history, so 3% is a placeholder"
    for n in (5, 3):
        start = fcf.get(year - n, np.nan)
        if not np.isnan(start) and start > 0 and fcf[year] > 0:
            raw = (fcf[year] / start) ** (1 / n) - 1
            growth = round(min(max(raw, 0.0), 0.15) * 200) / 200   # nearest 0.5%
            basis = f"{n}-year FCF growth was {raw:.1%} a year"
            break
    net_debt = k["net_debt"].get(year, np.nan) if "net_debt" in k else np.nan
    return {"year": year, "base_fcf": float(fcf[year]), "growth": float(growth),
            "growth_basis": basis, "shares": float(shares.iloc[-1]),
            "shares_year": int(shares.index[-1]),
            "net_debt": None if np.isnan(net_debt) else float(net_debt)}


def implied_growth(share_price: float, base_fcf: float, discount: float, terminal_growth: float,
                   shares: float, net_debt: float = 0.0, years: int = 5,
                   low: float = -0.5, high: float = 1.0) -> Optional[float]:
    """Reverse DCF: the FCF growth rate for the next `years` years at which this model's
    value per share equals today's price. None when FCF is not positive or no rate between
    `low` and `high` matches. `share_price` is per ordinary share in the reporting currency."""
    if not (share_price and share_price > 0 and base_fcf and base_fcf > 0 and shares and shares > 0):
        return None

    def gap(g: float) -> float:
        return dcf_equity_value(base_fcf, g, discount, terminal_growth, years, net_debt) / shares - share_price

    lo, hi = gap(low), gap(high)
    if math.isnan(lo) or math.isnan(hi) or lo > 0 or hi < 0:
        return None
    a, b = low, high
    for _ in range(60):
        mid = (a + b) / 2
        if gap(mid) < 0:
            a = mid
        else:
            b = mid
    return (a + b) / 2
