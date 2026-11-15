"""Health checks from filed numbers: a red-flag checklist, the Piotroski F-score and the
Altman Z-score. Pure functions, no network, no AI.

A flag is a prompt to read the filing, not a verdict: each one has an innocent
explanation as often as a worrying one. Thresholds live in FLAG_THRESHOLDS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.metrics import models

HEALTH_CONCEPTS = tuple(dict.fromkeys(models.MODEL_CONCEPTS + (
    "receivables", "inventory", "current_assets", "current_liabilities", "total_liabilities",
    "goodwill", "retained_earnings", "sbc")))

FLAG_THRESHOLDS = {
    "accruals": 0.05,            # (net income - operating cash flow) / average assets
    "receivables_gap": 0.10,     # receivables growth minus revenue growth
    "inventory_gap": 0.10,       # inventory growth minus revenue growth
    "sbc_to_revenue": 0.10,
    "goodwill_to_assets": 0.40,
    "net_debt_to_ebitda": 4.0,
    "interest_coverage": 3.0,    # flagged when below
    "current_ratio": 0.8,        # flagged when below
    "dividends_to_fcf": 1.0,
    "share_growth": 0.03,        # per year, over two years
    "margin_decline_years": 3,   # consecutive years of falling operating margin
    "auditor_lookback_years": 3,
}
ALTMAN_ZONES = ((2.99, "Safe zone"), (1.81, "Grey zone"), (-math.inf, "Distress zone"))


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    status: str               # "flag", "ok" or "no data"
    detail: str               # what the numbers are
    why: str                  # why it is worth a look


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _at(df: pd.DataFrame, col: str, year: int) -> Optional[float]:
    if col not in df or year not in df.index:
        return None
    return _num(df.at[year, col])


def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None or b <= 0 else a / b


def _growth(df: pd.DataFrame, col: str, year: int) -> Optional[float]:
    now, before = _at(df, col, year), _at(df, col, year - 1)
    return None if now is None or before is None or before <= 0 else now / before - 1


def _avg(df: pd.DataFrame, col: str, year: int) -> Optional[float]:
    now, before = _at(df, col, year), _at(df, col, year - 1)
    return None if now is None or before is None else (now + before) / 2


def _pct(v: float) -> str:
    return f"{'−' if v < 0 else ''}{abs(v):.1%}"


def _check(key, label, flagged: Optional[bool], detail: str, why: str) -> Check:
    if flagged is None:
        return Check(key, label, "no data", "The filings do not report the numbers this needs.", why)
    return Check(key, label, "flag" if flagged else "ok", detail, why)


def latest_year(wide: pd.DataFrame) -> Optional[int]:
    rev = wide["revenue"].dropna() if "revenue" in wide else pd.Series(dtype=float)
    return int(rev.index[-1]) if len(rev) else None


# ---- red-flag checklist --------------------------------------------------------------
def red_flags(wide: pd.DataFrame, k: pd.DataFrame, filings: Optional[pd.DataFrame] = None,
              today: Optional[str] = None) -> List[Check]:
    """`wide` holds HEALTH_CONCEPTS by fiscal year, `k` is kpis.compute(wide), `filings` is
    the filing list (for auditor events) and `today` an ISO date."""
    y = latest_year(wide)
    if y is None:
        return []
    T = FLAG_THRESHOLDS
    out: List[Check] = []

    ni, cfo, assets = _at(wide, "net_income", y), _at(wide, "cfo", y), _avg(wide, "total_assets", y)
    accr = _div(ni - cfo, assets) if None not in (ni, cfo) else None
    out.append(_check("accruals", "Profit running ahead of cash",
                      None if accr is None else accr > T["accruals"],
                      "" if accr is None else f"Net income minus operating cash flow was {_pct(accr)} of average assets "
                      f"(flagged above {_pct(T['accruals'])}).",
                      "Profit that is not backed by cash can come from aggressive accounting or from "
                      "customers who have not paid yet."))

    rev_g = _growth(wide, "revenue", y)
    for key, col, label, why in (
            ("receivables_gap", "receivables", "Receivables growing faster than sales",
             "Customers taking longer to pay, or sales booked early, show up here first."),
            ("inventory_gap", "inventory", "Inventory growing faster than sales",
             "Unsold stock building up can lead to write-downs or discounting later.")):
        g = _growth(wide, col, y)
        gap = None if g is None or rev_g is None else g - rev_g
        out.append(_check(key, label, None if gap is None else gap > T[key],
                          "" if gap is None else f"{col.capitalize()} changed {_pct(g)} while revenue changed "
                          f"{_pct(rev_g)} (flagged when the gap is above {_pct(T[key])}).", why))

    sbc = _div(_at(wide, "sbc", y), _at(wide, "revenue", y))
    out.append(_check("sbc_to_revenue", "Heavy stock-based pay", None if sbc is None else sbc > T["sbc_to_revenue"],
                      "" if sbc is None else f"Stock-based compensation was {_pct(sbc)} of revenue "
                      f"(flagged above {_pct(T['sbc_to_revenue'])}).",
                      "Pay in shares is a real cost that does not reduce operating cash flow, so cash "
                      "flow can look better than the economics."))

    gw = _div(_at(wide, "goodwill", y), _at(wide, "total_assets", y))
    out.append(_check("goodwill_to_assets", "Goodwill is a large share of assets",
                      None if gw is None else gw > T["goodwill_to_assets"],
                      "" if gw is None else f"Goodwill was {_pct(gw)} of total assets "
                      f"(flagged above {_pct(T['goodwill_to_assets'])}).",
                      "Goodwill is what was paid above book value in past acquisitions; it is written "
                      "off if those deals disappoint."))

    lev, ebitda = _at(k, "net_debt_to_ebitda", y), _at(k, "ebitda", y)
    lev_flag = None if lev is None and ebitda is None else (ebitda is not None and ebitda <= 0) or (
        lev is not None and lev > T["net_debt_to_ebitda"])
    out.append(_check("net_debt_to_ebitda", "High debt relative to earnings", lev_flag,
                      "EBITDA was not positive, so debt cannot be covered from earnings." if lev is None
                      else f"Net debt was {lev:.1f} times EBITDA (flagged above {T['net_debt_to_ebitda']:.1f}).",
                      "The more years of earnings it would take to repay debt, the less room there is "
                      "if earnings fall."))

    cov = _at(k, "interest_coverage", y)
    out.append(_check("interest_coverage", "Thin interest cover", None if cov is None else cov < T["interest_coverage"],
                      "" if cov is None else f"Operating income covered interest expense {cov:.1f} times "
                      f"(flagged below {T['interest_coverage']:.1f}).",
                      "Low cover means a modest fall in profit could make interest hard to pay."))

    cr = _div(_at(wide, "current_assets", y), _at(wide, "current_liabilities", y))
    out.append(_check("current_ratio", "Short-term bills exceed short-term assets",
                      None if cr is None else cr < T["current_ratio"],
                      "" if cr is None else f"Current assets were {cr:.2f} times current liabilities "
                      f"(flagged below {T['current_ratio']:.2f}).",
                      "A low ratio means relying on new cash coming in, or on borrowing, to pay what "
                      "falls due within a year. Common and harmless for some retailers."))

    fcf, div = _at(k, "fcf", y), _at(wide, "dividends_paid", y)
    if div is None or fcf is None:
        div_flag, div_detail = None, ""
    elif fcf <= 0:
        div_flag, div_detail = div > 0, "Dividends were paid while free cash flow was not positive."
    else:
        div_flag = div / fcf > T["dividends_to_fcf"]
        div_detail = f"Dividends were {_pct(div / fcf)} of free cash flow (flagged above {_pct(T['dividends_to_fcf'])})."
    out.append(_check("dividends_to_fcf", "Dividends not covered by free cash flow", div_flag, div_detail,
                      "A dividend larger than the cash left after investment has to be funded from "
                      "savings or borrowing."))

    s_now, s_then = _at(wide, "shares_diluted", y), _at(wide, "shares_diluted", y - 2)
    sg = None if s_now is None or not s_then or s_then <= 0 else (s_now / s_then) ** 0.5 - 1
    out.append(_check("share_growth", "Share count rising", None if sg is None else sg > T["share_growth"],
                      "" if sg is None else f"Diluted shares changed {_pct(sg)} a year over two years "
                      f"(flagged above {_pct(T['share_growth'])}).",
                      "More shares means each one owns a smaller slice of the business."))

    n = int(T["margin_decline_years"])
    margins = [_at(k, "operating_margin", y - i) for i in range(n + 1)]
    falling = None if None in margins else all(margins[i] < margins[i + 1] for i in range(n))
    out.append(_check("margin_decline", "Operating margin falling year after year", falling,
                      "" if falling is None else f"Operating margin went from {_pct(margins[-1])} to {_pct(margins[0])} "
                      f"over {n} years, " + ("falling every year." if falling else "not falling every year."),
                      "A steady slide can mean rising costs or weakening pricing power."))

    if filings is not None and today:
        since = str(int(today[:4]) - int(T["auditor_lookback_years"])) + today[4:]
        recent = filings[(filings["base_form"] == "8-K") & (filings["filed"] >= since)]
        hits = [f"{r.filed} (item {code})" for r in recent.itertuples()
                for code in ("4.01", "4.02") if code in str(r.items).split(",")]
        out.append(_check("auditor_events", "Auditor change or withdrawn financial statements", bool(hits),
                          ("8-K filings: " + ", ".join(hits) + ". Item 4.01 is a change of auditor; item 4.02 "
                           "means earlier statements should not be relied on.") if hits
                          else f"No such 8-K filing in the last {int(T['auditor_lookback_years'])} years.",
                          "Both are rare and worth reading in full when they happen."))
    return out


# ---- Piotroski F-score -----------------------------------------------------------------
def piotroski(wide: pd.DataFrame) -> Dict[str, object]:
    """Nine pass/fail tests of profitability, balance-sheet direction and efficiency.
    Returns {'year', 'score', 'available', 'tests': [(label, passed or None)]}."""
    y = latest_year(wide)
    if y is None:
        return {"year": None, "score": None, "available": 0, "tests": []}

    def roa(year):
        return _div(_at(wide, "net_income", year), _avg(wide, "total_assets", year))

    def ratio(a, b, year):
        return _div(_at(wide, a, year), _at(wide, b, year))

    def turnover(year):
        return _div(_at(wide, "revenue", year), _avg(wide, "total_assets", year))

    def cmp(now, before, better_if_lower=False):
        if now is None or before is None:
            return None
        return now < before if better_if_lower else now > before

    ni, cfo = _at(wide, "net_income", y), _at(wide, "cfo", y)
    shares_now, shares_before = _at(wide, "shares_diluted", y), _at(wide, "shares_diluted", y - 1)
    tests = [
        ("Positive net income", None if ni is None else ni > 0),
        ("Positive operating cash flow", None if cfo is None else cfo > 0),
        ("Return on assets higher than last year", cmp(roa(y), roa(y - 1))),
        ("Operating cash flow above net income", None if None in (ni, cfo) else cfo > ni),
        ("Debt lower relative to assets", cmp(ratio("total_debt", "total_assets", y),
                                              ratio("total_debt", "total_assets", y - 1), True)),
        ("Current ratio higher than last year", cmp(ratio("current_assets", "current_liabilities", y),
                                                    ratio("current_assets", "current_liabilities", y - 1))),
        ("No increase in share count", None if None in (shares_now, shares_before)
         else shares_now <= shares_before * 1.001),
        ("Gross margin higher than last year", cmp(ratio("gross_profit", "revenue", y),
                                                   ratio("gross_profit", "revenue", y - 1))),
        ("Asset turnover higher than last year", cmp(turnover(y), turnover(y - 1))),
    ]
    done = [t for t in tests if t[1] is not None]
    return {"year": y, "score": sum(1 for _, ok in done if ok) if done else None,
            "available": len(done), "tests": tests}


# ---- Altman Z-score --------------------------------------------------------------------
def altman_z(wide: pd.DataFrame, market_cap: Optional[float]) -> Dict[str, object]:
    """Original Altman Z for listed companies. `market_cap` must be in the reporting currency.
    Built for manufacturers: it is not meaningful for banks, insurers or asset-light firms."""
    y = latest_year(wide)
    empty = {"year": y, "z": None, "zone": "Not available", "parts": {}}
    if y is None:
        return empty
    ta, tl = _at(wide, "total_assets", y), _at(wide, "total_liabilities", y)
    ca, cl = _at(wide, "current_assets", y), _at(wide, "current_liabilities", y)
    re_, ebit, sales = _at(wide, "retained_earnings", y), _at(wide, "operating_income", y), _at(wide, "revenue", y)
    cap = _num(market_cap)
    if None in (ta, tl, ca, cl, re_, ebit, sales, cap) or ta <= 0 or tl <= 0:
        return empty
    parts = {"Working capital ÷ assets": (1.2, (ca - cl) / ta),
             "Retained earnings ÷ assets": (1.4, re_ / ta),
             "Operating income ÷ assets": (3.3, ebit / ta),
             "Market value ÷ liabilities": (0.6, cap / tl),
             "Revenue ÷ assets": (1.0, sales / ta)}
    z = sum(w * v for w, v in parts.values())
    zone = next(name for floor, name in ALTMAN_ZONES if z > floor)
    return {"year": y, "z": z, "zone": zone, "parts": parts}
