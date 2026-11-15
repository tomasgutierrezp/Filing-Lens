"""Rule-based ratings: a 0-100 score and a letter grade for each model, for the news
climate, and for the business's current, historical and future-readiness state.

Pure functions, no network, no AI. Every score comes from fixed thresholds in THRESHOLDS:
a metric at or beyond its "strong" anchor scores 100, at or beyond its "poor" anchor
scores 0, and in between it is a straight line. The same thresholds apply to every
company; they are not adjusted for industry. A rating describes the numbers — it is not
a forecast and not a recommendation.

The three overall grades use filing numbers only. The climate rating is kept apart and
never feeds them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.metrics import models

# metric key -> (poor anchor, strong anchor). Lower-is-better metrics have poor > strong.
THRESHOLDS = {
    "operating_margin": (0.05, 0.25),
    "net_margin": (0.03, 0.20),
    "margin_change_3y": (-0.03, 0.03),
    "fcf_to_net_income": (0.5, 1.1),
    "fcf_margin": (0.03, 0.20),
    "roe": (0.05, 0.25),
    "roic": (0.05, 0.20),
    "equity_multiplier": (5.0, 2.0),
    "payouts_to_fcf_5y": (1.5, 0.8),
    "uses_to_cfo_5y": (1.4, 1.0),
    "share_change_per_year": (0.03, -0.02),
    "dcf_scenarios_above_price": (0.0, 1.0),
    "tone_30d": (-4.0, 2.0),
    "tone_trend": (-1.5, 1.5),
    "net_debt_to_ebitda": (4.0, 0.5),
    "interest_coverage": (2.0, 12.0),
    "revenue_cagr_5y": (0.0, 0.10),
    "fcf_positive_years": (0.5, 1.0),
    "margin_volatility": (0.06, 0.01),
    "roe_average_5y": (0.05, 0.20),
    "growth_years": (0.4, 0.9),
    "revenue_cagr_3y": (-0.02, 0.08),
    "reinvestment_rate": (0.02, 0.12),
    "dividends_to_fcf": (1.0, 0.3),
}
GRADE_BANDS = (("A", 80), ("B", 65), ("C", 50), ("D", 35), ("E", 0))
MIN_COVERAGE = 0.5   # share of a rating's weight that must have data, or it is "Not rated"
DCF_DISCOUNT, DCF_TERMINAL = 0.09, 0.025


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    value: Optional[float]
    kind: str                 # "pct", "pts", "multiple", "ratio", "share" or "score"
    score: Optional[float]    # 0-100, None when the value is missing
    poor: float
    strong: float
    weight: float = 1.0


@dataclass(frozen=True)
class Rating:
    key: str
    label: str
    score: Optional[float]
    grade: str                # "A".."E" or "Not rated"
    metrics: List[Metric] = field(default_factory=list)
    basis: str = ""           # what the rating looks at, in a sentence
    note: str = ""            # anything left out, and why


# ---- building blocks ----------------------------------------------------------------
def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def scale(value: Optional[float], poor: float, strong: float) -> Optional[float]:
    """Linear 0-100 score between two anchors, clipped. Works for lower-is-better metrics
    when poor > strong."""
    v = _num(value)
    if v is None or poor == strong:
        return None
    return float(min(100.0, max(0.0, (v - poor) / (strong - poor) * 100.0)))


def grade_for(score: Optional[float]) -> str:
    if score is None:
        return "Not rated"
    for letter, floor in GRADE_BANDS:
        if score >= floor:
            return letter
    return "E"


def metric(key: str, label: str, value, kind: str, weight: float = 1.0) -> Metric:
    poor, strong = THRESHOLDS[key]
    v = _num(value)
    return Metric(key, label, v, kind, scale(v, poor, strong), poor, strong, weight)


def combine(key: str, label: str, metrics: Sequence[Metric], basis: str = "", note: str = "") -> Rating:
    """Weighted average of the metrics that have data. Not rated when too few do."""
    total = sum(m.weight for m in metrics)
    scored = [m for m in metrics if m.score is not None]
    have = sum(m.weight for m in scored)
    if not scored or total <= 0 or have / total < MIN_COVERAGE:
        return Rating(key, label, None, "Not rated", list(metrics), basis, note)
    score = sum(m.score * m.weight for m in scored) / have
    return Rating(key, label, round(score, 1), grade_for(score), list(metrics), basis, note)


def first_restated_year(annual_long: pd.DataFrame, concept: str = "revenue") -> Optional[int]:
    """First year of the most recent run of consecutive fiscal years whose figure was
    restated in a later filing. Years before it were not re-presented at that time, so they
    may be on a different basis (for example before a disposal), and growth measured across
    that point is not reliable."""
    rows = annual_long[(annual_long["concept"] == concept) & annual_long["restated"].astype(bool)]
    years = sorted(int(y) for y in rows["fiscal_year"].unique())
    if not years:
        return None
    start = years[-1]
    for y in reversed(years[:-1]):
        if y != start - 1:
            break
        start = y
    return start


def _spans_restatement(year: Optional[int], span: int, restated_from: Optional[int]) -> bool:
    return year is not None and restated_from is not None and year - span < restated_from <= year


_BASIS_NOTE = ("Revenue growth is left out where it would compare years before and after "
               "FY{year}, the earliest year restated in a later filing: the older figures "
               "are on a different basis.")


def _latest_year(k: pd.DataFrame) -> Optional[int]:
    if k.empty or "revenue" not in k:
        return None
    rev = k["revenue"].dropna()
    return int(rev.index[-1]) if len(rev) else None


def _at(k: pd.DataFrame, col: str, year: Optional[int]) -> Optional[float]:
    if year is None or col not in k or year not in k.index:
        return None
    return _num(k.at[year, col])


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    return None if num is None or den is None or den <= 0 else num / den


def _margin_change(k: pd.DataFrame, year: Optional[int], span: int = 3) -> Optional[float]:
    now, then = _at(k, "operating_margin", year), _at(k, "operating_margin", (year or 0) - span)
    return None if now is None or then is None else now - then


# ---- one rating per model -------------------------------------------------------------
def rate_profitability(k: pd.DataFrame) -> Rating:
    """Common-size model: how much of revenue is kept."""
    y = _latest_year(k)
    return combine("common_size", "Profitability", [
        metric("operating_margin", "Operating margin", _at(k, "operating_margin", y), "pct"),
        metric("net_margin", "Net margin", _at(k, "net_margin", y), "pct"),
        metric("margin_change_3y", "Operating margin change over 3 years", _margin_change(k, y), "pts", 0.5),
    ], "How much of each unit of revenue is kept as profit, and whether that is improving.")


def rate_cash_conversion(k: pd.DataFrame) -> Rating:
    """Revenue -> FCF bridge: how much profit and revenue becomes free cash."""
    y = _latest_year(k)
    return combine("bridge", "Cash conversion", [
        metric("fcf_to_net_income", "Free cash flow ÷ net income",
               _ratio(_at(k, "fcf", y), _at(k, "net_income", y)), "multiple"),
        metric("fcf_margin", "Free cash flow margin", _at(k, "fcf_margin", y), "pct"),
    ], "How much of reported profit and of revenue ends up as free cash flow.")


def rate_returns(k: pd.DataFrame, dupont: pd.DataFrame) -> Rating:
    """DuPont model: level of returns and how much of it comes from leverage."""
    y = _latest_year(k)
    return combine("dupont", "Returns on capital", [
        metric("roe", "Return on equity", _at(k, "roe", y), "pct"),
        metric("roic", "Return on invested capital", _at(k, "roic", y), "pct"),
        metric("equity_multiplier", "Equity multiplier (leverage)", _at(dupont, "equity_multiplier", y),
               "ratio", 0.5),
    ], "How much profit the business earns on the capital in it, with a penalty for returns "
       "that rely on heavy leverage.")


def rate_capital_allocation(wide: pd.DataFrame, years: int = 5) -> Rating:
    """Capital allocation model: are payouts and total uses covered by cash generated?"""
    uses = models.capital_uses(wide).dropna(subset=["cfo"]).tail(years)
    payouts_to_fcf = None
    if not uses.empty:
        fcf = (uses["cfo"] - uses["capex"]).dropna()
        paid = uses.loc[fcf.index, ["dividends_paid", "buybacks"]]
        if len(fcf) and fcf.sum() > 0 and paid.notna().any().any():
            payouts_to_fcf = float(paid.sum(min_count=1).sum() / fcf.sum())
    summary = models.allocation_summary(wide, years)
    return combine("capital_allocation", "Payout sustainability", [
        metric("payouts_to_fcf_5y", "Dividends + buybacks ÷ free cash flow (5 yrs)", payouts_to_fcf, "multiple"),
        metric("uses_to_cfo_5y", "All cash uses ÷ operating cash (5 yrs)",
               summary["total_share"] if summary else None, "multiple"),
    ], "Whether dividends, buybacks and other uses of cash were covered by the cash the "
       "business generated over the last five years.")


def rate_dilution(k: pd.DataFrame) -> Rating:
    """Share count model. Uses a 2-year span: those years normally come from the same
    filing, so they are on the same basis even after a split or consolidation."""
    change = None
    if "shares_diluted" in k:
        s = k["shares_diluted"].dropna()
        if len(s):
            y = int(s.index[-1])
            for span in (2, 1):
                start = _num(s.get(y - span))
                if start and start > 0:
                    change = (float(s[y]) / start) ** (1 / span) - 1
                    break
    return combine("shares", "Share count trend", [
        metric("share_change_per_year", "Diluted share count change per year", change, "pct"),
    ], "Whether each share's slice of the company is growing (fewer shares) or shrinking "
       "(more shares), over the last two years.")


def rate_dcf(k: pd.DataFrame, share_price: Optional[float]) -> Rating:
    """Simple DCF model at its default inputs: in how many of the 25 sensitivity scenarios
    is the model value above today's price? `share_price` is per ordinary share in the
    reporting currency."""
    share = None
    d = models.dcf_defaults(k)
    price = _num(share_price)
    if d is not None and d["base_fcf"] > 0 and price and price > 0:
        growths = [d["growth"] + step for step in (-0.04, -0.02, 0.0, 0.02, 0.04)]
        discounts = [DCF_DISCOUNT + step for step in (-0.02, -0.01, 0.0, 0.01, 0.02)]
        grid = models.dcf_grid(d["base_fcf"], growths, discounts, DCF_TERMINAL, d["shares"],
                               net_debt=d["net_debt"] or 0.0)
        share = float((grid.values > price).mean())
    return combine("dcf", "Price against the DCF range", [
        metric("dcf_scenarios_above_price", "Scenarios where model value is above today's price",
               share, "share"),
    ], "At the model's default inputs, the share of the 25 growth and discount-rate scenarios "
       "that give a value above today's share price. It says how demanding the price is "
       "relative to this simple model, nothing more.")


def model_ratings(k: pd.DataFrame, wide: pd.DataFrame, share_price: Optional[float] = None) -> Dict[str, Rating]:
    ratings = [rate_profitability(k), rate_cash_conversion(k), rate_returns(k, models.dupont(wide)),
               rate_capital_allocation(wide), rate_dilution(k), rate_dcf(k, share_price)]
    return {r.key: r for r in ratings}


# ---- news climate (context only; never part of the overall grades) ------------------------
def rate_climate(coverage: pd.DataFrame, today: str) -> Rating:
    """From GDELT's automated tone score: the average over the last 30 days and how it
    compares with the 60 days before. `coverage` has columns date (ISO) and tone."""
    recent_avg = trend = None
    if coverage is not None and not coverage.empty and "tone" in coverage:
        tone = coverage.dropna(subset=["tone"])
        end = pd.Timestamp(today)
        dates = pd.to_datetime(tone["date"])
        recent = tone.loc[(dates > end - pd.Timedelta(days=30)) & (dates <= end), "tone"]
        earlier = tone.loc[(dates > end - pd.Timedelta(days=90)) & (dates <= end - pd.Timedelta(days=30)), "tone"]
        if len(recent) >= 7:
            recent_avg = float(recent.mean())
            if len(earlier) >= 14:
                trend = recent_avg - float(earlier.mean())
    return combine("climate", "News climate", [
        metric("tone_30d", "Average tone of coverage, last 30 days", recent_avg, "score"),
        metric("tone_trend", "Change in tone versus the 60 days before", trend, "score", 0.5),
    ], "How positively or negatively the company has recently been written about, from "
       "GDELT's automated tone score. Context only: it is not part of the business grades.")


# ---- overall grades (filing numbers only) ----------------------------------------------
def rate_current(k: pd.DataFrame) -> Rating:
    y = _latest_year(k)
    return combine("current", "Current state", [
        metric("operating_margin", "Operating margin", _at(k, "operating_margin", y), "pct"),
        metric("net_margin", "Net margin", _at(k, "net_margin", y), "pct"),
        metric("fcf_to_net_income", "Free cash flow ÷ net income",
               _ratio(_at(k, "fcf", y), _at(k, "net_income", y)), "multiple"),
        metric("roic", "Return on invested capital", _at(k, "roic", y), "pct"),
        metric("net_debt_to_ebitda", "Net debt ÷ EBITDA", _at(k, "net_debt_to_ebitda", y), "multiple"),
        metric("interest_coverage", "Interest coverage", _at(k, "interest_coverage", y), "multiple"),
    ], "The latest fiscal year: profitability, cash conversion, returns and balance-sheet strength.")


def rate_historical(k: pd.DataFrame, years: int = 10, restated_from: Optional[int] = None) -> Rating:
    y = _latest_year(k)
    h = k.loc[k.index <= y].tail(years) if y is not None else k.iloc[0:0]

    def col(name):
        return h[name].dropna() if name in h else pd.Series(dtype=float)

    fcf, margin, growth, roe = col("fcf"), col("operating_margin"), col("revenue_growth"), col("roe").tail(5)
    cagr5 = _at(k, "revenue_cagr_5y", y)
    note = ""
    if restated_from is not None:
        kept = growth[growth.index != restated_from]      # that year's growth straddles the change
        if _spans_restatement(y, 5, restated_from) or len(kept) != len(growth):
            note = _BASIS_NOTE.format(year=restated_from)
        if _spans_restatement(y, 5, restated_from):
            cagr5 = None
        growth = kept[kept.index > restated_from] if len(kept[kept.index > restated_from]) >= 3 else kept
    return combine("historical", "Historical record", [
        metric("revenue_cagr_5y", "Revenue growth per year, 5 years", cagr5, "pct"),
        metric("fcf_positive_years", "Years with positive free cash flow",
               float((fcf > 0).mean()) if len(fcf) >= 3 else None, "share"),
        metric("margin_volatility", "Operating margin variability (standard deviation)",
               float(margin.std(ddof=0)) if len(margin) >= 3 else None, "pts"),
        metric("roe_average_5y", "Average return on equity, 5 years",
               float(roe.mean()) if len(roe) >= 2 else None, "pct"),
        metric("growth_years", "Years in which revenue grew",
               float((growth > 0).mean()) if len(growth) >= 3 else None, "share"),
    ], f"The track record over up to {years} years: growth, consistency of cash generation, "
       "stability of margins and returns earned.", note)


def rate_future(k: pd.DataFrame, wide: pd.DataFrame, restated_from: Optional[int] = None) -> Rating:
    """Future readiness: recent direction and financial capacity. Not a forecast."""
    y = _latest_year(k)
    cagr3 = _at(k, "revenue_cagr_3y", y)
    note = ""
    if _spans_restatement(y, 3, restated_from):
        cagr3, note = None, _BASIS_NOTE.format(year=restated_from)
    revenue = _at(k, "revenue", y)
    capex = _at(k, "capex", y)
    rnd = _at(wide, "rnd", y) if "rnd" in wide else None
    reinvest = _ratio((capex or 0.0) + (rnd or 0.0), revenue) if capex is not None else None
    fcf = _at(k, "fcf", y)
    dividends = _at(wide, "dividends_paid", y) if "dividends_paid" in wide else None
    return combine("future", "Future readiness", [
        metric("revenue_cagr_3y", "Revenue growth per year, last 3 years", cagr3, "pct"),
        metric("margin_change_3y", "Operating margin change over 3 years", _margin_change(k, y), "pts"),
        metric("reinvestment_rate", "Capex + R&D as a share of revenue", reinvest, "pct"),
        metric("net_debt_to_ebitda", "Net debt ÷ EBITDA (room to borrow)", _at(k, "net_debt_to_ebitda", y), "multiple"),
        metric("dividends_to_fcf", "Dividends ÷ free cash flow (cash left after dividends)",
               _ratio(dividends, fcf), "multiple"),
    ], "Where the business is heading on its recent numbers, and whether it has the financial "
       "room to keep investing. Built from past filings: it is not a forecast.", note)


def overall_ratings(k: pd.DataFrame, wide: pd.DataFrame,
                    restated_from: Optional[int] = None) -> Dict[str, Rating]:
    ratings = [rate_current(k), rate_historical(k, restated_from=restated_from),
               rate_future(k, wide, restated_from)]
    return {r.key: r for r in ratings}


def grade_history(k: pd.DataFrame, wide: pd.DataFrame, restated_from: Optional[int] = None,
                  years: int = 10) -> pd.DataFrame:
    """The three overall scores as they would have been at each past fiscal year end, using
    only the years up to it (with today's figures for those years, including restatements).
    Index: fiscal year; columns: current, historical, future (NaN where not rated)."""
    rows = {}
    last = _latest_year(k)
    if last is None:
        return pd.DataFrame(columns=["current", "historical", "future"])
    for y in [int(v) for v in k.index if v <= last][-years:]:
        cut = restated_from if restated_from is not None and restated_from <= y else None
        got = overall_ratings(k.loc[k.index <= y], wide.loc[wide.index <= y], cut)
        rows[y] = {key: (np.nan if r.score is None else r.score) for key, r in got.items()}
    return pd.DataFrame.from_dict(rows, orient="index")[["current", "historical", "future"]]


COMPARE_ROWS = (("revenue", "Revenue"), ("revenue_growth", "Revenue growth"),
                ("revenue_cagr_5y", "Revenue growth per year, 5 yrs"), ("gross_margin", "Gross margin"),
                ("operating_margin", "Operating margin"), ("net_margin", "Net margin"),
                ("fcf_margin", "Free cash flow margin"), ("roe", "Return on equity"),
                ("roic", "Return on invested capital"), ("net_debt_to_ebitda", "Net debt ÷ EBITDA"),
                ("interest_coverage", "Interest coverage"), ("share_change", "Share count change"))


def comparison(companies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Latest-year KPI values side by side. `companies` maps a display name to its
    kpis.compute output. Index: KPI key; one column per company, plus a 'fiscal_year' row."""
    data = {}
    for name, k in companies.items():
        y = _latest_year(k)
        col = {"fiscal_year": float(y) if y is not None else np.nan}
        for key, _ in COMPARE_ROWS:
            v = _at(k, key, y)
            col[key] = np.nan if v is None else v
        data[name] = col
    return pd.DataFrame(data, index=["fiscal_year"] + [key for key, _ in COMPARE_ROWS])
