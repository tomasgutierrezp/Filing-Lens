"""A plain verdict on how the evidence stacks up, for a long and a short horizon.

Pure functions, no network, no AI. Each horizon has a handful of signals; every signal is
+1 (favourable), 0 (neutral) or -1 (unfavourable) by a fixed rule in RULES, and the verdict
is their average. It summarises evidence already shown elsewhere in the app. It is not a
forecast, it knows nothing about the reader's circumstances, and it is not a
recommendation to buy or sell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import pandas as pd

RULES = {
    "grade_good": 65.0,            # a 0-100 score at or above this is favourable
    "grade_poor": 50.0,            # below this is unfavourable
    "flags_few": 0.10,             # share of red-flag checks flagged: at or below is favourable
    "flags_many": 0.30,            # at or above is unfavourable
    "flags_min_checks": 5,
    "piotroski_good": 0.75,        # share of available tests passed
    "piotroski_poor": 0.40,
    "piotroski_min_tests": 6,
    "growth_gap_poor": 0.05,       # price assumes growth this far above what was delivered
    "return_3m": 0.10,             # 3-month price change beyond +/- this counts
    "trading_days_3m": 63,
    "tone_good": 65.0,             # news-climate score
    "tone_poor": 35.0,
    "verdict_cut": 0.34,           # average signal above +cut is favourable, below -cut unfavourable
    "min_signals": 2,
}
LABELS = {1: "Favourable", 0: "Mixed", -1: "Unfavourable"}


@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    score: Optional[int]      # +1, 0, -1, or None when there is no data
    detail: str


@dataclass(frozen=True)
class Verdict:
    horizon: str              # "long" or "short"
    label: str                # Favourable / Mixed / Unfavourable / Not enough data
    average: Optional[float]
    signals: List[Signal] = field(default_factory=list)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _band(value: Optional[float], good: float, poor: float, higher_is_better: bool = True) -> Optional[int]:
    """+1 at or beyond `good`, -1 beyond `poor`, 0 in between, None without a value."""
    v = _num(value)
    if v is None:
        return None
    if higher_is_better:
        return 1 if v >= good else -1 if v < poor else 0
    return 1 if v <= good else -1 if v >= poor else 0


def _pct(v: float) -> str:
    return f"{'−' if v < 0 else ''}{abs(v):.1%}"


def decide(horizon: str, signals: Sequence[Signal]) -> Verdict:
    scored = [s.score for s in signals if s.score is not None]
    if len(scored) < RULES["min_signals"]:
        return Verdict(horizon, "Not enough data", None, list(signals))
    avg = sum(scored) / len(scored)
    cut = RULES["verdict_cut"]
    return Verdict(horizon, LABELS[1 if avg > cut else -1 if avg < -cut else 0], round(avg, 2), list(signals))


# ---- long-term signals (filings, plus price for the last one) --------------------------
def quality_signal(current: Optional[float], historical: Optional[float]) -> Signal:
    vals = [v for v in (_num(current), _num(historical)) if v is not None]
    avg = sum(vals) / len(vals) if vals else None
    detail = ("The business grades are not available." if avg is None else
              f"Current-state and historical grades average {avg:.0f} out of 100.")
    return Signal("quality", "Business quality", _band(avg, RULES["grade_good"], RULES["grade_poor"]), detail)


def direction_signal(future: Optional[float]) -> Signal:
    v = _num(future)
    detail = ("The future-readiness grade is not available." if v is None else
              f"Future-readiness grade is {v:.0f} out of 100 (recent growth, margins, reinvestment, borrowing room).")
    return Signal("direction", "Direction of travel", _band(v, RULES["grade_good"], RULES["grade_poor"]), detail)


def health_signal(flagged: int, known: int) -> Signal:
    if known < RULES["flags_min_checks"]:
        return Signal("health", "Financial red flags", None, "Too few health checks have data.")
    share = flagged / known
    return Signal("health", "Financial red flags",
                  _band(share, RULES["flags_few"], RULES["flags_many"], higher_is_better=False),
                  f"{flagged} of {known} red-flag checks are flagged.")


def piotroski_signal(score: Optional[int], available: int) -> Signal:
    if score is None or available < RULES["piotroski_min_tests"]:
        return Signal("piotroski", "Fundamentals improving or worsening", None,
                      "Too few Piotroski tests have data.")
    return Signal("piotroski", "Fundamentals improving or worsening",
                  _band(score / available, RULES["piotroski_good"], RULES["piotroski_poor"]),
                  f"Piotroski F-score is {score} of {available}.")


def expectations_signal(implied_growth: Optional[float], delivered_growth: Optional[float]) -> Signal:
    """Does today's price assume more growth than the business has delivered?"""
    imp, got = _num(implied_growth), _num(delivered_growth)
    if imp is None or got is None:
        return Signal("expectations", "What the price assumes", None,
                      "Needs a share price, positive free cash flow and five years of history.")
    gap = imp - got
    score = 1 if gap <= 0 else -1 if gap > RULES["growth_gap_poor"] else 0
    return Signal("expectations", "What the price assumes", score,
                  f"The price assumes free cash flow grows {_pct(imp)} a year; over the last five "
                  f"years it grew {_pct(got)} a year.")


def long_term(current: Optional[float], historical: Optional[float], future: Optional[float],
              flagged: int, known: int, piotroski_score: Optional[int], piotroski_available: int,
              implied_growth: Optional[float], delivered_growth: Optional[float]) -> Verdict:
    return decide("long", [
        quality_signal(current, historical), direction_signal(future), health_signal(flagged, known),
        piotroski_signal(piotroski_score, piotroski_available),
        expectations_signal(implied_growth, delivered_growth)])


# ---- short-term signals (price, latest results, news tone) ------------------------------
def price_trend(prices: Optional[pd.DataFrame]) -> dict:
    """Latest close, 50- and 200-day averages and the 3-month change, from a frame with a
    `close` column in date order. Values are None when there is too little history."""
    out = {"price": None, "ma50": None, "ma200": None, "return_3m": None}
    if prices is None or prices.empty:
        return out
    close = prices["close"].astype(float).reset_index(drop=True)
    out["price"] = float(close.iloc[-1])
    if len(close) >= 50:
        out["ma50"] = float(close.tail(50).mean())
    if len(close) >= 200:
        out["ma200"] = float(close.tail(200).mean())
    n = int(RULES["trading_days_3m"])
    if len(close) > n and close.iloc[-1 - n] > 0:
        out["return_3m"] = float(close.iloc[-1] / close.iloc[-1 - n] - 1)
    return out


def trend_signal(trend: dict) -> Signal:
    price, ma50, ma200 = trend.get("price"), trend.get("ma50"), trend.get("ma200")
    if None in (price, ma50, ma200):
        return Signal("trend", "Price trend", None, "Needs about 200 trading days of prices.")
    above, rising = price > ma200, ma50 > ma200
    score = 1 if above and rising else -1 if not above and not rising else 0
    return Signal("trend", "Price trend", score,
                  f"The price is {'above' if above else 'below'} its 200-day average and the 50-day "
                  f"average is {'above' if rising else 'below'} the 200-day average.")


def momentum_signal(trend: dict) -> Signal:
    r = _num(trend.get("return_3m"))
    if r is None:
        return Signal("momentum", "Last three months", None, "Needs about three months of prices.")
    cut = RULES["return_3m"]
    return Signal("momentum", "Last three months", 1 if r > cut else -1 if r < -cut else 0,
                  f"The share price changed {_pct(r)} over the last three months.")


def latest_period_growth(interim: pd.DataFrame) -> Optional[Tuple[str, float]]:
    """Growth of the most recent interim period over the same period a year earlier.
    `interim` is FactIndex.interim output. Returns (period label, growth) or None."""
    if interim is None or interim.empty:
        return None
    last = interim.iloc[-1]
    prior = interim[(interim["period"] == last["period"]) & (interim["fiscal_year"] == last["fiscal_year"] - 1)]
    if prior.empty:
        return None
    base, now = _num(prior.iloc[-1]["value"]), _num(last["value"])
    if base is None or now is None or base <= 0:
        return None
    return f"{last['period']} FY{int(last['fiscal_year'])}", now / base - 1


def results_signal(revenue: Optional[Tuple[str, float]], profit: Optional[Tuple[str, float]]) -> Signal:
    if revenue is None or profit is None:
        return Signal("results", "Latest reported results", None,
                      "The latest interim period cannot be compared with a year earlier.")
    (period, rev_g), (_, ni_g) = revenue, profit
    score = 1 if rev_g > 0 and ni_g > 0 else -1 if rev_g < 0 and ni_g < 0 else 0
    return Signal("results", "Latest reported results", score,
                  f"In {period}, revenue changed {_pct(rev_g)} and net income {_pct(ni_g)} on a year earlier.")


def tone_signal(climate_score: Optional[float]) -> Signal:
    v = _num(climate_score)
    detail = ("No news-tone data is stored yet (fetch it in the Climate tab)." if v is None else
              f"The news-climate score is {v:.0f} out of 100.")
    return Signal("tone", "News tone", _band(v, RULES["tone_good"], RULES["tone_poor"]), detail)


def short_term(trend: dict, revenue_growth: Optional[Tuple[str, float]],
               profit_growth: Optional[Tuple[str, float]], climate_score: Optional[float]) -> Verdict:
    return decide("short", [trend_signal(trend), momentum_signal(trend),
                            results_signal(revenue_growth, profit_growth), tone_signal(climate_score)])
