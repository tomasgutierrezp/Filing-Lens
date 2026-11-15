"""Rule-based explanations: deterministic templates driven by computed numbers. No AI.

Every function returns a sentence, or None when the numbers needed are missing.
Sentences only state what the numbers show.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.metrics.kpis import KPIS

THRESHOLDS = {
    "points": 0.01,    # a margin / return change is "meaningful" above 1 percentage point
    "relative": 0.05,  # a level change is "meaningful" above 5%
    "multiple": 0.1,   # a change in a multiple (e.g. 1.2x) is "meaningful" above 0.1x
}
MINUS = "−"


# ---- small helpers --------------------------------------------------------------
def _ok(*vals) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


def money(v: float, currency: str) -> str:
    """'USD 478.8bn', 'EUR 836m', '−USD 1.2bn'."""
    a = abs(v)
    if a >= 1e12:
        body = f"{a / 1e12:,.2f}tn"
    elif a >= 1e9:
        body = f"{a / 1e9:,.1f}bn"
    elif a >= 1e6:
        body = f"{a / 1e6:,.0f}m"
    else:
        body = f"{a:,.0f}"
    return f"{MINUS if v < 0 else ''}{currency} {body}"


def count(v: float) -> str:
    """Share counts: '14.72bn', '2,195m'."""
    return f"{v / 1e9:,.2f}bn" if abs(v) >= 1e10 else f"{v / 1e6:,.0f}m"


def pct(v: float, digits: int = 1) -> str:
    return f"{MINUS if v < 0 else ''}{abs(v):.{digits}%}"


def multiple(v: float) -> str:
    return f"{MINUS if v < 0 else ''}{abs(v):.1f}×"


def format_kpi(key: str, v: Optional[float], currency: str) -> str:
    if not _ok(v):
        return "Not available"
    kind = KPIS[key].kind
    if kind == "currency":
        return money(v, currency)
    if kind == "shares":
        return count(v)
    return pct(v) if kind == "pct" else multiple(v)


# ---- templates --------------------------------------------------------------------
def growth(label: str, prior: float, current: float, currency: str) -> Optional[str]:
    """Level change, e.g. revenue or net income."""
    if not _ok(prior, current):
        return None
    if prior <= 0 or current <= 0:
        return f"{label} was {money(current, currency)}, compared with {money(prior, currency)} the year before."
    change = current / prior - 1
    if abs(change) <= THRESHOLDS["relative"]:
        return (f"{label} was little changed at {money(current, currency)} "
                f"({pct(change)} on the year before).")
    verb = "rose" if change > 0 else "fell"
    return f"{label} {verb} {pct(abs(change))} to {money(current, currency)} from {money(prior, currency)}."


def cagr_note(label: str, rate: float, years: int) -> Optional[str]:
    if not _ok(rate):
        return None
    verb = "grew" if rate >= 0 else "shrank"
    return f"Over {years} years {label.lower()} {verb} {pct(abs(rate))} a year on average."


def points_change(label: str, prior: float, current: float) -> Optional[str]:
    """Change in a percentage metric, in percentage points."""
    if not _ok(prior, current):
        return None
    diff = current - prior
    if abs(diff) <= THRESHOLDS["points"]:
        return f"{label} was steady at {pct(current)} ({pct(prior)} the year before)."
    verb = "rose" if diff > 0 else "fell"
    return f"{label} {verb} {abs(diff) * 100:.1f} pts to {pct(current)} from {pct(prior)}."


def margin_change(label: str, prior: float, current: float, revenue_growth: float,
                  cost_growth: float, cost_label: str = "operating costs") -> Optional[str]:
    if not _ok(prior, current):
        return None
    if abs(current - prior) <= THRESHOLDS["points"]:
        first = f"{label} was steady at {pct(current)} ({pct(prior)} the year before)."
    else:
        first = f"{label} {'rose' if current > prior else 'fell'} from {pct(prior)} to {pct(current)}."
    if not _ok(revenue_growth, cost_growth):
        return first

    def moved(g):
        return f"{'grew' if g >= 0 else 'fell'} {pct(abs(g))}"

    return f"{first} Revenue {moved(revenue_growth)} while {cost_label} {moved(cost_growth)}."


def fcf_vs_net_income(fcf: float, net_income: float, cfo: float, capex: float,
                      d_and_a: float, currency: str) -> Optional[str]:
    if not _ok(fcf, net_income):
        return None
    if net_income <= 0:
        return (f"Free cash flow was {money(fcf, currency)} while net income was "
                f"{money(net_income, currency)}, so a ratio between them is not meaningful.")
    ratio = fcf / net_income
    amount = "more" if ratio > 1 else "less"
    first = f"Free cash flow was {ratio:.1f}× net income — {amount} of reported profit turned into cash"
    if abs(ratio - 1) <= THRESHOLDS["relative"]:
        return f"Free cash flow was {ratio:.1f}× net income — reported profit and cash generated were close."
    if not _ok(cfo, capex, d_and_a):
        return first + "."
    # FCF − net income = D&A (added back) + working capital and other items − capex
    drivers = {"depreciation and amortization": d_and_a,
               "working capital and other non-cash items": cfo - net_income - d_and_a,
               "capital expenditure": -capex}
    direction = 1 if ratio > 1 else -1
    aligned = {k: v for k, v in drivers.items() if v * direction > 0} or drivers
    main = max(aligned, key=lambda k: abs(aligned[k]))
    return f"{first}, mainly due to {main} ({money(abs(drivers[main]), currency)})."


def dilution(start: float, end: float, years: int) -> Optional[str]:
    if not _ok(start, end) or start <= 0 or years < 1:
        return None
    change = end / start - 1
    span = f"{years} year{'s' if years != 1 else ''}"
    if abs(change) < 0.001:
        return f"Diluted share count was unchanged over {span}."
    verb, owns = ("fell", "more") if change < 0 else ("rose", "less")
    return (f"Diluted share count {verb} {pct(abs(change))} over {span}, "
            f"so each share owns {owns} of the company.")


def leverage(net_debt: float, ebitda: float, currency: str) -> Optional[str]:
    if not _ok(net_debt, ebitda):
        return None
    if net_debt <= 0:
        return f"Cash exceeds debt by {money(abs(net_debt), currency)} (a net cash position)."
    if ebitda <= 0:
        return f"Net debt is {money(net_debt, currency)} and EBITDA is not positive, so the ratio is not meaningful."
    return (f"Net debt of {money(net_debt, currency)} equals {net_debt / ebitda:.1f} years "
            f"of EBITDA ({money(ebitda, currency)}).")


def coverage(operating_income: float, interest_expense: float, prior_ratio: float = float("nan")) -> Optional[str]:
    if not _ok(operating_income, interest_expense) or interest_expense <= 0:
        return None
    ratio = operating_income / interest_expense
    first = f"Operating income covers interest expense {ratio:.1f} times"
    if not _ok(prior_ratio) or abs(ratio - prior_ratio) <= THRESHOLDS["multiple"]:
        return first + "."
    return f"{first}, {'up' if ratio > prior_ratio else 'down'} from {prior_ratio:.1f} the year before."


# ---- dashboard dispatch -------------------------------------------------------------
def _get(k: pd.DataFrame, year: int, col: str) -> float:
    return float(k[col].get(year, float("nan"))) if col in k else float("nan")


def card_sentence(key: str, k: pd.DataFrame, year: int, currency: str) -> Optional[str]:
    """The one sentence shown on a KPI card for `year`. `k` is the output of kpis.compute."""
    sentence = _comparison_sentence(key, k, year, currency)
    value = _get(k, year, key)
    if sentence is None and _ok(value):   # nothing to compare with: state the level only
        sentence = f"{KPIS[key].label} was {format_kpi(key, value, currency)}."
    return sentence


def _comparison_sentence(key: str, k: pd.DataFrame, year: int, currency: str) -> Optional[str]:
    cur = lambda c: _get(k, year, c)       # noqa: E731
    pri = lambda c: _get(k, year - 1, c)   # noqa: E731
    label = KPIS[key].label
    if key == "revenue":
        parts = [growth(label, pri(key), cur(key), currency)]
        for n in (5, 3):
            note = cagr_note(label, cur(f"revenue_cagr_{n}y"), n)
            if note:
                parts.append(note)
                break
        return " ".join(p for p in parts if p) or None
    if key in ("operating_income", "net_income", "cfo", "capex", "cash", "total_debt"):
        return growth(label, pri(key), cur(key), currency)
    if key == "operating_margin":
        costs = k["revenue"] - k["operating_income"]
        prior_costs, cur_costs = costs.get(year - 1, float("nan")), costs.get(year, float("nan"))
        cost_growth = cur_costs / prior_costs - 1 if _ok(prior_costs, cur_costs) and prior_costs > 0 else float("nan")
        return margin_change(label, pri(key), cur(key), cur("revenue_growth"), cost_growth)
    if key in ("gross_margin", "net_margin", "fcf_margin", "roe", "roic"):
        return points_change(label, pri(key), cur(key))
    if key == "fcf":
        return fcf_vs_net_income(cur("fcf"), cur("net_income"), cur("cfo"), cur("capex"),
                                 cur("d_and_a"), currency)
    if key == "net_debt_to_ebitda":
        return leverage(cur("net_debt"), cur("ebitda"), currency)
    if key == "interest_coverage":
        return coverage(cur("operating_income"), cur("interest_expense"), pri("interest_coverage"))
    if key == "shares_diluted":
        shares = k["shares_diluted"].dropna()
        shares = shares[shares.index <= year]
        for n in (5, 4, 3, 2, 1):
            if year - n in shares.index and year in shares.index:
                return dilution(shares[year - n], shares[year], n)
    return None


# ---- model explanations ("What this means") -----------------------------------------
def common_size_note(current: pd.Series, prior: Optional[pd.Series], year: int,
                     labels: dict) -> Optional[str]:
    """`current` / `prior` are rows of models.common_size (fractions of revenue)."""
    net = current.get("net_income")
    if not _ok(net):
        return None
    costs = {k: float(v) for k, v in current.items() if k != "net_income" and _ok(v)}
    went = sorted(((k, v) for k, v in costs.items() if v >= 0.005), key=lambda kv: -kv[1])
    added = [(k, v) for k, v in costs.items() if v <= -0.005]
    parts = [f"{v * 100:.1f} went to {labels[k].lower()}" for k, v in went]
    text = f"In FY{year}, of every 100 of revenue, " + ", ".join(parts)
    text += f", leaving {net * 100:.1f} as net income" if parts else f"{net * 100:.1f} was net income"
    if prior is not None and _ok(prior.get("net_income")):
        text += f" ({prior['net_income'] * 100:.1f} the year before)"
    text += "."
    for k, v in added:
        text += f" {labels[k]} added {abs(v) * 100:.1f}."
    return text


def bridge_note(revenue: float, fcf: float, currency: str) -> Optional[str]:
    if not _ok(revenue, fcf) or revenue <= 0:
        return None
    if fcf <= 0:
        return (f"Revenue of {money(revenue, currency)} did not turn into free cash flow: "
                f"it was {money(fcf, currency)}.")
    return (f"Of {money(revenue, currency)} of revenue, {money(fcf, currency)} "
            f"({pct(fcf / revenue)}) ended up as free cash flow.")


def dupont_note(current: pd.Series, prior: Optional[pd.Series], driver: Optional[str],
                labels: dict) -> Optional[str]:
    """`current` / `prior` are rows of models.dupont; `driver` comes from models.dupont_driver."""
    if not _ok(current.get("roe")):
        return None
    formula = (f"net margin {pct(current['net_margin'])} × asset turnover "
               f"{current['asset_turnover']:.2f} × equity multiplier {current['equity_multiplier']:.2f}")
    if prior is None or not _ok(prior.get("roe")):
        return f"ROE of {pct(current['roe'])} = {formula}."
    diff = current["roe"] - prior["roe"]
    if abs(diff) <= THRESHOLDS["points"]:
        return f"ROE was steady at {pct(current['roe'])} ({pct(prior['roe'])} the year before) = {formula}."
    first = (f"ROE changed by {'+' if diff > 0 else MINUS}{abs(diff) * 100:.1f} pts "
             f"({pct(prior['roe'])} to {pct(current['roe'])})")
    if driver is None:
        return first + "."

    def show(row):
        return pct(row[driver]) if driver == "net_margin" else f"{row[driver]:.2f}"

    return f"{first}, driven mostly by {labels[driver].lower()} ({show(prior)} to {show(current)})."


def capital_allocation_note(summary: Optional[dict], labels: dict) -> Optional[str]:
    """`summary` comes from models.allocation_summary."""
    if not summary:
        return None
    n = summary["years"]
    span = f"the last {n} years" if n > 1 else "the last year"
    text = (f"Over {span}, the largest use of operating cash was "
            f"{labels[summary['top_use']].lower()} ({pct(summary['top_share'], 0)}).")
    total = summary["total_share"]
    if total > 1 + THRESHOLDS["relative"]:
        text += (f" The uses shown add up to {pct(total, 0)} of operating cash, so part was "
                 "funded from cash on hand, asset sales or borrowing.")
    else:
        text += f" The uses shown add up to {pct(total, 0)} of operating cash."
    return text


def dcf_note(low: float, high: float, central: float, currency: str) -> Optional[str]:
    if not _ok(low, high, central):
        return None

    def per_share(v):
        return f"{MINUS if v < 0 else ''}{currency} {abs(v):,.2f}"

    return (f"Across this range of assumptions the model gives {per_share(low)} to "
            f"{per_share(high)} per share ({per_share(central)} at your inputs). "
            "That spread is the point: small changes in assumptions move the answer a lot. "
            "It is arithmetic on assumptions, not a forecast or a price target.")


# ---- Tier 3 (market-dependent) --------------------------------------------------------
def market_sentence(key: str, m: dict, year: int, currency: str, net_income: float = float("nan"),
                    ebitda: float = float("nan"), fcf: float = float("nan")) -> Optional[str]:
    """One sentence per Tier 3 card. `m` is the output of market.market_multiples.
    States the arithmetic only — never whether the price is high or low."""
    cap = m.get("market_cap")
    if not _ok(cap):
        return None
    if key == "market_cap":
        return f"At today's price the market values the company's shares at {money(cap, currency)}."
    if key == "pe":
        if not _ok(net_income):
            return None
        if net_income <= 0:
            return f"FY{year} net income was {money(net_income, currency)}, so a P/E ratio is not meaningful."
        return f"The market value is {m['pe']:.1f} times FY{year} net income ({money(net_income, currency)})."
    if key == "ev_ebitda":
        ev = m.get("enterprise_value")
        if not _ok(ev, ebitda):
            return None
        if ebitda <= 0:
            return f"FY{year} EBITDA was {money(ebitda, currency)}, so EV / EBITDA is not meaningful."
        return (f"Enterprise value (market value plus net debt, {money(ev, currency)}) is "
                f"{m['ev_ebitda']:.1f} times FY{year} EBITDA ({money(ebitda, currency)}).")
    if key == "fcf_yield":
        if not _ok(fcf):
            return None
        if fcf <= 0:
            return f"FY{year} free cash flow was {money(fcf, currency)}, so an FCF yield is not meaningful."
        return (f"FY{year} free cash flow ({money(fcf, currency)}) equals {pct(m['fcf_yield'])} "
                "of the market value.")
    return None


# ---- ratings ------------------------------------------------------------------------------
def format_metric(value: Optional[float], kind: str) -> str:
    """Display a rating input: 'pct' 12.3%, 'pts' 1.5 pts, 'multiple' 1.2×, 'ratio' 2.10,
    'share' 80%, 'score' +0.42."""
    if not _ok(value):
        return "Not reported"
    if kind == "pct":
        return pct(value)
    if kind == "pts":
        return f"{MINUS if value < 0 else ''}{abs(value) * 100:.1f} pts"
    if kind == "multiple":
        return multiple(value)
    if kind == "share":
        return pct(value, 0)
    if kind == "score":
        return f"{'+' if value > 0 else MINUS if value < 0 else ''}{abs(value):.2f}"
    return f"{value:.2f}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:2].upper() != text[:2] or len(text) < 2 else text


def rating_sentence(rating, include_grade: bool = True) -> str:
    """One sentence for a ratings.Rating: the grade, and the strongest and weakest inputs."""
    scored = [m for m in rating.metrics if m.score is not None]
    if rating.score is None:
        missing = [_lower_first(m.label) for m in rating.metrics if m.score is None]
        return "Not rated: the data needed is missing (" + ", ".join(missing) + ")."
    head = f"Grade {rating.grade} ({rating.score:.0f} out of 100)." if include_grade else ""
    if len(scored) == 1:
        m = scored[0]
        return f"{head} {m.label}: {format_metric(m.value, m.kind)}.".strip()
    best, worst = max(scored, key=lambda m: m.score), min(scored, key=lambda m: m.score)
    if best.score - worst.score < 1:
        return f"{head} All inputs score about the same.".strip()
    return (f"{head} Strongest input: {_lower_first(best.label)} ({format_metric(best.value, best.kind)}). "
            f"Weakest: {_lower_first(worst.label)} ({format_metric(worst.value, worst.kind)}).").strip()


def reverse_dcf_note(implied: Optional[float], historical: Optional[float], discount: float,
                     terminal_growth: float, years: int = 5) -> Optional[str]:
    """What today's price assumes, next to what the company has delivered."""
    if not _ok(implied):
        return None
    text = (f"Today's price matches this model if free cash flow grows {pct(implied)} a year for "
            f"{years} years and {pct(terminal_growth)} after that, at a {pct(discount)} discount rate.")
    if _ok(historical):
        text += f" Over the last {years} years free cash flow actually grew {pct(historical)} a year."
    return text


# ---- verdicts -------------------------------------------------------------------------------
def verdict_sentence(v) -> str:
    """Counts behind a verdict.Verdict, e.g. '4 of 5 signals are favourable, ...'."""
    scored = [s.score for s in v.signals if s.score is not None]
    missing = len(v.signals) - len(scored)
    if v.average is None:
        return "Fewer than 2 signals have data, so there is no verdict."
    good, flat, bad = scored.count(1), scored.count(0), scored.count(-1)

    def n(count):
        return "none" if count == 0 else str(count)

    text = (f"{good} of {len(scored)} signals {'is' if good == 1 else 'are'} favourable, "
            f"{n(flat)} neutral and {n(bad)} unfavourable.")
    if missing:
        text += f" {missing} signal{'s have' if missing != 1 else ' has'} no data."
    return text
