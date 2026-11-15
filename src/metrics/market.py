"""Tier 3 — market-dependent multiples. Pure functions; they depend on today's price."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MarketKPI:
    key: str
    label: str
    kind: str      # "currency", "multiple" or "pct"
    formula: str


MARKET_KPIS: Dict[str, MarketKPI] = {k.key: k for k in [
    MarketKPI("market_cap", "Market cap", "currency", "Share price × shares outstanding"),
    MarketKPI("pe", "P/E", "multiple", "Market cap ÷ net income"),
    MarketKPI("ev_ebitda", "EV / EBITDA", "multiple",
              "(Market cap + total debt − cash) ÷ (operating income + D&A)"),
    MarketKPI("fcf_yield", "FCF yield", "pct", "Free cash flow ÷ market cap"),
]}


def _num(v) -> Optional[float]:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def price_in_reporting_currency(price: float, fx_rate: float = 1.0,
                                ordinary_per_adr: float = 1.0) -> float:
    """Price of one ordinary share in the company's reporting currency.
    `fx_rate` converts the trading currency to the reporting currency; an ADR that
    represents n ordinary shares is divided by n."""
    if ordinary_per_adr <= 0:
        raise ValueError("ordinary_per_adr must be positive")
    return price * fx_rate / ordinary_per_adr


def market_multiples(share_price: Optional[float], shares_outstanding: Optional[float],
                     net_income: Optional[float] = None, net_debt: Optional[float] = None,
                     ebitda: Optional[float] = None, fcf: Optional[float] = None) -> Dict[str, Optional[float]]:
    """All Tier 3 numbers. `share_price` must already be per ordinary share in the reporting
    currency. A value is None when an input is missing or the ratio is not meaningful
    (negative earnings, EBITDA or FCF)."""
    price, shares = _num(share_price), _num(shares_outstanding)
    out: Dict[str, Optional[float]] = {"market_cap": None, "pe": None, "enterprise_value": None,
                                       "ev_ebitda": None, "fcf_yield": None}
    if price is None or shares is None or price <= 0 or shares <= 0:
        return out
    cap = price * shares
    out["market_cap"] = cap
    ni, nd, eb, f = _num(net_income), _num(net_debt), _num(ebitda), _num(fcf)
    if ni is not None and ni > 0:
        out["pe"] = cap / ni
    if nd is not None:
        out["enterprise_value"] = cap + nd
        if eb is not None and eb > 0:
            out["ev_ebitda"] = (cap + nd) / eb
    if f is not None and f > 0:
        out["fcf_yield"] = f / cap
    return out
