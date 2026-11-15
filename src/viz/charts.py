"""Plotly figure builders. Provenance travels in the hover text."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.edgar.concept_map import CONCEPTS

TIER_COLORS = {1: "#3b82f6", 2: "#10b981", 3: "#f59e0b"}


def provenance_lines(annual_long: pd.DataFrame, inputs: Iterable[str], year: int) -> List[str]:
    """One line per input concept: form, period end, filing date and XBRL tag."""
    rows = annual_long[annual_long["fiscal_year"] == year].set_index("concept")
    lines = []
    for key in inputs:
        if key not in rows.index:
            lines.append(f"{CONCEPTS[key].label}: not reported")
            continue
        r = rows.loc[key]
        restated = " · restated" if bool(r["restated"]) else ""
        lines.append(f"{CONCEPTS[key].label}: {r['form']} · period ending {r['end']} · "
                     f"filed {r['filed']} · tag {r['tags']}{restated}")
    return lines


def provenance_table(annual_long: pd.DataFrame, inputs: Iterable[str], years: Iterable[int]) -> pd.DataFrame:
    """Every filed input behind a KPI: one row per fiscal year and input concept, newest first."""
    rows = annual_long[annual_long["concept"].isin(list(inputs))
                       & annual_long["fiscal_year"].isin(list(years))]
    rows = rows.sort_values(["fiscal_year", "label"], ascending=[False, True])
    return pd.DataFrame({
        "Year": ["FY%d" % y for y in rows["fiscal_year"]], "Input": rows["label"].values,
        "Value": rows["value"].values, "Unit": rows["unit"].values, "Form": rows["form"].values,
        "Period end": rows["end"].values, "Filed": rows["filed"].values,
        "XBRL tag": rows["tags"].values, "Restated": rows["restated"].astype(bool).values,
        "Filing": rows["url"].values})


def short_provenance(annual_long: pd.DataFrame, inputs: Iterable[str], year: int) -> List[str]:
    """Distinct 'form · filed date' lines for a year — short enough for a small chart's hover."""
    rows = annual_long[(annual_long["fiscal_year"] == year) & annual_long["concept"].isin(list(inputs))]
    return list(dict.fromkeys(f"{r.form} · filed {r.filed}" for r in rows.itertuples()))


def sparkline(years: Sequence[int], values: Sequence[float], labels: Sequence[str],
              hover: Sequence[str], color: str = "#3b82f6", height: int = 110) -> go.Figure:
    """Small line chart with no axes. `labels` are the formatted values; `hover` is extra
    HTML (formula, provenance) shown under them."""
    text = [f"<b>FY{y}: {lab}</b><br>{h}" for y, lab, h in zip(years, labels, hover)]
    fig = go.Figure(go.Scatter(
        x=list(years), y=list(values), mode="lines+markers", text=text,
        hovertemplate="%{text}<extra></extra>", line=dict(color=color, width=2),
        marker=dict(size=5, color=color), connectgaps=False))
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=6, b=6), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(align="left", font_size=12),
        xaxis=dict(visible=False, fixedrange=True), yaxis=dict(visible=False, fixedrange=True))
    return fig


# ---- model charts -----------------------------------------------------------------------
PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#ef4444", "#64748b", "#14b8a6"]
_BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             margin=dict(l=10, r=10, t=70, b=10), hoverlabel=dict(align="left"),
             legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))


def stacked_share_bars(shares: pd.DataFrame, labels: dict, y_title: str, height: int = 420) -> go.Figure:
    """Stacked bars by fiscal year. `shares` holds fractions (columns = series)."""
    fig = go.Figure()
    x = [f"FY{y}" for y in shares.index]
    for i, col in enumerate(c for c in shares.columns if shares[c].notna().any()):
        fig.add_bar(name=labels.get(col, col), x=x, y=shares[col], marker_color=PALETTE[i % len(PALETTE)],
                    hovertemplate="%{x} · " + labels.get(col, col) + ": %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="relative", height=height, **_BASE)
    fig.update_yaxes(tickformat=".0%", title=y_title)
    return fig


def waterfall(steps: Sequence[dict], unit_label: str, scale: float = 1e6, height: int = 440) -> go.Figure:
    """Bridge chart. Each step is {'label', 'value', 'kind'} as returned by models.fcf_bridge."""
    measures = ["absolute" if s["kind"] == "total" else "relative" for s in steps]
    values = [s["value"] / scale for s in steps]
    fig = go.Figure(go.Waterfall(
        x=[s["label"] for s in steps], y=values, measure=measures,
        text=[f"{v:,.0f}" for v in values], textposition="outside",
        increasing=dict(marker=dict(color="#10b981")), decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#3b82f6")), connector=dict(line=dict(color="#64748b", width=1)),
        hovertemplate="%{x}: %{y:,.0f} " + unit_label + "<extra></extra>"))
    fig.update_layout(height=height, showlegend=False, **{k: v for k, v in _BASE.items() if k != "legend"})
    fig.update_yaxes(title=unit_label)
    return fig


def line_chart(years: Sequence[int], values: Sequence[float], labels: Sequence[str], name: str,
               y_title: str, hover: Optional[Sequence[str]] = None, height: int = 360) -> go.Figure:
    """One line by fiscal year. `labels` are formatted values, `hover` extra provenance text."""
    hover = hover or [""] * len(years)
    text = [f"<b>FY{y}: {lab}</b>" + (f"<br>{h}" if h else "") for y, lab, h in zip(years, labels, hover)]
    fig = go.Figure(go.Scatter(x=[f"FY{y}" for y in years], y=list(values), mode="lines+markers",
                               name=name, text=text, hovertemplate="%{text}<extra></extra>",
                               line=dict(color=PALETTE[0], width=2)))
    fig.update_layout(height=height, showlegend=False, **{k: v for k, v in _BASE.items() if k != "legend"})
    fig.update_yaxes(title=y_title, rangemode="tozero")
    return fig


def sensitivity_heatmap(grid: pd.DataFrame, currency: str, height: int = 380) -> go.Figure:
    """DCF value per share. Rows: discount rate, columns: growth rate (both fractions)."""
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{g:.1%}" for g in grid.columns], y=[f"{d:.1%}" for d in grid.index],
        colorscale="Blues", texttemplate="%{z:,.0f}", showscale=False,
        hovertemplate="FCF growth %{x} · discount rate %{y}<br>" + currency
                      + " %{z:,.2f} per share<extra></extra>"))
    fig.update_layout(height=height, **{k: v for k, v in _BASE.items() if k != "legend"})
    fig.update_xaxes(title="FCF growth for 5 years", type="category")
    fig.update_yaxes(title="Discount rate", type="category")
    return fig


# ---- climate panel ------------------------------------------------------------------------
EVENT_COLORS = {"Results": "#3b82f6", "Leadership": "#a855f7", "Deals": "#f59e0b",
                "Shareholder matters": "#14b8a6", "Disclosure": "#ef4444",
                "Routine": "#64748b", "Other": "#94a3b8"}


def climate_figure(prices: Optional[pd.DataFrame], coverage: pd.DataFrame, events: pd.DataFrame,
                   price_currency: str = "USD", height: int = 640) -> go.Figure:
    """Price line on top, coverage volume and tone below on the same time axis, with a
    vertical marker for each filing event. Any of the three inputs may be empty."""
    has_coverage = not coverage.empty
    if has_coverage:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=("Share price", "News coverage volume",
                                            "Average tone of coverage"))
    else:   # no coverage stored yet: price and event markers only
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Share price",))
        height = int(height * 0.6)
    top = 1.0
    if prices is not None and not prices.empty:
        top = float(prices["close"].max())
        fig.add_trace(go.Scatter(x=prices["date"], y=prices["close"], mode="lines", name="Price",
                                 line=dict(color=PALETTE[0], width=2),
                                 hovertemplate="%{x}: " + price_currency + " %{y:,.2f}<extra></extra>"),
                      row=1, col=1)
    if has_coverage:
        vol = coverage.dropna(subset=["volume"])
        tone = coverage.dropna(subset=["tone"])
        fig.add_trace(go.Scatter(x=vol["date"], y=vol["volume"], mode="lines", fill="tozeroy",
                                 name="Coverage volume", line=dict(color="#64748b", width=1),
                                 hovertemplate="%{x}: %{y:.4f}% of monitored news<extra></extra>"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=tone["date"], y=tone["tone"], mode="lines", name="Tone",
                                 line=dict(color="#f59e0b", width=1.5),
                                 hovertemplate="%{x}: tone %{y:.2f}<extra></extra>"), row=3, col=1)
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"), row=3, col=1)
    for category, group in events.groupby("category", sort=False):
        color = EVENT_COLORS.get(category, "#94a3b8")
        for day in group["date"].unique():
            fig.add_shape(type="line", xref="x", yref="paper", x0=day, x1=day, y0=0, y1=1,
                          line=dict(color=color, width=1, dash="dot"), opacity=0.6)
        fig.add_trace(go.Scatter(
            x=group["date"], y=[top] * len(group), mode="markers", name=category,
            marker=dict(symbol="diamond", size=9, color=color),
            text=[f"{r.form} filed {r.date}<br>{r.label}" for r in group.itertuples()],
            hovertemplate="<b>%{text}</b><extra>" + category + "</extra>"), row=1, col=1)
    fig.update_layout(height=height, **_BASE)
    fig.update_layout(margin=dict(l=10, r=10, t=100, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0))
    fig.update_yaxes(title=price_currency, row=1, col=1)
    if has_coverage:
        fig.update_yaxes(title="% of news", row=2, col=1)
        fig.update_yaxes(title="tone", row=3, col=1)
    return fig


def multi_line(frame: pd.DataFrame, labels: dict, y_title: str, y_range=None, height: int = 340) -> go.Figure:
    """Several lines by fiscal year (index = year, one column per line)."""
    fig = go.Figure()
    x = [f"FY{y}" for y in frame.index]
    for i, col in enumerate(frame.columns):
        name = labels.get(col, col)
        fig.add_trace(go.Scatter(x=x, y=frame[col], mode="lines+markers", name=name,
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                 hovertemplate="%{x} · " + name + ": %{y:.0f}<extra></extra>"))
    fig.update_layout(height=height, **_BASE)
    fig.update_yaxes(title=y_title, range=y_range)
    return fig
