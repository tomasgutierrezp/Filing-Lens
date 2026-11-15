import pandas as pd

from src.viz import charts


def _long():
    return pd.DataFrame([
        {"concept": "revenue", "fiscal_year": 2024, "form": "10-K", "end": "2024-12-31",
         "filed": "2025-02-15", "tags": "Revenues", "restated": False},
        {"concept": "cfo", "fiscal_year": 2024, "form": "10-K/A", "end": "2024-12-31",
         "filed": "2025-04-01", "tags": "NetCashProvidedByUsedInOperatingActivities", "restated": True},
    ])


def test_provenance_lines():
    lines = charts.provenance_lines(_long(), ["revenue", "cfo", "capex"], 2024)
    assert lines[0] == "Revenue: 10-K · period ending 2024-12-31 · filed 2025-02-15 · tag Revenues"
    assert lines[1].endswith("tag NetCashProvidedByUsedInOperatingActivities · restated")
    assert lines[2] == "Capital expenditure: not reported"


def test_short_provenance_and_table():
    long = _long().assign(label=["Revenue", "Cash from operations"], value=[10.0, 3.0],
                          unit="USD", url=["u1", "u2"])
    assert charts.short_provenance(long, ["revenue", "cfo"], 2024) == [
        "10-K · filed 2025-02-15", "10-K/A · filed 2025-04-01"]
    assert charts.short_provenance(long, ["revenue"], 2023) == []
    t = charts.provenance_table(long, ["revenue", "cfo"], [2024])
    assert list(t["Input"]) == ["Cash from operations", "Revenue"]
    assert list(t["Year"]) == ["FY2024", "FY2024"] and list(t["Filing"]) == ["u2", "u1"]
    assert list(t["Restated"]) == [True, False]


def test_sparkline_carries_provenance_in_hover():
    fig = charts.sparkline([2023, 2024], [1.0, 2.0], ["USD 1m", "USD 2m"], ["a", "b<br>c"])
    trace = fig.data[0]
    assert list(trace.x) == [2023, 2024] and list(trace.y) == [1.0, 2.0]
    assert trace.text[1] == "<b>FY2024: USD 2m</b><br>b<br>c"
    assert fig.layout.xaxis.visible is False


def test_stacked_share_bars_skips_empty_series():
    shares = pd.DataFrame({"a": [0.5, 0.6], "b": [float("nan")] * 2, "c": [0.5, 0.4]}, index=[2023, 2024])
    fig = charts.stacked_share_bars(shares, {"a": "Alpha", "c": "Gamma"}, "Share")
    assert [t.name for t in fig.data] == ["Alpha", "Gamma"]
    assert list(fig.data[0].x) == ["FY2023", "FY2024"] and fig.layout.barmode == "relative"


def test_waterfall_measures_and_scale():
    steps = [{"label": "Revenue", "value": 100e6, "kind": "total"},
             {"label": "Costs", "value": -70e6, "kind": "delta"},
             {"label": "Profit", "value": 30e6, "kind": "total"}]
    trace = charts.waterfall(steps, "USD millions").data[0]
    assert list(trace.measure) == ["absolute", "relative", "absolute"]
    assert list(trace.y) == [100, -70, 30] and list(trace.text) == ["100", "-70", "30"]


def test_line_chart_and_heatmap():
    fig = charts.line_chart([2023, 2024], [10.0, 9.0], ["10m", "9m"], "Shares", "Shares", ["10-K", ""])
    assert fig.data[0].text[0] == "<b>FY2023: 10m</b><br>10-K" and fig.data[0].text[1] == "<b>FY2024: 9m</b>"
    grid = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=[0.08, 0.10], columns=[0.0, 0.05])
    heat = charts.sensitivity_heatmap(grid, "USD").data[0]
    assert list(heat.x) == ["0.0%", "5.0%"] and list(heat.y) == ["8.0%", "10.0%"]


def test_climate_figure_with_and_without_data():
    prices = pd.DataFrame({"date": ["2026-11-01", "2026-11-02"], "close": [10.0, 12.0]})
    cov = pd.DataFrame({"date": ["2026-11-01", "2026-11-02"], "volume": [0.1, 0.2], "tone": [-1.0, None]})
    ev = pd.DataFrame({"date": ["2026-11-02"], "form": ["8-K"], "category": ["Results"],
                       "label": ["Results of operations"], "url": ["u"]})
    fig = charts.climate_figure(prices, cov, ev)
    assert [t.name for t in fig.data] == ["Price", "Coverage volume", "Tone", "Results"]
    assert fig.data[3].y[0] == 12.0 and "Results of operations" in fig.data[3].text[0]
    assert len(fig.data[2].x) == 1                       # missing tone points are dropped
    assert any(s.type == "line" and s.x0 == "2026-11-02" for s in fig.layout.shapes)
    empty = charts.climate_figure(None, cov.iloc[0:0], ev.iloc[0:0])
    assert len(empty.data) == 0
    price_only = charts.climate_figure(prices, cov.iloc[0:0], ev)     # no coverage: one panel
    assert [t.name for t in price_only.data] == ["Price", "Results"]
    assert [a.text for a in price_only.layout.annotations] == ["Share price"]


def test_multi_line():
    frame = pd.DataFrame({"current": [50.0, 60.0], "future": [None, 40.0]}, index=[2025, 2026])
    fig = charts.multi_line(frame, {"current": "Current state"}, "Score", y_range=[0, 100])
    assert [t.name for t in fig.data] == ["Current state", "future"]
    assert list(fig.data[0].x) == ["FY2025", "FY2026"] and tuple(fig.layout.yaxis.range) == (0, 100)
