"""Filing Lens — Streamlit entry point."""
from __future__ import annotations

import calendar
import datetime as dt
import math
from typing import Optional

import pandas as pd
import streamlit as st

from src import config
from src.climate import events as climate_events, gdelt, headlines
from src.edgar import filings, tickers
from src.edgar.client import EdgarError
from src.edgar.facts import load_facts, shares_outstanding
from src.normalize.periods import FactIndex
from src.normalize.statements import STATEMENTS, available_years, build_statement
from src.explain import rules
from src.market import prices
from src.cache import watchlist
from src.metrics import health, kpis, market, models, ratings
from src.metrics import verdict as verdicts
from src.normalize import currency as fx
from src.viz import charts

st.set_page_config(page_title="Filing Lens", page_icon="📑", layout="wide")

TAXONOMY_NAMES = {"us-gaap": "US-GAAP", "ifrs-full": "IFRS"}
MINUS = "−"


# ---- data loading (cached per session; the SQLite cache sits underneath) ----------
@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_company(query: str) -> dict:
    company = tickers.lookup(query)
    profile = filings.company_profile(company.cik)
    profile["ticker"] = company.ticker or (profile["tickers"] or [""])[0]
    return profile


@st.cache_resource(ttl=config.HOUR, show_spinner=False)
def load_index(cik: int) -> FactIndex:
    return FactIndex(load_facts(cik), cik)


@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_filings(cik: int) -> pd.DataFrame:
    return filings.list_filings(cik)


# ---- formatting ---------------------------------------------------------------
def fiscal_year_end_text(mmdd: str) -> str:
    if len(mmdd) != 4 or not mmdd.isdigit():
        return "Not reported"
    return f"{calendar.month_name[int(mmdd[:2])]} {int(mmdd[2:])}"


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_value(v, unit: str, signed: bool = False) -> str:
    if _missing(v):
        return "Not reported"
    text = f"{abs(v):,.2f}" if unit == "per_share" else f"{abs(v) / 1e6:,.0f}"
    sign = MINUS if v < 0 else ("+" if signed and v > 0 else "")
    return sign + text


def fmt_pct(v) -> str:
    if _missing(v):
        return "–"
    return f"{'+' if v > 0 else MINUS if v < 0 else ''}{abs(v):.1%}"


def statement_markdown(table: pd.DataFrame, year: int) -> str:
    lines = [f"| Line item | FY{year} | FY{year - 1} | Change | Change&nbsp;% | Source |",
             "|:--|--:|--:|--:|--:|:--|"]
    for r in table.itertuples(index=False):
        cur = fmt_value(r.current, r.unit) + ("&nbsp;†" if r.restated else "")
        pri = fmt_value(r.prior, r.unit) + ("&nbsp;†" if r.prior_restated else "")
        change = "–" if _missing(r.change) else fmt_value(r.change, r.unit, signed=True)
        if r.url:  # hover shows the XBRL tag; non-breaking hyphens keep the cell on one line
            text = f"{r.form}&nbsp;·&nbsp;{r.filed}".replace("-", "\u2011")
            source = f'[{text}]({r.url} "XBRL tag: {r.tags} · period ending {r.end}")'
        else:
            source = "–"
        lines.append(f"| {r.label} | {cur} | {pri} | {change} | {fmt_pct(r.change_pct)} | {source} |")
    return "\n".join(lines)


# ---- views --------------------------------------------------------------------
def render_header(profile: dict, ix: Optional[FactIndex]) -> None:
    ticker = f" ({profile['ticker']})" if profile.get("ticker") else ""
    st.title(f"{profile['name']}{ticker}")
    if profile.get("sic_description"):
        st.caption(profile["sic_description"])
    facts = {
        "CIK": f"{profile['cik']:010d}",
        "Taxonomy": TAXONOMY_NAMES.get(ix.taxonomy, "Not reported") if ix else "Not reported",
        "Reporting currency": (ix.currency if ix else None) or "Not reported",
        "Fiscal year end": fiscal_year_end_text(profile.get("fiscal_year_end", "")),
    }
    for col, (label, value) in zip(st.columns(4), facts.items()):
        col.caption(label)
        col.markdown(f"**{value}**")


def render_timeline(cik: int) -> None:
    try:
        df = load_filings(cik)
    except EdgarError as exc:
        st.error(f"Could not load the filing list. {exc}")
        return
    present = [f for f in config.SUPPORTED_FORMS if f in set(df["base_form"])]
    left, right = st.columns([3, 1])
    chosen = left.multiselect("Form types", present, default=present)
    amendments = right.toggle("Include amendments (/A)", value=True)
    view = df[df["base_form"].isin(chosen)]
    if not amendments:
        view = view[~view["is_amendment"]]
    st.caption(f"{len(view):,} filings, newest first. Links open the original document on sec.gov.")
    st.dataframe(
        view[["filed", "form", "report_date", "description", "items", "doc_url", "index_url"]],
        hide_index=True, width="stretch", height=560,
        column_config={
            "filed": st.column_config.TextColumn("Filed", width="small"),
            "form": st.column_config.TextColumn("Form", width="small"),
            "report_date": st.column_config.TextColumn("Period", width="small"),
            "description": st.column_config.TextColumn("Description"),
            "items": st.column_config.TextColumn("8-K items", width="small"),
            "doc_url": st.column_config.LinkColumn("Document", display_text="Open document", width="small"),
            "index_url": st.column_config.LinkColumn("All files", display_text="Filing index", width="small"),
        })


def render_statements(ix: Optional[FactIndex]) -> None:
    years = available_years(ix) if ix else []
    if not years:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    year = st.selectbox("Fiscal year", years, format_func=lambda y: f"FY{y}")
    fy = ix.fiscal_years.set_index("fiscal_year").loc[year]
    st.caption(
        f"Fiscal year {fy['start']} to {fy['end']}. {ix.currency} millions; shares in millions; "
        "per-share figures as reported. † = restated in a later filing (latest filed value shown). "
        "Source = form and filing date, linked to the filing; hover for the XBRL tag.")
    for key, spec in STATEMENTS.items():
        st.subheader(spec["title"])
        st.markdown(statement_markdown(build_statement(ix, key, year), year))


@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_annual_inputs(cik: int) -> pd.DataFrame:
    return load_index(cik).annual_many(health.HEALTH_CONCEPTS)


def render_kpi_card(spec: kpis.KPI, k: pd.DataFrame, annual: pd.DataFrame, currency: str) -> None:
    series = k[spec.key].dropna().tail(10)
    with st.container(border=True):
        color = {1: "blue", 2: "green", 3: "orange"}[spec.tier]
        st.markdown(f":{color}-badge[{kpis.TIER_LABELS[spec.tier]}]")
        st.markdown(f"**{spec.label}**")
        if series.empty:
            st.markdown("#### Not reported")
            missing = [charts.CONCEPTS[c].label for c in spec.inputs if k[c].dropna().empty]
            st.caption("The filings do not report: " + ", ".join(missing) + "." if missing
                       else "Not enough years of data to compute this.")
            return
        year = int(series.index[-1])
        st.markdown(f"#### {rules.format_kpi(spec.key, series.iloc[-1], currency)}")
        st.caption(f"FY{year}" + ("" if year == int(k.index.max()) else " — latest year reported"))
        labels = [rules.format_kpi(spec.key, v, currency) for v in series]
        hover = ["<br>".join(charts.short_provenance(annual, spec.inputs, int(y)))
                 for y in series.index]
        fig = charts.sparkline(list(series.index), list(series), labels, hover,
                               color=charts.TIER_COLORS[spec.tier])
        st.plotly_chart(fig, config={"displayModeBar": False}, key=f"spark-{spec.key}")
        sentence = rules.card_sentence(spec.key, k, year, currency)
        if sentence:
            st.markdown(sentence)
        used = annual[annual["concept"].isin(spec.inputs) & annual["fiscal_year"].isin(series.index)]
        restated_years = sorted(used.loc[used["restated"].astype(bool), "fiscal_year"].unique())
        if restated_years and len(restated_years) < used["fiscal_year"].nunique():
            st.caption("† " + ", ".join(f"FY{y}" for y in restated_years) + " restated in a later "
                       "filing. Other years are as first filed and may be on a different basis.")
        with st.popover("Formula & sources"):
            st.markdown(f"**{spec.label}** — {spec.formula}")
            st.dataframe(
                charts.provenance_table(annual, spec.inputs, series.index), hide_index=True,
                column_config={
                    "Value": st.column_config.NumberColumn(format="localized"),
                    "Filing": st.column_config.LinkColumn(display_text="Open filing")})


@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_prices(ticker: str) -> prices.PriceHistory:
    return prices.get_history(ticker)


@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_fx(base: str, quote: str) -> fx.FxRate:
    return fx.get_rate(base, quote)


@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_shares_outstanding(cik: int) -> Optional[dict]:
    return shares_outstanding(load_facts(cik))


def render_market(ix: FactIndex, ticker: str, k: pd.DataFrame) -> None:
    """Tier 3: market-dependent multiples. Any failure here leaves the filing views intact."""
    st.subheader("Market-dependent")
    st.caption("These four numbers depend on today's share price and change every trading day. "
               "Everything else on this page comes only from filings.")
    if not ticker:
        st.info("No ticker is known for this company, so there is no price to look up.")
        return
    try:
        history = load_prices(ticker)
    except prices.PriceError as exc:
        st.warning(f"{exc} Market-dependent numbers are hidden; all filing-based views still work.")
        return
    price_date, price = history.latest
    rate = None
    if history.currency != ix.currency:
        try:
            rate = load_fx(history.currency, ix.currency)
        except fx.FxError as exc:
            st.warning(f"{exc} Market-dependent numbers are hidden; all filing-based views still work.")
            return
    ratio = 1.0
    if rate is not None or ix.taxonomy == "ifrs-full":
        known = config.ADR_ORDINARY_PER_ADR.get(ticker)
        ratio = st.number_input("Ordinary shares per US-listed share (ADR ratio)", min_value=0.001,
                                value=float(known or 1.0), step=0.5, key=f"adr-{ix.cik}")
        st.caption("Set in the app's config for this ticker; confirm it against the 20-F." if known
                   else "Not on file for this company, so 1 is only a starting value. "
                        "Check the 20-F before trusting these numbers.")
    cover = load_shares_outstanding(ix.cik)
    if cover:
        shares = cover["value"]
        shares_note = (f"{rules.count(shares)} shares outstanding as of {cover['end']} "
                       f"({cover['form']} cover page, filed {cover['filed']})")
    else:
        diluted = k["shares_diluted"].dropna()
        if diluted.empty:
            st.info("The filings do not report a share count, so market value cannot be computed.")
            return
        shares = float(diluted.iloc[-1])
        shares_note = (f"{rules.count(shares)} diluted weighted-average shares for FY{int(diluted.index[-1])} "
                       "(no single cover-page share count is reported)")
    year = int(k.index.max())
    row = k.loc[year]
    share_price = market.price_in_reporting_currency(price, rate.rate if rate else 1.0, ratio)
    m = market.market_multiples(share_price, shares, row["net_income"], row["net_debt"],
                                row["ebitda"], row["fcf"])
    parts = [f"Price {history.currency} {price:,.2f}, close on {price_date} ({history.source}"
             + (", last saved copy — refresh failed" if history.stale else "") + ")"]
    if rate is not None:
        parts.append(f"1 {rate.base} = {rate.rate:.4f} {rate.quote} on {rate.date} ({rate.source})")
    if rate is not None or ratio != 1.0:
        parts.append(f"= {ix.currency} {share_price:,.2f} per ordinary share")
    parts += [shares_note, f"financials for FY{year}"]
    st.caption("Inputs: " + " · ".join(parts) + ".")
    specs = list(market.MARKET_KPIS.values())
    for start in (0, 2):
        for col, spec in zip(st.columns(2), specs[start:start + 2]):
            with col, st.container(border=True):
                st.markdown(f":orange-badge[{kpis.TIER_LABELS[3]}]")
                st.markdown(f"**{spec.label}**")
                value = m[spec.key]
                if value is None:
                    st.markdown("#### Not available")
                else:
                    st.markdown("#### " + {"currency": lambda v: rules.money(v, ix.currency),
                                           "multiple": rules.multiple, "pct": rules.pct}[spec.kind](value))
                sentence = rules.market_sentence(spec.key, m, year, ix.currency, row["net_income"],
                                                 row["ebitda"], row["fcf"])
                st.markdown(sentence or "The filings do not report the numbers this needs.")
                st.caption(spec.formula)


def render_kpis(ix: Optional[FactIndex], ticker: str = "") -> None:
    if ix is None or ix.fiscal_years.empty:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    annual = load_annual_inputs(ix.cik)
    k = kpis.compute(kpis.to_wide(annual))
    st.caption("Tier 1 numbers are taken directly from filings. Tier 2 numbers are calculated "
               "from them. Tier 3 numbers, at the bottom, also depend on today's share price. Hover a chart point for the form and filing date; open "
               "Formula & sources on a card for the period, XBRL tag and a link to each filing.")
    cards = [kpis.KPIS[key] for key in kpis.DASHBOARD_CARDS]
    for start in range(0, len(cards), 3):
        for col, spec in zip(st.columns(3), cards[start:start + 3]):
            with col:
                render_kpi_card(spec, k, annual, ix.currency)
    render_market(ix, ticker, k)
    with st.expander("All KPIs by fiscal year"):
        table = k[list(kpis.KPIS)].tail(10).iloc[::-1]
        shown = pd.DataFrame(
            {f"FY{y}": [rules.format_kpi(key, table.loc[y, key], ix.currency) for key in kpis.KPIS]
             for y in table.index},
            index=[f"{spec.label} (T{spec.tier})" for spec in kpis.KPIS.values()])
        st.dataframe(shown, width="stretch", height=38 * (len(shown) + 1))


def what_this_means(text: Optional[str]) -> None:
    st.info("**What this means** — " + (text or "The filings do not report enough to say."))


def sources_expander(annual: pd.DataFrame, inputs, years) -> None:
    with st.expander("Sources (form, period, filing date, XBRL tag, link)"):
        st.dataframe(
            charts.provenance_table(annual, inputs, years), hide_index=True,
            column_config={"Value": st.column_config.NumberColumn(format="localized"),
                           "Filing": st.column_config.LinkColumn(display_text="Open filing")})


def _row(df: pd.DataFrame, year: int) -> Optional[pd.Series]:
    return df.loc[year] if year in df.index else None


def model_common_size(wide, annual, currency) -> None:
    cs = models.common_size(wide).tail(10)
    if cs.empty:
        st.info("Revenue, operating income and net income are needed for this view.")
        return
    st.caption("Each bar is 100% of that year's revenue, split by where it went. "
               "A slice below zero is a net gain (for example interest income).")
    st.plotly_chart(charts.stacked_share_bars(cs, models.COMMON_SIZE_LABELS, "Share of revenue"),
                    key="model-common-size")
    year = int(cs.index[-1])
    what_this_means(rules.common_size_note(cs.loc[year], _row(cs, year - 1), year,
                                           models.COMMON_SIZE_LABELS))
    sources_expander(annual, ["revenue", "cost_of_revenue", "gross_profit", "rnd", "sga",
                              "operating_income", "income_tax", "net_income"], cs.index)


def model_bridge(wide, annual, k, currency) -> None:
    years = [int(y) for y in wide.index[::-1] if models.fcf_bridge(wide, int(y))]
    if not years:
        st.info("Revenue, operating income, net income, operating cash flow and capex are needed.")
        return
    year = st.selectbox("Fiscal year", years, format_func=lambda y: f"FY{y}", key="bridge-year")
    st.plotly_chart(charts.waterfall(models.fcf_bridge(wide, year), f"{currency} millions"),
                    key="model-bridge")
    r = k.loc[year]
    notes = [rules.bridge_note(r["revenue"], r["fcf"], currency),
             rules.fcf_vs_net_income(r["fcf"], r["net_income"], r["cfo"], r["capex"], r["d_and_a"], currency)]
    what_this_means(" ".join(n for n in notes if n) or None)
    sources_expander(annual, ["revenue", "cost_of_revenue", "gross_profit", "operating_income",
                              "income_tax", "net_income", "d_and_a", "cfo", "capex"], [year])


def model_dupont(wide, annual, currency) -> None:
    d = models.dupont(wide).dropna(subset=["roe"])
    if d.empty:
        st.info("Net income, revenue, total assets and positive equity for two consecutive "
                "years are needed for this view.")
        return
    year = st.selectbox("Fiscal year", [int(y) for y in d.index[::-1]],
                        format_func=lambda y: f"FY{y}", key="dupont-year")
    cur, pri = d.loc[year], _row(d, year - 1)

    def delta(key, as_pct):
        if pri is None:
            return None
        diff = cur[key] - pri[key]
        return f"{diff * 100:+.1f} pts" if as_pct else f"{diff:+.2f}"

    cols = st.columns(4)
    cols[0].metric("ROE", rules.pct(cur["roe"]), delta("roe", True))
    cols[1].metric("= Net margin", rules.pct(cur["net_margin"]), delta("net_margin", True))
    cols[2].metric("× Asset turnover", f"{cur['asset_turnover']:.2f}", delta("asset_turnover", False))
    cols[3].metric("× Equity multiplier", f"{cur['equity_multiplier']:.2f}", delta("equity_multiplier", False))
    st.caption("Net margin = net income ÷ revenue. Asset turnover = revenue ÷ average assets. "
               "Equity multiplier = average assets ÷ average equity. Changes are versus the prior year.")
    driver = models.dupont_driver(pri, cur) if pri is not None else None
    what_this_means(rules.dupont_note(cur, pri, driver, models.DUPONT_LABELS))
    history = d.tail(10).iloc[::-1]
    st.dataframe(pd.DataFrame({
        "Year": [f"FY{y}" for y in history.index], "ROE": history["roe"].map(rules.pct),
        "Net margin": history["net_margin"].map(rules.pct),
        "Asset turnover": history["asset_turnover"].map("{:.2f}".format),
        "Equity multiplier": history["equity_multiplier"].map("{:.2f}".format)}), hide_index=True)
    sources_expander(annual, ["net_income", "revenue", "total_assets", "total_equity"],
                     [year, year - 1])


def model_capital_allocation(wide, annual, currency) -> None:
    ca = models.capital_allocation(wide).dropna(how="all").tail(10)
    if ca.empty:
        st.info("Operating cash flow and at least one cash use are needed for this view.")
        return
    st.caption("Each use of cash as a share of that year's cash from operations. A bar above "
               "100% means more was spent than operations brought in. Net debt repayment is "
               "repayments minus new borrowing, counted only when positive.")
    st.plotly_chart(charts.stacked_share_bars(ca, models.ALLOCATION_LABELS, "Share of operating cash"),
                    key="model-allocation")
    missing = [models.ALLOCATION_LABELS[c] for c in ("dividends_paid", "buybacks", "acquisitions")
               if wide[c].dropna().empty]
    if missing:
        st.caption("Not reported under a standard tag, so not shown: " + ", ".join(missing) + ".")
    what_this_means(rules.capital_allocation_note(models.allocation_summary(wide),
                                                  models.ALLOCATION_LABELS))
    sources_expander(annual, ["cfo", "dividends_paid", "buybacks", "capex", "acquisitions",
                              "debt_repaid", "debt_issued"], ca.index)


def model_shares(wide, annual, k, currency) -> None:
    shares = wide["shares_diluted"].dropna().tail(10)
    if shares.empty:
        st.info("Diluted share count is not reported.")
        return
    hover = ["<br>".join(charts.short_provenance(annual, ["shares_diluted"], int(y))) for y in shares.index]
    st.plotly_chart(charts.line_chart(list(shares.index), list(shares / 1e6),
                                      [rules.count(v) for v in shares], "Diluted shares",
                                      "Diluted shares (millions)", hover), key="model-shares")
    used = annual[(annual["concept"] == "shares_diluted") & annual["fiscal_year"].isin(shares.index)]
    restated = sorted(used.loc[used["restated"].astype(bool), "fiscal_year"])
    if restated and len(restated) < len(used):
        st.caption("† " + ", ".join(f"FY{y}" for y in restated) + " restated in a later filing "
                   "(for example after a share split or consolidation). Earlier years are as "
                   "first filed, so a sharp step in the line can be a change of basis.")
    what_this_means(rules.card_sentence("shares_diluted", k, int(shares.index[-1]), currency))
    sources_expander(annual, ["shares_diluted"], shares.index)


def model_dcf(k, annual, currency, cik, share_price=None) -> None:
    st.warning("A model under assumptions, not a price target. The output is a range.")
    d = models.dcf_defaults(k)
    if d is None or d["base_fcf"] <= 0:
        st.info("This model needs positive free cash flow in the latest year and a diluted share count.")
        return
    cols = st.columns(3)
    growth = cols[0].slider("FCF growth, next 5 years (%)", -5.0, 25.0, d["growth"] * 100, 0.5,
                            key=f"dcf-g-{cik}") / 100
    discount = cols[1].slider("Discount rate (%)", 5.0, 15.0, 9.0, 0.5, key=f"dcf-r-{cik}") / 100
    terminal = cols[2].slider("Terminal growth (%)", 0.0, 4.0, 2.5, 0.5, key=f"dcf-t-{cik}") / 100
    net_debt = d["net_debt"] or 0.0
    st.caption(
        f"Starting point from filings: FY{d['year']} free cash flow {rules.money(d['base_fcf'], currency)}, "
        + (f"net debt {rules.money(d['net_debt'], currency)}" if d["net_debt"] is not None
           else "net debt not reported (treated as zero)")
        + f", {rules.count(d['shares'])} diluted shares (FY{d['shares_year']}). "
        f"Default growth: {d['growth_basis']}, limited to 0–15%. The 9% discount rate and 2.5% "
        "terminal growth are generic starting values, not taken from filings.")
    growths = [growth + step for step in (-0.04, -0.02, 0.0, 0.02, 0.04)]
    discounts = [r for r in (discount + step for step in (-0.02, -0.01, 0.0, 0.01, 0.02)) if r > terminal]
    grid = models.dcf_grid(d["base_fcf"], growths, discounts, terminal, d["shares"], net_debt=net_debt)
    st.plotly_chart(charts.sensitivity_heatmap(grid, currency), key="model-dcf")
    st.caption(f"Value per share in {currency}, per ordinary share as counted in the filings. "
               "For a foreign company the US-listed ADR can represent a different number of shares.")
    st.markdown("**What today's price assumes (reverse DCF)**")
    implied = models.implied_growth(share_price, d["base_fcf"], discount, terminal, d["shares"], net_debt)
    fcf_then = k["fcf"].get(d["year"] - 5, float("nan"))
    past = ((d["base_fcf"] / fcf_then) ** 0.2 - 1) if fcf_then == fcf_then and fcf_then > 0 else None
    if share_price is None:
        st.caption("No share price is available right now, so this cannot be worked out.")
    else:
        st.info(rules.reverse_dcf_note(implied, past, discount, terminal)
                or f"At {currency} {share_price:,.2f} per share, no growth rate between −50% and "
                   "+100% a year makes this model match the price with these settings.")
        st.caption("This turns the model around: instead of asking what the shares are worth, it "
                   "shows the growth the price already assumes, using the discount rate and "
                   "terminal growth set above.")
    what_this_means(rules.dcf_note(float(grid.values.min()), float(grid.values.max()),
                                   float(grid.loc[discount, growth]), currency))
    sources_expander(annual, ["cfo", "capex", "total_debt", "cash", "shares_diluted"], [d["year"]])


def render_models(ix: Optional[FactIndex], ticker: str = "") -> None:
    if ix is None or ix.fiscal_years.empty:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    annual = load_annual_inputs(ix.cik)
    wide = kpis.to_wide(annual).reindex(columns=list(models.MODEL_CONCEPTS))
    k = kpis.compute(wide)
    tabs = st.tabs(["Common-size", "Revenue → FCF", "DuPont ROE", "Capital allocation",
                    "Share count", "Simple DCF"])
    with tabs[0]:
        model_common_size(wide, annual, ix.currency)
    with tabs[1]:
        model_bridge(wide, annual, k, ix.currency)
    with tabs[2]:
        model_dupont(wide, annual, ix.currency)
    with tabs[3]:
        model_capital_allocation(wide, annual, ix.currency)
    with tabs[4]:
        model_shares(wide, annual, k, ix.currency)
    with tabs[5]:
        model_dcf(k, annual, ix.currency, ix.cik, current_share_price(ix, ticker))


GRADE_COLORS = {"A": "green", "B": "green", "C": "orange", "D": "red", "E": "red", "Not rated": "gray"}


def current_share_price(ix: FactIndex, ticker: str) -> Optional[float]:
    """Today's price per ordinary share in the reporting currency, or None if unavailable."""
    if not ticker:
        return None
    try:
        history = load_prices(ticker)
        rate = load_fx(history.currency, ix.currency).rate if history.currency != ix.currency else 1.0
    except (prices.PriceError, fx.FxError):
        return None
    ratio = st.session_state.get(f"adr-{ix.cik}") or config.ADR_ORDINARY_PER_ADR.get(ticker, 1.0)
    return market.price_in_reporting_currency(history.latest[1], rate, float(ratio))


def render_rating_card(rating: ratings.Rating, context: bool = False) -> None:
    with st.container(border=True):
        if context:
            st.markdown(":gray-badge[Context — not financial data]")
        st.markdown(f"**{rating.label}**")
        color = GRADE_COLORS[rating.grade]
        if rating.score is None:
            st.markdown(f"### :{color}[Not rated]")
        else:
            st.markdown(f"### :{color}[{rating.grade}] &nbsp; {rating.score:.0f} / 100")
        st.markdown(rules.rating_sentence(rating, include_grade=False))
        if rating.note:
            st.caption(rating.note)
        with st.popover("How this is calculated"):
            st.markdown(rating.basis)
            st.dataframe(pd.DataFrame({
                "Input": [m.label for m in rating.metrics],
                "Value": [rules.format_metric(m.value, m.kind) for m in rating.metrics],
                "Score": ["–" if m.score is None else f"{m.score:.0f}" for m in rating.metrics],
                "Scores 0 at": [rules.format_metric(m.poor, m.kind) for m in rating.metrics],
                "Scores 100 at": [rules.format_metric(m.strong, m.kind) for m in rating.metrics],
                "Weight": [f"{m.weight:g}" for m in rating.metrics]}), hide_index=True)
            st.caption("Each input is scored on a straight line between the two anchors, then "
                       "averaged by weight. Inputs the filings do not report are left out.")


def render_scorecard(ix: Optional[FactIndex], profile: dict) -> None:
    if ix is None or ix.fiscal_years.empty:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    annual = load_annual_inputs(ix.cik)
    wide = kpis.to_wide(annual).reindex(columns=list(models.MODEL_CONCEPTS))
    k = kpis.compute(wide)
    year = int(k["revenue"].dropna().index[-1]) if k["revenue"].notna().any() else None
    st.warning(
        "Rule-based grades, not advice. Each grade is arithmetic on filed numbers against fixed "
        "thresholds that are the same for every company and are not adjusted for industry, so a "
        "bank, a retailer and a software company are held to the same bar. A grade describes "
        "the numbers; it does not say whether the shares are worth buying or selling.")
    st.caption("Grades: A = 80 or more, B = 65 or more, C = 50 or more, D = 35 or more, E = below 35."
               + (f" Latest fiscal year: FY{year}." if year else ""))

    st.subheader("The business")
    st.caption("From filings only. News coverage and the share price play no part in these three grades.")
    overall = ratings.overall_ratings(k, wide, ratings.first_restated_year(annual))
    for col, key in zip(st.columns(3), ("current", "historical", "future")):
        with col:
            render_rating_card(overall[key])
    st.caption("Future readiness is built from recent trends and financial capacity in past "
               "filings. It is not a forecast.")
    history = ratings.grade_history(k, wide, ratings.first_restated_year(annual)).dropna(how="all")
    if len(history) >= 2:
        st.markdown("**How these scores moved over time**")
        st.plotly_chart(charts.multi_line(
            history, {"current": "Current state", "historical": "Historical record",
                      "future": "Future readiness"}, "Score (0–100)", y_range=[0, 100]), key="grade-history")
        st.caption("Each point uses only the fiscal years up to it, with the figures as they stand "
                   "in the latest filings. Use it to judge whether the scores told you anything in "
                   "the past before leaning on them now.")

    st.subheader("By model")
    by_model = ratings.model_ratings(k, wide, current_share_price(ix, profile.get("ticker", "")))
    keys = list(by_model)
    for start in range(0, len(keys), 3):
        for col, key in zip(st.columns(3), keys[start:start + 3]):
            with col:
                render_rating_card(by_model[key])
    st.caption("The DCF rating uses the model's default inputs and today's share price, so it "
               "changes daily and is not rated when no price is available.")

    st.subheader("News climate")
    query = st.session_state.get(f"climate-q-{profile['cik']}") or gdelt.default_query(profile["name"])
    climate = ratings.rate_climate(gdelt.history(query), dt.date.today().isoformat())
    left, right = st.columns([1, 2])
    with left:
        render_rating_card(climate, context=True)
    right.caption(
        f"Based on the coverage history saved for the phrase {query} (set and fetched in the "
        "Climate tab). It needs at least 7 days of tone data in the last 30. It reflects how "
        "automated software scored the wording of news stories, which can be wrong and can pick "
        "up unrelated stories. It is kept out of the business grades above.")


def current_market_cap(ix: FactIndex, ticker: str, k: pd.DataFrame) -> Optional[float]:
    price = current_share_price(ix, ticker)
    if price is None:
        return None
    cover = load_shares_outstanding(ix.cik)
    diluted = k["shares_diluted"].dropna()
    shares = cover["value"] if cover else (float(diluted.iloc[-1]) if len(diluted) else None)
    return price * shares if shares else None


STATUS_BADGE = {"flag": ":red-badge[Flag]", "ok": ":green-badge[OK]", "no data": ":gray-badge[No data]"}


def render_health(ix: Optional[FactIndex], profile: dict) -> None:
    if ix is None or ix.fiscal_years.empty:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    annual = load_annual_inputs(ix.cik)
    wide = kpis.to_wide(annual).reindex(columns=list(health.HEALTH_CONCEPTS))
    k = kpis.compute(wide)
    year = health.latest_year(wide)
    try:
        filing_list = load_filings(ix.cik)
    except EdgarError:
        filing_list = None
    checks = health.red_flags(wide, k, filing_list, dt.date.today().isoformat())
    st.warning("A flag is a prompt to read the filing, not a verdict. Each one has an innocent "
               "explanation as often as a worrying one, and the thresholds are the same for every "
               "industry.")
    flagged = sum(c.status == "flag" for c in checks)
    known = sum(c.status != "no data" for c in checks)
    st.subheader(f"Red-flag checklist: {flagged} of {known} checks flagged" + (f" (FY{year})" if year else ""))
    for c in sorted(checks, key=lambda c: ["flag", "ok", "no data"].index(c.status)):
        with st.container(border=True):
            st.markdown(f"{STATUS_BADGE[c.status]} &nbsp; **{c.label}**")
            st.markdown(c.detail)
            st.caption(c.why)

    left, right = st.columns(2)
    with left:
        st.subheader("Piotroski F-score")
        p = health.piotroski(wide)
        if p["score"] is None:
            st.info("Not enough reported data for this score.")
        else:
            st.markdown(f"### {p['score']} of {p['available']}")
            st.caption(f"FY{p['year']} against FY{p['year'] - 1}. Nine pass/fail tests of profitability, "
                       "balance-sheet direction and efficiency; 7 or more is usually read as "
                       "improving fundamentals, 3 or fewer as deteriorating. Tests without data are left out.")
            st.dataframe(pd.DataFrame({"Test": [t for t, _ in p["tests"]],
                                       "Result": ["No data" if ok is None else "Pass" if ok else "Fail"
                                                  for _, ok in p["tests"]]}), hide_index=True)
    with right:
        st.subheader("Altman Z-score")
        st.markdown(":orange-badge[Tier 3 · Depends on today's price]")
        z = health.altman_z(wide, current_market_cap(ix, profile.get("ticker", ""), k))
        if z["z"] is None:
            st.info("Needs a share price plus current assets and liabilities, total liabilities and "
                    "retained earnings; at least one is not available.")
        else:
            st.markdown(f"### {z['z']:.2f} — {z['zone']}")
            st.caption("Above 2.99 is the safe zone, below 1.81 the distress zone. The formula was "
                       "built for manufacturers; it is not meaningful for banks or insurers and "
                       "flatters companies with a very high market value.")
            st.dataframe(pd.DataFrame({"Part": list(z["parts"]),
                                       "Weight": [f"{w:g}" for w, _ in z["parts"].values()],
                                       "Value": [f"{v:.2f}" for _, v in z["parts"].values()],
                                       "Points": [f"{w * v:.2f}" for w, v in z["parts"].values()]}),
                         hide_index=True)
    sources_expander(annual, list(health.HEALTH_CONCEPTS), [year, year - 1] if year else [])


def render_compare(ix: Optional[FactIndex], profile: dict) -> None:
    if ix is None or ix.fiscal_years.empty:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    raw = st.text_input("Compare with (up to 4 tickers, separated by commas)", key=f"peers-{ix.cik}",
                        placeholder="for example MSFT, GOOGL",
                        help="You choose the peers. Pick companies in the same line of business "
                             "for the comparison to mean something.")
    peers = [t.strip().upper() for t in raw.split(",") if t.strip()][:4]
    entries = [(profile.get("ticker") or profile["name"], ix)]
    for t in peers:
        try:
            other = load_company(t)
            entries.append((other.get("ticker") or t, load_index(other["cik"])))
        except EdgarError as exc:
            st.warning(f"{t}: {exc}")
    ks, grades, currencies = {}, {}, {}
    for name, index in entries:
        if index.fiscal_years.empty:
            st.warning(f"{name}: no annual financial data on EDGAR.")
            continue
        annual = load_annual_inputs(index.cik)
        wide = kpis.to_wide(annual).reindex(columns=list(models.MODEL_CONCEPTS))
        ks[name] = kpis.compute(wide)
        grades[name] = ratings.overall_ratings(ks[name], wide, ratings.first_restated_year(annual))
        currencies[name] = index.currency
    table = ratings.comparison(ks)
    shown = {"Fiscal year": {n: "–" if pd.isna(table.loc["fiscal_year", n]) else f"FY{int(table.loc['fiscal_year', n])}"
                             for n in ks},
             "Reporting currency": dict(currencies)}
    for key, label in ratings.COMPARE_ROWS:
        shown[label] = {n: rules.format_kpi(key, table.loc[key, n], currencies[n]) for n in ks}
    for key, label in (("current", "Grade: current state"), ("historical", "Grade: historical record"),
                       ("future", "Grade: future readiness")):
        shown[label] = {n: "Not rated" if grades[n][key].score is None
                        else f"{grades[n][key].grade} ({grades[n][key].score:.0f})" for n in ks}
    st.dataframe(pd.DataFrame(shown).T, height=38 * (len(shown) + 1) + 2)
    st.caption("Latest fiscal year for each company, from its own filings. Ratios can be compared "
               "directly; revenue is in each company's own reporting currency and fiscal years can "
               "end in different months. \"Not available\" means the filings do not report an input.")
    if len(ks) < 2:
        st.info("Add one or more tickers above to see companies side by side.")


@st.cache_data(ttl=config.HOUR, show_spinner=False)
def load_latest_period_growth(cik: int):
    """Latest interim period against a year earlier, for revenue and net income."""
    ix = load_index(cik)
    return (verdicts.latest_period_growth(ix.interim("revenue")),
            verdicts.latest_period_growth(ix.interim("net_income")))


VERDICT_COLORS = {"Favourable": "green", "Mixed": "orange", "Unfavourable": "red", "Not enough data": "gray"}
SIGNAL_MARKS = {1: ":green[▲ Favourable]", 0: ":orange[● Neutral]", -1: ":red[▼ Unfavourable]",
                None: ":gray[○ No data]"}


def render_verdict_card(title: str, subtitle: str, v: verdicts.Verdict) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(subtitle)
        st.markdown(f"## :{VERDICT_COLORS[v.label]}[{v.label}]")
        st.markdown(rules.verdict_sentence(v))
        st.divider()
        for s in v.signals:
            st.markdown(f"{SIGNAL_MARKS[s.score]} &nbsp; **{s.label}**")
            st.caption(s.detail)


def render_verdict(ix: Optional[FactIndex], profile: dict) -> None:
    if ix is None or ix.fiscal_years.empty:
        st.info("No annual financial data (XBRL) was found for this company on EDGAR.")
        return
    ticker = profile.get("ticker", "")
    annual = load_annual_inputs(ix.cik)
    wide = kpis.to_wide(annual).reindex(columns=list(health.HEALTH_CONCEPTS))
    k = kpis.compute(wide)

    # long term: grades, health checks, and what the price assumes
    overall = ratings.overall_ratings(k, wide, ratings.first_restated_year(annual))
    try:
        filing_list = load_filings(ix.cik)
    except EdgarError:
        filing_list = None
    checks = health.red_flags(wide, k, filing_list, dt.date.today().isoformat())
    pio = health.piotroski(wide)
    price = current_share_price(ix, ticker)
    d = models.dcf_defaults(k)
    implied = delivered = None
    if d is not None and d["base_fcf"] > 0:
        implied = models.implied_growth(price, d["base_fcf"], ratings.DCF_DISCOUNT, ratings.DCF_TERMINAL,
                                        d["shares"], d["net_debt"] or 0.0)
        then = k["fcf"].get(d["year"] - 5, float("nan"))
        if then == then and then > 0:
            delivered = (d["base_fcf"] / then) ** 0.2 - 1
    long_v = verdicts.long_term(
        overall["current"].score, overall["historical"].score, overall["future"].score,
        sum(c.status == "flag" for c in checks), sum(c.status != "no data" for c in checks),
        pio["score"], pio["available"], implied, delivered)

    # short term: price behaviour, latest results, news tone
    frame = None
    if ticker:
        try:
            frame = load_prices(ticker).frame
        except prices.PriceError:
            frame = None
    revenue_g, profit_g = load_latest_period_growth(ix.cik)
    query = st.session_state.get(f"climate-q-{profile['cik']}") or gdelt.default_query(profile["name"])
    climate = ratings.rate_climate(gdelt.history(query), dt.date.today().isoformat())
    short_v = verdicts.short_term(verdicts.price_trend(frame), revenue_g, profit_g, climate.score)

    st.warning(
        "This is a summary of evidence, not advice and not a forecast. It applies the same fixed "
        "rules to every company, knows nothing about your goals, time frame or what you already "
        "own, and cannot tell you what to buy or sell.")
    left, right = st.columns(2)
    with left:
        render_verdict_card("Long term", "Years. Business quality, financial health and what the price assumes.", long_v)
    with right:
        render_verdict_card("Short term", "Weeks to months. Recent price behaviour, latest results and news tone.", short_v)
    st.caption(
        "How it is decided: each signal is favourable (+1), neutral (0) or unfavourable (−1) by a "
        "fixed rule, and the verdict is their average — Favourable above +0.34, Unfavourable below "
        "−0.34, otherwise Mixed. Signals without data are left out. The rules are in "
        "src/metrics/verdict.py. Short-term price signals describe what has just happened; on their "
        "own they are a weak guide to what happens next. The detail behind every signal is in the "
        "Scorecard, Health checks, Models and Climate tabs.")


CLIMATE_WINDOWS = {"3 months": 92, "6 months": 183, "1 year": 366, "All stored": None}
CLIMATE_CAPTION = ("Coverage and tone describe how the company was being talked about. "
                   "They are not predictions.")


def render_climate(profile: dict) -> None:
    """News climate: context only, kept visually apart and never used in any KPI."""
    with st.container(border=True):
        st.markdown(":gray-badge[Context — not financial data]")
        st.subheader("News climate")
        left, right = st.columns([2, 3])
        query = left.text_input(
            "News search phrase", value=gdelt.default_query(profile["name"]), key=f"climate-q-{profile['cik']}",
            help="Used for GDELT coverage and Google News headlines. Keep the quotes for an exact "
                 "phrase; a short name can match unrelated stories.").strip()
        window = right.radio("Time window", list(CLIMATE_WINDOWS), horizontal=True, key="climate-window")
        today = dt.date.today()
        days = CLIMATE_WINDOWS[window]
        start = (today - dt.timedelta(days=days)).isoformat() if days else "0000-00-00"

        price_frame, price_currency = None, "USD"
        if profile.get("ticker"):
            try:
                history = load_prices(profile["ticker"])
                price_frame = history.frame[history.frame["date"] >= start]
                price_currency = history.currency
            except prices.PriceError as exc:
                st.warning(f"{exc} The price line is hidden.")

        coverage = pd.DataFrame(columns=["date", "volume", "tone"])
        if query:
            status = gdelt.pull_state(query)
            if status["status"] == "due":
                # Fetched only on request: GDELT is slow and strict about request rates.
                if st.button("Fetch news coverage from GDELT", key=f"climate-pull-{profile['cik']}",
                             help="Takes up to a minute. Results are saved, so each phrase is "
                                  "fetched at most twice a day."):
                    with st.spinner("Asking GDELT for coverage volume and tone…"):
                        status = gdelt.refresh(query)
            coverage = gdelt.history(query)
            coverage = coverage[coverage["date"] >= start]
            if status["status"] in ("failed", "waiting"):
                st.warning(status["message"] + (" Showing the coverage history already saved on this computer."
                                                if not coverage.empty else " No coverage history is saved yet."))
            elif status["status"] == "due":
                st.caption("News coverage has not been fetched for this phrase today."
                           + (" Showing saved history." if not coverage.empty else ""))
            elif coverage.empty:
                st.info("GDELT returned no coverage for this phrase. Try a different one.")

        try:
            all_events = climate_events.build_events(load_filings(profile["cik"]))
        except EdgarError as exc:
            st.warning(f"Filing events are unavailable. {exc}")
            all_events = pd.DataFrame(columns=climate_events.EVENT_COLUMNS)
        present = [c for c in climate_events.CATEGORIES if c in set(all_events["category"])]
        chosen = st.multiselect("Filing events to mark (8-K / 6-K)", present,
                                default=[c for c in present if c in climate_events.DEFAULT_CATEGORIES],
                                key=f"climate-cats-{profile['cik']}")
        shown = all_events[all_events["category"].isin(chosen) & (all_events["date"] >= start)]

        st.plotly_chart(charts.climate_figure(price_frame, coverage, shown, price_currency), key="climate-fig")
        st.caption(CLIMATE_CAPTION)
        st.caption("Coverage volume is the share of all news GDELT monitored that matched the phrase; "
                   "tone is GDELT's average tone score (below zero is more negative wording). GDELT only "
                   "reaches back about three months, so longer history builds up as you keep using the app. "
                   "Diamonds mark filing dates; hover for the event, and use the table for the link.")

        st.markdown(f"**Filing events in this window** ({len(shown)})")
        st.dataframe(shown[["date", "form", "category", "label", "url"]], hide_index=True,
                     height=min(320, 38 * (len(shown) + 1) + 2),
                     column_config={"date": "Filed", "form": "Form", "category": "Type", "label": "Event",
                                    "url": st.column_config.LinkColumn("Filing", display_text="Open filing")})

        st.markdown("**Headlines**")
        span = st.date_input("Date range", value=(today - dt.timedelta(days=14), today), max_value=today,
                             key=f"climate-range-{profile['cik']}")
        if not query:
            st.info("Enter a search phrase to list headlines.")
        elif not isinstance(span, tuple) or len(span) != 2:
            st.info("Pick an end date to list headlines.")
        else:
            try:
                news = headlines.get_headlines(query, span[0].isoformat(), span[1].isoformat())
            except headlines.HeadlinesError as exc:
                st.warning(str(exc))
            else:
                st.caption(f"{len(news)} headlines from Google News (at most about 100 per search). "
                           "Titles and links only; articles are not read or analysed.")
                st.dataframe(news, hide_index=True, height=320,
                             column_config={"date": "Date", "title": st.column_config.TextColumn("Headline", width="large"),
                                            "source": "Outlet",
                                            "link": st.column_config.LinkColumn("Link", display_text="Open")})


def _open_ticker(ticker: str) -> None:
    st.session_state["ticker"] = ticker


def main() -> None:
    with st.sidebar:
        st.header("📑 Filing Lens")
        st.session_state.setdefault("ticker", "AAPL")
        query = st.text_input("Ticker or CIK", key="ticker", help="A US ticker such as AAPL or UL, "
                              "or the SEC's numeric company ID (CIK).").strip()
        saved = watchlist.load()
        if saved:
            st.caption("Watchlist")
            for t in saved:
                pick, drop = st.columns([4, 1])
                pick.button(t, key=f"wl-open-{t}", on_click=_open_ticker, args=(t,), use_container_width=True)
                drop.button("✕", key=f"wl-drop-{t}", on_click=watchlist.remove, args=(t,),
                            help=f"Remove {t} from the watchlist")
        if query and query.upper() not in saved:
            st.button("Save to watchlist", on_click=watchlist.add, args=(query,))
        if config.SEC_USER_AGENT_IS_DEFAULT:
            st.warning("SEC_USER_AGENT is not set in .env. Ticker search needs it; "
                       "a numeric CIK still works (Apple 320193, Unilever 217410).")
        st.caption("Numbers come only from SEC EDGAR filings. Educational tool, not investment advice.")
    if not query:
        st.info("Enter a ticker to begin.")
        return
    try:
        with st.spinner("Loading from SEC EDGAR…"):
            profile = load_company(query)
    except EdgarError as exc:
        st.error(str(exc))
        return
    ix, facts_error = None, None
    try:
        with st.spinner("Loading financial data…"):
            ix = load_index(profile["cik"])
    except EdgarError as exc:
        facts_error = str(exc)
    render_header(profile, ix)
    if facts_error:
        st.warning(f"Financial data is unavailable: {facts_error}")
    (verdict_tab, kpi_tab, score_tab, health_tab, compare_tab, models_tab, timeline_tab,
     statements_tab, climate_tab) = st.tabs(
        ["Verdict", "KPI dashboard", "Scorecard", "Health checks", "Compare", "Models",
         "Filings timeline", "Statements", "Climate"])
    with models_tab:
        render_models(ix, profile.get("ticker", ""))
    with kpi_tab:
        render_kpis(ix, profile.get("ticker", ""))
    with timeline_tab:
        render_timeline(profile["cik"])
    with statements_tab:
        render_statements(ix)
    with climate_tab:
        render_climate(profile)
    with health_tab:
        render_health(ix, profile)
    with compare_tab:
        render_compare(ix, profile)
    with verdict_tab:   # after the other tabs so it can use the ADR ratio and news phrase set there
        render_verdict(ix, profile)
    with score_tab:   # after the other tabs so it can use the ADR ratio and news phrase set there
        render_scorecard(ix, profile)


main()
