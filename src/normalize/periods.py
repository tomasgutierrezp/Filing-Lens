"""Fiscal-period alignment: dedupe, restatements, annual / interim series, TTM.

A fact's period is identified by its (start, end) dates. The `fy`/`fp` fields in
companyfacts describe the *filing* a fact appeared in (comparatives carry the later
filing's fy), so they are not used to place a fact in time.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src import config
from src.edgar.concept_map import CONCEPTS, Concept, Sum
from src.edgar.facts import detect_taxonomy, reporting_currency
from src.edgar.filings import filing_index_url

# Period length in days (end - start) -> class
_CLASSES = (("Q", 80, 100), ("H", 170, 195), ("9M", 260, 285), ("FY", 350, 380))

SERIES_COLUMNS = ["concept", "label", "fiscal_year", "period", "start", "end", "value", "unit",
                  "tags", "taxonomy", "form", "filed", "accn", "url", "restated",
                  "original_value", "derived"]


def _d(iso: str) -> date:
    return date.fromisoformat(iso)


def period_class(start: str, end: str) -> str:
    """'I' for instants, 'Q' / 'H' / '9M' / 'FY' for durations, '' for anything else."""
    if not start:
        return "I"
    days = (_d(end) - _d(start)).days
    for name, lo, hi in _CLASSES:
        if lo <= days <= hi:
            return name
    return ""


def fiscal_year_label(end: str) -> int:
    """Fiscal year a period ending on `end` belongs to (52/53-week years can spill into January)."""
    return (_d(end) - timedelta(days=7)).year


def dedupe(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (tag, unit, start, end). The latest filing wins; if its value differs from
    the first-filed one the row is flagged `restated` and the original is kept alongside."""
    f = facts.sort_values(["filed", "accn"], kind="stable")
    out = f.groupby(["taxonomy", "tag", "unit", "start", "end"], sort=False).agg(
        val=("val", "last"), original_val=("val", "first"), accn=("accn", "last"),
        form=("form", "last"), filed=("filed", "last")).reset_index()
    out["restated"] = out["val"] != out["original_val"]
    out["pclass"] = [period_class(s, e) for s, e in zip(out["start"], out["end"])]
    return out


class FactIndex:
    """Deduped facts for one company, with fast lookup by (tag, start, end)."""

    def __init__(self, facts: pd.DataFrame, cik: int, taxonomy: Optional[str] = None,
                 currency: Optional[str] = None):
        self.cik = cik
        self.taxonomy = taxonomy or detect_taxonomy(facts)
        self.currency = currency or reporting_currency(facts, self.taxonomy)
        self.facts = dedupe(facts[facts["taxonomy"] == self.taxonomy])
        self._by: Dict[Tuple[str, str, str, str], tuple] = {
            (r.tag, r.unit, r.start, r.end): r for r in self.facts.itertuples(index=False)}
        self.fiscal_years = self._fiscal_years()

    # ---- fiscal calendar -------------------------------------------------
    def _fiscal_years(self) -> pd.DataFrame:
        """Fiscal years (label, start, end) as reported in annual forms."""
        f = self.facts
        ann = f[(f["pclass"] == "FY") & f["form"].map(config.base_form).isin(config.ANNUAL_FORMS)]
        if ann.empty:
            return pd.DataFrame(columns=["fiscal_year", "start", "end"])
        counts = ann.groupby(["start", "end"]).size().rename("n").reset_index()
        counts = counts[counts["n"] >= 0.25 * counts["n"].max()]
        best = counts.sort_values("n").drop_duplicates("end", keep="last")
        best["fiscal_year"] = best["end"].map(fiscal_year_label)
        best = best.sort_values("end").drop_duplicates("fiscal_year", keep="last")
        return best[["fiscal_year", "start", "end"]].reset_index(drop=True)

    def _interim_ends(self, fy_start: str) -> Dict[str, str]:
        """Year-to-date period ends for the fiscal year starting on fy_start, by class."""
        f = self.facts
        ytd = f[(f["start"] == fy_start) & f["pclass"].isin(["Q", "H", "9M"])]
        return {pc: grp["end"].value_counts().idxmax() for pc, grp in ytd.groupby("pclass")}

    # ---- tag resolution --------------------------------------------------
    def _unit(self, concept: Concept) -> Optional[str]:
        if concept.unit == "shares":
            return "shares"
        if concept.unit == "per_share":
            return f"{self.currency}/shares" if self.currency else None
        return self.currency

    def _get(self, tag: str, unit: str, start: str, end: str):
        return self._by.get((tag, unit, start, end))

    def resolve(self, concept_key: str, start: str, end: str) -> Optional[dict]:
        """Value of a concept for one period, from the first candidate tag that has data.
        Pass start='' for balance-sheet (stock) concepts."""
        concept = CONCEPTS[concept_key]
        unit = self._unit(concept)
        if unit is None:
            return None
        if concept.kind == "stock":
            start = ""
        for cand in concept.tags.get(self.taxonomy, ()):
            if isinstance(cand, Sum):
                rows = [self._get(t, unit, start, end) for t in cand.required]
                if any(r is None for r in rows):
                    continue
                rows += [r for r in (self._get(t, unit, start, end) for t in cand.optional) if r]
                rows += [r for r in (self._get(t, unit, start, end) for t in cand.one_of) if r][:1]
            else:
                row = self._get(cand, unit, start, end)
                if row is None:
                    continue
                rows = [row]
            lead = max(rows, key=lambda r: r.filed)
            return {
                "concept": concept.key, "label": concept.label, "start": start, "end": end,
                "value": sum(r.val for r in rows), "unit": unit,
                "tags": " + ".join(r.tag for r in rows), "taxonomy": self.taxonomy,
                "form": lead.form, "filed": lead.filed, "accn": lead.accn,
                "url": filing_index_url(self.cik, lead.accn),
                "restated": any(r.restated for r in rows),
                "original_value": sum(r.original_val for r in rows), "derived": False,
            }
        return None

    # ---- series ----------------------------------------------------------
    def annual(self, concept_key: str) -> pd.DataFrame:
        """One row per fiscal year in which the concept was reported."""
        rows = []
        for fy in self.fiscal_years.itertuples(index=False):
            hit = self.resolve(concept_key, fy.start, fy.end)
            if hit:
                hit.update(fiscal_year=int(fy.fiscal_year), period="FY")
                rows.append(hit)
        return pd.DataFrame(rows, columns=SERIES_COLUMNS)

    def annual_many(self, concept_keys) -> pd.DataFrame:
        """Annual series for several concepts, stacked (long format)."""
        frames = [f for f in (self.annual(k) for k in concept_keys) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SERIES_COLUMNS)

    def interim(self, concept_key: str) -> pd.DataFrame:
        """Quarterly (Q1-Q4) or half-yearly (H1/H2) series, whichever the company files.

        Discrete periods are used when reported. Otherwise a period is derived by
        subtracting year-to-date figures (e.g. Q4 = FY - 9M) and flagged `derived`.
        Nothing is derived across a restatement, where the two figures are not comparable.
        """
        concept = CONCEPTS[concept_key]
        years: List[Tuple[int, str, Optional[str]]] = [
            (int(r.fiscal_year), r.start, r.end) for r in self.fiscal_years.itertuples(index=False)]
        if years:  # the fiscal year in progress
            nxt = (_d(years[-1][2]) + timedelta(days=1)).isoformat()
            years.append((years[-1][0] + 1, nxt, None))
        rows = []
        prior_annual = None
        for fy, fy_start, fy_end in years:
            # If the prior year was restated between two year-to-date filings, the company
            # changed its reporting basis (e.g. a disposal) and the two are not comparable.
            basis_changed = prior_annual["filed"] if prior_annual and prior_annual["restated"] else ""
            prior_annual = self.resolve(concept_key, fy_start, fy_end) if fy_end else None
            ends = self._interim_ends(fy_start)
            if not ends:
                continue  # no interim reports for this year
            if fy_end:
                ends["FY"] = fy_end
            quarterly = "Q" in ends or "9M" in ends
            labels = ({"Q": "Q1", "H": "Q2", "9M": "Q3", "FY": "Q4"} if quarterly
                      else {"H": "H1", "FY": "H2"})
            order = [pc for pc in ("Q", "H", "9M", "FY") if pc in ends]
            prev_end, prev_cum, expected_prev = None, None, None
            for pc in order:
                end = ends[pc]
                seg_start = fy_start if prev_end is None else (
                    _d(prev_end) + timedelta(days=1)).isoformat()
                contiguous = expected_prev == prev_end  # no skipped boundary before this one
                if concept.kind == "stock":
                    hit = self.resolve(concept_key, "", end)
                else:
                    cum = self.resolve(concept_key, fy_start, end)
                    hit = self.resolve(concept_key, seg_start, end) if contiguous else None
                    if (hit is None and contiguous and cum and prev_cum
                            and cum["restated"] == prev_cum["restated"]
                            and not prev_cum["filed"] < basis_changed <= cum["filed"]):
                        hit = dict(cum, start=seg_start, derived=True,
                                   value=cum["value"] - prev_cum["value"],
                                   original_value=cum["original_value"] - prev_cum["original_value"],
                                   tags=cum["tags"])
                    prev_cum = cum
                if hit:
                    hit.update(fiscal_year=fy, period=labels[pc])
                    rows.append(hit)
                prev_end = expected_prev = end
                # A missing boundary (e.g. half-year filer inside a quarterly map) breaks the chain.
                nxt_idx = ("Q", "H", "9M", "FY").index(pc) + 1
                if quarterly and nxt_idx < 4 and ("Q", "H", "9M", "FY")[nxt_idx] not in ends:
                    expected_prev = None
        return pd.DataFrame(rows, columns=SERIES_COLUMNS)

    def ttm(self, concept_key: str) -> Optional[dict]:
        """Trailing twelve months: sum of the latest contiguous interim periods covering one
        year (flows); latest reported point (stocks). None when the filings don't support it."""
        series = self.interim(concept_key)
        if series.empty:
            return None
        if CONCEPTS[concept_key].kind == "stock":
            last = series.iloc[-1]
            return {"value": float(last["value"]), "end": last["end"],
                    "periods": [f"{last['period']} FY{last['fiscal_year']}"]}
        total, used, expect_end = 0.0, [], None
        for r in series.iloc[::-1].itertuples(index=False):
            if expect_end is not None and r.end != expect_end:
                return None
            total += r.value
            used.append(f"{r.period} FY{r.fiscal_year}")
            expect_end = (_d(r.start) - timedelta(days=1)).isoformat()
            days = (_d(series.iloc[-1]["end"]) - _d(r.start)).days
            if 350 <= days <= 380:
                return {"value": total, "start": r.start, "end": series.iloc[-1]["end"],
                        "periods": used[::-1]}
            if days > 380:
                return None
        return None
