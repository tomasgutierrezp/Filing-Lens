"""companyfacts JSON -> tidy DataFrame (one row per reported fact)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src import config
from src.edgar.client import EdgarClient, default_client

FACT_COLUMNS = ["taxonomy", "tag", "unit", "start", "end", "val", "accn", "fy", "fp",
                "form", "filed", "frame"]
TAXONOMIES = ("us-gaap", "ifrs-full")
_CURRENCY_PROBES = ("Assets", "Revenue", "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax", "ProfitLoss",
                    "NetIncomeLoss")


def facts_to_frame(companyfacts: dict) -> pd.DataFrame:
    """Flatten the nested companyfacts payload. Dates stay ISO strings; instants have start ''."""
    rows = []
    for taxonomy, tags in companyfacts.get("facts", {}).items():
        for tag, body in tags.items():
            for unit, items in body.get("units", {}).items():
                for it in items:
                    rows.append((taxonomy, tag, unit, it.get("start", ""), it["end"], it["val"],
                                 it.get("accn", ""), it.get("fy"), it.get("fp", ""),
                                 it.get("form", ""), it.get("filed", ""), it.get("frame", "")))
    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def load_facts(cik: int, client: Optional[EdgarClient] = None) -> pd.DataFrame:
    data = (client or default_client()).get_json(config.URL_COMPANY_FACTS.format(cik=cik))
    return facts_to_frame(data)


def detect_taxonomy(facts: pd.DataFrame) -> Optional[str]:
    """'us-gaap' or 'ifrs-full', whichever the company's recent annual filings use."""
    annual = facts[facts["form"].map(config.base_form).isin(config.ANNUAL_FORMS)
                   & facts["taxonomy"].isin(TAXONOMIES)]
    if annual.empty:
        return None
    recent = annual[annual["filed"] >= annual["filed"].max()[:4]]
    return recent["taxonomy"].value_counts().idxmax()


def reporting_currency(facts: pd.DataFrame, taxonomy: Optional[str] = None) -> Optional[str]:
    """ISO code the company reports in, read from the units of its headline tags."""
    taxonomy = taxonomy or detect_taxonomy(facts)
    probe = facts[(facts["taxonomy"] == taxonomy) & facts["tag"].isin(_CURRENCY_PROBES)
                  & facts["unit"].str.fullmatch(r"[A-Z]{3}")]
    if probe.empty:
        return None
    latest = probe[probe["filed"] == probe["filed"].max()]
    return latest["unit"].value_counts().idxmax()


def shares_outstanding(facts: pd.DataFrame) -> Optional[dict]:
    """Latest share count from the filing cover page (dei:EntityCommonStockSharesOutstanding).
    None when the company does not report a single figure there (e.g. several share classes)."""
    rows = facts[(facts["taxonomy"] == "dei") & (facts["tag"] == "EntityCommonStockSharesOutstanding")
                 & (facts["unit"] == "shares")]
    if rows.empty:
        return None
    r = rows.sort_values(["end", "filed"]).iloc[-1]
    return {"value": float(r["val"]), "end": r["end"], "form": r["form"], "filed": r["filed"],
            "accn": r["accn"], "tag": "dei:EntityCommonStockSharesOutstanding"}
