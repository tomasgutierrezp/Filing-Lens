"""Company profile and filing list from the EDGAR submissions API."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src import config
from src.edgar.client import EdgarClient, default_client

_COLUMNS = ["accessionNumber", "form", "filingDate", "reportDate", "primaryDocument",
            "primaryDocDescription", "items"]


def filing_index_url(cik: int, accn: str) -> str:
    return config.URL_FILING_INDEX.format(cik=cik, accn_nodash=accn.replace("-", ""), accn=accn)


def company_profile(cik: int, client: Optional[EdgarClient] = None) -> dict:
    sub = (client or default_client()).get_json(config.URL_SUBMISSIONS.format(cik=cik))
    return {
        "cik": cik,
        "name": sub.get("name", ""),
        "tickers": sub.get("tickers", []),
        "exchanges": sub.get("exchanges", []),
        "sic_description": sub.get("sicDescription", ""),
        "fiscal_year_end": sub.get("fiscalYearEnd") or "",  # "MMDD"
    }


def list_filings(cik: int, forms: Optional[Iterable[str]] = None, include_older: bool = True,
                 client: Optional[EdgarClient] = None) -> pd.DataFrame:
    """All filings, newest first. `forms` filters by base form ('10-K' also matches '10-K/A')."""
    client = client or default_client()
    sub = client.get_json(config.URL_SUBMISSIONS.format(cik=cik))
    pages = [sub["filings"]["recent"]]
    if include_older:
        for extra in sub["filings"].get("files", []):
            pages.append(client.get_json(config.URL_SUBMISSIONS_FILE.format(name=extra["name"])))
    frames = [pd.DataFrame({c: page.get(c, []) for c in _COLUMNS}) for page in pages]
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "accessionNumber": "accn", "filingDate": "filed", "reportDate": "report_date",
        "primaryDocument": "primary_doc", "primaryDocDescription": "description"})
    df["base_form"] = df["form"].map(config.base_form)
    df["is_amendment"] = df["form"].str.endswith("/A")
    if forms is not None:
        df = df[df["base_form"].isin(set(forms))]
    nodash = df["accn"].str.replace("-", "", regex=False)
    df["index_url"] = [filing_index_url(cik, a) for a in df["accn"]]
    df["doc_url"] = [
        config.URL_FILING_DOC.format(cik=cik, accn_nodash=n, doc=d) if d else u
        for n, d, u in zip(nodash, df["primary_doc"], df["index_url"])
    ]
    return (df.drop_duplicates("accn").sort_values(["filed", "accn"], ascending=False)
              .reset_index(drop=True))
