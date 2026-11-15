"""Official event markers from 8-K / 6-K filings. Pure: DataFrame in, DataFrame out."""
from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd

CATEGORIES = ("Results", "Leadership", "Deals", "Shareholder matters", "Disclosure", "Routine", "Other")
DEFAULT_CATEGORIES = ("Results", "Leadership", "Deals", "Shareholder matters", "Disclosure")
EVENT_COLUMNS = ["date", "form", "category", "label", "url"]

# 8-K item number -> (category, plain description). First match in this order sets the category.
ITEM_RULES: List[Tuple[str, str, str]] = [
    ("2.02", "Results", "Results of operations"),
    ("5.02", "Leadership", "Director or officer change"),
    ("2.01", "Deals", "Acquisition or disposal completed"),
    ("1.01", "Deals", "Material agreement signed"),
    ("1.02", "Deals", "Material agreement ended"),
    ("5.07", "Shareholder matters", "Shareholder vote results"),
    ("3.03", "Shareholder matters", "Change to shareholder rights"),
    ("5.03", "Shareholder matters", "Bylaw or fiscal-year change"),
    ("4.01", "Disclosure", "Change of auditor"),
    ("4.02", "Disclosure", "Earlier financial statements not to be relied on"),
    ("2.05", "Disclosure", "Restructuring costs"),
    ("2.06", "Disclosure", "Material impairment"),
    ("1.03", "Disclosure", "Bankruptcy or receivership"),
    ("1.05", "Disclosure", "Material cybersecurity incident"),
    ("3.01", "Disclosure", "Listing notice"),
    ("7.01", "Disclosure", "Regulation FD disclosure"),
    ("8.01", "Other", "Other events"),
]

# 6-K filings carry a free-text description instead of item numbers.
KEYWORD_RULES: List[Tuple[str, str]] = [
    (r"sharehold|voting rights|treasury shares|block listing|director declaration|"
     r"publication of|final terms|issuance of equity|information memorandum", "Routine"),
    (r"\bagm\b|general meeting", "Shareholder matters"),
    (r"result|trading statement|quarter|half[- ]year|interim|annual report|earnings", "Results"),
    (r"directorate|appointment|board|chief|\bceo\b|\bcfo\b|chair|secretary|\bned\b|\bsid\b", "Leadership"),
    (r"acqui|dispos|merger|demerger|\bsale of\b|divest|offer for", "Deals"),
    (r"dividend|buy-?back|vote", "Shareholder matters"),
    (r"investor|media|statement|announcement|update|guidance|outlook", "Disclosure"),
]


def classify_8k(items: str) -> Tuple[str, str]:
    """'2.02,9.01' -> ('Results', 'Results of operations')."""
    codes = {c.strip() for c in str(items or "").split(",") if c.strip()}
    for code, category, text in ITEM_RULES:
        if code in codes:
            return category, text
    return "Other", "Current report"


def classify_6k(description: str) -> Tuple[str, str]:
    """'DIRECTORATE CHANGE' -> ('Leadership', 'Directorate change')."""
    text = str(description or "").strip()
    if not text or re.fullmatch(r"(form\s+)?6-k(/a)?", text, re.IGNORECASE):
        return "Other", "Report of foreign issuer"
    label = text.capitalize() if text.isupper() else text
    for pattern, category in KEYWORD_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return category, label
    return "Other", label


def build_events(filings: pd.DataFrame) -> pd.DataFrame:
    """8-K and 6-K filings (and amendments) as dated, categorised events, newest first.
    `filings` is the output of edgar.filings.list_filings."""
    rows = []
    f = filings[filings["base_form"].isin(["8-K", "6-K"])]
    for r in f.itertuples(index=False):
        category, label = classify_8k(r.items) if r.base_form == "8-K" else classify_6k(r.description)
        if r.is_amendment:
            label += " (amendment)"
        rows.append((r.filed, r.form, category, label, r.doc_url))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
