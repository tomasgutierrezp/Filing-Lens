import pytest

from src.edgar.facts import facts_to_frame


def _f(val, end, form, filed, accn, start=None):
    row = {"end": end, "val": val, "accn": accn, "fy": int(filed[:4]), "fp": "FY",
           "form": form, "filed": filed}
    if start:
        row["start"] = start
    return row


K23, K24, K24A = ("10-K", "2024-02-15", "A-23"), ("10-K", "2025-02-15", "A-24"), ("10-K/A", "2025-04-01", "A-24A")
Q1, Q2, Q3 = ("10-Q", "2024-05-01", "Q-1"), ("10-Q", "2024-08-01", "Q-2"), ("10-Q", "2024-11-01", "Q-3")
Y23, Y24 = ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")


def _dur(val, period, filing):
    return _f(val, period[1], *filing, start=period[0])


def _inst(val, end, filing):
    return _f(val, end, *filing)


@pytest.fixture
def synthetic_companyfacts():
    """A small US-GAAP calendar-year filer with a restatement, YTD-only cash flow and debt parts."""
    usd = lambda rows: {"units": {"USD": rows}}
    return {"cik": 1, "entityName": "Test Co", "facts": {"us-gaap": {
        "Revenues": usd([
            _dur(400, Y23, K23), _dur(390, Y23, K24),          # FY2023 restated 400 -> 390
            _dur(100, ("2024-01-01", "2024-03-31"), Q1),
            _dur(210, ("2024-01-01", "2024-06-30"), Q2),
            _dur(110, ("2024-04-01", "2024-06-30"), Q2),       # discrete Q2 reported
            _dur(330, ("2024-01-01", "2024-09-30"), Q3),
            _dur(460, Y24, K24),
        ]),
        "NetCashProvidedByUsedInOperatingActivities": usd([    # YTD only, like real 10-Qs
            _dur(90, Y23, K23),
            _dur(20, ("2024-01-01", "2024-03-31"), Q1),
            _dur(45, ("2024-01-01", "2024-06-30"), Q2),
            _dur(75, ("2024-01-01", "2024-09-30"), Q3),
            _dur(110, Y24, K24),
        ]),
        "NetIncomeLoss": usd([
            _dur(70, Y23, K23),
            _dur(15, ("2024-01-01", "2024-03-31"), Q1),
            _dur(35, ("2024-01-01", "2024-06-30"), Q2),
            _dur(60, ("2024-01-01", "2024-09-30"), Q3),
            _dur(80, Y24, K24), _dur(75, Y24, K24A),           # FY2024 amended 80 -> 75
        ]),
        "Assets": usd([_inst(900, "2023-12-31", K23), _inst(950, "2024-03-31", Q1),
                       _inst(1000, "2024-12-31", K24)]),
        "LongTermDebtNoncurrent": usd([_inst(50, "2024-12-31", K24)]),
        "LongTermDebtCurrent": usd([_inst(5, "2024-12-31", K24)]),
        "ShortTermBorrowings": usd([_inst(3, "2024-12-31", K24)]),
        "CommercialPaper": usd([_inst(2, "2024-12-31", K24)]),
        "LongTermDebt": usd([_inst(40, "2023-12-31", K23)]),
    }}}


@pytest.fixture
def synthetic_facts(synthetic_companyfacts):
    return facts_to_frame(synthetic_companyfacts)
