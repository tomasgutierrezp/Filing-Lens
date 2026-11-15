"""Canonical metric -> ordered candidate XBRL tags per taxonomy.

For each period the first candidate with data wins, and the tag(s) used are recorded in
provenance. Candidates verified against the Apple (us-gaap) and Unilever (ifrs-full)
companyfacts payloads; extend the lists as new filers need them. If nothing matches,
callers show "Not reported" — never a guess. Later candidates can be broader than the
first (e.g. equity including minority interests, cash including restricted cash); the tag
actually used is always shown next to the number.

Also checked on 2026-11-15 against MSFT, GOOGL, AMZN, JNJ, PG, KO, WMT, NVDA, TSLA, HD, DIS,
NVS, SAP and TM. Banks and insurers (e.g. JPM) report a different income statement and are
only partly covered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

US, IFRS = "us-gaap", "ifrs-full"


@dataclass(frozen=True)
class Sum:
    """A metric built by adding tags: all `required`, any `optional`, first hit of `one_of`."""
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()
    one_of: Tuple[str, ...] = ()


Candidate = Union[str, Sum]


@dataclass(frozen=True)
class Concept:
    key: str
    label: str
    kind: str  # "flow" (over a period) or "stock" (at a date)
    unit: str  # "currency", "shares" or "per_share"
    tags: Dict[str, Tuple[Candidate, ...]]


def _c(key, label, kind, us, ifrs, unit="currency") -> Concept:
    return Concept(key, label, kind, unit, {US: tuple(us), IFRS: tuple(ifrs)})


_SHORT_TERM_US = ("ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings")

CONCEPTS: Dict[str, Concept] = {c.key: c for c in [
    _c("revenue", "Revenue", "flow",
       ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax"],
       ["Revenue", "RevenueFromContractsWithCustomers", "RevenueFromSaleOfGoods"]),
    _c("cost_of_revenue", "Cost of revenue", "flow",
       ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
       ["CostOfSales"]),
    _c("gross_profit", "Gross profit", "flow", ["GrossProfit"], ["GrossProfit"]),
    _c("rnd", "Research & development", "flow",
       ["ResearchAndDevelopmentExpense"], ["ResearchAndDevelopmentExpense"]),
    _c("sga", "Selling, general & admin", "flow",
       ["SellingGeneralAndAdministrativeExpense",
        Sum(("SellingAndMarketingExpense", "GeneralAndAdministrativeExpense")),
        Sum(("MarketingExpense", "GeneralAndAdministrativeExpense"))],
       ["SellingGeneralAndAdministrativeExpense",
        Sum(("SalesAndMarketingExpense", "AdministrativeExpense"))]),
    _c("operating_income", "Operating income", "flow",
       ["OperatingIncomeLoss"], ["ProfitLossFromOperatingActivities"]),
    _c("pretax_income", "Pre-tax income", "flow",
       ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
       ["ProfitLossBeforeTax"]),
    _c("income_tax", "Income tax expense", "flow",
       ["IncomeTaxExpenseBenefit"], ["IncomeTaxExpenseContinuingOperations"]),
    _c("net_income", "Net income", "flow",
       ["NetIncomeLoss", "ProfitLoss"],
       ["ProfitLossAttributableToOwnersOfParent", "ProfitLoss"]),
    _c("eps_diluted", "Diluted EPS", "flow",
       ["EarningsPerShareDiluted"], ["DilutedEarningsLossPerShare"], unit="per_share"),
    _c("cfo", "Cash from operations", "flow",
       ["NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
       ["CashFlowsFromUsedInOperatingActivities"]),
    _c("capex", "Capital expenditure", "flow",
       ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
       ["PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets"]),
    _c("dividends_paid", "Dividends paid", "flow",
       ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock", "PaymentsOfOrdinaryDividends"],
       ["DividendsPaidClassifiedAsFinancingActivities",
        "DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities"]),
    _c("buybacks", "Share repurchases", "flow",
       ["PaymentsForRepurchaseOfCommonStock"],
       ["PaymentsToAcquireOrRedeemEntitysShares"]),
    _c("acquisitions", "Acquisitions", "flow",
       ["PaymentsToAcquireBusinessesNetOfCashAcquired"],
       ["CashFlowsUsedInObtainingControlOfSubsidiariesOrOtherBusinessesClassifiedAsInvestingActivities"]),
    _c("cfi", "Cash from investing", "flow",
       ["NetCashProvidedByUsedInInvestingActivities"], ["CashFlowsFromUsedInInvestingActivities"]),
    _c("cff", "Cash from financing", "flow",
       ["NetCashProvidedByUsedInFinancingActivities"], ["CashFlowsFromUsedInFinancingActivities"]),
    _c("receivables", "Receivables", "stock",
       ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
       ["TradeAndOtherCurrentReceivables", "CurrentTradeReceivables"]),
    _c("inventory", "Inventory", "stock", ["InventoryNet"], ["Inventories"]),
    _c("current_assets", "Current assets", "stock", ["AssetsCurrent"], ["CurrentAssets"]),
    _c("ppe", "Property, plant & equipment", "stock",
       ["PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"],
       ["PropertyPlantAndEquipment", "PropertyPlantAndEquipmentIncludingRightofuseAssets"]),
    _c("debt_repaid", "Debt repaid", "flow",
       ["RepaymentsOfLongTermDebt", "RepaymentsOfDebt", "RepaymentsOfLongTermDebtAndCapitalSecurities",
        "RepaymentsOfDebtMaturingInMoreThanThreeMonths", "RepaymentsOfDebtAndCapitalLeaseObligations"],
       ["RepaymentsOfBorrowingsClassifiedAsFinancingActivities", "RepaymentsOfNoncurrentBorrowings"]),
    _c("debt_issued", "Debt issued", "flow",
       ["ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt",
        "ProceedsFromDebtMaturingInMoreThanThreeMonths", "ProceedsFromDebtNetOfIssuanceCosts",
        "ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet"],
       ["ProceedsFromBorrowingsClassifiedAsFinancingActivities", "ProceedsFromNoncurrentBorrowings"]),
    _c("goodwill", "Goodwill", "stock", ["Goodwill"], ["Goodwill"]),
    _c("retained_earnings", "Retained earnings", "stock",
       ["RetainedEarningsAccumulatedDeficit"], ["RetainedEarnings"]),
    _c("sbc", "Stock-based compensation", "flow",
       ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
       ["AdjustmentsForSharebasedPayments", "ExpenseFromSharebasedPaymentTransactionsWithEmployees"]),
    _c("total_assets", "Total assets", "stock", ["Assets"], ["Assets"]),
    _c("current_liabilities", "Current liabilities", "stock",
       ["LiabilitiesCurrent"], ["CurrentLiabilities"]),
    _c("total_liabilities", "Total liabilities", "stock", ["Liabilities"], ["Liabilities"]),
    _c("total_equity", "Shareholders' equity", "stock",
       ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
       ["EquityAttributableToOwnersOfParent", "Equity"]),
    _c("cash", "Cash and equivalents", "stock",
       ["CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
       ["CashAndCashEquivalents"]),
    _c("total_debt", "Total debt", "stock",
       [Sum(("LongTermDebtNoncurrent",), optional=("LongTermDebtCurrent",), one_of=_SHORT_TERM_US),
        Sum(("LongTermDebt",), one_of=_SHORT_TERM_US),
        Sum(("LongTermDebtAndCapitalLeaseObligations",),
            optional=("LongTermDebtAndCapitalLeaseObligationsCurrent",), one_of=_SHORT_TERM_US),
        "DebtLongtermAndShorttermCombinedAmount"],
       ["Borrowings",
        Sum(("NoncurrentBorrowings", "CurrentBorrowings")),
        Sum(("LongtermBorrowings",), optional=("ShorttermBorrowings",
                                               "CurrentPortionOfLongtermBorrowings"))]),
    _c("interest_expense", "Interest expense", "flow",
       ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
       ["FinanceCosts", "InterestExpense"]),
    _c("shares_diluted", "Diluted shares", "flow",
       ["WeightedAverageNumberOfDilutedSharesOutstanding"],
       ["AdjustedWeightedAverageShares", "WeightedAverageNumberOfSharesOutstandingDiluted"],
       unit="shares"),
    _c("d_and_a", "Depreciation & amortization", "flow",
       ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        Sum(("Depreciation",), optional=("AmortizationOfIntangibleAssets",))],
       ["DepreciationAndAmortisationExpense",
        "AdjustmentsForDepreciationAndAmortisationExpense",
        Sum(("DepreciationPropertyPlantAndEquipment",),
            optional=("DepreciationRightofuseAssets", "AmortisationIntangibleAssetsOtherThanGoodwill"))]),
]}
