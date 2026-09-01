"""Fundamental factor definitions."""

from __future__ import annotations

GROWTH_FACTORS = [
    "revenue_yoy_growth",
    "revenue_qoq_growth",
    "eps_yoy_growth",
    "ebitda_yoy_growth",
    "free_cash_flow_growth",
    "bookings_growth",
    "backlog_growth",
    "guidance_change",
]

QUALITY_FACTORS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "roic",
    "roe",
    "margin_expansion",
    "operating_leverage",
]

BALANCE_SHEET_FACTORS = [
    "net_debt_to_ebitda",
    "interest_coverage",
    "current_ratio",
    "cash_to_debt",
    "cash_runway_months",
    "debt_maturity_risk",
    "share_count_growth",
    "dilution_risk",
]

VALUATION_FACTORS = [
    "price_to_sales",
    "ev_to_sales",
    "ev_to_ebitda",
    "pe_ratio",
    "fcf_yield",
    "growth_adjusted_valuation",
]

REVISION_FACTORS = [
    "eps_estimate_revision_30d",
    "revenue_estimate_revision_30d",
    "target_price_revision",
    "guidance_up_or_down",
    "analyst_rating_change",
]

LOWER_IS_BETTER = {
    "net_debt_to_ebitda",
    "debt_maturity_risk",
    "share_count_growth",
    "dilution_risk",
    "price_to_sales",
    "ev_to_sales",
    "ev_to_ebitda",
    "pe_ratio",
    "growth_adjusted_valuation",
}

FACTOR_GROUPS = {
    "growth": GROWTH_FACTORS,
    "quality": QUALITY_FACTORS,
    "balance_sheet": BALANCE_SHEET_FACTORS,
    "valuation": VALUATION_FACTORS,
    "revision": REVISION_FACTORS,
}
