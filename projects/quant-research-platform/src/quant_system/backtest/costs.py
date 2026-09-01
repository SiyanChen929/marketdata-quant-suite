"""Transaction cost model."""

from __future__ import annotations


def apply_slippage(price: float, signed_quantity: float, slippage_bps: float) -> float:
    """Return fill price after slippage.

    signed_quantity > 0 means buy; signed_quantity < 0 means sell.
    """

    direction = 1.0 if signed_quantity > 0 else -1.0
    return price * (1.0 + direction * slippage_bps / 10_000.0)


def apply_slippage_and_impact(
    price: float,
    signed_quantity: float,
    base_slippage_bps: float,
    trade_notional: float,
    adv_dollars: float | None,
    impact_bps_per_1pct_adv: float,
    spread_bps: float = 0.0,
    impact_exponent: float = 1.0,
    min_liquidity_cost_bps: float = 0.0,
) -> float:
    """Return fill price after base slippage and ADV-based impact."""

    total_bps = total_execution_cost_bps(
        base_slippage_bps,
        trade_notional,
        adv_dollars,
        impact_bps_per_1pct_adv,
        spread_bps,
        impact_exponent,
        min_liquidity_cost_bps,
    )
    return apply_slippage(price, signed_quantity, total_bps)


def total_execution_cost_bps(
    base_slippage_bps: float,
    trade_notional: float,
    adv_dollars: float | None,
    impact_bps_per_1pct_adv: float,
    spread_bps: float = 0.0,
    impact_exponent: float = 1.0,
    min_liquidity_cost_bps: float = 0.0,
) -> float:
    """Estimate one-way execution cost in bps from spread and ADV participation."""

    participation = 0.0
    if adv_dollars and adv_dollars > 0:
        participation = max(0.0, trade_notional / adv_dollars)
    participation_pct = participation * 100.0
    exponent = max(0.25, float(impact_exponent or 1.0))
    impact_bps = float(impact_bps_per_1pct_adv) * (participation_pct**exponent)
    half_spread_bps = max(0.0, float(spread_bps)) / 2.0
    liquidity_floor = max(0.0, float(min_liquidity_cost_bps))
    return max(0.0, float(base_slippage_bps)) + max(liquidity_floor, half_spread_bps + impact_bps)


def commission(quantity: float, commission_per_share: float) -> float:
    """Per-share commission."""

    return abs(quantity) * commission_per_share


def daily_short_borrow_cost(short_market_value: float, annual_fee: float) -> float:
    """Daily borrow cost on absolute short market value."""

    return max(0.0, short_market_value) * annual_fee / 252.0
