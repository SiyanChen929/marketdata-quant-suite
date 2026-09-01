from __future__ import annotations

from quant_system.backtest.costs import apply_slippage_and_impact, total_execution_cost_bps


def test_capacity_costs_increase_non_linearly_with_adv_participation() -> None:
    small = total_execution_cost_bps(
        base_slippage_bps=5,
        trade_notional=10_000,
        adv_dollars=10_000_000,
        impact_bps_per_1pct_adv=2.5,
        spread_bps=4,
        impact_exponent=1.35,
    )
    large = total_execution_cost_bps(
        base_slippage_bps=5,
        trade_notional=500_000,
        adv_dollars=10_000_000,
        impact_bps_per_1pct_adv=2.5,
        spread_bps=4,
        impact_exponent=1.35,
    )
    assert small > 7
    assert large > small * 3


def test_capacity_costs_move_fill_against_trade_direction() -> None:
    buy = apply_slippage_and_impact(100, 100, 5, 100_000, 10_000_000, 2.5, spread_bps=4, impact_exponent=1.35)
    sell = apply_slippage_and_impact(100, -100, 5, 100_000, 10_000_000, 2.5, spread_bps=4, impact_exponent=1.35)
    assert buy > 100
    assert sell < 100
