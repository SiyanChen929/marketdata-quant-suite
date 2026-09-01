from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.execution import (
    ExecutableCostModel,
    ExecutionCostSpec,
    ExecutionMarketInputs,
    build_execution_market_inputs,
)


def _inputs() -> ExecutionMarketInputs:
    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = ["A", "B"]
    return ExecutionMarketInputs(
        dollar_adv=pd.DataFrame(10_000_000.0, index=dates, columns=columns),
        daily_volatility=pd.DataFrame(0.02, index=dates, columns=columns),
        half_spread_bps=pd.DataFrame(2.0, index=dates, columns=columns),
        annual_borrow_bps=pd.DataFrame({"A": 30.0, "B": 100.0}, index=dates),
        locate_available=pd.DataFrame(True, index=dates, columns=columns),
        annual_cash_rate=0.04,
    )


def test_execution_cost_decomposes_spread_commission_impact_and_carry():
    model = ExecutableCostModel(
        _inputs(),
        ExecutionCostSpec(
            portfolio_notional=1_000_000,
            commission_bps=1.0,
            impact_coefficient=0.10,
            maximum_participation_rate=0.10,
        ),
    )
    result = model.evaluate(
        "2024-01-03",
        pd.Series({"A": 0.10, "B": -0.10}),
        pd.Series({"A": 0.50, "B": -0.50}),
    )

    assert result.executable
    assert result.maximum_participation_rate == pytest.approx(0.01)
    assert result.commission_cost > 0
    assert result.spread_cost > 0
    assert result.impact_cost > 0
    assert result.borrow_cost > 0
    assert result.cash_income > 0
    assert result.short_rebate_income > 0
    assert result.net_cost == pytest.approx(
        result.transaction_cost
        + result.borrow_cost
        + result.financing_cost
        - result.cash_income
        - result.short_rebate_income
    )


def test_capacity_and_locate_breaches_fail_closed_when_requested():
    inputs = _inputs()
    locate = inputs.locate_available.copy()
    locate.loc[:, "B"] = False
    model = ExecutableCostModel(
        ExecutionMarketInputs(
            dollar_adv=inputs.dollar_adv,
            daily_volatility=inputs.daily_volatility,
            locate_available=locate,
        ),
        ExecutionCostSpec(
            portfolio_notional=10_000_000,
            maximum_participation_rate=0.01,
        ),
    )
    result = model.evaluate(
        "2024-01-03",
        pd.Series({"A": 0.10, "B": -0.10}),
        pd.Series({"A": 0.50, "B": -0.50}),
    )

    assert not result.executable
    assert result.capacity_breach_count == 2
    assert result.locate_failure_count == 1


def test_execution_input_builder_is_lagged_against_same_day_volume_and_range():
    dates = pd.bdate_range("2024-01-02", periods=5)
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=dates)
    volume = pd.DataFrame({"A": [10, 20, 30, 40, 50]}, index=dates)
    high = close * 1.01
    low = close * 0.99
    first = build_execution_market_inputs(
        close,
        volume,
        high=high,
        low=low,
        lookback=2,
        minimum_periods=2,
    )
    changed = volume.copy()
    changed.iloc[-1, 0] = 1_000_000
    second = build_execution_market_inputs(
        close,
        changed,
        high=high,
        low=low,
        lookback=2,
        minimum_periods=2,
    )

    assert first.dollar_adv.iloc[-1, 0] == pytest.approx(second.dollar_adv.iloc[-1, 0])
    assert first.half_spread_bps.iloc[-1, 0] == pytest.approx(
        second.half_spread_bps.iloc[-1, 0]
    )
    assert np.isnan(first.dollar_adv.iloc[0, 0])


def test_financing_applies_only_when_long_notional_exceeds_equity():
    model = ExecutableCostModel(_inputs())
    result = model.evaluate(
        "2024-01-03",
        pd.Series({"A": 0.0, "B": 0.0}),
        pd.Series({"A": 1.25, "B": 0.0}),
        equity=1.0,
    )

    assert result.financing_cost > 0
    assert result.cash_income == 0


def test_terminal_trade_pricing_does_not_duplicate_holding_carry():
    model = ExecutableCostModel(_inputs())
    result = model.evaluate(
        "2024-01-03",
        pd.Series({"A": -0.5, "B": 0.5}),
        pd.Series({"A": 0.0, "B": 0.0}),
        include_holding_costs=False,
    )

    assert result.transaction_cost > 0
    assert result.borrow_cost == pytest.approx(0.0)
    assert result.financing_cost == pytest.approx(0.0)
    assert result.cash_income == pytest.approx(0.0)
    assert result.short_rebate_income == pytest.approx(0.0)
    assert result.net_cost == pytest.approx(result.transaction_cost)
