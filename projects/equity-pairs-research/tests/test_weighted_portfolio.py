from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.execution import (
    ExecutableCostModel,
    ExecutionCostSpec,
    ExecutionMarketInputs,
)
from equity_pairs.weighted_portfolio import (
    alpha_tilted_risk_budgets_from_forecasts,
    backtest_weighted_pair_book,
    backtest_weighted_sleeves,
    build_gross_name_incidence,
    empirical_bayes_monthly_forecasts,
)


def test_name_incidence_uses_gross_normalized_hedge_coefficients():
    metadata = pd.DataFrame(
        {
            "pair": ["A__B", "C__D"],
            "dependent": ["A", "C"],
            "independent": ["B", "D"],
            "beta": [1.0, 3.0],
        }
    )
    incidence = build_gross_name_incidence(metadata)
    assert incidence.loc["A__B", "A"] == pytest.approx(0.5)
    assert incidence.loc["A__B", "B"] == pytest.approx(0.5)
    assert incidence.loc["C__D", "C"] == pytest.approx(0.25)
    assert incidence.loc["C__D", "D"] == pytest.approx(0.75)
    np.testing.assert_allclose(incidence.sum(axis=1), 1.0)


def test_empirical_bayes_forecasts_and_risk_budgets_are_bounded():
    dates = pd.bdate_range("2022-01-03", periods=504)
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(
        {
            "A__B": rng.normal(0.0003, 0.005, len(dates)),
            "C__D": rng.normal(0.0001, 0.004, len(dates)),
            "E__F": rng.normal(-0.0001, 0.006, len(dates)),
        },
        index=dates,
    )
    forecasts = empirical_bayes_monthly_forecasts(returns)
    budgets = alpha_tilted_risk_budgets_from_forecasts(forecasts)
    assert forecasts["capped_forecast_sharpe"].abs().max() <= 0.5 + 1e-12
    assert forecasts["posterior_data_weight"].between(0, 1).all()
    assert budgets.sum() == pytest.approx(1.0)
    assert (budgets * len(budgets)).between(0.70, 1.30).all()


def test_weighted_sleeve_backtest_charges_launch_rebalance_and_liquidation():
    dates = pd.bdate_range("2024-01-30", periods=25)
    returns = pd.DataFrame(
        {"A__B": [0.01] + [0.0] * 24, "C__D": [0.0] * 25},
        index=dates,
    )
    weights = pd.Series({"A__B": 0.6, "C__D": 0.4})
    active = pd.DataFrame(True, index=dates, columns=returns.columns)
    turnover = pd.DataFrame(0.0, index=dates, columns=returns.columns)
    result = backtest_weighted_sleeves(
        returns,
        weights,
        active_positions=active,
        unit_turnover=turnover,
        allocation_cost_bps=10.0,
        no_trade_band=0.0,
    )
    assert result.rebalance_log.iloc[0]["event"] == "launch"
    assert result.rebalance_log.iloc[-1]["event"] == "terminal_liquidation"
    assert result.metrics["scheduled_rebalance_count"] >= 1
    assert result.metrics["outer_allocation_cost_sum"] > 0
    assert result.metrics["outer_allocation_cost_bps_on_initial_capital"] > 0
    assert result.metrics["launch_turnover"] > 0
    assert result.deployed_weights.shape == result.realized_weights.shape
    np.testing.assert_allclose(result.deployed_weights.sum(axis=1), 1.0)
    assert result.daily["equity"].iloc[-1] < 1.006


def test_weighted_sleeve_input_validation():
    dates = pd.bdate_range("2024-01-02", periods=5)
    returns = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    with pytest.raises(ValueError, match="sum to one"):
        backtest_weighted_sleeves(returns, pd.Series({"A": 0.7, "B": 0.7}))


def test_exact_book_costs_entry_once_and_exit_once():
    dates = pd.bdate_range("2024-01-02", periods=3)
    signals = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "price_dependent": [100.0, 100.0, 100.0],
            "price_independent": [100.0, 100.0, 100.0],
            "weight_dependent": [0.5, 0.5, 0.0],
            "weight_independent": [-0.5, -0.5, 0.0],
            "gross_return": 0.0,
            "borrow_cost": 0.0,
        }
    )
    result = backtest_weighted_pair_book(
        signals,
        pd.Series({"A__B": 1.0}),
        transaction_cost_bps=10.0,
        rebalance_frequency="none",
        charge_terminal_liquidation=True,
    )

    # One unit of gross exposure is traded on entry and one on exit. There is
    # no additional outer-book charge and no terminal charge after the signal
    # has already flattened the pair.
    assert result.metrics["transaction_cost_sum"] == pytest.approx(0.002, abs=3e-6)
    assert result.metrics["transaction_cost_drag_bps"] == pytest.approx(20.01, abs=0.03)
    assert result.rebalance_log["event"].eq("terminal_liquidation").sum() == 0


def test_exact_book_rebalance_uses_underlying_notional_change():
    dates = pd.to_datetime(["2024-01-31", "2024-02-01"])
    rows = []
    for pair, dep, ind in (("A__B", 0.5, -0.5), ("C__D", 0.5, -0.5)):
        for date in dates:
            rows.append(
                {
                    "pair": pair,
                    "date": date,
                    "price_dependent": 100.0,
                    "price_independent": 100.0,
                    "weight_dependent": dep,
                    "weight_independent": ind,
                    "gross_return": 0.0,
                    "borrow_cost": 0.0,
                }
            )
    signals = pd.DataFrame(rows)
    result = backtest_weighted_pair_book(
        signals,
        pd.Series({"A__B": 0.6, "C__D": 0.4}),
        transaction_cost_bps=10.0,
        no_trade_band=0.0,
        charge_terminal_liquidation=False,
    )

    # With flat prices, the month boundary creates no capital rebalance.  The
    # only second-day trade is the tiny deleveraging required after day-one
    # transaction costs reduced NAV.
    assert result.daily["leg_turnover"].iloc[0] == pytest.approx(1.0)
    assert result.metrics["scheduled_rebalance_count"] == 0
    assert result.daily["leg_turnover"].iloc[1] == pytest.approx(0.001001001, abs=2e-6)


def test_exact_book_uses_ticker_level_executable_costs_and_terminal_liquidation():
    dates = pd.bdate_range("2024-01-02", periods=3)
    signals = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "price_dependent": 100.0,
            "price_independent": 100.0,
            "weight_dependent": 0.5,
            "weight_independent": -0.5,
            "gross_return": 0.0,
            "borrow_cost": 0.99,  # ignored when the executable model is active
        }
    )
    inputs = ExecutionMarketInputs(
        dollar_adv=pd.DataFrame(1_000_000_000.0, index=dates, columns=["A", "B"]),
        daily_volatility=pd.DataFrame(0.01, index=dates, columns=["A", "B"]),
        half_spread_bps=pd.DataFrame(1.0, index=dates, columns=["A", "B"]),
        annual_borrow_bps=pd.DataFrame(25.0, index=dates, columns=["A", "B"]),
        locate_available=pd.DataFrame(True, index=dates, columns=["A", "B"]),
        annual_cash_rate=0.0,
    )
    model = ExecutableCostModel(
        inputs,
        ExecutionCostSpec(
            portfolio_notional=1_000_000.0,
            commission_bps=0.1,
            impact_coefficient=0.0,
            maximum_participation_rate=0.05,
        ),
    )
    result = backtest_weighted_pair_book(
        signals,
        pd.Series({"A__B": 1.0}),
        transaction_cost_bps=9_999.0,
        rebalance_frequency="none",
        execution_cost_model=model,
        pair_metadata=pd.DataFrame(
            {"pair": ["A__B"], "dependent": ["A"], "independent": ["B"]}
        ),
    )

    assert result.metrics["execution_cost_model_enabled"]
    assert result.metrics["borrow_cost_sum"] < 0.001
    assert result.metrics["commission_cost_sum"] > 0
    assert result.metrics["spread_cost_sum"] > 0
    assert result.execution_log is not None
    assert set(result.execution_log["event"]) == {"regular", "terminal_liquidation"}
    assert result.execution_log.loc[
        result.execution_log["event"].eq("terminal_liquidation"),
        "borrow_cost_nav",
    ].sum() == pytest.approx(0.0)


def test_exact_book_fails_closed_on_capacity_breach():
    dates = pd.bdate_range("2024-01-02", periods=2)
    signals = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "price_dependent": 100.0,
            "price_independent": 100.0,
            "weight_dependent": 0.5,
            "weight_independent": -0.5,
            "gross_return": 0.0,
            "borrow_cost": 0.0,
        }
    )
    model = ExecutableCostModel(
        ExecutionMarketInputs(
            dollar_adv=pd.DataFrame(100_000.0, index=dates, columns=["A", "B"]),
            daily_volatility=pd.DataFrame(0.01, index=dates, columns=["A", "B"]),
        ),
        ExecutionCostSpec(
            portfolio_notional=10_000_000.0,
            maximum_participation_rate=0.01,
        ),
    )
    with pytest.raises(ValueError, match="execution constraints failed"):
        backtest_weighted_pair_book(
            signals,
            pd.Series({"A__B": 1.0}),
            execution_cost_model=model,
            pair_metadata=pd.DataFrame(
                {"pair": ["A__B"], "dependent": ["A"], "independent": ["B"]}
            ),
        )


def test_exact_book_applies_dated_target_schedule_at_monthly_rebalance():
    dates = pd.to_datetime(["2024-01-31", "2024-02-01"])
    signals = pd.DataFrame(
        [
            {
                "pair": pair,
                "date": date,
                "price_dependent": 100.0,
                "price_independent": 100.0,
                "weight_dependent": 0.5,
                "weight_independent": -0.5,
                "gross_return": 0.0,
                "borrow_cost": 0.0,
            }
            for pair in ("A__B", "C__D")
            for date in dates
        ]
    )
    schedule = pd.DataFrame(
        [[0.6, 0.4], [0.2, 0.8]],
        index=dates,
        columns=["A__B", "C__D"],
    )
    result = backtest_weighted_pair_book(
        signals,
        pd.Series({"A__B": 0.6, "C__D": 0.4}),
        transaction_cost_bps=0.0,
        no_trade_band=0.0,
        maximum_rebalance_turnover=2.0,
        charge_terminal_liquidation=False,
        target_weight_schedule=schedule,
    )

    np.testing.assert_allclose(result.deployed_weights.iloc[0], [0.6, 0.4])
    np.testing.assert_allclose(result.deployed_weights.iloc[1], [0.2, 0.8])
    assert result.metrics["target_weight_schedule_enabled"]
    assert result.metrics["target_weight_schedule_observations"] == 2
