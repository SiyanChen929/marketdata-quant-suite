from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.pair_health import (
    PairHealthConfig,
    PairHealthReference,
    apply_kill_switch,
    classify_pair_health,
    gate_target_positions,
    rolling_pair_diagnostics,
    run_pair_health,
)


def _cointegrated_prices(observations: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(29082026)
    log_x = 4.2 + np.cumsum(rng.normal(0.0, 0.012, observations))
    residual = np.zeros(observations)
    innovations = rng.normal(0.0, 0.006, observations)
    for position in range(1, observations):
        residual[position] = 0.55 * residual[position - 1] + innovations[position]
    log_y = 0.4 + 1.25 * log_x + residual
    return pd.DataFrame(
        {"Y": np.exp(log_y), "X": np.exp(log_x)},
        index=pd.bdate_range("2024-01-02", periods=observations),
    )


def _permissive_config(**overrides) -> PairHealthConfig:
    values = {
        "lookback_sessions": 60,
        "minimum_observations": 40,
        "decision_lag_sessions": 1,
        "adf_maxlag": 1,
        "maximum_cointegration_pvalue": None,
        "maximum_residual_adf_pvalue": None,
        "maximum_beta_relative_drift": None,
        "maximum_half_life_days": None,
        "maximum_half_life_multiple": None,
        "minimum_structural_break_pvalue": None,
    }
    values.update(overrides)
    return PairHealthConfig(**values)


def test_rolling_diagnostics_are_lagged_and_recover_the_hedge_relation():
    prices = _cointegrated_prices()
    reference = PairHealthReference("Y__X", "Y", "X", beta=1.25, half_life_days=2.0)

    diagnostics = rolling_pair_diagnostics(reference, prices, _permissive_config())
    last = diagnostics.iloc[-1]

    assert last["as_of_date"] == prices.index[-2]
    assert last["observations"] == 60
    assert last["beta"] == pytest.approx(1.25, abs=0.12)
    assert last["beta_relative_drift"] < 0.10
    assert 0.0 <= last["cointegration_pvalue"] <= 1.0
    assert 0.0 <= last["residual_adf_pvalue"] <= 1.0
    assert last["half_life_days"] > 0
    assert 0.0 <= last["structural_break_pvalue"] <= 1.0
    assert bool(last["diagnostic_valid"]) is True


def test_current_and_future_prices_cannot_change_same_day_health_signal():
    prices = _cointegrated_prices(100)
    reference = PairHealthReference("Y__X", "Y", "X", beta=1.25)
    config = _permissive_config()
    baseline = rolling_pair_diagnostics(reference, prices, config)

    change_position = 80
    shocked = prices.copy()
    shocked.iloc[change_position:, shocked.columns.get_loc("Y")] *= 3.0
    revised = rolling_pair_diagnostics(reference, shocked, config)

    # The row dated t only sees through t-1, so a changed close at t cannot
    # alter t's health signal.  It is visible beginning on t+1.
    columns = ["beta", "cointegration_pvalue", "half_life_days"]
    pd.testing.assert_series_equal(
        baseline.iloc[change_position][columns],
        revised.iloc[change_position][columns],
    )
    assert not np.isclose(
        baseline.iloc[change_position + 1]["beta"],
        revised.iloc[change_position + 1]["beta"],
    )

    extended = pd.concat(
        [
            prices,
            pd.DataFrame(
                {"Y": [1_000.0], "X": [1.0]},
                index=[prices.index[-1] + pd.offsets.BDay()],
            ),
        ]
    )
    extended_diagnostics = rolling_pair_diagnostics(reference, extended, config)
    pd.testing.assert_frame_equal(
        baseline,
        extended_diagnostics.loc[baseline.index],
        check_freq=False,
    )


def test_classification_identifies_each_health_failure():
    dates = pd.bdate_range("2025-01-02", periods=5)
    diagnostics = pd.DataFrame(
        {
            "diagnostic_valid": [True, True, True, True, False],
            "stale_sessions": [0, 0, 0, 0, 4],
            "beta": [1.0, 1.5, 1.0, 1.0, np.nan],
            "beta_relative_drift": [0.0, 0.5, 0.0, 0.0, np.nan],
            "cointegration_pvalue": [0.01, 0.01, 0.30, 0.01, np.nan],
            "residual_adf_pvalue": [0.01, 0.01, 0.30, 0.01, np.nan],
            "half_life_days": [5.0, 5.0, 5.0, 50.0, np.nan],
            "half_life_multiple": [1.0, 1.0, 1.0, 10.0, np.nan],
            "structural_break_pvalue": [0.50, 0.50, 0.50, 0.001, np.nan],
        },
        index=dates,
    )
    reference = PairHealthReference("Y__X", "Y", "X", 1.0, 5.0)
    config = PairHealthConfig(
        lookback_sessions=40,
        minimum_observations=20,
        maximum_half_life_days=20,
        maximum_half_life_multiple=3.0,
    )

    classified = classify_pair_health(diagnostics, reference, config)

    assert classified["unhealthy"].tolist() == [False, True, True, True, True]
    assert classified.iloc[1]["health_reasons"] == "beta"
    assert classified.iloc[2]["health_reasons"] == "stationarity"
    assert classified.iloc[3]["health_reasons"] == "half_life|structural_break"
    assert "data" in classified.iloc[4]["health_reasons"]


def test_kill_switch_warns_kills_cools_down_and_requires_recovery_streak():
    dates = pd.bdate_range("2025-01-02", periods=9)
    unhealthy = [False, True, True, False, False, False, True, False, False]
    classified = pd.DataFrame(
        {
            "diagnostic_valid": True,
            "unhealthy": unhealthy,
            "health_reasons": ["beta" if value else "" for value in unhealthy],
        },
        index=dates,
    )
    config = _permissive_config(
        bad_observations_to_kill=2,
        cooldown_sessions=2,
        good_observations_to_recover=2,
    )

    signals = apply_kill_switch(classified, config)

    assert signals["health_state"].tolist() == [
        "tradable",
        "warning",
        "killed",
        "cooldown",
        "cooldown",
        "recovery",
        "recovery",
        "recovery",
        "tradable",
    ]
    assert signals["allow_entry"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert signals["force_exit"].sum() == 1
    assert bool(signals.iloc[2]["kill_event"]) is True
    assert bool(signals.iloc[-1]["recovery_event"]) is True
    assert signals.iloc[3]["active_kill_reason"] == "beta"
    assert signals.iloc[-1]["active_kill_reason"] == ""


def test_run_pair_health_returns_integration_controls_events_and_summary():
    prices = _cointegrated_prices(90)
    reference = PairHealthReference("Y__X", "Y", "X", 1.25)
    config = _permissive_config()

    result = run_pair_health(reference, prices, config)

    assert result.reference == reference
    assert result.config == config
    assert {
        "allow_entry",
        "force_exit",
        "target_gross_multiplier",
        "health_state",
        "as_of_date",
    }.issubset(result.signals.columns)
    assert result.summary["pair"] == "Y__X"
    assert result.summary["observations"] == len(prices)
    assert result.summary["valid_diagnostic_observations"] > 0


def test_position_gate_blocks_warning_entries_but_only_kill_forces_existing_trade_flat():
    dates = pd.bdate_range("2025-02-03", periods=7)
    desired = pd.Series([1, 1, 1, 1, 0, -1, -1], index=dates)
    health = pd.DataFrame(
        {
            "allow_entry": [False, True, False, False, False, False, True],
            "target_gross_multiplier": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        },
        index=dates,
    )

    gated = gate_target_positions(desired, health)

    # Day 1 warning blocks entry; day 3 warning permits the already-open long
    # to remain; day 4 kill forces flat; the short waits for entry permission.
    assert gated.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback_sessions": 19},
        {"lookback_sessions": 30, "minimum_observations": 31},
        {"decision_lag_sessions": -1},
        {"maximum_cointegration_pvalue": 1.1},
        {"bad_observations_to_kill": 0},
        {"cooldown_sessions": -1},
    ],
)
def test_health_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PairHealthConfig(**kwargs)
