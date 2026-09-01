from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.allocation import (
    SUPPORTED_METHODS,
    AllocationError,
    allocate_pair_sleeves,
    allocation_diagnostics,
    build_allocation_candidates,
    estimate_covariance,
)


def _returns(seed: int = 7, observations: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0001, 0.004, observations)
    return pd.DataFrame(
        {
            "A__B": 0.15 * common + rng.normal(0.00020, 0.003, observations),
            "C__D": 0.25 * common + rng.normal(0.00015, 0.006, observations),
            "A__E": -0.10 * common + rng.normal(0.00010, 0.009, observations),
            "F__G": 0.05 * common + rng.normal(0.00005, 0.012, observations),
        }
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": ["A__B", "C__D", "A__E", "F__G"],
            "sector": ["Tech", "Tech", "Health", "Health"],
            "dependent": ["A", "C", "A", "F"],
            "independent": ["B", "D", "E", "G"],
        }
    )


def test_ledoit_wolf_covariance_is_labeled_annualized_and_positive_definite():
    returns = _returns(observations=200)
    covariance = estimate_covariance(returns, annualization=252)

    assert covariance.index.tolist() == returns.columns.tolist()
    assert covariance.columns.tolist() == returns.columns.tolist()
    assert covariance.attrs["estimator"] == "ledoit_wolf"
    assert 0 <= covariance.attrs["shrinkage"] <= 1
    assert covariance.attrs["observations"] == 200
    assert np.linalg.eigvalsh(covariance.to_numpy()).min() > 0


def test_inverse_volatility_is_non_equal_and_favors_lower_volatility():
    result = allocate_pair_sleeves(_returns(), method="inverse_volatility")

    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()
    assert result.weights["A__B"] > result.weights["F__G"]
    assert result.weights.nunique() > 1
    assert result.diagnostics["effective_number_of_pairs"] < 4
    assert result.effective_method == "inverse_volatility"


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_every_allocator_respects_pair_sector_and_shared_name_caps(method):
    expected = pd.Series(
        {"A__B": 0.12, "C__D": 0.09, "A__E": 0.10, "F__G": 0.07}
    )
    result = allocate_pair_sleeves(
        _returns(),
        method=method,
        expected_returns=expected,
        pair_betas={"A__B": 0.1, "C__D": -0.1, "A__E": 0.2, "F__G": 0.0},
        pair_metadata=_metadata(),
        lower_bounds=0.05,
        upper_bounds=0.45,
        sector_caps={"Tech": 0.60, "Health": 0.60},
        name_incidence_caps={"A": 0.40},
    )

    weights = result.weights
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights >= 0.05 - 1e-8).all()
    assert (weights <= 0.45 + 1e-8).all()
    assert weights[["A__B", "C__D"]].sum() <= 0.60 + 1e-7
    assert weights[["A__E", "F__G"]].sum() <= 0.60 + 1e-7
    assert weights[["A__B", "A__E"]].sum() <= 0.40 + 1e-7
    assert result.diagnostics["expected_market_beta"] == pytest.approx(
        float(weights @ pd.Series({"A__B": 0.1, "C__D": -0.1, "A__E": 0.2, "F__G": 0.0}))
    )
    assert set(result.weight_diagnostics.index) == set(weights.index)
    assert result.constraint_diagnostics["slack"].min() >= -1e-7


def test_equal_risk_contribution_is_nearly_equal_on_independent_assets():
    rng = np.random.default_rng(13)
    scales = np.array([0.004, 0.008, 0.012])
    returns = pd.DataFrame(
        rng.normal(size=(4000, 3)) * scales,
        columns=["A", "B", "C"],
    )
    result = allocate_pair_sleeves(
        returns,
        method="equal_risk_contribution",
        covariance_estimator="sample",
    )
    contributions = result.weight_diagnostics["risk_contribution_fraction"]

    np.testing.assert_allclose(contributions, np.full(3, 1 / 3), atol=0.015)
    assert result.weights["A"] > result.weights["B"] > result.weights["C"]


def test_minimum_variance_and_maximum_diversification_behave_differently():
    returns = _returns(observations=1000)
    minimum_variance = allocate_pair_sleeves(returns, method="minimum_variance")
    maximum_diversification = allocate_pair_sleeves(
        returns, method="maximum_diversification"
    )

    assert minimum_variance.weights["A__B"] > 0.60
    assert not np.allclose(minimum_variance.weights, maximum_diversification.weights)
    assert (
        minimum_variance.diagnostics["expected_annual_volatility"]
        <= maximum_diversification.diagnostics["expected_annual_volatility"] + 1e-8
    )


def test_minimum_covariance_alias_enforces_optional_target_return_floor():
    expected = {"A__B": 0.02, "C__D": 0.30, "A__E": 0.04, "F__G": 0.03}
    result = allocate_pair_sleeves(
        _returns(),
        method="minimum_covariance",
        expected_returns=expected,
        mean_shrinkage=0.0,
        target_return=0.15,
        upper_bounds=0.80,
    )

    assert result.requested_method == "minimum_variance"
    assert result.diagnostics["expected_annual_return"] >= 0.15 - 1e-8
    assert result.diagnostics["target_annual_return"] == pytest.approx(0.15)
    assert result.diagnostics["target_return_slack"] >= -1e-8
    target_row = result.constraint_diagnostics.query(
        "constraint_type == 'expected_return_lower_cap'"
    ).iloc[0]
    assert target_row["allocation"] == pytest.approx(
        result.diagnostics["expected_annual_return"]
    )
    assert target_row["limit"] == pytest.approx(0.15)


def test_infeasible_target_return_is_never_silently_relaxed():
    expected = {"A__B": 0.02, "C__D": 0.30, "A__E": 0.04, "F__G": 0.03}
    with pytest.raises(AllocationError, match="infeasible"):
        allocate_pair_sleeves(
            _returns(),
            method="minimum_variance",
            expected_returns=expected,
            mean_shrinkage=0.0,
            target_return=0.50,
        )


def test_shrunk_tangency_uses_forecasts_but_obeys_regularization_and_cap():
    expected = {"A__B": 0.04, "C__D": 0.30, "A__E": 0.02, "F__G": 0.01}
    result = allocate_pair_sleeves(
        _returns(),
        method="shrunk_tangency",
        expected_returns=expected,
        mean_shrinkage=0.25,
        upper_bounds=0.50,
        l2_penalty=0.05,
    )

    assert result.weights.idxmax() == "C__D"
    assert result.weights["C__D"] > 0.40
    assert result.weights.max() <= 0.50 + 1e-7
    assert result.diagnostics["expected_annual_return"] == pytest.approx(
        float(result.weights @ result.expected_returns)
    )


def test_custom_name_incidence_matrix_is_supported():
    incidence = pd.DataFrame(
        {"SHARED": [1, 1, 0, 0]}, index=_returns().columns
    )
    result = allocate_pair_sleeves(
        _returns(),
        method="minimum_variance",
        upper_bounds=0.60,
        name_incidence=incidence,
        name_incidence_caps={"SHARED": 0.45},
    )

    assert result.weights.iloc[:2].sum() <= 0.45 + 1e-7


def test_custom_name_incidence_above_one_is_rejected_not_clipped():
    incidence = pd.DataFrame(
        {"SHARED": [1.2, 0.0, 0.0, 0.0]}, index=_returns().columns
    )
    with pytest.raises(AllocationError, match=r"in \[0, 1\]"):
        allocate_pair_sleeves(
            _returns(),
            name_incidence=incidence,
            name_incidence_caps={"SHARED": 0.45},
        )


def test_generic_signed_exposure_band_supports_market_beta_control_in_either_orientation():
    beta_by_pair = pd.DataFrame(
        [[0.8, 0.5, -0.4, -0.7]],
        index=["market_beta"],
        columns=_returns().columns,
    )
    result = allocate_pair_sleeves(
        _returns(),
        method="minimum_variance",
        linear_exposures=beta_by_pair,
        linear_exposure_bounds={"market_beta": 0.02},
    )
    realized_beta = float(beta_by_pair.loc["market_beta"] @ result.weights)

    assert abs(realized_beta) <= 0.02 + 1e-7
    beta_rows = result.constraint_diagnostics.query(
        "constraint_type.str.startswith('linear_exposure')", engine="python"
    )
    assert len(beta_rows) == 2
    assert beta_rows["slack"].min() >= -1e-7


def test_explicit_gross_net_market_factor_and_group_controls_are_auditable():
    assets = _returns().columns
    pair_betas = {"A__B": 0.7, "C__D": 0.4, "A__E": -0.5, "F__G": -0.8}
    gross = {"A__B": 1.0, "C__D": 1.2, "A__E": 1.8, "F__G": 2.0}
    net = {"A__B": 0.5, "C__D": 0.2, "A__E": -0.4, "F__G": -0.7}
    factor_betas = pd.DataFrame(
        {
            "size": [0.4, -0.2, 0.1, -0.3],
            "value": [0.1, 0.6, -0.5, -0.2],
        },
        index=assets,
    )
    groups = pd.DataFrame(
        {
            "shared_ab": [1.0, 1.0, 0.0, 0.0],
            "other": [0.0, 0.0, 1.0, 1.0],
        },
        index=assets,
    )
    result = allocate_pair_sleeves(
        _returns(),
        method="minimum_variance",
        upper_bounds=0.60,
        group_incidence=groups,
        group_caps=0.65,
        pair_gross_exposures=gross,
        gross_exposure_bounds=(1.35, 1.75),
        pair_net_exposures=net,
        net_exposure_bounds=0.10,
        pair_betas=pair_betas,
        market_beta_bounds=0.05,
        factor_betas=factor_betas,
        factor_beta_bounds={"size": 0.10, "value": (-0.15, 0.15)},
    )

    weights = result.weights
    assert 1.35 - 1e-7 <= float(weights @ pd.Series(gross)) <= 1.75 + 1e-7
    assert abs(float(weights @ pd.Series(net))) <= 0.10 + 1e-7
    assert abs(float(weights @ pd.Series(pair_betas))) <= 0.05 + 1e-7
    assert (groups.T @ weights <= 0.65 + 1e-7).all()
    factor_realized = factor_betas.T @ weights
    assert abs(factor_realized["size"]) <= 0.10 + 1e-7
    assert -0.15 - 1e-7 <= factor_realized["value"] <= 0.15 + 1e-7
    assert result.diagnostics["portfolio_gross_exposure"] == pytest.approx(
        float(weights @ pd.Series(gross))
    )
    assert result.diagnostics["portfolio_net_exposure"] == pytest.approx(
        float(weights @ pd.Series(net))
    )
    assert result.diagnostics["factor_beta_exposures"] == pytest.approx(
        factor_realized.to_dict()
    )
    controlled_rows = result.constraint_diagnostics.query(
        "constraint_type.str.contains('gross|net|market_beta|factor_beta|group')",
        engine="python",
    )
    assert controlled_rows["slack"].min() >= -1e-7


def test_signed_exposure_can_have_a_strictly_negative_band():
    net = {"A__B": 0.5, "C__D": 0.2, "A__E": -0.4, "F__G": -0.7}
    result = allocate_pair_sleeves(
        _returns(),
        method="minimum_variance",
        pair_net_exposures=net,
        net_exposure_bounds=(-0.30, -0.10),
    )

    realized = float(result.weights @ pd.Series(net))
    assert -0.30 - 1e-7 <= realized <= -0.10 + 1e-7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gross_exposure_bounds": 1.5}, "gross exposures are required"),
        (
            {
                "pair_gross_exposures": {
                    "A__B": 1.0,
                    "C__D": -1.0,
                    "A__E": 1.0,
                    "F__G": 1.0,
                },
                "gross_exposure_bounds": 1.5,
            },
            "gross exposures must be non-negative",
        ),
        ({"market_beta_bounds": 0.05}, "market exposures are required"),
        ({"group_incidence": pd.DataFrame()}, "group_caps are required"),
    ],
)
def test_explicit_production_control_inputs_fail_closed(kwargs, message):
    with pytest.raises(AllocationError, match=message):
        allocate_pair_sleeves(_returns(), **kwargs)


def test_alpha_tilted_erc_targets_non_equal_risk_budgets():
    budgets = {"A__B": 0.325, "C__D": 0.275, "A__E": 0.225, "F__G": 0.175}
    result = allocate_pair_sleeves(
        _returns(observations=2000),
        method="equal_risk_contribution",
        covariance_estimator="sample",
        target_risk_budgets=budgets,
    )

    actual = result.weight_diagnostics["risk_contribution_fraction"]
    target = pd.Series(budgets).reindex(actual.index)
    np.testing.assert_allclose(actual, target, atol=0.015)
    assert "target_risk_budget" in result.weight_diagnostics
    assert result.diagnostics["maximum_absolute_risk_budget_deviation"] < 0.015


def test_diagnostics_report_hhi_effective_n_return_beta_and_risk():
    assets = pd.Index(["P1", "P2"])
    weights = pd.Series([0.75, 0.25], index=assets)
    covariance = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=assets, columns=assets)
    detail, summary = allocation_diagnostics(
        weights,
        covariance,
        expected_returns={"P1": 0.10, "P2": 0.20},
        pair_betas={"P1": 0.2, "P2": -0.2},
    )

    assert summary["weight_hhi"] == pytest.approx(0.625)
    assert summary["effective_number_of_pairs"] == pytest.approx(1.6)
    assert summary["expected_annual_return"] == pytest.approx(0.125)
    assert summary["expected_market_beta"] == pytest.approx(0.1)
    assert detail["risk_contribution_fraction"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lower_bounds": 0.30}, "fully invested"),
        ({"upper_bounds": 0.20}, "fully invested"),
        (
            {
                "pair_metadata": _metadata(),
                "lower_bounds": 0.20,
                "sector_caps": {"Tech": 0.30},
            },
            "mandatory lower-bound",
        ),
        (
            {"pair_metadata": _metadata(), "name_incidence_caps": {"TYPO": 0.2}},
            "unknown underlying",
        ),
    ],
)
def test_invalid_or_infeasible_constraints_fail_clearly(kwargs, message):
    with pytest.raises(AllocationError, match=message):
        allocate_pair_sleeves(_returns(), **kwargs)


def test_candidate_builder_canonicalizes_aliases_and_is_deterministic():
    first = build_allocation_candidates(
        _returns(), methods=["inverse_vol", "erc", "min_var"]
    )
    second = build_allocation_candidates(
        _returns(), methods=["inverse_volatility", "equal_risk_contribution", "minimum_variance"]
    )

    assert list(first) == [
        "inverse_volatility",
        "equal_risk_contribution",
        "minimum_variance",
    ]
    for method in first:
        pd.testing.assert_series_equal(first[method].weights, second[method].weights)


def test_missing_history_and_misaligned_forecasts_are_rejected():
    returns = _returns(observations=10)
    returns["A__B"] = np.nan
    with pytest.raises(AllocationError, match="entirely missing"):
        allocate_pair_sleeves(returns)

    with pytest.raises(AllocationError, match="labels do not match"):
        allocate_pair_sleeves(
            _returns(), expected_returns={"A__B": 0.1}, method="shrunk_tangency"
        )
