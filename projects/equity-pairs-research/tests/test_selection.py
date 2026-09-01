from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.selection import (
    PairSelectionSpec,
    alpha_tilted_risk_budgets,
    build_pair_selection_features,
    select_diversified_pairs,
    select_constrained_pairs_milp,
)


def _synthetic_inputs(pair_count: int = 12):
    dates = pd.bdate_range("2022-01-03", periods=260)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {
            f"A{i}__B{i}": 0.00005 * (pair_count - i) + rng.normal(0, 0.003 + i * 0.0001, len(dates))
            for i in range(pair_count)
        },
        index=dates,
    )
    metadata = pd.DataFrame(
        {
            "pair": returns.columns,
            "sector": [f"S{i // 2}" for i in range(pair_count)],
            "dependent": [f"A{i}" for i in range(pair_count)],
            "independent": [f"B{i}" for i in range(pair_count)],
        }
    )
    trades = pd.DataFrame(
        [
            {"pair": pair, "entry_date": date}
            for pair in returns
            for date in dates[::40]
        ]
    )
    return returns, metadata, trades


def test_feature_scores_and_diversified_selection_are_deterministic():
    returns, metadata, trades = _synthetic_inputs()
    features = build_pair_selection_features(returns, metadata, trades=trades)
    spec = PairSelectionSpec(
        pair_count=6,
        maximum_pairs_per_sector=1,
        minimum_sector_count=6,
    )
    first = select_diversified_pairs(features, spec)
    second = select_diversified_pairs(features, spec)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    assert first["sector"].nunique() == 6
    assert len(set(first["dependent"]) | set(first["independent"])) == 12
    assert first["selection_score"].between(0, 1).all()


def test_feature_score_honors_declared_component_weights():
    returns, metadata, trades = _synthetic_inputs()
    spec = PairSelectionSpec(
        sharpe_weight=1.0,
        cagr_weight=0.0,
        worst_half_sharpe_weight=0.0,
        positive_month_fraction_weight=0.0,
        drawdown_weight=0.0,
    )
    features = build_pair_selection_features(
        returns,
        metadata,
        trades=trades,
        spec=spec,
    )

    np.testing.assert_allclose(
        features["selection_score"],
        features["sharpe_percentile"],
    )


def test_selection_rejects_caps_that_cannot_fill_book():
    returns, metadata, trades = _synthetic_inputs(4)
    features = build_pair_selection_features(returns, metadata, trades=trades)
    with pytest.raises(ValueError, match="allow only"):
        select_diversified_pairs(
            features,
            PairSelectionSpec(
                pair_count=4,
                maximum_pairs_per_sector=1,
                minimum_sector_count=2,
            ),
        )


def test_alpha_tilted_budgets_are_bounded_and_normalized():
    selected = pd.DataFrame(
        {"pair": ["A__B", "C__D", "E__F"], "selection_score": [0.1, 0.5, 0.9]}
    )
    budgets = alpha_tilted_risk_budgets(selected)

    assert budgets.sum() == pytest.approx(1.0)
    assert budgets.loc["E__F"] > budgets.loc["C__D"] > budgets.loc["A__B"]
    multipliers = budgets * len(budgets)
    assert multipliers.min() >= 0.70 - 1e-12
    assert multipliers.max() <= 1.30 + 1e-12


def test_milp_selection_enforces_beta_ticker_and_sector_constraints():
    returns, metadata, trades = _synthetic_inputs()
    features = build_pair_selection_features(returns, metadata, trades=trades)
    betas = pd.Series(
        {pair: (-0.02 if index % 2 else 0.02) for index, pair in enumerate(returns.columns)}
    )
    correlations = returns.corr()
    selected = select_constrained_pairs_milp(
        features,
        PairSelectionSpec(
            pair_count=6,
            maximum_pairs_per_sector=1,
            minimum_sector_count=6,
        ),
        pair_betas=betas,
        equal_weight_beta_bound=0.01,
        pair_correlations=correlations,
        maximum_pair_correlation=0.95,
    )
    assert len(selected) == 6
    assert selected["sector"].nunique() == 6
    assert abs(selected["tuning_market_beta"].mean()) <= 0.01 + 1e-12
    tickers = pd.concat([selected["dependent"], selected["independent"]])
    assert tickers.nunique() == 12
