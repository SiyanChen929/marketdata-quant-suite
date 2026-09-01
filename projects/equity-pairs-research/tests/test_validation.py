from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from equity_pairs.validation import (
    NestedWalkForwardSpec,
    bootstrap_performance_confidence_intervals,
    build_purged_walk_forward_folds,
    folds_to_audit_table,
    reality_check,
    run_nested_walk_forward,
    run_nested_walk_forward_from_returns,
)


def _small_spec(**overrides: object) -> NestedWalkForwardSpec:
    values: dict[str, object] = {
        "outer_train_size": 35,
        "outer_test_size": 6,
        "inner_train_size": 12,
        "inner_validation_size": 4,
        "outer_step_size": 7,
        "inner_step_size": 5,
        "purge_size": 2,
        "embargo_size": 1,
        "maximum_outer_folds": 3,
        "maximum_inner_folds": 3,
        "selection_metric": "mean_return",
        "minimum_evaluation_observations": 2,
        "bootstrap_samples": 80,
        "bootstrap_block_size": 3,
        "random_seed": 19,
    }
    values.update(overrides)
    return NestedWalkForwardSpec(**values)


def test_purged_fold_boundaries_and_embargo_are_explicit_and_chronological():
    index = pd.RangeIndex(60, name="observation")
    folds = build_purged_walk_forward_folds(
        index,
        initial_train_size=10,
        test_size=4,
        step_size=7,
        purge_size=1,
        embargo_size=3,
        expanding=True,
        maximum_folds=3,
    )

    assert len(folds) == 3
    first, second = folds[:2]
    assert first.train_index.tolist() == list(range(10))
    assert first.purge_index.tolist() == [10]
    assert first.test_index.tolist() == [11, 12, 13, 14]
    assert first.embargo_index.tolist() == [15, 16, 17]
    assert second.test_index.tolist() == [18, 19, 20, 21]
    assert second.prior_embargo_excluded.tolist() == [15, 16]
    assert first.train_index.intersection(first.test_index).empty

    audit = folds_to_audit_table(folds, layer="outer")
    assert audit["chronological"].all()
    assert audit["train_evaluation_overlap"].eq(0).all()
    assert audit.iloc[0]["purged_observations"] == 1
    assert audit.iloc[1]["prior_embargo_excluded_from_train"] == 2


def test_cached_nested_walk_forward_selects_only_on_inner_history():
    dates = pd.bdate_range("2023-01-02", periods=80)
    returns = pd.DataFrame(-0.001, index=dates, columns=["stable", "jackpot"])
    returns["stable"] = 0.001
    # The outer test starts after 45 training observations plus a two-day purge.
    returns.iloc[47:53, returns.columns.get_loc("stable")] = -0.08
    returns.iloc[47:53, returns.columns.get_loc("jackpot")] = 0.10
    spec = _small_spec(
        outer_train_size=45,
        maximum_outer_folds=1,
        bootstrap_samples=60,
    )

    result = run_nested_walk_forward_from_returns(returns, spec)

    selected = result.selection_audit.query("selected")
    assert selected["candidate_id"].tolist() == ["stable"]
    assert result.oos_returns["candidate_id"].unique().tolist() == ["stable"]
    assert result.oos_returns["return"].eq(-0.08).all()
    assert result.outer_evaluations.iloc[0]["test_start"] == dates[47]
    assert result.pbo_fold_audit.iloc[0]["selected_outer_rank"] == 2
    pbo = result.data_snooping_diagnostics.query(
        "method == 'chronological_outer_rank_pbo_style'"
    ).iloc[0]
    assert pbo["pbo_style_probability"] == pytest.approx(1.0)


def test_callback_outer_evaluation_invokes_only_the_frozen_inner_winner():
    dates = pd.bdate_range("2024-01-02", periods=75)
    calls: list[tuple[str, int, int]] = []
    candidates = {
        "alpha": {"name": "alpha", "mean": 0.002},
        "beta": {"name": "beta", "mean": -0.001},
    }

    def evaluator(
        parameter: dict[str, object],
        train_index: pd.Index,
        evaluation_index: pd.Index,
    ) -> pd.Series:
        calls.append((str(parameter["name"]), len(train_index), len(evaluation_index)))
        time = np.arange(len(evaluation_index), dtype=float)
        values = float(parameter["mean"]) + 0.0001 * np.sin(time)
        return pd.Series(values, index=evaluation_index)

    result = run_nested_walk_forward(dates, candidates, evaluator, _small_spec())

    outer_calls = [call for call in calls if call[2] == 6]
    assert len(outer_calls) == 3
    assert {call[0] for call in outer_calls} == {"alpha"}
    assert result.outer_evaluations["candidate_id"].eq("alpha").all()
    assert result.selection_audit.query("selected")["candidate_id"].eq("alpha").all()
    assert result.selection_audit["parameter_fingerprint"].str.len().eq(16).all()
    assert set(result.fold_audit["layer"]) == {"outer", "inner"}
    assert result.fold_audit["train_evaluation_overlap"].eq(0).all()


def test_cached_and_callback_paths_produce_identical_reported_oos_returns():
    dates = pd.bdate_range("2022-01-03", periods=75)
    time = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame(
        {
            "a": 0.001 + 0.0005 * np.sin(time / 3.0),
            "b": -0.0002 + 0.0005 * np.cos(time / 4.0),
        },
        index=dates,
    )
    spec = _small_spec(bootstrap_samples=0)

    cached = run_nested_walk_forward_from_returns(frame, spec)

    def evaluator(column: str, _train: pd.Index, evaluation: pd.Index) -> pd.Series:
        return frame[column].reindex(evaluation)

    callback = run_nested_walk_forward(
        dates,
        {"a": "a", "b": "b"},
        evaluator,
        spec,
    )

    pdt.assert_series_equal(cached.oos_returns["return"], callback.oos_returns["return"])
    assert cached.outer_evaluations["candidate_id"].tolist() == callback.outer_evaluations[
        "candidate_id"
    ].tolist()


def test_bootstrap_confidence_intervals_are_seeded_and_reproducible():
    time = np.arange(90, dtype=float)
    returns = pd.Series(0.0004 + 0.008 * np.sin(time / 5.0))

    first = bootstrap_performance_confidence_intervals(
        returns,
        samples=150,
        block_size=7,
        seed=123,
    )
    second = bootstrap_performance_confidence_intervals(
        returns,
        samples=150,
        block_size=7,
        seed=123,
    )
    different_seed = bootstrap_performance_confidence_intervals(
        returns,
        samples=150,
        block_size=7,
        seed=124,
    )

    pdt.assert_frame_equal(first, second)
    assert not first[["lower", "upper"]].equals(different_seed[["lower", "upper"]])
    assert (first["lower"] <= first["upper"]).all()
    assert first["seed"].eq(123).all()


def test_reality_check_penalizes_search_and_is_deterministic():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-02", periods=180)
    frame = pd.DataFrame(
        {
            "signal": 0.0025 + rng.normal(0.0, 0.002, len(dates)),
            "noise_a": rng.normal(0.0, 0.002, len(dates)),
            "noise_b": rng.normal(0.0, 0.002, len(dates)),
        },
        index=dates,
    )

    first = reality_check(frame, samples=199, block_size=8, seed=71)
    second = reality_check(frame, samples=199, block_size=8, seed=71)

    assert first == second
    assert first["best_candidate"] == "signal"
    assert first["candidate_count"] == 3
    assert first["null_pvalue"] <= 0.05
    assert "style" in first["method"]


def test_invalid_fold_and_evaluator_inputs_fail_clearly():
    with pytest.raises(ValueError, match="sorted"):
        build_purged_walk_forward_folds(
            pd.Index([2, 1, 3]), initial_train_size=1, test_size=1
        )
    with pytest.raises(ValueError, match="outer_step_size"):
        _small_spec(outer_step_size=6, embargo_size=1)

    dates = pd.bdate_range("2024-01-02", periods=60)

    def wrong_length(_parameter: object, _train: pd.Index, _test: pd.Index) -> np.ndarray:
        return np.zeros(1)

    with pytest.raises(ValueError, match="same length"):
        run_nested_walk_forward(
            dates,
            {"bad": object()},
            wrong_length,
            _small_spec(maximum_outer_folds=1, bootstrap_samples=0),
        )
