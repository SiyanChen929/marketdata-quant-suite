"""No-lookahead property: data after date T never changes factor values at or before T."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lfm_helpers import make_bars, perturb_after
from llm_factor_mining.data import panel_from_bars
from llm_factor_mining.dsl import LIBRARY, OPERATORS, TERMINALS, ArgKind
from llm_factor_mining.evaluate import Evaluator


def _operator_expression(name: str) -> str:
    spec = OPERATORS[name]
    series = iter(["returns", "volume"])
    args = []
    for kind in spec.arg_kinds:
        if kind is ArgKind.SERIES:
            args.append(next(series))
        elif kind is ArgKind.WINDOW:
            args.append("5")
        else:
            args.append("2")
    return f"{name}({', '.join(args)})"


COMPOSITES = [
    "cs_rank(ts_corr(delay(close, 3), ts_rank(volume, 5), 7))",
    "decay_linear(cs_zscore(ts_argmax(high - low, 6)), 4) / ts_mean(dollar_volume, 10)",
    "cs_demean(ts_cov(returns, delta(log(volume), 2), 8)) * sign(ts_zscore(vwap_proxy, 5))",
]


@pytest.fixture(scope="module")
def perturbed_pair():
    bars = make_bars(70, 10, seed=11)
    cutoff = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))[45]
    original = Evaluator(panel_from_bars(bars))
    changed = Evaluator(panel_from_bars(perturb_after(bars, cutoff)))
    return original, changed, cutoff


def _assert_causal(original: Evaluator, changed: Evaluator, cutoff: pd.Timestamp, expression: str) -> None:
    a = original.evaluate(expression)
    b = changed.evaluate(expression)
    past_a, past_b = a.loc[:cutoff].to_numpy(), b.loc[:cutoff].to_numpy()
    assert np.array_equal(past_a, past_b, equal_nan=True), expression
    future_a, future_b = a.loc[a.index > cutoff].to_numpy(), b.loc[b.index > cutoff].to_numpy()
    assert not np.array_equal(future_a, future_b, equal_nan=True), f"vacuous test: {expression}"


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_every_operator_is_causal(perturbed_pair, name: str) -> None:
    original, changed, cutoff = perturbed_pair
    _assert_causal(original, changed, cutoff, _operator_expression(name))


@pytest.mark.parametrize("name", sorted(TERMINALS))
def test_every_terminal_is_causal(perturbed_pair, name: str) -> None:
    original, changed, cutoff = perturbed_pair
    _assert_causal(original, changed, cutoff, name)


@pytest.mark.parametrize("expression", COMPOSITES)
def test_nested_expressions_are_causal(perturbed_pair, expression: str) -> None:
    original, changed, cutoff = perturbed_pair
    _assert_causal(original, changed, cutoff, expression)


def test_library_factors_are_causal() -> None:
    bars = make_bars(300, 8, seed=21)
    cutoff = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))[270]
    original = Evaluator(panel_from_bars(bars))
    changed = Evaluator(panel_from_bars(perturb_after(bars, cutoff)))
    for factor in LIBRARY:
        _assert_causal(original, changed, cutoff, factor.expression)


def test_cross_sectional_operators_are_date_local() -> None:
    bars = make_bars(30, 10, seed=5)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))
    target = dates[12]
    one_day = bars.copy()
    rows = pd.to_datetime(one_day["date"]).eq(target)
    factor = np.linspace(0.7, 1.3, int(rows.sum()))
    for column in ("open", "high", "low", "close"):
        one_day.loc[rows, column] = one_day.loc[rows, column].to_numpy() * factor
    for name in ("cs_rank", "cs_zscore", "cs_demean"):
        a = Evaluator(panel_from_bars(bars)).evaluate(f"{name}(close)")
        b = Evaluator(panel_from_bars(one_day)).evaluate(f"{name}(close)")
        other = a.index != target
        assert np.array_equal(a.loc[other].to_numpy(), b.loc[other].to_numpy(), equal_nan=True)
        assert not np.array_equal(a.loc[target].to_numpy(), b.loc[target].to_numpy())


def test_warmup_rows_equal_lookback() -> None:
    """With full windows only, the first lookback(expr) sessions are NaN and no more."""

    from llm_factor_mining.dsl import lookback, parse

    panel = panel_from_bars(make_bars(60, 6, seed=2))
    engine = Evaluator(panel)
    for expression in ("ts_mean(returns, 10)", "delay(ts_std(close, 5), 3)", "ts_corr(delta(close, 2), volume, 6)"):
        values = engine.evaluate(expression)
        warmup = lookback(parse(expression))
        assert values.iloc[:warmup].isna().all().all(), expression
        assert values.iloc[warmup].notna().all(), expression
