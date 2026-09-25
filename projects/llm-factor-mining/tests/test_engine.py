from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from lfm_helpers import tiny_bars
from llm_factor_mining.data import Panel, panel_from_bars
from llm_factor_mining.dsl import OPERATORS
from llm_factor_mining.evaluate import KERNELS, Evaluator, ExpressionError, evaluate_expression


DATES = [f"2031-03-{day:02d}" for day in (3, 4, 5, 6, 7, 10, 11, 12)]
CLOSE = {
    "AAA": [10, 11, 12, 11, 13, 13, 12, 14],
    "BBB": [20, 19, 21, 22, 22, 20, 23, 21],
    "CCC": [5, 5, 5, 5, 6, 4, 5, 7],
}
VOLUME = {
    "AAA": [100, 120, 90, 150, 130, 110, 160, 140],
    "BBB": [300, 310, 290, 280, 330, 320, 300, 310],
    "CCC": [50, 55, 60, 65, 70, 75, 80, 85],
}


def _rows(skip: set[tuple[int, str]] | None = None) -> list[tuple[str, str, float, float, float, float, float]]:
    rows = []
    for symbol in CLOSE:
        for index, date in enumerate(DATES):
            if skip and (index, symbol) in skip:
                continue
            close = float(CLOSE[symbol][index])
            rows.append(
                (date, symbol, close * 0.99, close * 1.02, close * 0.97, close, float(VOLUME[symbol][index]))
            )
    return rows


@pytest.fixture()
def tiny() -> Panel:
    return panel_from_bars(tiny_bars(_rows()))


def _frame(panel: Panel, data: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(data, index=panel.dates).reindex(columns=panel.close.columns).astype(float)


def _manual_rolling(frame: pd.DataFrame, window: int, fn: Callable[[np.ndarray], float]) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    for column in range(values.shape[1]):
        for row in range(window - 1, values.shape[0]):
            chunk = values[row - window + 1 : row + 1, column]
            if not np.isnan(chunk).any():
                out[row, column] = fn(chunk)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _manual_rolling2(
    left: pd.DataFrame, right: pd.DataFrame, window: int, fn: Callable[[np.ndarray, np.ndarray], float]
) -> pd.DataFrame:
    a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    for column in range(a.shape[1]):
        for row in range(window - 1, a.shape[0]):
            x = a[row - window + 1 : row + 1, column]
            y = b[row - window + 1 : row + 1, column]
            if not (np.isnan(x).any() or np.isnan(y).any()):
                out[row, column] = fn(x, y)
    return pd.DataFrame(out, index=left.index, columns=left.columns)


def _check(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert actual.index.equals(expected.index) and actual.columns.equals(expected.columns)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True)


def _std(chunk: np.ndarray) -> float:
    return float(np.std(chunk, ddof=1))


def _ts_rank(chunk: np.ndarray) -> float:
    last = chunk[-1]
    return float(((chunk < last).sum() + ((chunk == last).sum() + 1) / 2.0) / chunk.size)


def _zscore(chunk: np.ndarray) -> float:
    std = _std(chunk)
    return float("nan") if std == 0 else float((chunk[-1] - chunk.mean()) / std)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _decay(chunk: np.ndarray) -> float:
    weights = np.arange(1, chunk.size + 1, dtype=float)  # oldest=1 ... today=d
    return float((weights * chunk).sum() / weights.sum())


def _argmax(chunk: np.ndarray) -> float:
    return float(np.argmax(chunk[::-1]))  # first hit in reversed order = most recent


def _argmin(chunk: np.ndarray) -> float:
    return float(np.argmin(chunk[::-1]))


def test_terminals(tiny: Panel) -> None:
    engine = Evaluator(tiny)
    close = _frame(tiny, CLOSE)
    volume = _frame(tiny, VOLUME)
    _check(engine.evaluate("close"), close)
    _check(engine.evaluate("returns"), close / close.shift(1) - 1)
    _check(engine.evaluate("vwap_proxy"), (close * 1.02 + close * 0.97 + close) / 3)
    _check(engine.evaluate("dollar_volume"), close * volume)
    _check(engine.evaluate("open"), close * 0.99)


@pytest.mark.parametrize(
    ("expression", "window", "fn"),
    [
        ("ts_mean(close, 3)", 3, np.mean),
        ("ts_sum(close, 4)", 4, np.sum),
        ("ts_std(close, 3)", 3, _std),
        ("ts_min(close, 3)", 3, np.min),
        ("ts_max(close, 3)", 3, np.max),
        ("ts_rank(close, 4)", 4, _ts_rank),
        ("ts_zscore(close, 3)", 3, _zscore),
        ("decay_linear(close, 4)", 4, _decay),
        ("ts_argmax(close, 4)", 4, _argmax),
        ("ts_argmin(close, 4)", 4, _argmin),
    ],
)
def test_rolling_operators_match_hand_computation(tiny: Panel, expression: str, window: int, fn) -> None:
    close = _frame(tiny, CLOSE)
    _check(evaluate_expression(tiny, expression), _manual_rolling(close, window, fn))


def test_constant_windows_are_degenerate(tiny: Panel) -> None:
    engine = Evaluator(tiny)
    std = engine.evaluate("ts_std(close, 3)")
    assert std.loc[tiny.dates[2], "CCC"] == 0.0
    assert np.isnan(engine.evaluate("ts_zscore(close, 3)").loc[tiny.dates[2], "CCC"])
    assert np.isnan(engine.evaluate("ts_corr(close, volume, 3)").loc[tiny.dates[3], "CCC"])


def test_pairwise_rolling_operators(tiny: Panel) -> None:
    close, volume = _frame(tiny, CLOSE), _frame(tiny, VOLUME)
    _check(evaluate_expression(tiny, "ts_corr(close, volume, 4)"), _manual_rolling2(close, volume, 4, _corr))
    _check(
        evaluate_expression(tiny, "ts_cov(close, volume, 3)"),
        _manual_rolling2(close, volume, 3, lambda x, y: float(np.cov(x, y, ddof=1)[0, 1])),
    )
    corr = evaluate_expression(tiny, "ts_corr(close, volume, 4)").to_numpy()
    assert np.nanmax(np.abs(corr)) <= 1.0


def test_shift_and_elementwise_operators(tiny: Panel) -> None:
    close, volume = _frame(tiny, CLOSE), _frame(tiny, VOLUME)
    engine = Evaluator(tiny)
    _check(engine.evaluate("delay(close, 2)"), close.shift(2))
    _check(engine.evaluate("delta(close, 3)"), close - close.shift(3))
    _check(engine.evaluate("close - volume / 100"), close - volume / 100)
    _check(engine.evaluate("-close * 2 + 1"), -close * 2 + 1)
    _check(engine.evaluate("abs(delta(close, 1))"), (close - close.shift(1)).abs())
    _check(engine.evaluate("sign(delta(close, 1))"), np.sign(close - close.shift(1)))
    r = close / close.shift(1) - 1
    _check(engine.evaluate("log(returns)"), np.sign(r) * np.log1p(r.abs()))
    _check(engine.evaluate("sqrt(returns)"), np.sign(r) * np.sqrt(r.abs()))
    _check(engine.evaluate("power(returns, 3)"), np.sign(r) * r.abs() ** 3)


def test_safe_division_and_non_finite_cleanup(tiny: Panel) -> None:
    engine = Evaluator(tiny)
    assert engine.evaluate("close / (close - close)").isna().all().all()
    inverse = engine.evaluate("power(delta(close, 1), -1)")
    assert np.isfinite(inverse.to_numpy()[~np.isnan(inverse.to_numpy())]).all()
    zero_move = (_frame(tiny, CLOSE).diff() == 0).to_numpy()
    assert np.isnan(inverse.to_numpy()[zero_move]).all()


def test_cross_sectional_operators(tiny: Panel) -> None:
    close = _frame(tiny, CLOSE)
    values = close.to_numpy()
    ranks = np.vstack([rankdata(row) / row.size for row in values])
    _check(evaluate_expression(tiny, "cs_rank(close)"), pd.DataFrame(ranks, index=close.index, columns=close.columns))
    z = (values - values.mean(axis=1, keepdims=True)) / values.std(axis=1, ddof=1, keepdims=True)
    _check(evaluate_expression(tiny, "cs_zscore(close)"), pd.DataFrame(z, index=close.index, columns=close.columns))
    demeaned = values - values.mean(axis=1, keepdims=True)
    _check(
        evaluate_expression(tiny, "cs_demean(close)"),
        pd.DataFrame(demeaned, index=close.index, columns=close.columns),
    )


def test_missing_bar_invalidates_windows_and_masks_root() -> None:
    panel = panel_from_bars(tiny_bars(_rows(skip={(3, "BBB")})))
    mean = evaluate_expression(panel, "ts_mean(close, 3)")
    assert mean["BBB"].iloc[3:6].isna().all()
    assert mean["BBB"].iloc[[2, 6, 7]].notna().all()
    delayed = Evaluator(panel).evaluate("delay(close, 1)")
    assert np.isnan(delayed.loc[panel.dates[3], "BBB"])
    unmasked = Evaluator(panel, mask_unavailable=False).evaluate("delay(close, 1)")
    assert unmasked.loc[panel.dates[3], "BBB"] == 21.0  # BBB close on the prior session


def test_memoization_by_canonical_hash(tiny: Panel) -> None:
    engine = Evaluator(tiny)
    engine.evaluate("ts_mean(close, 5) + ts_mean(close, 5)")
    info = engine.cache_info()
    assert (info.hits, info.misses) == (1, 3)
    first = engine.evaluate("close + volume")
    before = engine.cache_info()
    second = engine.evaluate("volume + close")
    after = engine.cache_info()
    assert after.hits == before.hits + 1 and after.misses == before.misses
    pd.testing.assert_frame_equal(first, second)
    corr_a = engine.evaluate("ts_corr(close, volume, 4)")
    corr_b = engine.evaluate("ts_corr(volume, close, 4)")
    pd.testing.assert_frame_equal(corr_a, corr_b)
    second.iloc[:, :] = 0.0  # returned frames are copies; the cache is not mutated
    assert engine.evaluate("close + volume").equals(first)
    bounded = Evaluator(tiny, max_cache_entries=2)
    bounded.evaluate("ts_mean(close, 3) + ts_std(close, 3)")
    assert bounded.cache_info().size == 2
    engine.clear_cache()
    assert engine.cache_info().size == 0


def test_invalid_expressions_raise(tiny: Panel) -> None:
    with pytest.raises(ExpressionError) as info:
        Evaluator(tiny).evaluate("ts_mean(close, 1)")
    assert info.value.codes == ("WINDOW_RANGE",)
    with pytest.raises(ExpressionError) as unknown:
        Evaluator(tiny).evaluate("forward_return(close)")
    assert unknown.value.codes == ("UNKNOWN_OP",)
    with pytest.raises(ExpressionError) as syntax:
        Evaluator(tiny).evaluate("ts_mean(close, 5")
    assert syntax.value.codes == ("PARSE",)


def test_every_operator_has_a_kernel() -> None:
    assert set(KERNELS) == set(OPERATORS)
