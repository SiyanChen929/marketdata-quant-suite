from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from lfm_helpers import bars_from_close, calendar, make_bars, perturb_after, symbols
from llm_factor_mining.data import panel_from_bars
from llm_factor_mining.evaluate import (
    EvaluationWindow,
    MetricConfig,
    default_nw_lags,
    embargoed_signal_dates,
    evaluate_factor,
    evaluate_signal,
    forward_returns,
    holding_weights,
    ic_summary,
    long_short_backtest,
    long_short_weights,
    newey_west_se,
    quantile_buckets,
    rank_ic,
    top_quantile_turnover,
)


def _frame(values: np.ndarray) -> pd.DataFrame:
    n_dates, n_symbols = values.shape
    return pd.DataFrame(values, index=calendar(n_dates), columns=pd.Index(symbols(n_symbols), name="symbol"))


def _close_from_returns(returns: np.ndarray) -> pd.DataFrame:
    return _frame(100.0 * np.cumprod(1.0 + returns, axis=0))


def _planted(delay_sessions: int, *, n_dates: int = 400, n_symbols: int = 40, seed: int = 5):
    """Returns R[t + delay] = 0.01 * z[t] + noise, so z at t predicts that session's move."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_dates, n_symbols))
    returns = rng.normal(0.0, 0.01, size=(n_dates, n_symbols))
    returns[delay_sessions:] += 0.01 * z[:-delay_sessions]
    returns[0] = 0.0
    return _frame(z), _close_from_returns(returns)


# --------------------------------------------------------------------------
# forward returns and execution lag
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 5])
def test_forward_return_alignment(lag: int, horizon: int) -> None:
    n = 20
    close = _frame(np.outer(np.arange(1, n + 1, dtype=float), [1.0, 2.0, 3.0]))
    fwd = forward_returns(close, horizon, lag)
    for t in range(n - lag - horizon):
        expected = (t + lag + horizon + 1) / (t + lag + 1) - 1.0
        assert fwd.iloc[t].to_numpy() == pytest.approx([expected] * 3, rel=1e-14)
    assert fwd.iloc[n - lag - horizon :].isna().all().all()
    assert fwd.iloc[: n - lag - horizon].notna().all().all()


@pytest.mark.parametrize("lag", [0, -1])
def test_same_session_execution_is_rejected(lag: int) -> None:
    close = _frame(np.ones((5, 3)))
    with pytest.raises(ValueError, match="t\\+1"):
        forward_returns(close, 1, lag)
    with pytest.raises(ValueError):
        MetricConfig(lag=lag)
    with pytest.raises(ValueError):
        forward_returns(close, 0, 1)


def test_signal_that_decays_within_one_session_earns_nothing() -> None:
    """A signal about the t -> t+1 move is not tradeable at the t+1 close."""

    signal, close = _planted(delay_sessions=1)
    tradeable = evaluate_signal(signal, close, config=MetricConfig(min_names=10)).report
    assert abs(tradeable.ic_mean) < 0.03
    same_close = close.shift(-1) / close - 1.0  # what a lag-0 (lookahead) evaluation would score
    leaked = rank_ic(signal, same_close, min_names=10).mean()
    assert leaked > 0.5


# --------------------------------------------------------------------------
# information coefficient
# --------------------------------------------------------------------------


def test_rank_ic_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    signal = _frame(rng.normal(size=(15, 12)))
    forward = _frame(0.3 * signal.to_numpy() + rng.normal(size=(15, 12)))
    signal.iloc[2, :5] = np.nan
    forward.iloc[3, 7] = np.nan
    ic = rank_ic(signal, forward, min_names=5)
    for t in range(15):
        mask = signal.iloc[t].notna() & forward.iloc[t].notna()
        expected = spearmanr(signal.iloc[t][mask], forward.iloc[t][mask]).statistic
        assert ic.iloc[t] == pytest.approx(expected, abs=1e-12)
    assert rank_ic(signal, forward, min_names=8).isna().iloc[2]  # only 7 names on that date


def test_rank_ic_without_rank_variation_is_nan() -> None:
    signal = _frame(np.ones((3, 10)))
    forward = _frame(np.random.default_rng(0).normal(size=(3, 10)))
    assert rank_ic(signal, forward, min_names=5).isna().all()


def test_planted_signal_has_high_ic_and_noise_has_none() -> None:
    signal, close = _planted(delay_sessions=2)
    config = MetricConfig(min_names=10, cost_bps=5.0)
    report = evaluate_signal(signal, close, config=config).report
    assert report.ic_mean > 0.5
    assert report.ic_tstat_nw > 20
    assert report.ic_hit_rate > 0.95
    assert report.quantile_monotonicity == pytest.approx(1.0)
    assert report.long_short_mean > 0
    assert report.ls_net_ann_return < report.ls_gross_ann_return
    assert report.coverage == pytest.approx(1.0)

    noise = _frame(np.random.default_rng(99).normal(size=signal.shape))
    null = evaluate_signal(noise, close, config=config).report
    assert abs(null.ic_mean) < 0.02
    assert abs(null.ic_tstat_nw) < 3.0
    assert null.ls_net_sharpe < report.ls_net_sharpe


def test_dsl_factor_on_panel_recovers_planted_autocorrelation() -> None:
    """returns[t+2] = 0.6 returns[t] + noise, so the DSL factor `returns` is predictive at lag 1."""

    rng = np.random.default_rng(3)
    n_dates, n_symbols = 300, 30
    returns = np.zeros((n_dates, n_symbols))
    noise = rng.normal(0.0, 0.01, size=(n_dates, n_symbols))
    for t in range(1, n_dates):
        returns[t] = noise[t] + (0.6 * returns[t - 2] if t >= 2 else 0.0)
    panel = panel_from_bars(bars_from_close(_close_from_returns(returns), seed=4))
    config = MetricConfig(min_names=10)
    positive = evaluate_factor(panel, "returns", config=config)["full"].report
    negative = evaluate_factor(panel, "-returns", config=config)["full"].report
    assert positive.ic_mean > 0.4
    assert negative.ic_mean == pytest.approx(-positive.ic_mean, abs=1e-12)
    assert positive.expression == "returns" and negative.expression == "neg(returns)"
    assert positive.panel_sha256 == panel.content_sha256


# --------------------------------------------------------------------------
# windows and embargo
# --------------------------------------------------------------------------


def test_embargo_drops_last_lag_plus_horizon_sessions() -> None:
    dates = calendar(40)
    window = EvaluationWindow("test", dates[10], dates[29])
    kept = embargoed_signal_dates(dates, window, lag=1, horizon=5)
    assert kept[0] == dates[10] and kept[-1] == dates[23]
    assert len(kept) == 20 - (1 + 5)
    for position in range(10, 30):
        assert (dates[position] in kept) == (position + 1 + 5 <= 29)
    full = embargoed_signal_dates(dates, None, lag=2, horizon=3)
    assert full[-1] == dates[34]
    beyond = EvaluationWindow("open-ended", dates[30], dates[-1] + pd.Timedelta(days=90))
    assert embargoed_signal_dates(dates, beyond, lag=1, horizon=1)[-1] == dates[37]
    empty = EvaluationWindow("future", dates[-1] + pd.Timedelta(days=1), None)
    assert len(embargoed_signal_dates(dates, empty, lag=1, horizon=1)) == 0
    tiny = EvaluationWindow("tiny", dates[5], dates[6])
    assert len(embargoed_signal_dates(dates, tiny, lag=1, horizon=1)) == 0
    with pytest.raises(ValueError):
        EvaluationWindow("bad", dates[5], dates[4])


def test_window_metrics_ignore_prices_after_window_end() -> None:
    bars = make_bars(160, 20, seed=13)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))
    windows = [
        EvaluationWindow("formation", dates[0], dates[79]),
        EvaluationWindow("validation", dates[80], dates[119]),
    ]
    config = MetricConfig(horizon=3, lag=1, min_names=10)
    expression = "cs_rank(-ts_mean(returns, 5)) + ts_corr(close, volume, 10)"
    original = evaluate_factor(panel_from_bars(bars), expression, windows=windows, config=config)
    changed_panel = panel_from_bars(perturb_after(bars, dates[119]))
    changed = evaluate_factor(changed_panel, expression, windows=windows, config=config)
    for name in ("formation", "validation"):
        a, b = original[name].report.to_dict(), changed[name].report.to_dict()
        a.pop("panel_sha256"), b.pop("panel_sha256")
        assert a == b, name
        assert a["n_signal_dates"] == (80 if name == "formation" else 40) - 4

    # Without the embargo, the last lag + h signal dates would read post-window prices.
    signal_a = original["validation"].ic.index
    naive_dates = dates[80:120]
    close_a = panel_from_bars(bars).close
    close_b = changed_panel.close
    from llm_factor_mining.evaluate import Evaluator

    fa = Evaluator(panel_from_bars(bars)).evaluate(expression)
    fb = Evaluator(changed_panel).evaluate(expression)
    naive_a = rank_ic(fa, forward_returns(close_a, 3, 1), min_names=10).reindex(naive_dates)
    naive_b = rank_ic(fb, forward_returns(close_b, 3, 1), min_names=10).reindex(naive_dates)
    assert not np.allclose(naive_a.to_numpy(), naive_b.to_numpy(), equal_nan=True)
    assert signal_a.max() == dates[115]


# --------------------------------------------------------------------------
# Newey-West
# --------------------------------------------------------------------------


def test_newey_west_zero_lags_is_iid_formula() -> None:
    x = np.random.default_rng(0).normal(size=50)
    expected = math.sqrt(np.mean((x - x.mean()) ** 2) / x.size)
    assert newey_west_se(x, 0) == pytest.approx(expected, rel=1e-12)


def test_newey_west_hand_computed() -> None:
    # u = [-2, 0, -1, 2, 1]; gamma0 = 2, gamma1 = 0, gamma2 = 0.2; Bartlett L=2
    long_run = 2.0 + 2.0 * (2.0 / 3.0) * 0.0 + 2.0 * (1.0 / 3.0) * 0.2
    assert newey_west_se([1.0, 3.0, 2.0, 5.0, 4.0], 2) == pytest.approx(math.sqrt(long_run / 5.0), rel=1e-12)
    assert newey_west_se(pd.Series([1.0, np.nan, 3.0, 2.0, 5.0, 4.0]), 2) == pytest.approx(
        math.sqrt(long_run / 5.0), rel=1e-12
    )
    assert math.isnan(newey_west_se([1.0], 3))
    with pytest.raises(ValueError):
        newey_west_se([1.0, 2.0], -1)


def test_newey_west_recovers_ar1_long_run_variance() -> None:
    rng = np.random.default_rng(42)
    phi, n = 0.5, 50_000
    shocks = rng.normal(size=n)
    x = np.empty(n)
    x[0] = shocks[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    theory = math.sqrt(1.0 / (1.0 - phi) ** 2 / n)
    hac = newey_west_se(x, 60)
    iid = newey_west_se(x, 0)
    assert 0.9 < hac / theory < 1.1
    assert iid / theory < 0.65


def test_newey_west_corrects_overlapping_horizons() -> None:
    rng = np.random.default_rng(7)
    h, n = 5, 20_000
    shocks = rng.normal(size=n + h)
    overlapping = np.convolve(shocks, np.ones(h), mode="valid")[:n]
    theory = math.sqrt(h**2 / n)
    hac = newey_west_se(overlapping, default_nw_lags(n, h))
    assert 0.8 < hac / theory < 1.1
    assert newey_west_se(overlapping, 0) / theory < 0.5


def test_default_lags_and_summary_guards() -> None:
    assert default_nw_lags(100, 1) == 4
    assert default_nw_lags(100, 10) == 9
    assert default_nw_lags(0, 1) == 0
    ic = pd.Series([0.1, 0.2, -0.05, 0.15, 0.3])
    summary = ic_summary(ic, horizon=1, nw_lags=0)
    assert summary.mean == pytest.approx(0.14)
    assert summary.icir == pytest.approx(0.14 / np.std(ic.to_numpy(), ddof=1))
    assert summary.hit_rate == pytest.approx(0.8)
    with pytest.raises(ValueError):
        ic_summary(ic, horizon=5, nw_lags=2)
    with pytest.raises(ValueError):
        MetricConfig(horizon=5, nw_lags=3)
    assert ic_summary(pd.Series([], dtype=float)).n == 0


# --------------------------------------------------------------------------
# quantiles, turnover, costs
# --------------------------------------------------------------------------


def test_quantile_buckets() -> None:
    signal = _frame(np.vstack([np.arange(1.0, 11.0), np.ones(10), np.arange(10.0, 0.0, -1.0)]))
    buckets = quantile_buckets(signal, 5, min_names=10)
    assert buckets.iloc[0].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert buckets.iloc[1].tolist() == [3] * 10  # all tied -> middle bucket, no legs
    assert buckets.iloc[2].tolist() == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    sparse = signal.copy()
    sparse.iloc[0, 0] = np.nan
    assert quantile_buckets(sparse, 5, min_names=10).iloc[0].isna().all()


def test_top_quantile_turnover() -> None:
    up = np.arange(1.0, 11.0)
    static = _frame(np.vstack([up] * 4))
    flipping = _frame(np.vstack([up, up[::-1], up, up[::-1]]))
    assert top_quantile_turnover(quantile_buckets(static, 5, min_names=10), 5).iloc[1:].tolist() == [0.0] * 3
    assert top_quantile_turnover(quantile_buckets(flipping, 5, min_names=10), 5).iloc[1:].tolist() == [1.0] * 3
    assert np.isnan(top_quantile_turnover(quantile_buckets(static, 5, min_names=10), 5).iloc[0])


def test_long_short_backtest_by_hand() -> None:
    n_dates, n_symbols = 6, 10
    rng = np.random.default_rng(8)
    close = _frame(100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(n_dates, n_symbols)), axis=0))
    signal = _frame(np.vstack([np.arange(1.0, 11.0)] * n_dates))
    config = MetricConfig(horizon=1, lag=1, n_quantiles=5, min_names=10, cost_bps=10.0)
    result = long_short_backtest(signal, close, config)
    weights = np.array([-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5])
    next_returns = (close.shift(-2) / close.shift(-1) - 1.0).to_numpy()
    for t in range(n_dates - 2):
        assert result["gross"].iloc[t] == pytest.approx(float(weights @ next_returns[t]), abs=1e-15)
    assert result["turnover"].tolist() == [2.0] + [0.0] * (n_dates - 1)
    assert result["cost"].iloc[0] == pytest.approx(2.0 * 10.0e-4)
    assert (result["net"] == result["gross"] - result["cost"]).all()
    assert result["gross_exposure"].iloc[0] == pytest.approx(2.0)
    assert result["missing_return_weight"].iloc[-1] == pytest.approx(2.0)


def test_overlapping_holding_weights() -> None:
    signal = _frame(np.vstack([np.arange(1.0, 11.0), np.arange(10.0, 0.0, -1.0), np.arange(1.0, 11.0)]))
    formation = long_short_weights(signal, 5, min_names=10)
    held = holding_weights(formation, 2)
    assert held.iloc[0].tolist() == pytest.approx((formation.iloc[0] / 2).tolist())
    assert held.iloc[1].abs().sum() == pytest.approx(0.0)  # opposite formations cancel
    assert held.iloc[2].tolist() == pytest.approx(((formation.iloc[1] + formation.iloc[2]) / 2).tolist())


def test_report_is_json_safe_and_handles_empty_windows() -> None:
    signal, close = _planted(delay_sessions=2, n_dates=60, n_symbols=12)
    config = MetricConfig(min_names=10)
    report = evaluate_signal(signal, close, config=config, expression="z").report
    json.dumps(report.to_dict(), allow_nan=False)
    empty = evaluate_signal(
        signal, close, config=config, window=EvaluationWindow("future", pd.Timestamp("2099-01-01"))
    ).report
    payload = empty.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["n_signal_dates"] == 0 and payload["ic_mean"] is None
    with pytest.raises(ValueError, match="same dates"):
        evaluate_signal(signal.iloc[:-1], close)


def test_metric_config_validation() -> None:
    for kwargs in ({"n_quantiles": 1}, {"min_names": 2}, {"cost_bps": -1.0}, {"horizon": 0}, {"periods_per_year": 0}):
        with pytest.raises(ValueError):
            MetricConfig(**kwargs)
    assert MetricConfig(n_quantiles=10, min_names=5).effective_min_names == 10


def test_multi_window_evaluation_matches_single_window_calls() -> None:
    from llm_factor_mining.evaluate import evaluate_windows

    signal, close = _planted(delay_sessions=2, n_dates=120, n_symbols=15)
    dates = close.index
    windows = [
        EvaluationWindow("formation", dates[0], dates[59]),
        EvaluationWindow("test", dates[60], dates[-1]),
    ]
    config = MetricConfig(horizon=2, min_names=10)
    combined = evaluate_windows(signal, close, windows, config=config, expression="z")
    for window in windows:
        single = evaluate_signal(signal, close, config=config, window=window, expression="z")
        assert combined[window.name].report == single.report
    assert combined["formation"].report.last_signal_date == str(dates[59 - 3].date())
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_windows(signal, close, [windows[0], windows[0]], config=config)
