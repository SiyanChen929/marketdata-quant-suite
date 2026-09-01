"""Close-to-close, no-look-ahead pairs backtester with explicit trading frictions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass(frozen=True)
class PairModel:
    pair: str
    sector: str
    dependent: str
    independent: str
    alpha: float
    beta: float


@dataclass
class BacktestResult:
    model: PairModel
    signals: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def _cagr(equity: pd.Series) -> float:
    clean = equity.dropna()
    if len(clean) < 2 or clean.iloc[-1] <= 0:
        return math.nan
    years = len(clean) / 252.0
    return float(clean.iloc[-1] ** (1.0 / years) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return math.nan
    drawdown = clean / clean.cummax() - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series, equity: pd.Series | None = None) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = equity if equity is not None else (1.0 + returns).cumprod()
    annual_return = _cagr(equity)
    annual_volatility = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else math.nan
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else math.nan
    )
    downside = returns[returns < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(252.0)) if len(downside) else 0.0
    sortino = (
        float(returns.mean() * 252.0 / downside_deviation)
        if downside_deviation > 0
        else math.nan
    )
    max_drawdown = _max_drawdown(equity)
    calmar = (
        float(annual_return / abs(max_drawdown))
        if np.isfinite(annual_return) and np.isfinite(max_drawdown) and max_drawdown < 0
        else math.nan
    )
    return {
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else math.nan,
        "cagr": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _target_state(zscore: pd.Series, config: StrategyConfig) -> tuple[pd.Series, pd.Series]:
    """Generate close-time target states. Positions become effective next session."""
    positions: list[int] = []
    events: list[str] = []
    position = 0
    held = 0
    for raw_value in zscore.to_numpy(dtype=float):
        event = "hold" if position else "flat"
        if not np.isfinite(raw_value):
            if position:
                event = "exit_missing_signal"
            position = 0
            held = 0
        elif position == 0:
            if raw_value <= -config.entry_z:
                position = 1
                held = 0
                event = "enter_long_spread"
            elif raw_value >= config.entry_z:
                position = -1
                held = 0
                event = "enter_short_spread"
        else:
            held += 1
            if abs(raw_value) <= config.exit_z:
                position = 0
                held = 0
                event = "exit_mean_reversion"
            elif abs(raw_value) >= config.stop_z:
                position = 0
                held = 0
                event = "exit_stop"
            elif held >= config.maximum_holding_days:
                position = 0
                held = 0
                event = "exit_max_holding"
        positions.append(position)
        events.append(event)
    return (
        pd.Series(positions, index=zscore.index, dtype=int, name="target_position"),
        pd.Series(events, index=zscore.index, dtype="string", name="decision_event"),
    )


def compute_zscore(
    spread: pd.Series,
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return causal trailing location, scale, and z-score for a spread.

    ``sma`` is the original equal-weighted rolling estimator. ``ewma`` uses
    ``adjust=False`` so the recursively updated statistic gives recent
    observations more weight without looking ahead.
    """
    if config.zscore_method == "sma":
        center = spread.rolling(
            config.zscore_lookback,
            min_periods=config.zscore_min_periods,
        ).mean()
        scale = spread.rolling(
            config.zscore_lookback,
            min_periods=config.zscore_min_periods,
        ).std(ddof=1)
    elif config.zscore_method == "ewma":
        estimator = spread.ewm(
            span=config.zscore_lookback,
            min_periods=config.zscore_min_periods,
            adjust=False,
        )
        center = estimator.mean()
        scale = estimator.std(bias=False)
    else:  # validation normally catches this; keep the computation fail-closed.
        raise ValueError(f"Unsupported z-score method: {config.zscore_method}")
    zscore = ((spread - center) / scale.replace(0.0, np.nan)).rename("zscore")
    return center, scale, zscore


def _extract_trades(frame: pd.DataFrame, model: PairModel) -> pd.DataFrame:
    position = frame["position"].astype(int)
    previous = position.shift(1, fill_value=0)
    entry_locs = np.flatnonzero((previous.eq(0) & position.ne(0)).to_numpy())
    rows: list[dict[str, Any]] = []
    for sequence, entry_loc in enumerate(entry_locs, start=1):
        direction = int(position.iloc[entry_loc])
        later = position.iloc[entry_loc + 1 :]
        exit_candidates = np.flatnonzero(later.eq(0).to_numpy())
        if len(exit_candidates):
            exit_loc = entry_loc + 1 + int(exit_candidates[0])
            exit_reason = str(frame["effective_event"].iloc[exit_loc])
            closed = True
        else:
            exit_loc = len(frame) - 1
            exit_reason = "end_of_test"
            closed = False
        capital_before = float(frame["net_equity"].iloc[entry_loc - 1]) if entry_loc else 1.0
        capital_after = float(frame["net_equity"].iloc[exit_loc])
        segment = frame.iloc[entry_loc : exit_loc + 1]
        rows.append(
            {
                "pair": model.pair,
                "sector": model.sector,
                "trade_number": sequence,
                "direction": "long_spread" if direction == 1 else "short_spread",
                "entry_date": frame.index[entry_loc],
                "exit_date": frame.index[exit_loc],
                "calendar_days": int((frame.index[exit_loc] - frame.index[entry_loc]).days),
                "trading_days": int(exit_loc - entry_loc) if closed else int(exit_loc - entry_loc + 1),
                "entry_zscore": float(frame["decision_zscore"].iloc[entry_loc]),
                "exit_zscore": float(frame["decision_zscore"].iloc[exit_loc])
                if np.isfinite(frame["decision_zscore"].iloc[exit_loc])
                else math.nan,
                "exit_reason": exit_reason,
                "closed": closed,
                "net_return": capital_after / capital_before - 1.0,
                "gross_return_sum": float(segment["gross_return"].sum()),
                "transaction_cost_sum": float(segment["transaction_cost"].sum()),
                "borrow_cost_sum": float(segment["borrow_cost"].sum()),
                "maximum_adverse_daily_return": float(segment["net_return"].min()),
            }
        )
    return pd.DataFrame(rows)


def run_pair_backtest(
    model: PairModel,
    prices: pd.DataFrame,
    test_start: pd.Timestamp | str,
    config: StrategyConfig,
) -> BacktestResult:
    """Backtest one fixed formation-window relation on a held-out test window."""
    if model.dependent not in prices or model.independent not in prices:
        raise KeyError(f"Missing price leg for {model.pair}")
    if not np.isfinite(model.beta) or model.beta <= 0:
        raise ValueError(f"Pair {model.pair} requires a positive finite hedge ratio")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("prices index must contain unique dates")
    prices = prices.sort_index()
    legs = prices[[model.dependent, model.independent]].apply(pd.to_numeric, errors="coerce")
    valid_prices = legs.where(legs > 0)
    spread = (
        np.log(valid_prices[model.dependent])
        - model.alpha
        - model.beta * np.log(valid_prices[model.independent])
    ).rename("spread")
    rolling_mean, rolling_std, zscore = compute_zscore(spread, config)
    target, decision_event = _target_state(zscore, config)

    leg_returns = valid_prices.pct_change(fill_method=None)
    base_dependent = 1.0 / (1.0 + abs(model.beta))
    base_independent = -model.beta / (1.0 + abs(model.beta))
    effective_position = target.shift(1, fill_value=0).astype(int).rename("position")
    effective_event = decision_event.shift(1, fill_value="flat").rename("effective_event")
    decision_zscore = zscore.shift(1).rename("decision_zscore")
    weight_dependent = (effective_position * base_dependent).rename("weight_dependent")
    weight_independent = (effective_position * base_independent).rename("weight_independent")
    gross_return = (
        weight_dependent * leg_returns[model.dependent].fillna(0.0)
        + weight_independent * leg_returns[model.independent].fillna(0.0)
    ).rename("gross_return")
    previous_dep = weight_dependent.shift(1, fill_value=0.0)
    previous_ind = weight_independent.shift(1, fill_value=0.0)
    previous_gross_return = gross_return.shift(1, fill_value=0.0)
    previous_denominator = (1.0 + previous_gross_return).replace(0.0, np.nan)
    post_return_dep = (
        previous_dep
        * (1.0 + leg_returns[model.dependent].shift(1).fillna(0.0))
        / previous_denominator
    ).fillna(previous_dep)
    post_return_ind = (
        previous_ind
        * (1.0 + leg_returns[model.independent].shift(1).fillna(0.0))
        / previous_denominator
    ).fillna(previous_ind)
    # Rebalance to the configured gross-normalized hedge weights at each close;
    # the drift correction is real turnover and is charged accordingly.
    turnover = (
        (weight_dependent - post_return_dep).abs()
        + (weight_independent - post_return_ind).abs()
    ).rename("turnover")
    transaction_cost = (
        turnover * config.transaction_cost_bps / 10_000.0
    ).rename("transaction_cost")
    short_notional = (
        weight_dependent.clip(upper=0).abs() + weight_independent.clip(upper=0).abs()
    )
    borrow_cost = (
        short_notional * config.annual_short_borrow_bps / 10_000.0 / 252.0
    ).rename("borrow_cost")
    net_return = (gross_return - transaction_cost - borrow_cost).rename("net_return")

    full = pd.concat(
        [
            valid_prices.rename(
                columns={model.dependent: "price_dependent", model.independent: "price_independent"}
            ),
            spread,
            rolling_mean.rename("spread_rolling_mean"),
            rolling_std.rename("spread_rolling_std"),
            zscore,
            decision_zscore,
            target,
            decision_event,
            effective_position,
            effective_event,
            weight_dependent,
            weight_independent,
            turnover,
            gross_return,
            transaction_cost,
            borrow_cost,
            net_return,
        ],
        axis=1,
    )
    frame = full.loc[full.index >= pd.Timestamp(test_start)].copy()
    if frame.empty:
        raise ValueError(f"No test observations for {model.pair} from {test_start}")
    frame["gross_equity"] = (1.0 + frame["gross_return"].fillna(0.0)).cumprod()
    frame["net_equity"] = (1.0 + frame["net_return"].fillna(0.0)).cumprod()
    frame["drawdown"] = frame["net_equity"] / frame["net_equity"].cummax() - 1.0
    frame.index.name = "date"

    trades = _extract_trades(frame, model)
    net_metrics = performance_metrics(frame["net_return"], frame["net_equity"])
    gross_metrics = performance_metrics(frame["gross_return"], frame["gross_equity"])
    trade_returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    winners = trade_returns[trade_returns > 0]
    losers = trade_returns[trade_returns < 0]
    profit_factor = (
        float(winners.sum() / abs(losers.sum()))
        if len(losers) and abs(losers.sum()) > 0
        else math.nan
    )
    metrics: dict[str, Any] = {
        **asdict(model),
        "test_start": str(frame.index.min().date()),
        "test_end": str(frame.index.max().date()),
        "test_observations": int(len(frame)),
        "gross_total_return": gross_metrics["total_return"],
        "gross_cagr": gross_metrics["cagr"],
        "net_total_return": net_metrics["total_return"],
        "net_cagr": net_metrics["cagr"],
        "annual_volatility": net_metrics["annual_volatility"],
        "sharpe": net_metrics["sharpe"],
        "sortino": net_metrics["sortino"],
        "max_drawdown": net_metrics["max_drawdown"],
        "calmar": net_metrics["calmar"],
        "trade_count": int(len(trades)),
        "closed_trade_count": int(trades["closed"].sum()) if not trades.empty else 0,
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else math.nan,
        "profit_factor": profit_factor,
        "average_holding_days": float(trades["trading_days"].mean()) if not trades.empty else math.nan,
        "exposure_fraction": float(frame["position"].ne(0).mean()),
        "annualized_turnover": float(frame["turnover"].mean() * 252.0),
        "transaction_cost_sum": float(frame["transaction_cost"].sum()),
        "borrow_cost_sum": float(frame["borrow_cost"].sum()),
        "gross_minus_net_total_return": gross_metrics["total_return"] - net_metrics["total_return"],
    }
    return BacktestResult(model=model, signals=frame, trades=trades, metrics=metrics)


def backtest_selected_pairs(
    selected_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    test_start: pd.Timestamp | str,
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest all selected rows and return metrics, signals, trades, pair returns."""
    metrics: list[dict[str, Any]] = []
    signals: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    returns: dict[str, pd.Series] = {}
    for row in selected_pairs.to_dict(orient="records"):
        model = PairModel(
            pair=str(row["pair"]),
            sector=str(row["sector"]),
            dependent=str(row["dependent"]),
            independent=str(row["independent"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
        )
        try:
            result = run_pair_backtest(model, prices, test_start, config)
        except Exception as exc:  # one broken price series must not destroy a full research run
            metrics.append(
                {
                    **asdict(model),
                    "backtest_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        result.metrics["backtest_status"] = "ok"
        metrics.append(result.metrics)
        signal_frame = result.signals.reset_index()
        signal_frame.insert(0, "sector", model.sector)
        signal_frame.insert(0, "pair", model.pair)
        signals.append(signal_frame)
        if not result.trades.empty:
            trades.append(result.trades)
        returns[model.pair] = result.signals["net_return"]
    metrics_frame = pd.DataFrame(metrics)
    signals_frame = pd.concat(signals, ignore_index=True) if signals else pd.DataFrame()
    trades_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    returns_frame = pd.DataFrame(returns).sort_index()
    returns_frame.index.name = "date"
    return metrics_frame, signals_frame, trades_frame, returns_frame


def build_portfolios(
    pair_returns: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    best_pairs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build static all-pairs, sector-balanced, and ex-post top-10 research portfolios."""
    if pair_returns.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid_pairs = [column for column in pair_returns.columns if column in set(selected_pairs["pair"])]
    all_selected = pair_returns[valid_pairs].fillna(0.0).mean(axis=1)
    sector_returns: dict[str, pd.Series] = {}
    pair_sector = selected_pairs.drop_duplicates("pair").set_index("pair")["sector"].to_dict()
    for sector in sorted(set(pair_sector.get(pair) for pair in valid_pairs)):
        names = [pair for pair in valid_pairs if pair_sector.get(pair) == sector]
        if names:
            sector_returns[str(sector)] = pair_returns[names].fillna(0.0).mean(axis=1)
    sector_frame = pd.DataFrame(sector_returns)
    sector_balanced = sector_frame.mean(axis=1)
    usable_best = [pair for pair in best_pairs if pair in pair_returns.columns]
    top_returns = (
        pair_returns[usable_best].fillna(0.0).mean(axis=1)
        if usable_best
        else pd.Series(0.0, index=pair_returns.index)
    )
    portfolio_returns = pd.DataFrame(
        {
            "all_formation_selected": all_selected,
            "sector_balanced": sector_balanced,
            "expost_top_performers": top_returns,
        }
    )
    equity = (1.0 + portfolio_returns).cumprod()
    equity.columns = [f"{column}_equity" for column in equity.columns]
    curves = pd.concat([portfolio_returns, equity], axis=1)
    curves.index.name = "date"
    metric_rows: list[dict[str, Any]] = []
    for column in portfolio_returns:
        series = portfolio_returns[column]
        values = performance_metrics(series, (1.0 + series).cumprod())
        metric_rows.append({"portfolio": column, **values})
    return curves, pd.DataFrame(metric_rows)


def cost_sensitivity(
    signals: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    best_pairs: list[str],
    scenarios_bps: tuple[float, ...] | list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reprice the exact same trades under alternative one-way turnover costs."""
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    working = signals.copy()
    working["date"] = pd.to_datetime(working["date"])
    pair_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    for bps in sorted(set(float(value) for value in scenarios_bps)):
        working["scenario_net_return"] = (
            working["gross_return"].fillna(0.0)
            - working["turnover"].fillna(0.0) * bps / 10_000.0
            - working["borrow_cost"].fillna(0.0)
        )
        scenario_returns = working.pivot(index="date", columns="pair", values="scenario_net_return").sort_index()
        for pair in scenario_returns:
            series = scenario_returns[pair].fillna(0.0)
            metrics = performance_metrics(series)
            pair_rows.append({"transaction_cost_bps": bps, "pair": pair, **metrics})
        _, portfolio_metrics = build_portfolios(
            scenario_returns,
            selected_pairs,
            best_pairs,
        )
        if not portfolio_metrics.empty:
            portfolio_metrics.insert(0, "transaction_cost_bps", bps)
            portfolio_rows.extend(portfolio_metrics.to_dict(orient="records"))
    return pd.DataFrame(pair_rows), pd.DataFrame(portfolio_rows)
