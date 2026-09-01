"""Pure event-ledger construction, portfolio aggregation, and date inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .events import validate_event_frame
from .marketdata import validate_confirmed_daily_bars


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    if "_session_date" in frame:
        return frame["_session_date"]
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return timestamp.dt.normalize()


def _clock(frame: pd.DataFrame) -> pd.Series:
    if "_clock" in frame:
        return frame["_clock"]
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp.dt.strftime("%H:%M")


def _intraday_window_return(
    frame: pd.DataFrame,
    session_date: pd.Timestamp,
) -> dict[str, float] | None:
    """Decompose a complete 15:49--15:59 continuous-minute window.

    This function does not infer an official auction price or allow the 15:49
    decision to use information disseminated at 15:50 or later.
    """

    session = frame.loc[_session_dates(frame).eq(pd.Timestamp(session_date).normalize())].copy()
    if session.empty:
        return None
    session["clock"] = _clock(session)
    entry = session.loc[session["clock"].eq("15:49")]
    dissemination = session.loc[session["clock"].eq("15:50")]
    last = session.sort_values("timestamp").tail(1)
    if entry.empty or dissemination.empty or last.empty or str(last["clock"].iloc[0]) != "15:59":
        return None
    entry_price = float(entry.iloc[0]["open"])
    entry_close = float(entry.iloc[0]["close"])
    dissemination_open = float(dissemination.iloc[0]["open"])
    close_price = float(last.iloc[0]["close"])
    prices = (entry_price, entry_close, dissemination_open, close_price)
    if not all(np.isfinite(price) for price in prices) or min(prices) <= 0:
        return None
    last_ten = session.loc[session["clock"].between("15:50", "15:59")]
    total_volume = float(session["volume"].sum()) if "volume" in session else np.nan
    last_ten_volume = float(last_ten["volume"].sum()) if "volume" in session else np.nan
    last_volume = float(last.iloc[0]["volume"]) if "volume" in session else np.nan
    return {
        "entry_1549": entry_price,
        "close_1549": entry_close,
        "entry_1550": dissemination_open,
        "last_minute_close": close_price,
        "underlying_1549_bar_return": entry_close / entry_price - 1.0,
        "underlying_1549_to_1550_return": dissemination_open / entry_price - 1.0,
        "underlying_1550_to_last_minute_return": close_price / dissemination_open - 1.0,
        "underlying_last_minute_return": close_price / entry_price - 1.0,
        "day_volume": total_volume,
        "last10_volume": last_ten_volume,
        "last_minute_volume": last_volume,
        "last10_volume_share": last_ten_volume / total_volume if total_volume > 0 else np.nan,
        "last_minute_volume_share": last_volume / total_volume if total_volume > 0 else np.nan,
    }


def _minute_to_daily_price_scale(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    event_date: pd.Timestamp,
) -> float:
    """Detect a large split-basis mismatch without absorbing auction moves."""

    if minute.empty or daily.empty:
        return 1.0
    event_date = pd.Timestamp(event_date).normalize()
    minute_session = minute.loc[_session_dates(minute).eq(event_date)].sort_values("timestamp")
    daily_dates = (
        pd.to_datetime(daily["date"]).dt.normalize()
        if "date" in daily
        else _session_dates(daily)
    )
    daily_session = daily.loc[daily_dates.eq(event_date)]
    if minute_session.empty or daily_session.empty:
        return 1.0
    daily_row = daily_session.iloc[-1]
    candidates = (
        float(daily_row["open"]) / float(minute_session.iloc[0]["open"]),
        float(daily_row["high"]) / float(minute_session["high"].max()),
        float(daily_row["low"]) / float(minute_session["low"].min()),
    )
    valid = [value for value in candidates if np.isfinite(value) and value > 0]
    if len(valid) < 2:
        return 1.0
    candidate = float(np.median(valid))
    return candidate if candidate < 0.9 or candidate > 1.1 else 1.0


def build_daily_event_ledger(
    events: pd.DataFrame,
    confirmed_daily_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a causal announcement-to-implementation ledger.

    Entry is the first confirmed session open strictly after the announcement.
    Exit is the confirmed close on the implementation date. Missing sessions
    fail closed and are reported separately.
    """

    normalized_events = validate_event_frame(events)
    bars = validate_confirmed_daily_bars(confirmed_daily_bars)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for event in normalized_events.itertuples(index=False):
        announcement = pd.Timestamp(event.announcement_date).normalize()
        effective = pd.Timestamp(event.effective_close_date).normalize()
        symbol_bars = bars.loc[bars["symbol"].eq(event.symbol)].sort_values("date")
        effective_bar = symbol_bars.loc[symbol_bars["date"].eq(effective)]
        entry_candidates = symbol_bars.loc[
            symbol_bars["date"].gt(announcement) & symbol_bars["date"].le(effective)
        ]
        if effective_bar.empty:
            failures.append(
                {
                    "effective_close_date": effective,
                    "symbol": event.symbol,
                    "security_name": event.security_name,
                    "reason": "missing_confirmed_effective_session",
                }
            )
            continue
        if entry_candidates.empty:
            failures.append(
                {
                    "effective_close_date": effective,
                    "symbol": event.symbol,
                    "security_name": event.security_name,
                    "reason": "missing_post_announcement_entry_session",
                }
            )
            continue
        entry = entry_candidates.iloc[0]
        exit_row = effective_bar.iloc[-1]
        direction = 1.0 if event.action == "add" else -1.0
        entry_open = float(entry["open"])
        effective_open = float(exit_row["open"])
        effective_close = float(exit_row["close"])
        record = event._asdict()
        record.update(
            {
                "direction": direction,
                "entry_date": entry["date"],
                "entry_open": entry_open,
                "effective_open": effective_open,
                "effective_close": effective_close,
                "announcement_to_effective_return_gross": direction
                * (effective_close / entry_open - 1.0),
                "effective_session_return_gross": direction
                * (effective_close / effective_open - 1.0),
                "marketdata_source": exit_row["source"],
                "marketdata_finality": exit_row["finality"],
            }
        )
        later = symbol_bars.loc[symbol_bars["date"].gt(effective)]
        if later.empty:
            record.update(
                next_session_date=pd.NaT,
                next_open=np.nan,
                post_effective_open_reversal_gross=np.nan,
            )
        else:
            next_row = later.iloc[0]
            next_open = float(next_row["open"])
            record.update(
                next_session_date=next_row["date"],
                next_open=next_open,
                post_effective_open_reversal_gross=-direction
                * (next_open / effective_close - 1.0),
            )
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(failures)


def aggregate_date_portfolio(
    trades: pd.DataFrame,
    return_column: str,
    *,
    cost_bps: float = 0.0,
    single_leg_weight: float = 0.5,
) -> pd.DataFrame:
    """Balance add/delete legs and produce one independent row per date."""

    if return_column not in trades:
        raise ValueError(f"Missing return column: {return_column}")
    records: list[dict[str, object]] = []
    usable = trades.dropna(subset=[return_column]).copy()
    for event_date, group in usable.groupby("effective_close_date"):
        additions = group.loc[group["action"].eq("add"), return_column]
        deletions = group.loc[group["action"].eq("delete"), return_column]
        if not additions.empty and not deletions.empty:
            gross = 0.5 * float(additions.mean()) + 0.5 * float(deletions.mean())
            gross_exposure = 1.0
        elif not additions.empty:
            gross = single_leg_weight * float(additions.mean())
            gross_exposure = single_leg_weight
        elif not deletions.empty:
            gross = single_leg_weight * float(deletions.mean())
            gross_exposure = single_leg_weight
        else:
            continue
        records.append(
            {
                "effective_close_date": pd.Timestamp(event_date),
                "gross_return": gross,
                "net_return": gross - gross_exposure * float(cost_bps) / 10_000.0,
                "cost_bps": float(cost_bps),
                "gross_exposure": gross_exposure,
                "n_adds": len(additions),
                "n_deletes": len(deletions),
                "n_names": len(group),
                "add_leg_return": float(additions.mean()) if not additions.empty else np.nan,
                "delete_leg_return": float(deletions.mean()) if not deletions.empty else np.nan,
            }
        )
    columns = [
        "effective_close_date",
        "gross_return",
        "net_return",
        "cost_bps",
        "gross_exposure",
        "n_adds",
        "n_deletes",
        "n_names",
        "add_leg_return",
        "delete_leg_return",
    ]
    return pd.DataFrame(records, columns=columns).sort_values("effective_close_date").reset_index(
        drop=True
    )


def moving_block_bootstrap_mean(
    returns: np.ndarray,
    *,
    block_size: int = 4,
    samples: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    """Return a deterministic 95% moving-block interval for the mean."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks = [
        np.take(values, np.arange(start, start + block_size) % len(values))
        for start in range(len(values))
    ]
    means = np.empty(samples)
    blocks_needed = math.ceil(len(values) / block_size)
    for index in range(samples):
        chosen = rng.integers(0, len(blocks), size=blocks_needed)
        sample = np.concatenate([blocks[item] for item in chosen])[: len(values)]
        means[index] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def performance_metrics(
    portfolio: pd.DataFrame,
    return_column: str = "net_return",
) -> dict[str, object]:
    """Compute transparent date-level research diagnostics."""

    if return_column not in portfolio or portfolio.empty:
        return {}
    frame = portfolio.dropna(subset=[return_column]).sort_values("effective_close_date")
    returns = frame[return_column].to_numpy(float)
    if not len(returns):
        return {}
    equity = np.cumprod(1.0 + returns)
    equity_with_initial = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(equity_with_initial)
    drawdowns = equity_with_initial / peaks - 1.0
    span_years = (
        pd.Timestamp(frame["effective_close_date"].iloc[-1])
        - pd.Timestamp(frame["effective_close_date"].iloc[0])
    ).days / 365.25
    years = max(span_years, 1 / 365.25)
    standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
    downside = returns[returns < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else np.nan
    mean = float(np.mean(returns))
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    ci_low, ci_high = moving_block_bootstrap_mean(returns)
    t_result = stats.ttest_1samp(returns, 0.0, alternative="greater") if len(returns) > 1 else None
    sign_result = stats.binomtest(int((returns > 0).sum()), len(returns), 0.5, alternative="greater")
    total_positive = float(wins.sum())
    best_contribution = (
        float(wins.max() / total_positive) if len(wins) and total_positive > 0 else np.nan
    )
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    max_drawdown = float(drawdowns.min())
    break_even_cost = (
        float(frame["gross_return"].mean() / frame["gross_exposure"].mean() * 10_000.0)
        if {"gross_return", "gross_exposure"}.issubset(frame.columns)
        else np.nan
    )
    return {
        "start_date": frame["effective_close_date"].iloc[0],
        "end_date": frame["effective_close_date"].iloc[-1],
        "event_dates": len(returns),
        "total_return": float(equity[-1] - 1.0),
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "event_sharpe_sqrt4": (
            float(np.sqrt(4) * mean / standard_deviation)
            if standard_deviation and standard_deviation > 0
            else np.nan
        ),
        "event_sortino_sqrt4": (
            float(np.sqrt(4) * mean / downside_deviation)
            if downside_deviation and downside_deviation > 0
            else np.nan
        ),
        "mean_return_per_event": mean,
        "median_return_per_event": float(np.median(returns)),
        "volatility_per_event": standard_deviation,
        "hit_rate": float((returns > 0).mean()),
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) else np.nan,
        "worst_event": float(returns.min()),
        "best_event": float(returns.max()),
        "expected_shortfall_95": float(
            np.mean(np.sort(returns)[: max(1, math.ceil(0.05 * len(returns)))])
        ),
        "t_stat": float(t_result.statistic) if t_result else np.nan,
        "one_sided_t_pvalue": float(t_result.pvalue) if t_result else np.nan,
        "sign_test_pvalue": float(sign_result.pvalue),
        "block_bootstrap_mean_ci_low": ci_low,
        "block_bootstrap_mean_ci_high": ci_high,
        "best_date_share_of_positive_pnl": best_contribution,
        "break_even_cost_bps": break_even_cost,
    }


def add_equity_columns(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Append equity, running peak, and drawdown columns."""

    output = portfolio.sort_values("effective_close_date").copy()
    output["equity"] = (1.0 + output["net_return"]).cumprod()
    output["peak"] = output["equity"].cummax().clip(lower=1.0)
    output["drawdown"] = output["equity"] / output["peak"] - 1.0
    return output
