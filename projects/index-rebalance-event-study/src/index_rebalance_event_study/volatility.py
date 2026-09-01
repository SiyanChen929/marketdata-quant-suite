"""Underlying-only diagnostics for a pre-close long-volatility hypothesis.

These pure functions deliberately do not claim option P&L. Historical
executable option returns require private, point-in-time quote/trade data,
contract definitions, sizes, fees, and hedge fills.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def _clock(frame: pd.DataFrame) -> pd.Series:
    if "clock" in frame:
        return frame["clock"].astype(str)
    timestamp = pd.to_datetime(frame["timestamp"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York")
    return timestamp.dt.strftime("%H:%M")


def _window_frame(session: pd.DataFrame) -> pd.DataFrame | None:
    """Return a validated, ordered 15:49--15:59 one-minute window."""

    if session.empty:
        return None
    frame = session.copy()
    frame["clock"] = _clock(frame)
    frame = frame.loc[frame["clock"].between("15:49", "15:59")].sort_values(
        "timestamp"
    )
    expected = [f"15:{minute:02d}" for minute in range(49, 60)]
    if len(frame) != len(expected) or frame["clock"].tolist() != expected:
        return None
    price_columns = ["open", "high", "low", "close"]
    frame[price_columns] = frame[price_columns].apply(pd.to_numeric, errors="coerce")
    if frame[price_columns].isna().any().any():
        return None
    if not np.isfinite(frame[price_columns].to_numpy(float)).all():
        return None
    if float(frame[price_columns].min().min()) <= 0:
        return None
    if pd.to_datetime(frame["timestamp"]).dt.date.nunique() != 1:
        return None
    if pd.to_datetime(frame["timestamp"]).duplicated().any():
        return None
    if (
        (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
        or (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame["high"]).any()
    ):
        return None
    return frame


def window_statistics(session: pd.DataFrame) -> dict[str, float] | None:
    """Return 15:49-open to 15:59-close path statistics for one session."""

    frame = _window_frame(session)
    if frame is None:
        return None
    entry = float(frame.iloc[0]["open"])
    exit_price = float(frame.iloc[-1]["close"])
    path = np.r_[entry, frame["close"].to_numpy(float)]
    minute_log_returns = np.diff(np.log(path))
    terminal_log_return = math.log(exit_price / entry)
    max_high = float(frame["high"].max())
    min_low = float(frame["low"].min())
    return {
        "entry_1549_open": entry,
        "exit_1559_close": exit_price,
        "terminal_log_return": terminal_log_return,
        "absolute_terminal_move": abs(terminal_log_return),
        "sqrt_realized_variance_11m": float(
            np.sqrt(np.square(minute_log_returns).sum())
        ),
        "high_low_log_range": math.log(max_high / min_low),
        "max_absolute_excursion": max(abs(math.log(max_high / entry)), abs(math.log(min_low / entry))),
    }


def breakout_proxy(
    session: pd.DataFrame,
    barrier_bps: float = 25.0,
    cost_bps: float = 10.0,
) -> dict[str, object] | None:
    """Conservative minute-bar OCO breakout proxy.

    A stop observed inside minute ``t`` fills no earlier than minute ``t+1``.
    The fill is adverse relative to the stop barrier.  If both barriers occur
    in one minute and their ordering is unknowable, the worse final direction
    is assigned.  This is an equity convexity proxy, not an option strategy.
    """

    frame = _window_frame(session)
    if frame is None:
        return None
    entry_reference = float(frame.iloc[0]["open"])
    exit_price = float(frame.iloc[-1]["close"])
    barrier = float(barrier_bps) / 10_000.0
    upper = entry_reference * (1.0 + barrier)
    lower = entry_reference * (1.0 - barrier)

    for index in range(len(frame) - 1):
        bar = frame.iloc[index]
        next_bar = frame.iloc[index + 1]
        upper_hit = float(bar["high"]) >= upper
        lower_hit = float(bar["low"]) <= lower
        if not upper_hit and not lower_hit:
            continue
        next_open = float(next_bar["open"])
        long_entry = max(next_open, upper)
        short_entry = min(next_open, lower)
        long_return = exit_price / long_entry - 1.0
        short_return = -(exit_price / short_entry - 1.0)
        ambiguous = bool(upper_hit and lower_hit)
        if ambiguous:
            direction = 1 if long_return <= short_return else -1
            gross_return = min(long_return, short_return)
            fill_price = long_entry if direction == 1 else short_entry
        elif upper_hit:
            direction = 1
            gross_return = long_return
            fill_price = long_entry
        else:
            direction = -1
            gross_return = short_return
            fill_price = short_entry
        return {
            "triggered": True,
            "direction": direction,
            "ambiguous_same_bar": ambiguous,
            "trigger_clock": str(bar["clock"]),
            "fill_clock": str(next_bar["clock"]),
            "fill_price": fill_price,
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": gross_return - float(cost_bps) / 10_000.0,
            "gross_exposure": 1.0,
        }
    return {
        "triggered": False,
        "direction": 0,
        "ambiguous_same_bar": False,
        "trigger_clock": None,
        "fill_clock": None,
        "fill_price": np.nan,
        "exit_price": exit_price,
        "gross_return": 0.0,
        "net_return": 0.0,
        "gross_exposure": 0.0,
    }


def black_scholes_straddle(
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
) -> float:
    """Zero-rate, zero-dividend European call-plus-put value."""

    if spot <= 0 or strike <= 0 or volatility <= 0:
        return np.nan
    if time_years <= 0:
        return abs(spot - strike)
    root_time = math.sqrt(time_years)
    d1 = (math.log(spot / strike) + 0.5 * volatility**2 * time_years) / (
        volatility * root_time
    )
    d2 = d1 - volatility * root_time
    call = spot * norm.cdf(d1) - strike * norm.cdf(d2)
    put = strike * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return float(call + put)


def black_scholes_straddle_delta(
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
) -> float:
    """Zero-rate delta of one call plus one put."""

    if spot <= 0 or strike <= 0 or volatility <= 0 or time_years <= 0:
        return np.nan
    d1 = (
        math.log(spot / strike) + 0.5 * volatility**2 * time_years
    ) / (volatility * math.sqrt(time_years))
    return float(2.0 * norm.cdf(d1) - 1.0)


def hypothetical_straddle_return(
    underlying_log_return: float,
    dte: int,
    entry_volatility: float,
    exit_volatility_change: float = 0.0,
    friction_fraction_of_premium: float = 0.0,
    holding_minutes: int = 11,
    delta_hedged: bool = True,
) -> float:
    """Model sensitivity using a unit-spot ATM European straddle.

    This is not executable option P&L.  It holds the strike fixed at entry,
    reprices with a hypothetical volatility change, and subtracts friction as a
    fraction of entry premium.
    """

    entry_time = max(float(dte) / 365.0, 1e-9)
    exit_time = max(entry_time - float(holding_minutes) / (365.0 * 24.0 * 60.0), 0.0)
    exit_volatility = max(entry_volatility + exit_volatility_change, 0.01)
    entry_value = black_scholes_straddle(1.0, 1.0, entry_time, entry_volatility)
    entry_delta = black_scholes_straddle_delta(
        1.0, 1.0, entry_time, entry_volatility
    )
    exit_spot = math.exp(float(underlying_log_return))
    exit_value = black_scholes_straddle(
        exit_spot,
        1.0,
        exit_time,
        exit_volatility,
    )
    if not np.isfinite(entry_value) or entry_value <= 0 or not np.isfinite(exit_value):
        return np.nan
    hedge_pnl = -entry_delta * (exit_spot - 1.0) if delta_hedged else 0.0
    return float(
        (exit_value + hedge_pnl) / entry_value
        - 1.0
        - friction_fraction_of_premium
    )
