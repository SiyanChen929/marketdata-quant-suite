"""Executable transaction, holding-cost, and cash-carry model.

The model operates on normalized portfolio-dollar notionals.  A value of 0.05
means five percent of current NAV; ``portfolio_notional`` converts that weight
to dollars solely for ADV participation and nonlinear impact calculations.
All rolling liquidity inputs are shifted one session by the provided builder so
the execution estimate never uses today's close, range, or volume to price a
trade assumed at the start of that session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionCostSpec:
    portfolio_notional: float = 10_000_000.0
    commission_bps: float = 0.10
    fallback_half_spread_bps: float = 2.50
    impact_coefficient: float = 0.10
    maximum_participation_rate: float = 0.05
    default_adv_dollars: float = 25_000_000.0
    default_daily_volatility: float = 0.02
    default_annual_borrow_bps: float = 50.0
    financing_spread_bps: float = 100.0
    cash_yield_haircut_bps: float = 25.0
    short_rebate_spread_bps: float = 50.0
    annualization: int = 252
    reject_capacity_breaches: bool = True
    reject_locate_failures: bool = True


@dataclass(frozen=True)
class ExecutionMarketInputs:
    dollar_adv: pd.DataFrame
    daily_volatility: pd.DataFrame
    half_spread_bps: pd.DataFrame | None = None
    annual_borrow_bps: pd.DataFrame | None = None
    locate_available: pd.DataFrame | None = None
    annual_cash_rate: pd.Series | float = 0.0
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionCostBreakdown:
    transaction_cost: float
    commission_cost: float
    spread_cost: float
    impact_cost: float
    borrow_cost: float
    financing_cost: float
    cash_income: float
    short_rebate_income: float
    net_cost: float
    maximum_participation_rate: float
    capacity_breach_count: int
    locate_failure_count: int
    executable: bool
    detail: pd.DataFrame


def _validate_spec(spec: ExecutionCostSpec) -> None:
    finite_nonnegative = {
        "portfolio_notional": spec.portfolio_notional,
        "commission_bps": spec.commission_bps,
        "fallback_half_spread_bps": spec.fallback_half_spread_bps,
        "impact_coefficient": spec.impact_coefficient,
        "maximum_participation_rate": spec.maximum_participation_rate,
        "default_adv_dollars": spec.default_adv_dollars,
        "default_daily_volatility": spec.default_daily_volatility,
        "default_annual_borrow_bps": spec.default_annual_borrow_bps,
        "financing_spread_bps": spec.financing_spread_bps,
        "cash_yield_haircut_bps": spec.cash_yield_haircut_bps,
        "short_rebate_spread_bps": spec.short_rebate_spread_bps,
    }
    if any(not math.isfinite(value) or value < 0 for value in finite_nonnegative.values()):
        raise ValueError("execution-cost parameters must be finite and non-negative")
    if spec.portfolio_notional <= 0 or spec.default_adv_dollars <= 0:
        raise ValueError("portfolio_notional and default_adv_dollars must be positive")
    if not 0 < spec.maximum_participation_rate <= 1:
        raise ValueError("maximum_participation_rate must be in (0, 1]")
    if not isinstance(spec.annualization, int) or isinstance(spec.annualization, bool) or spec.annualization < 1:
        raise ValueError("annualization must be a positive integer")


def _clean_wide(frame: pd.DataFrame, *, label: str, nonnegative: bool = True) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{label} must use a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be sorted and unique")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} columns must be unique")
    clean = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if nonnegative and (clean < 0).any().any():
        raise ValueError(f"{label} cannot contain negative values")
    clean.columns = clean.columns.astype(str)
    return clean


def build_execution_market_inputs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    annual_borrow_bps: pd.DataFrame | None = None,
    locate_available: pd.DataFrame | None = None,
    annual_cash_rate: pd.Series | float = 0.0,
    lookback: int = 20,
    minimum_periods: int = 10,
    range_to_half_spread: float = 0.25,
) -> ExecutionMarketInputs:
    """Create lagged ADV, volatility, and optional range-spread proxies.

    Daily OHLCV bars do not contain historical executable quotes. When ``high``
    and ``low`` are supplied, the half-spread input is explicitly a conservative
    range-based proxy, not an observed bid/ask series.
    """
    close_clean = _clean_wide(close, label="close", nonnegative=True)
    volume_clean = _clean_wide(volume, label="volume", nonnegative=True)
    if (close_clean <= 0).any().any():
        raise ValueError("close prices must be positive")
    if set(close_clean.columns) != set(volume_clean.columns):
        raise ValueError("close and volume columns must match")
    if not close_clean.index.equals(volume_clean.index):
        raise ValueError("close and volume indexes must match")
    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer of at least two")
    if not isinstance(minimum_periods, int) or not 2 <= minimum_periods <= lookback:
        raise ValueError("minimum_periods must be between two and lookback")
    if not math.isfinite(range_to_half_spread) or range_to_half_spread < 0:
        raise ValueError("range_to_half_spread must be finite and non-negative")

    volume_clean = volume_clean.reindex(columns=close_clean.columns)
    dollar_volume = close_clean * volume_clean
    dollar_adv = dollar_volume.rolling(lookback, min_periods=minimum_periods).mean().shift(1)
    daily_volatility = (
        close_clean.pct_change(fill_method=None)
        .rolling(lookback, min_periods=minimum_periods)
        .std(ddof=1)
        .shift(1)
    )
    half_spread: pd.DataFrame | None = None
    if (high is None) ^ (low is None):
        raise ValueError("high and low must be supplied together")
    if high is not None and low is not None:
        high_clean = _clean_wide(high, label="high", nonnegative=True).reindex(
            index=close_clean.index, columns=close_clean.columns
        )
        low_clean = _clean_wide(low, label="low", nonnegative=True).reindex(
            index=close_clean.index, columns=close_clean.columns
        )
        if high_clean.isna().all().any() or low_clean.isna().all().any():
            raise ValueError("high and low must cover every close column")
        if (high_clean < low_clean).any().any():
            raise ValueError("high cannot be below low")
        range_bps = (high_clean - low_clean) / close_clean * 10_000.0
        half_spread = (
            (range_to_half_spread * range_bps)
            .rolling(lookback, min_periods=minimum_periods)
            .median()
            .shift(1)
            .clip(lower=0.05, upper=100.0)
        )
    return ExecutionMarketInputs(
        dollar_adv=dollar_adv,
        daily_volatility=daily_volatility,
        half_spread_bps=half_spread,
        annual_borrow_bps=annual_borrow_bps,
        locate_available=locate_available,
        annual_cash_rate=annual_cash_rate,
        provenance={
            "liquidity_lookback_sessions": lookback,
            "minimum_periods": minimum_periods,
            "adv_source": "lagged rolling close times volume",
            "volatility_source": "lagged rolling close-to-close volatility",
            "spread_source": "lagged high-low range proxy" if half_spread is not None else "fallback assumption",
        },
    )


class ExecutableCostModel:
    """Evaluate spread, impact, capacity, borrow, financing, and cash carry."""

    def __init__(
        self,
        inputs: ExecutionMarketInputs,
        spec: ExecutionCostSpec = ExecutionCostSpec(),
    ) -> None:
        _validate_spec(spec)
        self.spec = spec
        self.inputs = inputs
        self.dollar_adv = _clean_wide(inputs.dollar_adv, label="dollar_adv")
        self.daily_volatility = _clean_wide(
            inputs.daily_volatility, label="daily_volatility"
        )
        if set(self.dollar_adv.columns) != set(self.daily_volatility.columns):
            raise ValueError("dollar ADV and volatility columns must match")
        self.daily_volatility = self.daily_volatility.reindex(
            index=self.dollar_adv.index, columns=self.dollar_adv.columns
        )
        self.half_spread_bps = self._optional_numeric(
            inputs.half_spread_bps, "half_spread_bps"
        )
        self.annual_borrow_bps = self._optional_numeric(
            inputs.annual_borrow_bps, "annual_borrow_bps"
        )
        self.locate_available = self._optional_boolean(inputs.locate_available)

    def _optional_numeric(
        self,
        frame: pd.DataFrame | None,
        label: str,
    ) -> pd.DataFrame | None:
        if frame is None:
            return None
        clean = _clean_wide(frame, label=label).reindex(
            index=self.dollar_adv.index, columns=self.dollar_adv.columns
        )
        return clean

    def _optional_boolean(self, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("locate_available must be a DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("locate_available must use a DatetimeIndex")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("locate_available index must be sorted and unique")
        return frame.astype("boolean").reindex(
            index=self.dollar_adv.index, columns=self.dollar_adv.columns
        )

    @property
    def tickers(self) -> pd.Index:
        return self.dollar_adv.columns

    def _row(self, frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        eligible = frame.index[frame.index <= date]
        if len(eligible) == 0:
            return pd.Series(np.nan, index=frame.columns, dtype=float)
        return frame.loc[eligible[-1]].reindex(frame.columns)

    def _cash_rate(self, date: pd.Timestamp) -> float:
        value = self.inputs.annual_cash_rate
        if np.isscalar(value):
            rate = float(value)
        else:
            series = pd.to_numeric(value, errors="coerce").sort_index()
            eligible = series.index[series.index <= date]
            rate = float(series.loc[eligible[-1]]) if len(eligible) else 0.0
        if not math.isfinite(rate):
            return 0.0
        return rate

    def evaluate(
        self,
        date: pd.Timestamp | str,
        trade_dollars: pd.Series,
        holdings: pd.Series,
        *,
        equity: float = 1.0,
        include_holding_costs: bool = True,
    ) -> ExecutionCostBreakdown:
        """Evaluate one session's costs using normalized NAV notionals."""
        timestamp = pd.Timestamp(date)
        trades = pd.to_numeric(trade_dollars, errors="coerce")
        held = pd.to_numeric(holdings, errors="coerce")
        if trades.index.has_duplicates or held.index.has_duplicates:
            raise ValueError("trade and holding ticker labels must be unique")
        tickers = self.tickers.union(trades.index.astype(str)).union(held.index.astype(str))
        trades.index = trades.index.astype(str)
        held.index = held.index.astype(str)
        trades = trades.reindex(tickers, fill_value=0.0)
        held = held.reindex(tickers, fill_value=0.0)
        if trades.isna().any() or held.isna().any() or not np.isfinite(
            np.r_[trades.to_numpy(), held.to_numpy()]
        ).all():
            raise ValueError("trade and holding notionals must be finite")
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("equity must be finite and positive")

        adv = self._row(self.dollar_adv, timestamp).reindex(tickers)
        volatility = self._row(self.daily_volatility, timestamp).reindex(tickers)
        adv = adv.fillna(self.spec.default_adv_dollars).clip(lower=1.0)
        volatility = volatility.fillna(self.spec.default_daily_volatility).clip(lower=0.0)
        if self.half_spread_bps is None:
            half_spread = pd.Series(
                self.spec.fallback_half_spread_bps, index=tickers, dtype=float
            )
            spread_is_proxy = pd.Series(True, index=tickers)
        else:
            observed_spread = self._row(self.half_spread_bps, timestamp).reindex(tickers)
            spread_is_proxy = observed_spread.isna()
            half_spread = observed_spread.fillna(self.spec.fallback_half_spread_bps).clip(lower=0.0)
        if self.annual_borrow_bps is None:
            borrow_bps = pd.Series(
                self.spec.default_annual_borrow_bps, index=tickers, dtype=float
            )
            borrow_is_proxy = pd.Series(True, index=tickers)
        else:
            observed_borrow = self._row(self.annual_borrow_bps, timestamp).reindex(tickers)
            borrow_is_proxy = observed_borrow.isna()
            borrow_bps = observed_borrow.fillna(self.spec.default_annual_borrow_bps).clip(lower=0.0)
        if self.locate_available is None:
            locate = pd.Series(True, index=tickers)
            locate_is_proxy = pd.Series(True, index=tickers)
        else:
            raw_locate = self._row(self.locate_available.astype(float), timestamp).reindex(tickers)
            locate_is_proxy = raw_locate.isna()
            locate = raw_locate.fillna(1.0).astype(bool)

        absolute_trade = trades.abs()
        actual_trade_dollars = absolute_trade * self.spec.portfolio_notional
        participation = actual_trade_dollars / adv
        commission = absolute_trade * self.spec.commission_bps / 10_000.0
        spread = absolute_trade * half_spread / 10_000.0
        impact_rate = self.spec.impact_coefficient * volatility * np.sqrt(
            participation.clip(lower=0.0)
        )
        impact = absolute_trade * impact_rate
        short_notional = held.clip(upper=0.0).abs()
        borrow = (
            short_notional * borrow_bps / 10_000.0 / self.spec.annualization
            if include_holding_costs
            else pd.Series(0.0, index=tickers)
        )
        capacity_breach = participation > self.spec.maximum_participation_rate
        locate_failure = held.lt(0.0) & ~locate

        if include_holding_costs:
            long_total = float(held.clip(lower=0.0).sum())
            short_total = float(short_notional.sum())
            cash_rate = self._cash_rate(timestamp)
            free_cash = max(equity - long_total, 0.0)
            debit = max(long_total - equity, 0.0)
            cash_income = free_cash * max(
                cash_rate - self.spec.cash_yield_haircut_bps / 10_000.0,
                0.0,
            ) / self.spec.annualization
            short_rebate_income = short_total * max(
                cash_rate - self.spec.short_rebate_spread_bps / 10_000.0,
                0.0,
            ) / self.spec.annualization
            financing_cost = debit * (
                max(cash_rate, 0.0) + self.spec.financing_spread_bps / 10_000.0
            ) / self.spec.annualization
        else:
            long_total = 0.0
            short_total = 0.0
            cash_income = 0.0
            short_rebate_income = 0.0
            financing_cost = 0.0
        commission_total = float(commission.sum())
        spread_total = float(spread.sum())
        impact_total = float(impact.sum())
        transaction_total = commission_total + spread_total + impact_total
        borrow_total = float(borrow.sum())
        net_cost = (
            transaction_total
            + borrow_total
            + financing_cost
            - cash_income
            - short_rebate_income
        )
        capacity_count = int(capacity_breach.sum())
        locate_count = int(locate_failure.sum())
        executable = not (
            (self.spec.reject_capacity_breaches and capacity_count)
            or (self.spec.reject_locate_failures and locate_count)
        )
        detail = pd.DataFrame(
            {
                "ticker": tickers,
                "signed_trade_nav": trades.to_numpy(),
                "absolute_trade_dollars": actual_trade_dollars.to_numpy(),
                "dollar_adv": adv.to_numpy(),
                "participation_rate": participation.to_numpy(),
                "daily_volatility": volatility.to_numpy(),
                "half_spread_bps": half_spread.to_numpy(),
                "spread_is_proxy": spread_is_proxy.to_numpy(),
                "commission_cost_nav": commission.to_numpy(),
                "spread_cost_nav": spread.to_numpy(),
                "impact_cost_nav": impact.to_numpy(),
                "holding_nav": held.to_numpy(),
                "annual_borrow_bps": borrow_bps.to_numpy(),
                "borrow_is_proxy": borrow_is_proxy.to_numpy(),
                "borrow_cost_nav": borrow.to_numpy(),
                "locate_available": locate.to_numpy(),
                "locate_is_proxy": locate_is_proxy.to_numpy(),
                "capacity_breach": capacity_breach.to_numpy(),
                "locate_failure": locate_failure.to_numpy(),
            }
        )
        return ExecutionCostBreakdown(
            transaction_cost=transaction_total,
            commission_cost=commission_total,
            spread_cost=spread_total,
            impact_cost=impact_total,
            borrow_cost=borrow_total,
            financing_cost=float(financing_cost),
            cash_income=float(cash_income),
            short_rebate_income=float(short_rebate_income),
            net_cost=float(net_cost),
            maximum_participation_rate=float(participation.max()),
            capacity_breach_count=capacity_count,
            locate_failure_count=locate_count,
            executable=bool(executable),
            detail=detail,
        )
