"""Causal rolling diagnostics and kill-switch state for an equity pair.

The module is intentionally independent of the trading engine.  It emits one
row per requested signal date with explicit ``as_of_date`` provenance and
execution-ready controls (``allow_entry``, ``force_exit`` and
``target_gross_multiplier``).  With the default one-session decision lag, the
row dated *t* only uses prices through *t-1*.

Rolling Engle-Granger and residual ADF p-values are monitoring statistics, not
a substitute for a separately frozen formation screen or a multiple-testing
procedure.  They are best interpreted as pair-decay alarms.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from statsmodels.tools.sm_exceptions import CollinearityWarning, InterpolationWarning
from statsmodels.tsa.stattools import adfuller, coint

from .statistics import estimate_half_life, fit_log_hedge


@dataclass(frozen=True)
class PairHealthReference:
    """Formation-window quantities frozen before online monitoring starts."""

    pair: str
    dependent: str
    independent: str
    beta: float
    half_life_days: float | None = None

    def __post_init__(self) -> None:
        if not self.pair.strip():
            raise ValueError("pair must be non-empty")
        if not self.dependent.strip() or not self.independent.strip():
            raise ValueError("dependent and independent tickers must be non-empty")
        if self.dependent == self.independent:
            raise ValueError("dependent and independent tickers must differ")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("reference beta must be positive and finite")
        if self.half_life_days is not None and (
            not np.isfinite(self.half_life_days) or self.half_life_days <= 0
        ):
            raise ValueError("reference half-life must be positive and finite when supplied")


@dataclass(frozen=True)
class PairHealthConfig:
    """Parameters for rolling diagnostics and the quarantine state machine.

    Thresholds may be set to ``None`` to disable the corresponding rule.
    ``cooldown_sessions`` counts sessions *after* the kill-trigger session.
    """

    lookback_sessions: int = 252
    minimum_observations: int = 126
    decision_lag_sessions: int = 1
    adf_maxlag: int = 5
    maximum_cointegration_pvalue: float | None = 0.10
    maximum_residual_adf_pvalue: float | None = 0.10
    maximum_beta_relative_drift: float | None = 0.35
    maximum_half_life_days: float | None = 126.0
    maximum_half_life_multiple: float | None = 3.0
    minimum_structural_break_pvalue: float | None = 0.01
    maximum_stale_sessions: int = 2
    bad_observations_to_kill: int = 2
    cooldown_sessions: int = 20
    good_observations_to_recover: int = 3
    warning_blocks_new_entries: bool = True

    def __post_init__(self) -> None:
        if self.lookback_sessions < 20:
            raise ValueError("lookback_sessions must be at least 20")
        if not 20 <= self.minimum_observations <= self.lookback_sessions:
            raise ValueError(
                "minimum_observations must be between 20 and lookback_sessions"
            )
        if self.decision_lag_sessions < 0:
            raise ValueError("decision_lag_sessions cannot be negative")
        if self.adf_maxlag < 0:
            raise ValueError("adf_maxlag cannot be negative")
        for name in (
            "maximum_cointegration_pvalue",
            "maximum_residual_adf_pvalue",
            "minimum_structural_break_pvalue",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1] when supplied")
        for name in (
            "maximum_beta_relative_drift",
            "maximum_half_life_days",
            "maximum_half_life_multiple",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite when supplied")
        if self.maximum_stale_sessions < 0:
            raise ValueError("maximum_stale_sessions cannot be negative")
        if self.bad_observations_to_kill < 1:
            raise ValueError("bad_observations_to_kill must be at least one")
        if self.cooldown_sessions < 0:
            raise ValueError("cooldown_sessions cannot be negative")
        if self.good_observations_to_recover < 1:
            raise ValueError("good_observations_to_recover must be at least one")


@dataclass
class PairHealthResult:
    """Rolling health output plus transition events and a compact summary."""

    reference: PairHealthReference
    config: PairHealthConfig
    signals: pd.DataFrame
    events: pd.DataFrame
    summary: dict[str, Any]


def _safe_adf(residual: np.ndarray, maxlag: int) -> tuple[float, float]:
    values = np.asarray(residual, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(20, maxlag + 10) or np.std(values) <= 1e-12:
        return math.nan, math.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            statistic, pvalue, *_ = adfuller(
                values,
                maxlag=maxlag,
                regression="n",
                autolag="BIC",
            )
        return float(statistic), float(pvalue)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return math.nan, math.nan


def _safe_cointegration(
    dependent: np.ndarray,
    independent: np.ndarray,
    maxlag: int,
) -> tuple[float, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CollinearityWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            statistic, pvalue, _ = coint(
                np.log(dependent),
                np.log(independent),
                trend="c",
                maxlag=maxlag,
                autolag="BIC",
            )
        return float(statistic), float(pvalue)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return math.nan, math.nan


def _chow_break_test(log_y: np.ndarray, log_x: np.ndarray) -> tuple[float, float]:
    """Return a midpoint Chow-test structural-break proxy.

    The split is fixed at the midpoint of the trailing window, so it is not
    chosen by searching for the most significant break.
    """

    y = np.asarray(log_y, dtype=float)
    x = np.asarray(log_x, dtype=float)
    nobs = len(y)
    parameters = 2
    split = nobs // 2
    if split <= parameters or nobs - split <= parameters:
        return math.nan, math.nan

    def residual_sum_squares(segment_y: np.ndarray, segment_x: np.ndarray) -> float:
        design = np.column_stack([np.ones(len(segment_x), dtype=float), segment_x])
        coefficients = np.linalg.lstsq(design, segment_y, rcond=None)[0]
        residual = segment_y - design @ coefficients
        return float(residual @ residual)

    pooled_rss = residual_sum_squares(y, x)
    split_rss = residual_sum_squares(y[:split], x[:split]) + residual_sum_squares(
        y[split:], x[split:]
    )
    denominator_df = nobs - 2 * parameters
    if split_rss <= 1e-18 or denominator_df <= 0:
        return math.nan, math.nan
    numerator = max(0.0, (pooled_rss - split_rss) / parameters)
    denominator = split_rss / denominator_df
    statistic = numerator / denominator
    pvalue = float(f_distribution.sf(statistic, parameters, denominator_df))
    return float(statistic), pvalue


def _empty_diagnostic(
    signal_date: pd.Timestamp,
    as_of_date: pd.Timestamp | pd.NaT,
    window_start: pd.Timestamp | pd.NaT,
    observations: int,
    stale_sessions: int | float,
) -> dict[str, Any]:
    return {
        "signal_date": signal_date,
        "as_of_date": as_of_date,
        "window_start": window_start,
        "observations": observations,
        "stale_sessions": stale_sessions,
        "alpha": math.nan,
        "beta": math.nan,
        "beta_drift": math.nan,
        "beta_relative_drift": math.nan,
        "cointegration_statistic": math.nan,
        "cointegration_pvalue": math.nan,
        "residual_adf_statistic": math.nan,
        "residual_adf_pvalue": math.nan,
        "half_life_days": math.nan,
        "half_life_multiple": math.nan,
        "structural_break_statistic": math.nan,
        "structural_break_pvalue": math.nan,
        "diagnostic_valid": False,
    }


def rolling_pair_diagnostics(
    reference: PairHealthReference,
    prices: pd.DataFrame,
    config: PairHealthConfig | None = None,
) -> pd.DataFrame:
    """Compute trailing pair diagnostics without using future observations.

    With ``decision_lag_sessions=1``, each output row uses a window ending one
    row before its signal date.  A zero lag is supported for close-time signals
    that are explicitly executed no earlier than the next session.
    """

    config = config or PairHealthConfig()
    missing = {reference.dependent, reference.independent} - set(prices.columns)
    if missing:
        raise KeyError(f"Missing pair-health price columns: {sorted(missing)}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("prices index must contain unique dates")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    legs = prices[[reference.dependent, reference.independent]].apply(
        pd.to_numeric, errors="coerce"
    )
    legs = legs.where(legs > 0)
    rows: list[dict[str, Any]] = []
    index = legs.index

    for signal_position, signal_date in enumerate(index):
        as_of_position = signal_position - config.decision_lag_sessions
        if as_of_position < 0:
            rows.append(
                _empty_diagnostic(signal_date, pd.NaT, pd.NaT, 0, math.nan)
            )
            continue

        start_position = max(0, as_of_position - config.lookback_sessions + 1)
        raw_window = legs.iloc[start_position : as_of_position + 1]
        complete = raw_window.dropna()
        observations = len(complete)
        if complete.empty:
            stale_sessions: int | float = math.inf
            latest_complete_position = None
        else:
            latest_complete_label = complete.index[-1]
            latest_complete_position = int(index.get_loc(latest_complete_label))
            stale_sessions = as_of_position - latest_complete_position
        row = _empty_diagnostic(
            signal_date,
            index[as_of_position],
            raw_window.index[0],
            observations,
            stale_sessions,
        )
        if observations < config.minimum_observations:
            rows.append(row)
            continue

        dependent = complete[reference.dependent].to_numpy(dtype=float)
        independent = complete[reference.independent].to_numpy(dtype=float)
        alpha, beta, residual = fit_log_hedge(dependent, independent)
        coint_stat, coint_pvalue = _safe_cointegration(
            dependent, independent, config.adf_maxlag
        )
        adf_stat, adf_pvalue = _safe_adf(residual, config.adf_maxlag)
        half_life = estimate_half_life(residual)
        chow_stat, chow_pvalue = _chow_break_test(
            np.log(dependent), np.log(independent)
        )
        beta_drift = beta - reference.beta
        beta_relative_drift = abs(beta_drift) / abs(reference.beta)
        half_life_multiple = (
            half_life / reference.half_life_days
            if reference.half_life_days is not None and np.isfinite(half_life)
            else math.nan
        )
        required = (
            beta,
            coint_pvalue,
            adf_pvalue,
            half_life,
            chow_pvalue,
        )
        row.update(
            {
                "alpha": alpha,
                "beta": beta,
                "beta_drift": beta_drift,
                "beta_relative_drift": beta_relative_drift,
                "cointegration_statistic": coint_stat,
                "cointegration_pvalue": coint_pvalue,
                "residual_adf_statistic": adf_stat,
                "residual_adf_pvalue": adf_pvalue,
                "half_life_days": half_life,
                "half_life_multiple": half_life_multiple,
                "structural_break_statistic": chow_stat,
                "structural_break_pvalue": chow_pvalue,
                "diagnostic_valid": bool(all(np.isfinite(value) for value in required)),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows).set_index("signal_date")
    result.index = pd.DatetimeIndex(result.index, name=prices.index.name)
    return result


def classify_pair_health(
    diagnostics: pd.DataFrame,
    reference: PairHealthReference,
    config: PairHealthConfig | None = None,
) -> pd.DataFrame:
    """Apply deterministic breach flags to a diagnostic frame."""

    config = config or PairHealthConfig()
    required_columns = {
        "diagnostic_valid",
        "stale_sessions",
        "beta",
        "beta_relative_drift",
        "cointegration_pvalue",
        "residual_adf_pvalue",
        "half_life_days",
        "half_life_multiple",
        "structural_break_pvalue",
    }
    missing = required_columns - set(diagnostics.columns)
    if missing:
        raise ValueError(f"diagnostics missing required columns: {sorted(missing)}")

    result = diagnostics.copy()
    result["data_breach"] = (
        ~result["diagnostic_valid"].astype(bool)
        | (pd.to_numeric(result["stale_sessions"], errors="coerce") > config.maximum_stale_sessions)
    )
    result["stationarity_breach"] = False
    if config.maximum_cointegration_pvalue is not None:
        result["stationarity_breach"] |= (
            pd.to_numeric(result["cointegration_pvalue"], errors="coerce")
            > config.maximum_cointegration_pvalue
        ).fillna(False)
    if config.maximum_residual_adf_pvalue is not None:
        result["stationarity_breach"] |= (
            pd.to_numeric(result["residual_adf_pvalue"], errors="coerce")
            > config.maximum_residual_adf_pvalue
        ).fillna(False)
    beta = pd.to_numeric(result["beta"], errors="coerce")
    result["beta_breach"] = (~np.isfinite(beta)) | (beta <= 0)
    if config.maximum_beta_relative_drift is not None:
        result["beta_breach"] |= (
            pd.to_numeric(result["beta_relative_drift"], errors="coerce")
            > config.maximum_beta_relative_drift
        ).fillna(False)
    half_life = pd.to_numeric(result["half_life_days"], errors="coerce")
    result["half_life_breach"] = (~np.isfinite(half_life)) | (half_life <= 0)
    if config.maximum_half_life_days is not None:
        result["half_life_breach"] |= (half_life > config.maximum_half_life_days).fillna(
            False
        )
    if (
        config.maximum_half_life_multiple is not None
        and reference.half_life_days is not None
    ):
        result["half_life_breach"] |= (
            pd.to_numeric(result["half_life_multiple"], errors="coerce")
            > config.maximum_half_life_multiple
        ).fillna(False)
    result["structural_break_breach"] = False
    if config.minimum_structural_break_pvalue is not None:
        result["structural_break_breach"] = (
            pd.to_numeric(result["structural_break_pvalue"], errors="coerce")
            < config.minimum_structural_break_pvalue
        ).fillna(False)

    breach_columns = [
        "data_breach",
        "stationarity_breach",
        "beta_breach",
        "half_life_breach",
        "structural_break_breach",
    ]
    result["unhealthy"] = result[breach_columns].any(axis=1)

    def reasons(row: pd.Series) -> str:
        return "|".join(name.removesuffix("_breach") for name in breach_columns if row[name])

    result["health_reasons"] = result.apply(reasons, axis=1).astype("string")
    return result


def apply_kill_switch(
    classified: pd.DataFrame,
    config: PairHealthConfig | None = None,
) -> pd.DataFrame:
    """Convert classified daily health into causal trading controls.

    This function is kept separate from statistical estimation so state
    transitions can be unit-tested and operationally replayed exactly.
    """

    config = config or PairHealthConfig()
    required = {"diagnostic_valid", "unhealthy", "health_reasons"}
    missing = required - set(classified.columns)
    if missing:
        raise ValueError(f"classified diagnostics missing columns: {sorted(missing)}")

    result = classified.copy()
    states: list[str] = []
    allow_entries: list[bool] = []
    force_exits: list[bool] = []
    multipliers: list[float] = []
    kill_events: list[bool] = []
    recovery_events: list[bool] = []
    consecutive_bad_values: list[int] = []
    consecutive_good_values: list[int] = []
    cooldown_values: list[int] = []
    active_reasons: list[str] = []

    quarantined = False
    cooldown_remaining = 0
    consecutive_bad = 0
    consecutive_good = 0
    active_reason = ""
    ever_valid = False

    for _, row in result.iterrows():
        diagnostic_valid = bool(row["diagnostic_valid"])
        unhealthy = bool(row["unhealthy"])
        health_reason = str(row["health_reasons"] or "")
        kill_event = False
        recovery_event = False

        if not ever_valid and not diagnostic_valid:
            state = "warming_up"
            allow_entry = False
            force_exit = False
            multiplier = 0.0
            consecutive_bad = 0
            consecutive_good = 0
        else:
            ever_valid = ever_valid or diagnostic_valid
            if not quarantined:
                if unhealthy:
                    consecutive_bad += 1
                    consecutive_good = 0
                    if consecutive_bad >= config.bad_observations_to_kill:
                        quarantined = True
                        cooldown_remaining = config.cooldown_sessions
                        active_reason = health_reason or "unknown"
                        state = "killed"
                        kill_event = True
                        allow_entry = False
                        force_exit = True
                        multiplier = 0.0
                    else:
                        state = "warning"
                        allow_entry = not config.warning_blocks_new_entries
                        force_exit = False
                        multiplier = 1.0
                else:
                    consecutive_bad = 0
                    consecutive_good = 0
                    active_reason = ""
                    state = "tradable"
                    allow_entry = True
                    force_exit = False
                    multiplier = 1.0
            else:
                consecutive_bad = consecutive_bad + 1 if unhealthy else 0
                if cooldown_remaining > 0:
                    state = "cooldown"
                    cooldown_remaining -= 1
                    consecutive_good = 0
                else:
                    if unhealthy:
                        consecutive_good = 0
                    else:
                        consecutive_good += 1
                    if consecutive_good >= config.good_observations_to_recover:
                        quarantined = False
                        active_reason = ""
                        consecutive_bad = 0
                        state = "tradable"
                        recovery_event = True
                    else:
                        state = "recovery"
                allow_entry = state == "tradable"
                force_exit = False
                multiplier = 1.0 if state == "tradable" else 0.0

        states.append(state)
        allow_entries.append(allow_entry)
        force_exits.append(force_exit)
        multipliers.append(multiplier)
        kill_events.append(kill_event)
        recovery_events.append(recovery_event)
        consecutive_bad_values.append(consecutive_bad)
        consecutive_good_values.append(consecutive_good)
        cooldown_values.append(cooldown_remaining)
        active_reasons.append(active_reason)

    result["health_state"] = pd.Series(states, index=result.index, dtype="string")
    result["allow_entry"] = allow_entries
    result["force_exit"] = force_exits
    result["target_gross_multiplier"] = multipliers
    result["kill_event"] = kill_events
    result["recovery_event"] = recovery_events
    result["consecutive_bad"] = consecutive_bad_values
    result["consecutive_good"] = consecutive_good_values
    result["cooldown_remaining"] = cooldown_values
    result["active_kill_reason"] = pd.Series(
        active_reasons, index=result.index, dtype="string"
    )
    return result


def gate_target_positions(
    target_position: pd.Series,
    health_signals: pd.DataFrame,
) -> pd.Series:
    """Apply daily health controls to an existing strategy target series.

    A warning may block a new position while allowing an existing position to
    follow its normal exit logic.  A kill/cooldown/recovery quarantine forces
    the target flat.  Missing health rows fail closed.
    """

    required = {"allow_entry", "target_gross_multiplier"}
    missing = required - set(health_signals.columns)
    if missing:
        raise ValueError(f"health_signals missing required columns: {sorted(missing)}")
    if target_position.index.has_duplicates:
        raise ValueError("target_position index must contain unique dates")
    if health_signals.index.has_duplicates:
        raise ValueError("health_signals index must contain unique dates")

    desired = pd.to_numeric(target_position, errors="coerce").fillna(0.0)
    aligned = health_signals.reindex(desired.index)
    allow_entry = aligned["allow_entry"].fillna(False).astype(bool)
    multiplier = pd.to_numeric(
        aligned["target_gross_multiplier"], errors="coerce"
    ).fillna(0.0)
    gated_values: list[float] = []
    previous = 0.0
    for value, entry_allowed, gross_multiplier in zip(
        desired.to_numpy(dtype=float),
        allow_entry.to_numpy(dtype=bool),
        multiplier.to_numpy(dtype=float),
        strict=True,
    ):
        if gross_multiplier <= 0.0 or value == 0.0:
            gated = 0.0
        elif previous == 0.0 or np.sign(value) != np.sign(previous):
            gated = value * max(0.0, gross_multiplier) if entry_allowed else 0.0
        else:
            gated = value * max(0.0, gross_multiplier)
        gated_values.append(gated)
        previous = gated
    return pd.Series(gated_values, index=desired.index, name="health_gated_target")


def run_pair_health(
    reference: PairHealthReference,
    prices: pd.DataFrame,
    config: PairHealthConfig | None = None,
) -> PairHealthResult:
    """Run rolling diagnostics and the kill switch for one pair."""

    config = config or PairHealthConfig()
    diagnostics = rolling_pair_diagnostics(reference, prices, config)
    classified = classify_pair_health(diagnostics, reference, config)
    signals = apply_kill_switch(classified, config)
    event_columns = [
        "health_state",
        "health_reasons",
        "active_kill_reason",
        "kill_event",
        "recovery_event",
        "cooldown_remaining",
    ]
    events = signals.loc[signals["kill_event"] | signals["recovery_event"], event_columns].copy()
    valid = signals["diagnostic_valid"].astype(bool)
    summary: dict[str, Any] = {
        "pair": reference.pair,
        "observations": int(len(signals)),
        "valid_diagnostic_observations": int(valid.sum()),
        "kill_count": int(signals["kill_event"].sum()),
        "recovery_count": int(signals["recovery_event"].sum()),
        "entry_eligible_fraction": float(signals["allow_entry"].mean())
        if len(signals)
        else math.nan,
        "forced_flat_fraction": float(
            (signals["target_gross_multiplier"] == 0.0).mean()
        )
        if len(signals)
        else math.nan,
        "latest_state": str(signals["health_state"].iloc[-1]) if len(signals) else None,
        "latest_as_of_date": signals["as_of_date"].iloc[-1] if len(signals) else pd.NaT,
    }
    return PairHealthResult(reference, config, signals, events, summary)
