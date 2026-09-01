"""Leakage-controlled parameter search for the held-out pairs strategy.

The pair universe and each pair's formation-window ``alpha``/``beta`` are inputs;
this module never reselects pairs or refits hedge ratios.  The first part of the
held-out sample is used for tuning.  Only the tuning winner and the unmodified
baseline are then reported on the later lockbox sample.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .backtest import backtest_selected_pairs, compute_zscore, performance_metrics
from .config import StrategyConfig


@dataclass(frozen=True)
class GridSearchSpec:
    """Cartesian search space.

    The defaults are a focused 128-cell frequency/decay grid when both SMA and
    EWMA are supported by :class:`StrategyConfig`.
    """

    zscore_methods: tuple[str, ...] = ("sma", "ewma")
    zscore_lookbacks: tuple[int, ...] = (20, 40, 60, 90)
    entry_zs: tuple[float, ...] = (1.25, 1.5, 1.75, 2.0)
    exit_zs: tuple[float, ...] = (0.25, 0.5)
    maximum_holding_days: tuple[int, ...] = (30, 60)


@dataclass(frozen=True)
class ParameterSet:
    config_id: str
    zscore_method: str
    zscore_lookback: int
    zscore_min_periods: int
    entry_z: float
    exit_z: float
    maximum_holding_days: int
    is_baseline: bool = False

    def strategy_values(self) -> dict[str, Any]:
        return {
            "zscore_method": self.zscore_method,
            "zscore_lookback": self.zscore_lookback,
            "zscore_min_periods": self.zscore_min_periods,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "maximum_holding_days": self.maximum_holding_days,
        }


@dataclass
class GridEvaluation:
    """Detailed output for one fixed parameter set."""

    parameter_set: ParameterSet
    strategy_config: StrategyConfig
    summary: pd.DataFrame
    pair_metrics: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    pair_returns: pd.DataFrame


@dataclass
class GridSearchResult:
    """Tuning table, untouched lockbox comparison, and reproducible details."""

    summary: pd.DataFrame
    ranking: pd.DataFrame
    comparison: pd.DataFrame
    failures: pd.DataFrame
    best_parameters: dict[str, Any]
    best_config: StrategyConfig
    baseline_config: StrategyConfig
    best_evaluation: GridEvaluation
    baseline_evaluation: GridEvaluation


@dataclass
class ParameterReturnCache:
    """Causal daily grid results for a fixed pair universe and allocation.

    ``returns`` and ``turnover`` are date-by-configuration portfolio panels.
    ``pair_returns`` retains each configuration's date-by-pair net returns so a
    nested validator can fit alternative allocations on its training dates and
    apply the frozen weights out of sample. ``positions`` and ``entries`` expose
    effective daily position states and causal entry indicators, while
    ``entry_counts`` is their window total. A position carried into the window
    is not a new entry.
    """

    returns: pd.DataFrame
    turnover: pd.DataFrame
    pair_returns: dict[str, pd.DataFrame]
    entries: dict[str, pd.DataFrame]
    positions: dict[str, pd.DataFrame]
    entry_counts: pd.DataFrame
    parameters: pd.DataFrame
    pair_weights: pd.Series


ProgressCallback = Callable[[int, int, ParameterSet], None]


class _FastTuningEvaluator:
    """Cache fixed-price inputs and vectorize one tuning simulation across pairs.

    This is intentionally used only while ranking grid cells.  The chosen cell
    and baseline are subsequently rerun through ``backtest_selected_pairs`` for
    canonical signals, ledgers, and lockbox results.
    """

    def __init__(
        self,
        selected_pairs: pd.DataFrame,
        prices: pd.DataFrame,
        heldout_start: pd.Timestamp,
        lockbox_start: pd.Timestamp,
        base_config: StrategyConfig,
    ) -> None:
        self.selected_pairs = selected_pairs.copy()
        self.prices = prices.sort_index()
        self.heldout_start = heldout_start
        self.lockbox_start = lockbox_start
        self.base_config = base_config
        self.dates = self.prices.index
        self.test_mask = (self.dates >= heldout_start) & (self.dates < lockbox_start)
        self.pairs = self.selected_pairs["pair"].astype(str).tolist()
        self.sectors = self.selected_pairs["sector"].astype(str).to_numpy()
        self.spreads: list[pd.Series] = []
        dependent_returns: list[np.ndarray] = []
        independent_returns: list[np.ndarray] = []
        base_dependent: list[float] = []
        base_independent: list[float] = []
        for row in self.selected_pairs.to_dict(orient="records"):
            dependent = str(row["dependent"])
            independent = str(row["independent"])
            beta = float(row["beta"])
            if dependent not in self.prices or independent not in self.prices:
                raise KeyError(f"Missing price leg for {row['pair']}")
            if not np.isfinite(beta) or beta <= 0:
                raise ValueError(f"Pair {row['pair']} requires a positive finite hedge ratio")
            legs = self.prices[[dependent, independent]].apply(pd.to_numeric, errors="coerce")
            valid = legs.where(legs > 0)
            spread = (
                np.log(valid[dependent])
                - float(row["alpha"])
                - beta * np.log(valid[independent])
            )
            returns = valid.pct_change(fill_method=None)
            self.spreads.append(spread)
            dependent_returns.append(returns[dependent].to_numpy(dtype=float))
            independent_returns.append(returns[independent].to_numpy(dtype=float))
            base_dependent.append(1.0 / (1.0 + abs(beta)))
            base_independent.append(-beta / (1.0 + abs(beta)))
        self.dependent_returns = np.nan_to_num(
            np.column_stack(dependent_returns), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.independent_returns = np.nan_to_num(
            np.column_stack(independent_returns), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.base_dependent = np.asarray(base_dependent, dtype=float)
        self.base_independent = np.asarray(base_independent, dtype=float)
        self._zscore_cache: dict[tuple[str, int, int], np.ndarray] = {}

    def _zscores(
        self,
        parameter_set: ParameterSet,
        config: StrategyConfig,
    ) -> np.ndarray:
        key = (
            parameter_set.zscore_method,
            parameter_set.zscore_lookback,
            parameter_set.zscore_min_periods,
        )
        cached = self._zscore_cache.get(key)
        if cached is not None:
            return cached
        matrix = np.column_stack(
            [compute_zscore(spread, config)[2].to_numpy(dtype=float) for spread in self.spreads]
        )
        self._zscore_cache[key] = matrix
        return matrix

    @staticmethod
    def _target_positions(zscores: np.ndarray, config: StrategyConfig) -> np.ndarray:
        rows, columns = zscores.shape
        target = np.zeros((rows, columns), dtype=np.int8)
        position = np.zeros(columns, dtype=np.int8)
        held = np.zeros(columns, dtype=np.int32)
        entry_z = float(config.entry_z)
        exit_z = float(config.exit_z)
        stop_z = float(config.stop_z)
        maximum_holding_days = int(config.maximum_holding_days)
        for location in range(rows):
            values = zscores[location]
            finite = np.isfinite(values)
            start_active = position != 0
            missing = ~finite
            position[missing] = 0
            held[missing] = 0

            active = start_active & finite
            held[active] += 1
            absolute = np.abs(values)
            exits = active & (
                (absolute <= exit_z)
                | (absolute >= stop_z)
                | (held >= maximum_holding_days)
            )
            position[exits] = 0
            held[exits] = 0

            flat = (~start_active) & finite
            position[flat & (values <= -entry_z)] = 1
            position[flat & (values >= entry_z)] = -1
            held[flat] = 0
            target[location] = position
        return target

    @staticmethod
    def _matrix_cagr(returns: np.ndarray) -> np.ndarray:
        if len(returns) < 2:
            return np.full(returns.shape[1], np.nan)
        ending = np.prod(1.0 + returns, axis=0)
        result = np.full(returns.shape[1], np.nan)
        valid = ending > 0
        result[valid] = ending[valid] ** (252.0 / len(returns)) - 1.0
        return result

    def _simulate(
        self,
        parameter_set: ParameterSet,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return full-history pair net returns, turnover, and positions."""

        config = strategy_config_for_parameter_set(self.base_config, parameter_set)
        zscores = self._zscores(parameter_set, config)
        target = self._target_positions(zscores, config)
        effective = np.zeros_like(target)
        effective[1:] = target[:-1]

        weight_dependent = effective * self.base_dependent
        weight_independent = effective * self.base_independent
        gross_return = (
            weight_dependent * self.dependent_returns
            + weight_independent * self.independent_returns
        )
        previous_dependent = np.zeros_like(weight_dependent)
        previous_independent = np.zeros_like(weight_independent)
        previous_gross_return = np.zeros_like(gross_return)
        shifted_dependent_returns = np.zeros_like(self.dependent_returns)
        shifted_independent_returns = np.zeros_like(self.independent_returns)
        previous_dependent[1:] = weight_dependent[:-1]
        previous_independent[1:] = weight_independent[:-1]
        previous_gross_return[1:] = gross_return[:-1]
        shifted_dependent_returns[1:] = self.dependent_returns[:-1]
        shifted_independent_returns[1:] = self.independent_returns[:-1]
        denominator = 1.0 + previous_gross_return
        with np.errstate(divide="ignore", invalid="ignore"):
            post_return_dependent = (
                previous_dependent * (1.0 + shifted_dependent_returns) / denominator
            )
            post_return_independent = (
                previous_independent * (1.0 + shifted_independent_returns) / denominator
            )
        post_return_dependent = np.where(
            np.isfinite(post_return_dependent), post_return_dependent, previous_dependent
        )
        post_return_independent = np.where(
            np.isfinite(post_return_independent), post_return_independent, previous_independent
        )
        turnover = (
            np.abs(weight_dependent - post_return_dependent)
            + np.abs(weight_independent - post_return_independent)
        )
        short_notional = np.maximum(-weight_dependent, 0.0) + np.maximum(
            -weight_independent, 0.0
        )
        net_return = (
            gross_return
            - turnover * float(config.transaction_cost_bps) / 10_000.0
            - short_notional * float(config.annual_short_borrow_bps) / 10_000.0 / 252.0
        )
        return net_return, turnover, effective

    def evaluate(self, parameter_set: ParameterSet) -> pd.DataFrame:
        net_return, turnover, effective = self._simulate(parameter_set)

        window_returns = net_return[self.test_mask]
        window_turnover = turnover[self.test_mask]
        window_positions = effective[self.test_mask]
        previous_positions = np.zeros_like(window_positions)
        previous_positions[1:] = window_positions[:-1]
        trade_counts = ((previous_positions == 0) & (window_positions != 0)).sum(axis=0)
        pair_cagrs = self._matrix_cagr(window_returns)
        positive_pair_fraction = float(np.nanmean(pair_cagrs > 0))

        equal_return = window_returns.mean(axis=1)
        equal_turnover = window_turnover.mean(axis=1)
        sector_returns: list[np.ndarray] = []
        sector_turnover: list[np.ndarray] = []
        for sector in sorted(set(self.sectors)):
            mask = self.sectors == sector
            sector_returns.append(window_returns[:, mask].mean(axis=1))
            sector_turnover.append(window_turnover[:, mask].mean(axis=1))
        balanced_return = np.column_stack(sector_returns).mean(axis=1)
        balanced_turnover = np.column_stack(sector_turnover).mean(axis=1)
        first_date = self.dates[self.test_mask][0]
        last_date = self.dates[self.test_mask][-1]
        common = {
            **asdict(parameter_set),
            "split": "tuning",
            "window_start": str(first_date.date()),
            "window_end": str(last_date.date()),
            "observations": int(self.test_mask.sum()),
            "pair_count": int(len(self.pairs)),
            "sector_count": int(len(set(self.sectors))),
            "total_trades": int(trade_counts.sum()),
            "median_trades_per_pair": float(np.median(trade_counts)),
            "mean_trades_per_pair": float(np.mean(trade_counts)),
            "positive_pair_fraction": positive_pair_fraction,
        }
        rows: list[dict[str, Any]] = []
        for portfolio, portfolio_return, portfolio_turnover in (
            ("equal_weight", equal_return, equal_turnover),
            ("sector_balanced", balanced_return, balanced_turnover),
        ):
            series = pd.Series(portfolio_return, index=self.dates[self.test_mask])
            rows.append(
                {
                    **common,
                    "portfolio": portfolio,
                    **performance_metrics(series),
                    "annual_turnover": float(portfolio_turnover.mean() * 252.0),
                }
            )
        return pd.DataFrame(rows)


def _unique_sorted(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(values)))


def _number_token(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _config_id(
    method: str,
    lookback: int,
    min_periods: int,
    entry_z: float,
    exit_z: float,
    maximum_holding_days: int,
) -> str:
    return "__".join(
        [
            method,
            f"lb{lookback}",
            f"min{min_periods}",
            f"en{_number_token(entry_z)}",
            f"ex{_number_token(exit_z)}",
            f"hold{maximum_holding_days}",
        ]
    )


def _baseline_parameter(config: StrategyConfig) -> ParameterSet:
    method = str(getattr(config, "zscore_method", "sma")).lower()
    return ParameterSet(
        config_id=_config_id(
            method,
            config.zscore_lookback,
            config.zscore_min_periods,
            config.entry_z,
            config.exit_z,
            config.maximum_holding_days,
        ),
        zscore_method=method,
        zscore_lookback=int(config.zscore_lookback),
        zscore_min_periods=int(config.zscore_min_periods),
        entry_z=float(config.entry_z),
        exit_z=float(config.exit_z),
        maximum_holding_days=int(config.maximum_holding_days),
        is_baseline=True,
    )


def build_parameter_sets(
    spec: GridSearchSpec,
    base_config: StrategyConfig,
    *,
    include_baseline: bool = True,
) -> tuple[ParameterSet, ...]:
    """Return a deterministic, validated Cartesian parameter grid."""

    methods = tuple(str(value).strip().lower() for value in _unique_sorted(spec.zscore_methods))
    lookbacks = _unique_sorted(spec.zscore_lookbacks)
    entries = _unique_sorted(spec.entry_zs)
    exits = _unique_sorted(spec.exit_zs)
    holding_periods = _unique_sorted(spec.maximum_holding_days)
    if not all((methods, lookbacks, entries, exits, holding_periods)):
        raise ValueError("Every grid dimension must contain at least one value")
    if any(method not in {"sma", "ewma"} for method in methods):
        raise ValueError("zscore_methods may contain only 'sma' and 'ewma'")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 2 for value in lookbacks):
        raise ValueError("zscore_lookbacks must contain integers of at least 2")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in holding_periods
    ):
        raise ValueError("maximum_holding_days must contain positive integers")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in entries + exits):
        raise ValueError("entry_zs and exit_zs must contain finite non-negative values")

    supports_method = "zscore_method" in {field.name for field in fields(base_config)}
    if not supports_method and any(method != "sma" for method in methods):
        raise ValueError(
            "This StrategyConfig does not expose zscore_method; only the SMA grid can be run"
        )

    baseline = _baseline_parameter(base_config)
    rows: list[ParameterSet] = []
    seen: set[str] = set()
    for method, lookback, entry_z, exit_z, maximum_holding_days in product(
        methods, lookbacks, entries, exits, holding_periods
    ):
        entry_z = float(entry_z)
        exit_z = float(exit_z)
        if not 0 <= exit_z < entry_z < float(base_config.stop_z):
            continue
        # Short lookbacks cannot retain a larger baseline warm-up.  Holding the
        # warm-up at min(base, lookback) is deterministic and is disclosed as a
        # searched-row field rather than hidden inside the backtester.
        min_periods = min(int(base_config.zscore_min_periods), int(lookback))
        identifier = _config_id(
            method,
            int(lookback),
            min_periods,
            entry_z,
            exit_z,
            int(maximum_holding_days),
        )
        if identifier in seen:
            continue
        seen.add(identifier)
        rows.append(
            ParameterSet(
                config_id=identifier,
                zscore_method=method,
                zscore_lookback=int(lookback),
                zscore_min_periods=min_periods,
                entry_z=entry_z,
                exit_z=exit_z,
                maximum_holding_days=int(maximum_holding_days),
                is_baseline=identifier == baseline.config_id,
            )
        )
    if include_baseline and baseline.config_id not in seen:
        rows.append(baseline)
    if not rows:
        raise ValueError("The grid has no combinations satisfying exit_z < entry_z < stop_z")
    return tuple(sorted(rows, key=lambda row: row.config_id))


def strategy_config_for_parameter_set(
    base_config: StrategyConfig,
    parameter_set: ParameterSet,
) -> StrategyConfig:
    """Apply searched values while leaving costs and stop rules unchanged."""

    supported = {field.name for field in fields(base_config)}
    values = parameter_set.strategy_values()
    if "zscore_method" not in supported:
        if parameter_set.zscore_method != "sma":
            raise ValueError("EWMA requires StrategyConfig.zscore_method support")
        values.pop("zscore_method")
    values = {key: value for key, value in values.items() if key in supported}
    return replace(base_config, **values)


def _validate_fixed_pair_weights(
    selected_pairs: pd.DataFrame,
    pair_weights: pd.Series | Mapping[str, float],
) -> pd.Series:
    pairs = selected_pairs["pair"].astype(str).tolist()
    if len(set(pairs)) != len(pairs):
        raise ValueError("selected_pairs must contain unique pair identifiers")
    if isinstance(pair_weights, pd.Series):
        weights = pair_weights.copy()
    elif isinstance(pair_weights, Mapping):
        weights = pd.Series(dict(pair_weights), dtype=float)
    else:
        raise TypeError("pair_weights must be a pandas Series or mapping")
    weights.index = weights.index.map(str)
    if weights.index.has_duplicates:
        raise ValueError("pair_weights must have unique pair identifiers")
    missing = sorted(set(pairs) - set(weights.index))
    extra = sorted(set(weights.index) - set(pairs))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError(
            "pair_weights must align exactly with selected_pairs ("
            + "; ".join(details)
            + ")"
        )
    weights = pd.to_numeric(weights.reindex(pairs), errors="coerce").astype(float)
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("pair_weights must be finite")
    if (weights < 0).any():
        raise ValueError("pair_weights cannot be negative")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("pair_weights must sum to one")
    weights.name = "pair_weight"
    weights.index.name = "pair"
    return weights


def build_parameter_return_cache(
    selected_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    base_config: StrategyConfig,
    spec: GridSearchSpec,
    pair_weights: pd.Series | Mapping[str, float],
    *,
    include_baseline: bool = True,
    progress: ProgressCallback | None = None,
) -> ParameterReturnCache:
    """Build a fast causal return cache for every grid cell on ``[start, end]``.

    The supplied fixed pair weights must align exactly with ``selected_pairs``,
    be non-negative, and sum to one.  The full price history before ``start`` is
    retained as causal signal warm-up, while observations after ``end`` are
    discarded before any calculation.  Each pair is charged the flat turnover
    and annual short-borrow costs in ``base_config`` before aggregation.

    This function uses the vectorized grid simulator directly.  It deliberately
    does not call the canonical backtester once per parameter combination.
    """

    required = {"pair", "sector", "dependent", "independent", "alpha", "beta"}
    missing_columns = sorted(required - set(selected_pairs.columns))
    if missing_columns:
        raise ValueError(
            "selected_pairs is missing columns: " + ", ".join(missing_columns)
        )
    if selected_pairs.empty:
        raise ValueError("selected_pairs cannot be empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("prices index must contain unique dates")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start > end:
        raise ValueError("start must be on or before end")
    weights = _validate_fixed_pair_weights(selected_pairs, pair_weights)

    # Sorting is deterministic, and clipping first makes the no-future-data
    # boundary structural rather than merely an assumption of the caller.
    causal_prices = prices.sort_index().loc[lambda frame: frame.index <= end]
    window_mask = (causal_prices.index >= start) & (causal_prices.index <= end)
    if not bool(window_mask.any()):
        raise ValueError("No price observations fall inside the requested window")
    evaluator = _FastTuningEvaluator(
        selected_pairs=selected_pairs,
        prices=causal_prices,
        heldout_start=start,
        lockbox_start=end,
        base_config=base_config,
    )
    parameter_sets = build_parameter_sets(
        spec,
        base_config,
        include_baseline=include_baseline,
    )
    dates = evaluator.dates[window_mask]
    weight_array = weights.to_numpy(dtype=float)
    portfolio_returns: dict[str, pd.Series] = {}
    portfolio_turnover: dict[str, pd.Series] = {}
    pair_return_cache: dict[str, pd.DataFrame] = {}
    entry_cache: dict[str, pd.DataFrame] = {}
    position_cache: dict[str, pd.DataFrame] = {}
    count_rows: list[dict[str, Any]] = []
    for completed, parameter_set in enumerate(parameter_sets, start=1):
        net_return, turnover, effective = evaluator._simulate(parameter_set)
        window_returns = net_return[window_mask]
        window_turnover = turnover[window_mask]

        previous_positions = np.zeros_like(effective)
        previous_positions[1:] = effective[:-1]
        entries = ((previous_positions == 0) & (effective != 0))[window_mask]
        counts = entries.sum(axis=0).astype(int)
        identifier = parameter_set.config_id
        portfolio_returns[identifier] = pd.Series(
            window_returns @ weight_array,
            index=dates,
            dtype=float,
        )
        portfolio_turnover[identifier] = pd.Series(
            window_turnover @ weight_array,
            index=dates,
            dtype=float,
        )
        pair_return_cache[identifier] = pd.DataFrame(
            window_returns,
            index=dates,
            columns=evaluator.pairs,
            dtype=float,
        )
        entry_cache[identifier] = pd.DataFrame(
            entries,
            index=dates,
            columns=evaluator.pairs,
            dtype=bool,
        )
        position_cache[identifier] = pd.DataFrame(
            effective[window_mask],
            index=dates,
            columns=evaluator.pairs,
            dtype=np.int8,
        )
        count_rows.append(
            {
                "config_id": identifier,
                **dict(zip(evaluator.pairs, counts, strict=True)),
                "total_entries": int(counts.sum()),
            }
        )
        if progress is not None:
            progress(completed, len(parameter_sets), parameter_set)

    returns_frame = pd.DataFrame(portfolio_returns, index=dates)
    turnover_frame = pd.DataFrame(portfolio_turnover, index=dates)
    returns_frame.index.name = "date"
    returns_frame.columns.name = "config_id"
    turnover_frame.index.name = "date"
    turnover_frame.columns.name = "config_id"
    for frame in (
        *pair_return_cache.values(),
        *entry_cache.values(),
        *position_cache.values(),
    ):
        frame.index.name = "date"
        frame.columns.name = "pair"
    entry_counts = pd.DataFrame(count_rows).set_index("config_id")
    entry_counts.index.name = "config_id"
    parameters = pd.DataFrame([asdict(row) for row in parameter_sets]).set_index(
        "config_id"
    )
    parameters.index.name = "config_id"
    return ParameterReturnCache(
        returns=returns_frame,
        turnover=turnover_frame,
        pair_returns=pair_return_cache,
        entries=entry_cache,
        positions=position_cache,
        entry_counts=entry_counts,
        parameters=parameters,
        pair_weights=weights,
    )


def _validate_inputs(
    selected_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    heldout_start: pd.Timestamp,
    lockbox_start: pd.Timestamp,
    heldout_end: pd.Timestamp,
) -> None:
    required = {"pair", "sector", "dependent", "independent", "alpha", "beta"}
    missing = sorted(required - set(selected_pairs.columns))
    if missing:
        raise ValueError(f"selected_pairs is missing columns: {', '.join(missing)}")
    if selected_pairs.empty:
        raise ValueError("selected_pairs cannot be empty")
    if selected_pairs["pair"].duplicated().any():
        raise ValueError("selected_pairs must contain unique pair identifiers")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("prices index must contain unique dates")
    if not heldout_start < lockbox_start <= heldout_end:
        raise ValueError("require heldout_start < lockbox_start <= heldout_end")
    tuning_count = int(((prices.index >= heldout_start) & (prices.index < lockbox_start)).sum())
    lockbox_count = int(((prices.index >= lockbox_start) & (prices.index <= heldout_end)).sum())
    if tuning_count < 2 or lockbox_count < 2:
        raise ValueError("Both tuning and lockbox windows require at least two observations")


def _portfolio_series(
    pair_values: pd.DataFrame,
    pair_sector: dict[str, str],
) -> dict[str, pd.Series]:
    columns = [column for column in pair_values.columns if column in pair_sector]
    if not columns:
        return {}
    working = pair_values[columns].fillna(0.0)
    equal_weight = working.mean(axis=1)
    sector_values: dict[str, pd.Series] = {}
    for sector in sorted({pair_sector[pair] for pair in columns}):
        names = [pair for pair in columns if pair_sector[pair] == sector]
        sector_values[sector] = working[names].mean(axis=1)
    sector_balanced = pd.DataFrame(sector_values).mean(axis=1)
    return {"equal_weight": equal_weight, "sector_balanced": sector_balanced}


def _window_summary(
    parameter_set: ParameterSet,
    split: str,
    pair_returns: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    end_inclusive: bool,
) -> pd.DataFrame:
    return_mask = pair_returns.index >= start
    return_mask &= pair_returns.index <= end if end_inclusive else pair_returns.index < end
    window_returns = pair_returns.loc[return_mask].fillna(0.0)
    if window_returns.empty:
        raise ValueError(f"No pair returns in {split} window")
    pair_sector = (
        selected_pairs.drop_duplicates("pair").set_index("pair")["sector"].astype(str).to_dict()
    )
    columns = [column for column in window_returns.columns if column in pair_sector]
    window_returns = window_returns[columns]
    portfolio_returns = _portfolio_series(window_returns, pair_sector)
    if not portfolio_returns:
        raise ValueError(f"No successful pairs in {split} window")

    pair_cagrs = pd.Series(
        {pair: performance_metrics(window_returns[pair])["cagr"] for pair in columns},
        dtype=float,
    )
    positive_pair_fraction = float((pair_cagrs > 0).mean()) if len(pair_cagrs) else math.nan

    pair_trade_counts = pd.Series(0, index=columns, dtype=int)
    if not trades.empty and {"pair", "entry_date"}.issubset(trades.columns):
        entry_dates = pd.to_datetime(trades["entry_date"])
        trade_mask = entry_dates >= start
        trade_mask &= entry_dates <= end if end_inclusive else entry_dates < end
        counts = trades.loc[trade_mask, "pair"].value_counts()
        pair_trade_counts = counts.reindex(columns, fill_value=0).astype(int)

    turnover_frame = pd.DataFrame(0.0, index=window_returns.index, columns=columns)
    if not signals.empty and {"date", "pair", "turnover"}.issubset(signals.columns):
        working = signals[["date", "pair", "turnover"]].copy()
        working["date"] = pd.to_datetime(working["date"])
        signal_mask = working["date"] >= start
        signal_mask &= working["date"] <= end if end_inclusive else working["date"] < end
        pivot = working.loc[signal_mask].pivot_table(
            index="date", columns="pair", values="turnover", aggfunc="sum"
        )
        turnover_frame = pivot.reindex(index=window_returns.index, columns=columns).fillna(0.0)
    portfolio_turnover = _portfolio_series(turnover_frame, pair_sector)

    common = {
        **asdict(parameter_set),
        "split": split,
        "window_start": str(window_returns.index.min().date()),
        "window_end": str(window_returns.index.max().date()),
        "observations": int(len(window_returns)),
        "pair_count": int(len(columns)),
        "sector_count": int(len({pair_sector[pair] for pair in columns})),
        "total_trades": int(pair_trade_counts.sum()),
        "median_trades_per_pair": float(pair_trade_counts.median()),
        "mean_trades_per_pair": float(pair_trade_counts.mean()),
        "positive_pair_fraction": positive_pair_fraction,
    }
    rows: list[dict[str, Any]] = []
    for portfolio, returns in portfolio_returns.items():
        metrics = performance_metrics(returns)
        turnover = portfolio_turnover[portfolio]
        rows.append(
            {
                **common,
                "portfolio": portfolio,
                **metrics,
                "annual_turnover": float(turnover.mean() * 252.0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_parameter_set(
    selected_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    heldout_start: pd.Timestamp | str,
    base_config: StrategyConfig,
    parameter_set: ParameterSet,
    windows: Iterable[tuple[str, pd.Timestamp | str, pd.Timestamp | str, bool]],
    *,
    require_all_pairs: bool = True,
) -> GridEvaluation:
    """Backtest and summarize one parameter set over explicitly supplied windows.

    Each window tuple is ``(name, start, end, end_inclusive)``.  Prices passed to
    this function should already be clipped at the desired information boundary.
    """

    strategy_config = strategy_config_for_parameter_set(base_config, parameter_set)
    pair_metrics, signals, trades, pair_returns = backtest_selected_pairs(
        selected_pairs=selected_pairs,
        prices=prices,
        test_start=pd.Timestamp(heldout_start),
        config=strategy_config,
    )
    successful = set(pair_returns.columns)
    expected = set(selected_pairs["pair"].astype(str))
    if require_all_pairs and successful != expected:
        failures = pair_metrics.loc[
            pair_metrics.get("backtest_status", pd.Series(dtype=str)).ne("ok")
        ]
        details = "; ".join(
            f"{row.get('pair')}: {row.get('error', 'failed')}"
            for row in failures.to_dict(orient="records")[:5]
        )
        raise RuntimeError(
            f"{len(expected - successful)} of {len(expected)} fixed pairs failed for "
            f"{parameter_set.config_id}. {details}"
        )
    summaries = []
    for name, start, end, end_inclusive in windows:
        summaries.append(
            _window_summary(
                parameter_set=parameter_set,
                split=str(name),
                pair_returns=pair_returns,
                signals=signals,
                trades=trades,
                selected_pairs=selected_pairs,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                end_inclusive=bool(end_inclusive),
            )
        )
    return GridEvaluation(
        parameter_set=parameter_set,
        strategy_config=strategy_config,
        summary=pd.concat(summaries, ignore_index=True),
        pair_metrics=pair_metrics,
        signals=signals,
        trades=trades,
        pair_returns=pair_returns,
    )


def select_best_parameters(
    summary: pd.DataFrame,
    *,
    portfolio: str = "sector_balanced",
    objective: str = "sharpe",
    minimum_total_trades: int = 0,
    minimum_median_trades_per_pair: float = 0.0,
) -> pd.DataFrame:
    """Rank tuning rows only; lockbox columns are deliberately never consulted."""

    if objective not in {"sharpe", "cagr", "sortino", "calmar"}:
        raise ValueError("objective must be one of sharpe, cagr, sortino, or calmar")
    required = {
        "split",
        "portfolio",
        "config_id",
        objective,
        "cagr",
        "max_drawdown",
        "annual_turnover",
        "total_trades",
        "median_trades_per_pair",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing columns: {', '.join(missing)}")
    candidates = summary.loc[
        summary["split"].eq("tuning")
        & summary["portfolio"].eq(portfolio)
        & summary["total_trades"].ge(minimum_total_trades)
        & summary["median_trades_per_pair"].ge(minimum_median_trades_per_pair)
    ].copy()
    candidates = candidates.loc[np.isfinite(pd.to_numeric(candidates[objective], errors="coerce"))]
    if candidates.empty:
        raise ValueError("No tuning candidates satisfy the ranking constraints")
    candidates = candidates.sort_values(
        [objective, "cagr", "max_drawdown", "annual_turnover", "config_id"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def _comparison_table(
    summary: pd.DataFrame,
    baseline_id: str,
    best_id: str,
) -> pd.DataFrame:
    keys = ["split", "portfolio"]
    metrics = [
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "total_trades",
        "median_trades_per_pair",
        "positive_pair_fraction",
    ]
    baseline = summary.loc[summary["config_id"].eq(baseline_id), keys + metrics].drop_duplicates(keys)
    best = summary.loc[summary["config_id"].eq(best_id), keys + metrics].drop_duplicates(keys)
    comparison = baseline.merge(best, on=keys, suffixes=("_baseline", "_best"), how="inner")
    for metric in metrics:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_best"] - comparison[f"{metric}_baseline"]
        )
    comparison.insert(2, "baseline_config_id", baseline_id)
    comparison.insert(3, "best_config_id", best_id)
    return comparison


def run_parameter_grid(
    selected_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    heldout_start: pd.Timestamp | str,
    lockbox_start: pd.Timestamp | str,
    heldout_end: pd.Timestamp | str,
    base_config: StrategyConfig,
    spec: GridSearchSpec,
    *,
    objective: str = "sharpe",
    ranking_portfolio: str = "sector_balanced",
    minimum_total_trades: int = 0,
    minimum_median_trades_per_pair: float = 0.0,
    require_all_pairs: bool = True,
    continue_on_error: bool = False,
    fast_tuning: bool = True,
    progress: ProgressCallback | None = None,
) -> GridSearchResult:
    """Tune on the early held-out sample and open the lockbox only once.

    All grid cells are evaluated using prices strictly before ``lockbox_start``.
    After ranking, only the tuning winner and baseline are rerun through
    ``heldout_end`` and summarized on the lockbox.  Transaction and borrow costs
    come directly from ``base_config`` and therefore enter every objective value.
    """

    heldout_start = pd.Timestamp(heldout_start)
    lockbox_start = pd.Timestamp(lockbox_start)
    heldout_end = pd.Timestamp(heldout_end)
    prices = prices.sort_index()
    _validate_inputs(selected_pairs, prices, heldout_start, lockbox_start, heldout_end)
    parameter_sets = build_parameter_sets(spec, base_config, include_baseline=True)
    baseline_parameter = _baseline_parameter(base_config)
    parameters_by_id = {row.config_id: row for row in parameter_sets}

    tuning_prices = prices.loc[prices.index < lockbox_start]
    fast_evaluator = (
        _FastTuningEvaluator(
            selected_pairs,
            tuning_prices,
            heldout_start,
            lockbox_start,
            base_config,
        )
        if fast_tuning
        else None
    )
    tuning_rows: list[pd.DataFrame] = []
    failure_rows: list[dict[str, str]] = []
    for completed, parameter_set in enumerate(parameter_sets, start=1):
        try:
            if fast_evaluator is not None:
                tuning_summary_for_parameter = fast_evaluator.evaluate(parameter_set)
            else:
                evaluation = evaluate_parameter_set(
                    selected_pairs=selected_pairs,
                    prices=tuning_prices,
                    heldout_start=heldout_start,
                    base_config=base_config,
                    parameter_set=parameter_set,
                    windows=(("tuning", heldout_start, lockbox_start, False),),
                    require_all_pairs=require_all_pairs,
                )
                tuning_summary_for_parameter = evaluation.summary
        except Exception as exc:
            if not continue_on_error:
                raise
            failure_rows.append(
                {
                    "config_id": parameter_set.config_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            tuning_rows.append(tuning_summary_for_parameter)
        if progress is not None:
            progress(completed, len(parameter_sets), parameter_set)
    if not tuning_rows:
        raise RuntimeError("Every parameter combination failed")
    tuning_summary = pd.concat(tuning_rows, ignore_index=True)
    ranking = select_best_parameters(
        tuning_summary,
        portfolio=ranking_portfolio,
        objective=objective,
        minimum_total_trades=minimum_total_trades,
        minimum_median_trades_per_pair=minimum_median_trades_per_pair,
    )
    best_id = str(ranking.iloc[0]["config_id"])
    best_parameter = parameters_by_id[best_id]

    full_prices = prices.loc[prices.index <= heldout_end]
    reporting_windows = (
        ("tuning", heldout_start, lockbox_start, False),
        ("lockbox", lockbox_start, heldout_end, True),
    )
    best_evaluation = evaluate_parameter_set(
        selected_pairs,
        full_prices,
        heldout_start,
        base_config,
        best_parameter,
        reporting_windows,
        require_all_pairs=require_all_pairs,
    )
    if best_id == baseline_parameter.config_id:
        baseline_evaluation = best_evaluation
    else:
        baseline_evaluation = evaluate_parameter_set(
            selected_pairs,
            full_prices,
            heldout_start,
            base_config,
            baseline_parameter,
            reporting_windows,
            require_all_pairs=require_all_pairs,
        )

    selected_ids = {best_id, baseline_parameter.config_id}
    reporting_summary = pd.concat(
        [best_evaluation.summary, baseline_evaluation.summary], ignore_index=True
    ).drop_duplicates(["config_id", "split", "portfolio"])
    # Preserve every tuning row from the grid, but publish lockbox rows only for
    # the predeclared baseline and the tuning-selected winner.
    summary = pd.concat(
        [
            tuning_summary,
            reporting_summary.loc[
                reporting_summary["split"].eq("lockbox")
                & reporting_summary["config_id"].isin(selected_ids)
            ],
        ],
        ignore_index=True,
    ).sort_values(["split", "config_id", "portfolio"], kind="mergesort")
    comparison = _comparison_table(summary, baseline_parameter.config_id, best_id)
    return GridSearchResult(
        summary=summary.reset_index(drop=True),
        ranking=ranking,
        comparison=comparison,
        failures=pd.DataFrame(failure_rows, columns=["config_id", "error_type", "error"]),
        best_parameters=asdict(best_parameter),
        best_config=best_evaluation.strategy_config,
        baseline_config=base_config,
        best_evaluation=best_evaluation,
        baseline_evaluation=baseline_evaluation,
    )
